"""
Phase 3 — collaborative similarity (item-item, TF-IDF-weighted cosine).

Two games are "collaboratively similar" here if the SAME users play both. Each
game becomes a vector over the 150,000 sampled users (1 if that user has any
interaction with the game, 0 otherwise) and we measure cosine between those
vectors.

Raw binary cosine has a well-known popularity-bias problem: a blockbuster
co-occurs with everything, so it looks "similar" to everything. The standard
fix is TF-IDF weighting BEFORE cosine — items appearing in many users get a
low IDF, which damps their contribution to every pair they're in. We use
`sklearn.feature_extraction.text.TfidfTransformer` on the user×item matrix:
    - rows are users (documents),
    - columns are items (terms),
    - IDF down-weights items that appear in many users (i.e. popular games).
Then we transpose to item×user, L2-normalise each item row, and compute cosine.

Pipeline:
    interactions.npz    (users × items, binary, CSR)
      -> TfidfTransformer(norm=None) -> popular items get low IDF column-wise
      -> .T                          -> items × users
      -> L2-normalise rows           -> unit item vectors (dot == cosine)
      -> items @ items.T             -> 6000×6000 cosine matrix in [0, 1]

`collab_scores(game_id)` returns the row of this matrix aligned to
artifacts/game_index.json — the SAME ordering as `content_scores` in Phase 2.
Phase 4 blends the two row-for-row with one α, no rescaling needed.

Cold-start: a game with too few interactions can't produce a meaningful CF
neighbour set — its column vector is so sparse that its cosine to anything
is dominated by 1–2 random co-players. We flag every item below
LOW_SIGNAL_THRESHOLD (50 raw interactions) and Phase 4 routes those seeds to
the content / popularity fallback.

Locked design: TF-IDF-weighted item-item cosine, period. No matrix
factorization, no ALS, no neural CF. The hybrid is two cosine arms blended
by one α — keep it that way.
"""

from __future__ import annotations

import json
from difflib import get_close_matches
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

INTERACTIONS_NPZ = ART / "interactions.npz"
GAME_INDEX_JSON = ART / "game_index.json"
GAMES_JSON = ART / "games.json"
ID_NAME_JSON = ART / "id_name_lookup.json"

COLLAB_SIMS_NPZ = ART / "collab_sims.npz"
COLLAB_NEIGHBORS_JSON = ART / "collab_neighbors.json"
CF_LOW_SIGNAL_JSON = ART / "cf_low_signal.json"

# A game with fewer than this many user interactions in our sampled matrix
# can't generate a meaningful CF neighbour set. 50 is a soft floor: high
# enough that a few co-players don't dominate, low enough that we keep
# coverage on niche games. Phase 4 routes flagged seeds to the fallback.
LOW_SIGNAL_THRESHOLD = 50


# --- core build --------------------------------------------------------------
def build_collab_sims(M: sparse.csr_matrix) -> np.ndarray:
    """
    Build the 6000×6000 item-item cosine matrix from the user×item interactions.

    Why TfidfTransformer with norm=None:
    - rows of M are users (documents), columns are items (terms).
    - smooth_idf=True (default) gives idf_i = ln((1 + n_users) / (1 + df_i)) + 1,
      so very popular items (high df) get a small IDF — the popularity fix.
    - norm=None: we do our own L2 normalisation AFTER transposing, on the item
      vectors. Sklearn's default norm='l2' would L2 each USER row, which is not
      what we want.
    - sublinear_tf=False: input is binary (TF ∈ {0, 1}), 1 + log(1) = 1, so the
      sublinear flag has no effect here — leave it off for clarity.

    After transposing + row-normalising, item vectors are unit length, so the
    dot product IS cosine. Output range [0, 1] matches the content arm.
    """
    transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=True)
    M_w = transformer.fit_transform(M)                       # users × items, IDF-weighted
    X_items = normalize(M_w.T.tocsr(), norm="l2", axis=1)    # items × users, unit rows
    # 6000×6000 result; toarray() ~144 MB float32, fine on a laptop.
    sims = (X_items @ X_items.T).toarray().astype(np.float32, copy=False)
    np.fill_diagonal(sims, 0.0)
    # Defensive clamp — non-negative inputs should keep us in [0, 1], but
    # float roundoff occasionally produces ~-1e-7. Keep the contract clean.
    np.clip(sims, 0.0, 1.0, out=sims)
    return sims


# --- module-level singletons (loaded on first use) ---------------------------
_S: np.ndarray | None = None
_GAME_INDEX: list[int] | None = None
_ID_TO_ROW: dict[int, int] | None = None
_ID_TO_NAME: dict[int, str] | None = None
_NAME_TO_ID: dict[str, int] | None = None
_INTERACTION_COUNTS: np.ndarray | None = None
_LOW_SIGNAL_IDS: set[int] | None = None


def _ensure_loaded() -> None:
    """Lazy-load sims + counts + lookups. Falls back to a re-build if needed."""
    global _S, _GAME_INDEX, _ID_TO_ROW, _ID_TO_NAME, _NAME_TO_ID
    global _INTERACTION_COUNTS, _LOW_SIGNAL_IDS
    if _S is not None:
        return

    _GAME_INDEX = json.loads(GAME_INDEX_JSON.read_text())
    _ID_TO_ROW = {gid: i for i, gid in enumerate(_GAME_INDEX)}

    lookup = json.loads(ID_NAME_JSON.read_text())
    _ID_TO_NAME = {int(k): v for k, v in lookup["id_to_name"].items()}
    _NAME_TO_ID = {name.lower(): int(gid) for name, gid in lookup["name_to_id"].items()}

    # Per-item interaction count — measured on the RAW binary matrix, since
    # popularity for cold-start purposes is "how many users touched it",
    # not "how much IDF-weighted mass".
    M = sparse.load_npz(INTERACTIONS_NPZ).tocsr()
    counts = np.asarray(M.sum(axis=0)).ravel().astype(np.int64)
    _INTERACTION_COUNTS = counts
    low_idx = np.where(counts < LOW_SIGNAL_THRESHOLD)[0]
    _LOW_SIGNAL_IDS = {int(_GAME_INDEX[i]) for i in low_idx}

    if COLLAB_SIMS_NPZ.exists():
        with np.load(COLLAB_SIMS_NPZ) as npz:
            _S = npz["sims"].astype(np.float32, copy=False)
    else:
        _S = build_collab_sims(M)


# --- public API --------------------------------------------------------------
def collab_scores(game_id: int) -> np.ndarray:
    """
    Length-n_games cosine vector aligned to artifacts/game_index.json. Self = 0.
    Values in [0, 1] (same scale as content_scores). Phase 4 blends row-for-row.
    """
    _ensure_loaded()
    assert _S is not None and _ID_TO_ROW is not None
    if game_id not in _ID_TO_ROW:
        raise KeyError(f"game_id {game_id} is not in the catalog")
    # Return a copy so callers can't accidentally mutate the cached matrix.
    return _S[_ID_TO_ROW[game_id]].copy()


def collab_similarity(game_id: int, k: int = 10) -> list[tuple[int, float]]:
    """Top-k CF neighbours of `game_id` as (game_id, cosine), descending."""
    _ensure_loaded()
    assert _GAME_INDEX is not None
    sims = collab_scores(game_id)
    k = min(k, len(sims) - 1)
    # argpartition picks the k largest indices in O(n) without sorting the
    # whole 6000-long vector; we then sort just those k.
    top_idx = np.argpartition(-sims, k)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(int(_GAME_INDEX[i]), float(sims[i])) for i in top_idx]


def cf_low_signal_ids() -> set[int]:
    """Set of game_ids with raw interaction count < LOW_SIGNAL_THRESHOLD."""
    _ensure_loaded()
    assert _LOW_SIGNAL_IDS is not None
    return _LOW_SIGNAL_IDS


def is_cf_low_signal(game_id: int) -> bool:
    return game_id in cf_low_signal_ids()


def interaction_count(game_id: int) -> int:
    """Raw number of sampled users that interacted with `game_id`."""
    _ensure_loaded()
    assert _INTERACTION_COUNTS is not None and _ID_TO_ROW is not None
    if game_id not in _ID_TO_ROW:
        raise KeyError(f"game_id {game_id} is not in the catalog")
    return int(_INTERACTION_COUNTS[_ID_TO_ROW[game_id]])


def _resolve_name(query: str) -> int | None:
    """Fuzzy name -> game_id (used only by the sanity-check script below)."""
    _ensure_loaded()
    assert _NAME_TO_ID is not None
    q = (query or "").strip().lower()
    if not q:
        return None
    if q in _NAME_TO_ID:
        return _NAME_TO_ID[q]
    m = get_close_matches(q, list(_NAME_TO_ID.keys()), n=1, cutoff=0.6)
    return _NAME_TO_ID[m[0]] if m else None


# --- artifact build + sanity check (run as a script) -------------------------
def _save_neighbors(sims: np.ndarray, game_index: list[int], id_to_name: dict[int, str], k: int = 20) -> None:
    out: dict[str, list[dict]] = {}
    n = sims.shape[0]
    for i in range(n):
        row = sims[i]
        kk = min(k, n - 1)
        top = np.argpartition(-row, kk)[:kk]
        top = top[np.argsort(-row[top])]
        out[str(game_index[i])] = [
            {
                "game_id": int(game_index[j]),
                "name": id_to_name.get(int(game_index[j]), ""),
                "score": float(row[j]),
            }
            for j in top
        ]
    COLLAB_NEIGHBORS_JSON.write_text(json.dumps(out))


def _save_low_signal(counts: np.ndarray, game_index: list[int], id_to_name: dict[int, str]) -> list[int]:
    low_idx = np.where(counts < LOW_SIGNAL_THRESHOLD)[0]
    low_ids = [int(game_index[i]) for i in low_idx]
    examples = [
        {"game_id": int(game_index[i]),
         "name": id_to_name.get(int(game_index[i]), ""),
         "n": int(counts[i])}
        for i in low_idx[:20]
    ]
    CF_LOW_SIGNAL_JSON.write_text(json.dumps({
        "threshold": LOW_SIGNAL_THRESHOLD,
        "low_signal_count": len(low_ids),
        "low_signal_ids": low_ids,
        "examples": examples,
    }, indent=2))
    return low_ids


def _print_seed_block(seed: str, k: int = 10) -> None:
    gid = _resolve_name(seed)
    if gid is None:
        print(f"  [{seed!r}] not found")
        return
    assert _ID_TO_NAME is not None
    print(f"  Seed: {_ID_TO_NAME.get(gid, str(gid))}  "
          f"(game_id={gid}, interactions={interaction_count(gid):,}, "
          f"low_signal={is_cf_low_signal(gid)})")
    for nid, score in collab_similarity(gid, k=k):
        print(f"    {score:.3f}  [n={interaction_count(nid):>6,}]  "
              f"{_ID_TO_NAME.get(nid, str(nid))}")


def _anti_popularity_test(seeds: list[str], k: int = 10) -> bool:
    """
    Decisive test: pick `seeds` distinct mid-popularity games and check that
    their top-k neighbour sets DO NOT collapse to "the same 10 blockbusters".

    Measure:
      - |union of all top-k sets|: should approach n_seeds * k if neighbour
        sets are mostly distinct; collapses to ~k if they all return the same
        blockbusters.
      - mean pairwise Jaccard overlap: low (<= 0.3) means the seeds live in
        different neighbourhoods.

    Pass criterion (deliberately strict): union >= 1.5*k AND avg Jaccard <= 0.5.
    """
    all_neighbours: list[set[int]] = []
    assert _ID_TO_NAME is not None
    for seed in seeds:
        gid = _resolve_name(seed)
        if gid is None:
            print(f"  [{seed!r}] not found — skipping")
            continue
        nset = {nid for nid, _ in collab_similarity(gid, k=k)}
        all_neighbours.append(nset)
        names = sorted(_ID_TO_NAME.get(n, str(n)) for n in nset)
        print(f"  {_ID_TO_NAME.get(gid, str(gid))}  ->  {names}")
    if not all_neighbours:
        print("  no seeds resolved")
        return False
    union = set().union(*all_neighbours)
    n = len(all_neighbours)
    pair_overlap = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = all_neighbours[i], all_neighbours[j]
            if a or b:
                pair_overlap += len(a & b) / len(a | b)
                pair_count += 1
    avg_jaccard = pair_overlap / pair_count if pair_count else 0.0
    print(f"\n  union over {n} seeds' top-{k}: {len(union)} distinct games "
          f"(max possible = {n*k})")
    print(f"  mean pairwise Jaccard overlap: {avg_jaccard:.3f}")
    return len(union) >= int(1.5 * k) and avg_jaccard <= 0.5


def main() -> None:
    print("Loading user×item interactions ...")
    M = sparse.load_npz(INTERACTIONS_NPZ).tocsr()
    print(f"  shape: {M.shape}  nnz: {M.nnz:,}  dtype: {M.dtype}")

    print("\nBuilding TF-IDF-weighted item-item cosine ...")
    sims = build_collab_sims(M)
    print(f"  sims shape: {sims.shape}  dtype: {sims.dtype}")
    print(f"  diagonal: max={sims.diagonal().max():.4f} (should be 0)")
    print(f"  off-diag: max={sims.max():.4f}  mean={sims.mean():.6f}")

    print("\nSaving artifacts ...")
    np.savez_compressed(COLLAB_SIMS_NPZ, sims=sims)
    print(f"  wrote {COLLAB_SIMS_NPZ.name}  "
          f"({COLLAB_SIMS_NPZ.stat().st_size / 1e6:.1f} MB compressed)")

    game_index = json.loads(GAME_INDEX_JSON.read_text())
    lookup = json.loads(ID_NAME_JSON.read_text())
    id_to_name = {int(k): v for k, v in lookup["id_to_name"].items()}

    print("Computing top-20 CF neighbours per game ...")
    _save_neighbors(sims, game_index, id_to_name, k=20)
    print(f"  wrote {COLLAB_NEIGHBORS_JSON.name}")

    print("Computing CF-low-signal flag set ...")
    counts = np.asarray(M.sum(axis=0)).ravel().astype(np.int64)
    low_ids = _save_low_signal(counts, game_index, id_to_name)
    print(f"  wrote {CF_LOW_SIGNAL_JSON.name}  ({len(low_ids)} flagged games)")
    print(f"  interaction count range: min={counts.min()} max={counts.max():,}")
    print(f"  fraction below threshold ({LOW_SIGNAL_THRESHOLD}): "
          f"{len(low_ids)}/{len(counts)} ({100*len(low_ids)/len(counts):.2f}%)")

    # Force-reload from disk so the sanity checks exercise the load path.
    global _S
    _S = None
    _ensure_loaded()

    print("\n=== Sanity check: CF neighbours (top 10) — judge BEHAVIOUR, not theme ===")
    for seed in [
        "Counter-Strike",
        "Stardew Valley",
        "The Witcher 3: Wild Hunt",
        "Dota 2",
        "Hades",
        "Lost Ark",
    ]:
        _print_seed_block(seed, k=10)
        print()

    # Are the 5 zero-content F2P MMOs CF-low-signal? Expectation per the task
    # prompt: probably NOT, because they're popular MMOs. That's the whole
    # point of hybridising — the content arm and CF arm have weak spots on
    # different sets, so the blend covers both.
    print("=== Zero-content F2P MMOs: CF coverage check ===")
    f2p_mmos = ["Lost Ark", "Conqueror's Blade", "MapleStory", "Vindictus", "Mabinogi"]
    assert _ID_TO_NAME is not None
    for seed in f2p_mmos:
        gid = _resolve_name(seed)
        if gid is None:
            print(f"  [{seed!r}] not found")
            continue
        print(f"  {_ID_TO_NAME.get(gid, str(gid))}: "
              f"n={interaction_count(gid):,}, low_signal={is_cf_low_signal(gid)}")

    print("\n=== Anti-popularity test (5 DIFFERENT mid-popularity seeds) ===")
    # Chosen because each has interaction support well above the low-signal
    # threshold but well below the blockbuster tier — if the popularity fix
    # failed, all of them would return roughly the same blockbuster cluster.
    mid_seeds = [
        "FTL: Faster Than Light",
        "Slay the Spire",
        "Don't Starve Together",
        "Subnautica",
        "Cuphead",
    ]
    passed = _anti_popularity_test(mid_seeds, k=10)
    print(f"\n  RESULT: anti-popularity test "
          f"{'PASSED — neighbour sets are distinct' if passed else 'FAILED — neighbour sets collapsed to popular games'}")


if __name__ == "__main__":
    main()
