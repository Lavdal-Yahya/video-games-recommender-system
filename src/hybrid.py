"""
Phase 4 — weighted hybrid recommender.

Burke's *weighted* hybrid: blend the content arm (Phase 2) and the
item-item CF arm (Phase 3) with a single mixing weight α:

    score(seed, other) = α · content_cosine(seed, other) + (1 - α) · cf_cosine(seed, other)

Both arms produce length-6000 cosine vectors aligned to artifacts/game_index.json
(self = 0, values in [0, 1]) — so the blend is a straight elementwise sum, no
rescaling needed. That's the whole "model".

Per-seed routing (the important part — ~34% of the catalog is CF-low-signal):
- seed has a zero content vector (the 5 F2P MMOs Steam appdetails refused)
    -> CF-only (effective α = 0)
- seed is CF-low-signal (< 50 interactions, Phase 3 flag)
    -> content-only (effective α = 1)
- seed has neither arm
    -> popularity fallback (top by n_interactions, filtered)
- otherwise
    -> normal α-blend

The chosen path is returned alongside the results so the demo / API can show
"why" — useful for debugging and for the report's cold-start discussion.

Filters apply AFTER scoring, BEFORE top-k:
- max_price: drop games above the price ceiling.
- tags / genres: keep games whose tag set intersects the requested tags
  (compared in the underscored form: "Open World" == "open_world").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.collaborative import (
    cf_low_signal_ids,
    collab_scores,
    interaction_count,
    is_cf_low_signal,
)
from src.content import content_scores, find_game_id


# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

GAMES_JSON = ART / "games.json"
GAME_INDEX_JSON = ART / "game_index.json"
ID_NAME_JSON = ART / "id_name_lookup.json"
ALPHA_SWEEP_JSON = ART / "alpha_sweep.json"


# --- tunable -----------------------------------------------------------------
# DEFAULT_ALPHA = 0.25 is PROVISIONAL — chosen so the blend path is a genuine
# weighted hybrid, not a CF passthrough. The LOO sweep (artifacts/alpha_sweep.json,
# with the per-seed max-rescale in _blended_scores) still peaks at α=0 on
# precision@10, but that protocol predicts a held-out interaction from a user's
# other interactions — co-play prediction, which is CF's home turf — so it
# structurally favours pure CF. α=0.25 costs ~4 hits/1000 versus α=0 and gives
# us a real blend whose value should show up on coverage / diversity, to be
# confirmed in Phase 8. /recommend takes alpha as a parameter, so this default
# is cheap to revisit once the diversity numbers are in.
DEFAULT_ALPHA: float = 0.25


# --- module singletons -------------------------------------------------------
_GAME_INDEX: list[int] | None = None
_ID_TO_ROW: dict[int, int] | None = None
_GAMES_BY_ID: dict[int, dict] | None = None
_ID_TO_NAME: dict[int, str] | None = None
_CONTENT_NORMS: np.ndarray | None = None  # per-row L2 norm of TF-IDF (0 -> zero vector)


def _ensure_loaded() -> None:
    """Load the catalog index + games.json once."""
    global _GAME_INDEX, _ID_TO_ROW, _GAMES_BY_ID, _ID_TO_NAME
    if _GAME_INDEX is not None:
        return
    _GAME_INDEX = json.loads(GAME_INDEX_JSON.read_text())
    _ID_TO_ROW = {gid: i for i, gid in enumerate(_GAME_INDEX)}
    games = json.loads(GAMES_JSON.read_text())
    _GAMES_BY_ID = {g["game_id"]: g for g in games}
    lookup = json.loads(ID_NAME_JSON.read_text())
    _ID_TO_NAME = {int(k): v for k, v in lookup["id_to_name"].items()}


def _content_norms() -> np.ndarray:
    """Per-row L2 norm of the TF-IDF matrix. A norm of 0 == zero content vector."""
    global _CONTENT_NORMS
    if _CONTENT_NORMS is not None:
        return _CONTENT_NORMS
    # Import lazily so we don't pay the TF-IDF load cost unless we need it.
    from src import content as content_mod
    content_mod._ensure_loaded()
    X = content_mod._X
    assert X is not None
    # sqrt(sum(X^2)) per row. TfidfVectorizer L2-normalises by default, so
    # rows are either 0 (empty soup) or 1. Compute defensively anyway.
    sq = X.multiply(X).sum(axis=1)
    _CONTENT_NORMS = np.sqrt(np.asarray(sq).ravel()).astype(np.float32)
    return _CONTENT_NORMS


def _has_content(game_id: int) -> bool:
    _ensure_loaded()
    assert _ID_TO_ROW is not None
    return bool(_content_norms()[_ID_TO_ROW[game_id]] > 0.0)


# --- tag/filter helpers ------------------------------------------------------
def _tag_token(t: str) -> str:
    """Match content.py's tag normalisation: 'Open World' -> 'open_world'."""
    return t.strip().lower().replace(" ", "_")


def _game_tag_set(game: dict) -> set[str]:
    return {_tag_token(t) for t in game.get("tags", []) if t and t.strip()}


def _passes_filters(game: dict, filters: dict[str, Any] | None) -> bool:
    """Apply per-game filters. Unknown keys are ignored gracefully."""
    if not filters:
        return True
    max_price = filters.get("max_price")
    if max_price is not None:
        price = game.get("price")
        if price is not None and float(price) > float(max_price):
            return False
    # Accept "tags" or "genres" (in this dataset genres is empty so both go
    # through the tag intersection check).
    wanted: list[str] = []
    for key in ("tags", "genres"):
        v = filters.get(key)
        if v:
            wanted.extend(_tag_token(x) for x in v)
    if wanted:
        if not _game_tag_set(game).intersection(wanted):
            return False
    return True


# --- core blend --------------------------------------------------------------
def _blended_scores(seed_id: int, alpha: float) -> np.ndarray:
    """
    Straight elementwise blend (no routing). Used by the alpha sweep so the
    endpoints α=0 and α=1 are genuine ablation endpoints (CF-only / content-only).

    Per-seed max-rescale before blending: each arm's score vector is divided
    by its own max so both arms live in [0,1] *for this seed*. Without it the
    arms are on incomparable scales (content cosines run ~3-4× CF cosines on
    average), so α stops interpolating — at nominal α=0.25 the effective
    content weight is already ~78%. Rescale → α actually interpolates. Swap
    `.max()` for `mean(nonzero)` to use mean-scaling instead.
    """
    cs = content_scores(seed_id).astype(np.float32, copy=True)
    fs = collab_scores(seed_id).astype(np.float32, copy=True)
    c_max = float(cs.max())
    f_max = float(fs.max())
    if c_max > 0.0:
        cs /= c_max
    if f_max > 0.0:
        fs /= f_max
    out = alpha * cs + (1.0 - alpha) * fs
    _ensure_loaded()
    assert _ID_TO_ROW is not None
    out[_ID_TO_ROW[seed_id]] = 0.0  # belt-and-braces; both arms already zero self
    return out


def _topk_with_filter(
    scores: np.ndarray,
    k: int,
    filters: dict[str, Any] | None,
    exclude_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    """
    Mask out filtered-out games (and any exclude_ids), then take top-k.

    We mask by setting their score to -inf rather than slicing, so the row
    stays aligned to game_index.
    """
    _ensure_loaded()
    assert _GAME_INDEX is not None and _GAMES_BY_ID is not None
    masked = scores.astype(np.float32, copy=True)
    if filters or exclude_ids:
        ex = exclude_ids or set()
        for i, gid in enumerate(_GAME_INDEX):
            if gid in ex:
                masked[i] = -np.inf
                continue
            if filters and not _passes_filters(_GAMES_BY_ID[gid], filters):
                masked[i] = -np.inf
    # argpartition for the top-k indices without sorting the whole vector
    n = len(masked)
    kk = min(k, n - 1)
    top_idx = np.argpartition(-masked, kk)[:kk]
    top_idx = top_idx[np.argsort(-masked[top_idx])]
    out: list[tuple[int, float]] = []
    for i in top_idx:
        s = float(masked[i])
        if s == -np.inf:
            continue
        out.append((int(_GAME_INDEX[i]), s))
    return out


# --- public API --------------------------------------------------------------
def recommend(
    seed: int | str,
    k: int = 10,
    alpha: float | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Hybrid recommendation, with cold-start routing.

    Args:
        seed: game_id (int) or game name (str — resolved via the Phase-2 fuzzy
              matcher); the game to find neighbours of.
        k: number of results to return.
        alpha: blend weight; defaults to DEFAULT_ALPHA. Only used when path
               == 'blend'.
        filters: dict of {max_price, tags|genres} applied after scoring.

    Returns:
        {
          "seed_id": int, "seed_name": str,
          "path": "blend" | "content_only" | "cf_only" | "popularity",
          "alpha_effective": float,   # the α actually used (0/1 for the pure arms)
          "results": [{game_id, name, score}, ...]  # length up to k
        }
    """
    _ensure_loaded()
    assert _ID_TO_NAME is not None and _GAMES_BY_ID is not None

    if alpha is None:
        alpha = DEFAULT_ALPHA

    # Resolve the seed.
    if isinstance(seed, str):
        sid = find_game_id(seed)
        if sid is None:
            raise KeyError(f"could not resolve seed name {seed!r}")
    else:
        sid = int(seed)
    if sid not in _GAMES_BY_ID:
        raise KeyError(f"game_id {sid} is not in the catalog")

    has_content = _has_content(sid)
    cf_weak = is_cf_low_signal(sid)

    # Decide the path.
    if not has_content and cf_weak:
        path = "popularity"
        eff_alpha = float("nan")
        scores = _popularity_scores()
    elif not has_content:
        path = "cf_only"
        eff_alpha = 0.0
        scores = collab_scores(sid)
    elif cf_weak:
        path = "content_only"
        eff_alpha = 1.0
        scores = content_scores(sid)
    else:
        path = "blend"
        eff_alpha = float(alpha)
        scores = _blended_scores(sid, alpha)

    results = _topk_with_filter(scores, k, filters, exclude_ids={sid})
    return {
        "seed_id": sid,
        "seed_name": _ID_TO_NAME.get(sid, str(sid)),
        "path": path,
        "alpha_effective": eff_alpha,
        "results": [
            {"game_id": gid, "name": _ID_TO_NAME.get(gid, str(gid)), "score": s}
            for gid, s in results
        ],
    }


# --- popularity fallback -----------------------------------------------------
_POP_SCORES: np.ndarray | None = None


def _popularity_scores() -> np.ndarray:
    """
    Score vector for the popularity fallback. We just use n_interactions
    aligned to game_index. Filters then trim it; top-k picks the leaders.
    """
    global _POP_SCORES
    if _POP_SCORES is not None:
        return _POP_SCORES.copy()
    _ensure_loaded()
    assert _GAME_INDEX is not None and _GAMES_BY_ID is not None
    pop = np.array(
        [float(_GAMES_BY_ID[gid].get("n_interactions", 0)) for gid in _GAME_INDEX],
        dtype=np.float32,
    )
    _POP_SCORES = pop
    return pop.copy()


def popularity_only_count() -> int:
    """
    How many catalog games would fall into the 'popularity' fallback
    (no content vector AND CF-low-signal)? Useful for the Phase 4 outputs note.
    """
    _ensure_loaded()
    assert _GAME_INDEX is not None
    norms = _content_norms()
    low = cf_low_signal_ids()
    n = 0
    for i, gid in enumerate(_GAME_INDEX):
        if norms[i] == 0.0 and gid in low:
            n += 1
    return n


def routing_counts() -> dict[str, int]:
    """How many catalog games fall into each routing bucket — for the report."""
    _ensure_loaded()
    assert _GAME_INDEX is not None
    norms = _content_norms()
    low = cf_low_signal_ids()
    counts = {"blend": 0, "content_only": 0, "cf_only": 0, "popularity": 0}
    for i, gid in enumerate(_GAME_INDEX):
        has_c = norms[i] > 0.0
        cf_w = gid in low
        if not has_c and cf_w:
            counts["popularity"] += 1
        elif not has_c:
            counts["cf_only"] += 1
        elif cf_w:
            counts["content_only"] += 1
        else:
            counts["blend"] += 1
    return counts


# --- alpha tuning (LOO-style eval) -------------------------------------------
def _build_dense_content_cosine() -> np.ndarray:
    """
    Precompute the full 6000×6000 content cosine matrix once so the alpha sweep
    can pick rows by index without redoing the sparse multiply each time.
    Matches the CF matrix's shape and ordering; ~144 MB float32, fine on a laptop.
    """
    from src import content as content_mod
    content_mod._ensure_loaded()
    X = content_mod._X
    assert X is not None
    C = (X @ X.T).toarray().astype(np.float32, copy=False)
    np.fill_diagonal(C, 0.0)
    return C


def _build_dense_cf_cosine() -> np.ndarray:
    """Load the precomputed CF cosine matrix from artifacts (it already exists)."""
    from src import collaborative as cf_mod
    cf_mod._ensure_loaded()
    assert cf_mod._S is not None
    return cf_mod._S


def _sample_eval_users(n_users: int, rng: np.random.Generator) -> list[int]:
    """Sample n_users distinct user rows with >= 2 interactions."""
    from scipy import sparse
    M = sparse.load_npz(ART / "interactions.npz").tocsr()
    counts = np.asarray(M.sum(axis=1)).ravel()
    eligible = np.where(counts >= 2)[0]
    if len(eligible) < n_users:
        n_users = len(eligible)
    return list(rng.choice(eligible, size=n_users, replace=False))


def alpha_sweep(
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    n_users: int = 1000,
    n_seeds_per_user: int = 3,
    k: int = 10,
    seed: int = 42,
) -> list[dict]:
    """
    Leave-one-out-style sweep. For each sampled user:
      1. hold out one of their interactions (the "target");
      2. pick up to `n_seeds_per_user` of their OTHER interactions as seeds
         (most-popular among the user's history — deterministic picks);
      3. for each α, sum per-seed (α·content + (1-α)·CF) score vectors, mask
         the user's known interactions (except the held-out target), take top-k;
      4. hit = 1 if target is in top-k.

    Metrics:
      - precision@k = hits / (n_trials * k)   (each top-k list has 1 relevant)
      - recall@k    = hits / n_trials         (|relevant| = 1 per user)

    Straight blend, no routing — so α=0 and α=1 are pure CF / content ablation
    endpoints. Phase 8 reuses these numbers.
    """
    from scipy import sparse

    _ensure_loaded()
    assert _GAME_INDEX is not None and _GAMES_BY_ID is not None

    print(f"  loading interactions + dense cosine matrices...")
    M = sparse.load_npz(ART / "interactions.npz").tocsr()
    C = _build_dense_content_cosine()
    F = _build_dense_cf_cosine()

    rng = np.random.default_rng(seed)
    user_rows = _sample_eval_users(n_users, rng)
    print(f"  sampled {len(user_rows)} users (seed={seed}); "
          f"using up to {n_seeds_per_user} seeds/user, k={k}")

    pop_by_row = np.array(
        [float(_GAMES_BY_ID[gid].get("n_interactions", 0)) for gid in _GAME_INDEX],
        dtype=np.float32,
    )

    # Build trials once. Each trial: (target_col, seed_cols, mask_cols).
    trials: list[tuple[int, list[int], np.ndarray]] = []
    for u in user_rows:
        cols = M.indices[M.indptr[u]:M.indptr[u + 1]]
        if len(cols) < 2:
            continue
        target = int(rng.choice(cols))
        others = np.array([c for c in cols if c != target], dtype=np.int64)
        if len(others) == 0:
            continue
        order = np.argsort(-pop_by_row[others])
        seed_cols = list(others[order][:n_seeds_per_user].astype(int))
        # Mask: all of the user's interactions EXCEPT the held-out target,
        # so seeds themselves don't win their own top-k slot.
        mask_cols = np.array([c for c in cols if c != target], dtype=np.int64)
        trials.append((target, seed_cols, mask_cols))

    print(f"  built {len(trials)} eval trials")

    results: list[dict] = []
    n_games = C.shape[0]
    for a in alphas:
        hits = 0
        n_trials = 0
        for target, seed_cols, mask_cols in trials:
            score = np.zeros(n_games, dtype=np.float32)
            for sc in seed_cols:
                # Same per-seed max-rescale as _blended_scores so the sweep
                # measures the actual blend the runtime uses.
                cv = C[sc]
                fv = F[sc]
                cm = float(cv.max())
                fm = float(fv.max())
                cv = cv / cm if cm > 0.0 else cv
                fv = fv / fm if fm > 0.0 else fv
                score += a * cv + (1.0 - a) * fv
            score[mask_cols] = -np.inf
            kk = min(k, n_games - 1)
            top = np.argpartition(-score, kk)[:kk]
            if target in top:
                hits += 1
            n_trials += 1
        p_at_k = hits / (n_trials * k) if n_trials else 0.0
        r_at_k = hits / n_trials if n_trials else 0.0
        print(f"  α={a:.2f}  hits={hits:>4}/{n_trials}  "
              f"precision@{k}={p_at_k:.4f}  recall@{k}={r_at_k:.4f}")
        results.append({
            "alpha": a,
            "hits": hits,
            "n_trials": n_trials,
            f"precision@{k}": p_at_k,
            f"recall@{k}": r_at_k,
        })
    return results


# --- script entrypoint -------------------------------------------------------
def _print_recommend(seed: str | int, **kw) -> None:
    r = recommend(seed, **kw)
    a = r["alpha_effective"]
    a_s = f"{a:.2f}" if a == a else "n/a"  # NaN check
    print(f"  Seed: {r['seed_name']}  (id={r['seed_id']})  "
          f"path={r['path']}  α_eff={a_s}")
    for row in r["results"]:
        print(f"    {row['score']:.3f}  {row['name']}")


def main() -> None:
    print("=== Phase 4 — hybrid recommender ===\n")

    print("Routing-bucket sizes over the 6000-game catalog:")
    rc = routing_counts()
    for k, v in rc.items():
        print(f"  {k:<14}  {v:>5}")
    pop_only = popularity_only_count()
    print(f"  (popularity-only count cross-check: {pop_only})")

    print("\n--- α sweep (LOO-style, seed=42) ---")
    sweep = alpha_sweep(
        alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
        n_users=1000,
        n_seeds_per_user=3,
        k=10,
        seed=42,
    )

    # Pick the best α by precision@10 (tie-break on recall@10).
    best = max(sweep, key=lambda r: (r["precision@10"], r["recall@10"]))
    chosen_alpha = float(best["alpha"])
    print(f"\n  best α by precision@10: {chosen_alpha}  "
          f"(p@10={best['precision@10']:.4f}, r@10={best['recall@10']:.4f})")

    ALPHA_SWEEP_JSON.write_text(json.dumps({
        "alphas": sweep,
        "chosen_alpha": chosen_alpha,
        "n_users": 1000,
        "n_seeds_per_user": 3,
        "k": 10,
        "seed": 42,
        "protocol": "per user: hold out 1 interaction; use up to 3 most-popular "
                    "other interactions as seeds; sum (α·content + (1-α)·CF) "
                    "score vectors; mask known interactions; check whether the "
                    "held-out target is in the top-k. Straight blend (no routing) "
                    "so α=0 and α=1 are pure CF / content ablation endpoints.",
    }, indent=2))
    print(f"  wrote {ALPHA_SWEEP_JSON.name}")

    # Patch DEFAULT_ALPHA at runtime so the sanity-check block uses the tuned
    # value. The constant in source is updated by hand after the first run.
    global DEFAULT_ALPHA
    DEFAULT_ALPHA = chosen_alpha
    print(f"\n  >> set DEFAULT_ALPHA = {DEFAULT_ALPHA}  "
          "(remember to bake into the source after the first run)")

    print("\n--- Sanity: recommend() for known seeds (tuned α) ---")
    for s in ["Stardew Valley", "The Witcher 3: Wild Hunt",
              "Counter-Strike", "Hades", "Lost Ark"]:
        _print_recommend(s, k=10)
        print()

    print("--- Ablation: same seed at α=0 / α=1 / α=tuned (blend path) ---")
    seed_for_ablate = "Hades"
    for a in [0.0, 1.0, DEFAULT_ALPHA]:
        _print_recommend(seed_for_ablate, k=10, alpha=a)
        print()

    print("--- Filtered query: Hades, max_price=20 ---")
    _print_recommend("Hades", k=10, filters={"max_price": 20})


if __name__ == "__main__":
    main()
