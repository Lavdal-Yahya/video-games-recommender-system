"""
Phase 8 — evaluation and ablation.

This module is **measurement only**. It does not change the model: it consumes
the same two cosine arms (content TF-IDF + item-item CF) and the same per-seed
max-rescale blend that `src/hybrid.py` uses at runtime, and reports numbers.

The protocol is leave-one-out (LOO) on the user-game interaction matrix, seeded
with `seed=42`:

    for each sampled user u:
      pick one of u's interactions as the held-out TARGET;
      use the rest as the user's "profile" (which seeds the recommender);
      for each alpha in the grid:
        compute a length-6000 score vector by summing, over each seed in the
          profile, alpha*content_cosine + (1-alpha)*cf_cosine — with the same
          per-seed max-rescale that lives in hybrid._blended_scores;
        mask the user's known interactions (everything except the target);
        take top-k.
      hit = 1 if the target is in top-k.

Two seed-aggregation modes are evaluated side-by-side because the choice was
not obvious from Phase 4:

    "3seed"    — use up to 3 most-popular items in the profile as seeds
                 (Phase 4's tuning protocol).
    "fullprof" — use ALL items in the profile as seeds
                 (standard item-kNN scoring).

Phase 8's STEP 1 validation compares the two on the same sample; whichever is
materially honest gets used for the headline numbers.

Coverage and diversity are derived from the top-k lists at k=10 across users:
    catalog coverage = |union of top-k items across users| / |catalog|
    intra-list diversity = mean over users of mean pairwise (1 - content_cosine)
                           within their top-k list.
The diversity metric is mildly circular (it uses the content arm to judge a
list that the content arm helped produce), but is reported with that caveat
because it is the cheapest principled diversity signal we have.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse

# Avoid importing src.hybrid here to keep this module focused on measurement.
# We reload the same artifacts hybrid.py uses, so the dense cosine matrices we
# build here match what hybrid._blended_scores would compute online.


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

GAME_INDEX_JSON = ART / "game_index.json"
INTERACTIONS_NPZ = ART / "interactions.npz"
ALPHA_SWEEP_JSON = ART / "alpha_sweep.json"
EVAL_METRICS_JSON = ART / "eval_metrics.json"


# ---------- data loading -----------------------------------------------------
def load_interactions() -> sparse.csr_matrix:
    return sparse.load_npz(INTERACTIONS_NPZ).tocsr()


def load_game_index() -> list[int]:
    return json.loads(GAME_INDEX_JSON.read_text())


def build_dense_content_cosine() -> np.ndarray:
    """Full 6000x6000 content cosine matrix, self-zeroed. ~144 MB float32."""
    from src import content as content_mod
    content_mod._ensure_loaded()
    X = content_mod._X
    assert X is not None
    C = (X @ X.T).toarray().astype(np.float32, copy=False)
    np.fill_diagonal(C, 0.0)
    return C


def load_dense_cf_cosine() -> np.ndarray:
    """The CF cosine matrix lives on disk already."""
    from src import collaborative as cf_mod
    cf_mod._ensure_loaded()
    assert cf_mod._S is not None
    return cf_mod._S


# ---------- trial construction ----------------------------------------------
def sample_users(M: sparse.csr_matrix, n_users: int, seed: int = 42) -> list[int]:
    """Sample `n_users` rows with >= 2 interactions (deterministic)."""
    counts = np.asarray(M.sum(axis=1)).ravel()
    eligible = np.where(counts >= 2)[0]
    rng = np.random.default_rng(seed)
    if len(eligible) < n_users:
        n_users = len(eligible)
    return list(rng.choice(eligible, size=n_users, replace=False))


def build_trials(
    M: sparse.csr_matrix,
    user_rows: list[int],
    pop_by_col: np.ndarray,
    seed: int = 42,
) -> list[dict]:
    """
    For each user, pick a random held-out target and record their profile
    (the other interactions). We also pre-compute the 3-seed list (top-3 by
    catalog popularity from the user's profile) so both seed-modes share a
    trial set — the only difference is which seed_cols list is used.

    Each trial is a dict so the notebook can read the fields by name:
        {
          "user": int,
          "target": int,           # column index = catalog row index
          "profile": np.ndarray,   # all columns except target (the "full profile")
          "seeds3": list[int],     # top-3 most-popular from profile (Phase 4 mode)
        }
    """
    rng = np.random.default_rng(seed)
    trials: list[dict] = []
    for u in user_rows:
        cols = M.indices[M.indptr[u]:M.indptr[u + 1]]
        if len(cols) < 2:
            continue
        target = int(rng.choice(cols))
        profile = np.array([c for c in cols if c != target], dtype=np.int64)
        if len(profile) == 0:
            continue
        order = np.argsort(-pop_by_col[profile])
        seeds3 = [int(x) for x in profile[order][:3]]
        trials.append({
            "user": int(u),
            "target": target,
            "profile": profile,
            "seeds3": seeds3,
        })
    return trials


# ---------- scoring ----------------------------------------------------------
def _per_seed_rescaled_sum(
    seed_cols: Iterable[int],
    C: np.ndarray,
    F: np.ndarray,
    alpha: float,
    n_games: int,
) -> np.ndarray:
    """
    Sum, over seeds, alpha*content_norm + (1-alpha)*cf_norm. Each arm is
    divided by its own per-seed max so the two arms live in the same [0,1]
    range for each seed before mixing — matches hybrid._blended_scores.
    """
    score = np.zeros(n_games, dtype=np.float32)
    for sc in seed_cols:
        cv = C[sc]
        fv = F[sc]
        cm = float(cv.max())
        fm = float(fv.max())
        if cm > 0.0:
            cv = cv / cm
        if fm > 0.0:
            fv = fv / fm
        score += alpha * cv + (1.0 - alpha) * fv
    return score


def eval_alpha_grid(
    trials: list[dict],
    C: np.ndarray,
    F: np.ndarray,
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    ks: tuple[int, ...] = (10, 20),
    seeds_mode: str = "fullprof",
    collect_top_lists_at: int | None = 10,
) -> dict:
    """
    Run the LOO ablation across the alpha grid.

    Returns a dict shaped like:
      {
        "seeds_mode": ...,
        "n_trials":   ...,
        "alphas": [
          {"alpha": 0.0, "hits@10": ..., "hits@20": ...,
           "hit_rate@10": ..., "precision@10": ..., "recall@10": ...,
           "hit_rate@20": ..., "precision@20": ..., "recall@20": ...,
           "top_lists": [...]   # only present for the alpha if collect=True
          }, ...
        ]
      }

    For each (trial, alpha): compute one score vector, mask the user's known
    interactions (except target), then derive top-max(ks). Slice it for each k.

    Hit-rate@k == recall@k here because each trial has exactly one held-out
    positive. We report both for clarity; precision@k = hits / (n_trials*k).
    Hit-rate is the honest headline (precision@k is capped at 1/k).
    """
    n_games = C.shape[0]
    k_max = max(ks)
    out_alphas = []
    for a in alphas:
        # per-k counters
        hits = {k: 0 for k in ks}
        top_lists_for_coverage: list[np.ndarray] = []
        for tr in trials:
            seed_cols = tr["profile"] if seeds_mode == "fullprof" else tr["seeds3"]
            if len(seed_cols) == 0:
                continue
            score = _per_seed_rescaled_sum(seed_cols, C, F, a, n_games)
            # Mask all of the user's known interactions EXCEPT the held-out
            # target so the system has a chance to surface the target.
            mask_cols = np.asarray(tr["profile"], dtype=np.int64)
            score[mask_cols] = -np.inf
            # top-k_max once
            kk = min(k_max, n_games - 1)
            top = np.argpartition(-score, kk)[:kk]
            top = top[np.argsort(-score[top])]  # full sort within top-kk
            tgt = tr["target"]
            for k in ks:
                if tgt in top[:k]:
                    hits[k] += 1
            if collect_top_lists_at is not None:
                top_lists_for_coverage.append(top[:collect_top_lists_at].copy())
        n_trials = len(trials)
        entry: dict = {"alpha": float(a)}
        for k in ks:
            h = hits[k]
            entry[f"hits@{k}"] = h
            entry[f"hit_rate@{k}"] = h / n_trials if n_trials else 0.0
            entry[f"recall@{k}"] = h / n_trials if n_trials else 0.0
            entry[f"precision@{k}"] = h / (n_trials * k) if n_trials else 0.0
        if collect_top_lists_at is not None:
            entry["top_lists"] = top_lists_for_coverage
        out_alphas.append(entry)
    return {
        "seeds_mode": seeds_mode,
        "n_trials": len(trials),
        "ks": list(ks),
        "alphas": out_alphas,
    }


# ---------- coverage & diversity --------------------------------------------
def catalog_coverage(top_lists: list[np.ndarray], n_catalog: int) -> float:
    """Fraction of catalog games appearing in at least one user's top-k list."""
    seen: set[int] = set()
    for tl in top_lists:
        seen.update(int(x) for x in tl)
    return len(seen) / n_catalog


def intra_list_diversity(top_lists: list[np.ndarray], C: np.ndarray) -> float:
    """
    Mean over users of the mean pairwise (1 - content_cosine) within each
    top-k list.

    NOTE on circularity: using the content arm to score diversity of a list
    the content arm helped produce biases the diversity number DOWNWARD for
    high-alpha (content-heavy) configurations and UPWARD for low-alpha
    (CF-heavy) ones. That is the expected direction of the effect — we just
    have to be honest that this metric has a built-in tilt in favour of CF.
    A purely categorical (tag-Jaccard) diversity would be cleaner but the
    Phase 8 brief is fine with the content-cosine version for a learning
    project; we surface the caveat in the notebook.
    """
    if not top_lists:
        return 0.0
    per_user = []
    for tl in top_lists:
        k = len(tl)
        if k < 2:
            continue
        sub = C[np.ix_(tl, tl)]   # k x k content-cosine block
        # Off-diagonal entries; symmetric, so take the upper triangle.
        iu = np.triu_indices(k, k=1)
        dissim = 1.0 - sub[iu]
        per_user.append(float(dissim.mean()))
    return float(np.mean(per_user)) if per_user else 0.0


# ---------- summary helpers used by the notebook ----------------------------
def hittability(trials: list[dict], n_catalog: int) -> dict:
    """
    By construction, every column in the interaction matrix corresponds to a
    catalog game (Phase 1 built `interactions.npz` with shape users x 6000,
    where the 6000 columns are exactly artifacts/game_index.json). So targets
    are 100% in the catalog. This helper computes the number to make the
    invariant visible in the report rather than assumed.
    """
    in_catalog = sum(0 <= tr["target"] < n_catalog for tr in trials)
    return {
        "n_trials": len(trials),
        "in_catalog": in_catalog,
        "fraction": in_catalog / max(1, len(trials)),
    }


def build_pop_by_col(M: sparse.csr_matrix) -> np.ndarray:
    """Catalog-aligned popularity vector (interactions per game). seed=42 not
    needed because this is deterministic."""
    return np.asarray(M.sum(axis=0)).ravel().astype(np.float32)


# ---------- dumping ---------------------------------------------------------
def metrics_to_jsonable(d: dict) -> dict:
    """Strip np arrays (top_lists) out so the metrics dict is JSON-safe."""
    out = {k: v for k, v in d.items() if k != "alphas"}
    out["alphas"] = []
    for a in d["alphas"]:
        entry = {k: v for k, v in a.items() if k != "top_lists"}
        out["alphas"].append(entry)
    return out
