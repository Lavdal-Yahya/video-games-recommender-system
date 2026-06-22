"""Phase 1 — turn raw Steam data into the artifacts both arms will consume.

The two arms (content similarity, item-item collaborative filtering) must agree
on **one** catalog of games — otherwise their similarity matrices index
different things and you can't blend them. So this script:

  1. Picks the catalog: top-N games by interaction count that also have tags.
  2. Builds the content "soup" (tags + description) for each catalog game.
  3. Builds a sampled (user x game) binary interaction matrix on that same
     catalog, keeping only users with >= 5 interactions inside it.

Design decisions locked in by Phase 1 brief:
  - game_id = app_id (already a stable int, shared join key — no re-keying).
  - catalog = top-N (N=6000) most-interacted games with non-empty tags.
  - interaction signal = binary presence (1 if user reviewed game, else 0).
    `hours` and `is_recommended` are *not* used yet — see Phase 4 refinements.
  - Sample users with >= 5 in-catalog interactions; cap to USER_CAP if larger.
  - Random seed = 42 for any sampling, so reruns are identical.

Run with: `conda run -n ds python src/data_prep.py`
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


# ---------- paths & knobs --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"

CATALOG_SIZE = 6000           # top-N games by interaction count (with tags)
MIN_USER_INTERACTIONS = 5     # drop users with < 5 in-catalog interactions
USER_CAP = 150_000            # cap on sampled users (keeps the matrix laptop-sized)
SEED = 42
CHUNK_ROWS = 2_000_000        # recommendations.csv chunk size (memory-bounded)


# ---------- step 1: count interactions per app_id --------------------------

def count_interactions_per_app() -> Counter:
    """Stream recommendations.csv once, counting reviews per app_id.

    We don't load the full ~41M-row file into memory. Only `app_id` is needed
    here, and we read it as int32 (Steam app_ids fit comfortably).
    """
    print("[1/5] counting interactions per app_id (streaming)...")
    counts: Counter = Counter()
    reader = pd.read_csv(
        RAW / "recommendations.csv",
        usecols=["app_id"],
        dtype={"app_id": np.int32},
        chunksize=CHUNK_ROWS,
    )
    n_rows = 0
    for chunk in reader:
        # Counter.update on a numpy array is fast enough; value_counts also works.
        vc = chunk["app_id"].value_counts()
        for app_id, c in vc.items():
            counts[int(app_id)] += int(c)
        n_rows += len(chunk)
    print(f"      scanned {n_rows:,} interaction rows across {len(counts):,} games")
    return counts


# ---------- step 2: pick the catalog --------------------------------------

def build_catalog(interaction_counts: Counter) -> pd.DataFrame:
    """Pick the top-N most-interacted games that also have non-empty tags.

    Returns one DataFrame with the columns we'll persist in games.json plus
    the `soup` field (tags + description) that the content arm will vectorize.
    """
    print("[2/5] building catalog (top-N games with tags + metadata join)...")

    games = pd.read_csv(
        RAW / "games.csv",
        usecols=[
            "app_id", "title", "price_final", "positive_ratio",
            "user_reviews", "date_release",
        ],
        dtype={
            "app_id": np.int32,
            "title": "string",
            "price_final": np.float32,
            "positive_ratio": np.float32,
            "user_reviews": np.int32,
        },
    )

    # JSON-lines — one game per line — same shape as Phase 0 confirmed.
    metadata = pd.read_json(RAW / "games_metadata.json", lines=True)
    metadata["app_id"] = metadata["app_id"].astype(np.int32)

    # Inner-join: a game must have both a row in games.csv (for name/price) and
    # a row in games_metadata.json (for tags/description) to be eligible.
    df = games.merge(metadata, on="app_id", how="inner")

    # Phase 1.5 — the tag gate has been removed. Catalog is strictly top-N
    # by interaction count; empty-tag games are enriched from Steam appdetails
    # in src/enrich_metadata.py after build.
    df["tags"] = df["tags"].apply(lambda t: t if isinstance(t, list) else [])

    # Attach interaction counts; games with zero interactions get 0.
    df["n_interactions"] = df["app_id"].map(interaction_counts).fillna(0).astype(np.int64)

    # Rank by popularity — the top-N most-played games with tags become the
    # working catalog. Both arms (content + CF) will index this exact set.
    df = df.sort_values("n_interactions", ascending=False).head(CATALOG_SIZE).reset_index(drop=True)

    # Build the content soup. tags + description; if description is missing or
    # blank, fall back to tags alone (tags are ~98% present so soup is never
    # empty). The soup is what Phase 2's TF-IDF vectorizer will see.
    def make_soup(row) -> str:
        tag_part = " ".join(row["tags"])
        desc = row.get("description")
        if isinstance(desc, str) and desc.strip():
            return f"{tag_part} {desc.strip()}"
        return tag_part

    df["soup"] = df.apply(make_soup, axis=1)

    # Normalize price + description for the export (avoid NaN in JSON).
    df["price_final"] = df["price_final"].fillna(0.0).astype(float)
    df["description"] = df["description"].where(df["description"].notna(), "").astype(str)

    print(f"      catalog size: {len(df):,} games "
          f"(target was {CATALOG_SIZE:,})")
    print(f"      interactions: min={df['n_interactions'].min()}, "
          f"median={int(df['n_interactions'].median()):,}, "
          f"max={df['n_interactions'].max():,}")
    return df


# ---------- step 3: build the sampled interaction matrix ------------------

def build_interaction_matrix(catalog_ids: np.ndarray):
    """Stream recommendations.csv a second time, keep in-catalog rows only,
    then sample users with >= MIN_USER_INTERACTIONS in-catalog reviews.

    Returns: (sparse CSR matrix, row->user_id list, col->game_id list).
    """
    print("[3/5] collecting in-catalog (user, game) pairs (streaming)...")

    catalog_set = set(int(x) for x in catalog_ids)

    # Stream and keep only rows whose app_id is in our catalog. We read
    # user_id + app_id only; both fit in int32. is_recommended/hours are
    # ignored at this phase (binary signal — see header).
    reader = pd.read_csv(
        RAW / "recommendations.csv",
        usecols=["app_id", "user_id"],
        dtype={"app_id": np.int32, "user_id": np.int32},
        chunksize=CHUNK_ROWS,
    )
    kept_chunks = []
    n_rows = 0
    for chunk in reader:
        n_rows += len(chunk)
        mask = chunk["app_id"].isin(catalog_set)
        if mask.any():
            kept_chunks.append(chunk[mask])
    pairs = pd.concat(kept_chunks, ignore_index=True)
    print(f"      scanned {n_rows:,} rows; "
          f"{len(pairs):,} are in-catalog interactions")

    # Drop accidental dup (user, game) rows — binary signal, presence only.
    pairs = pairs.drop_duplicates(subset=["user_id", "app_id"])

    # Keep users with enough signal. Too-few-interaction users don't help
    # item-item CF and just inflate the matrix.
    per_user = pairs.groupby("user_id").size()
    active_users = per_user[per_user >= MIN_USER_INTERACTIONS].index
    pairs = pairs[pairs["user_id"].isin(active_users)]
    print(f"      kept {pairs['user_id'].nunique():,} users with "
          f">= {MIN_USER_INTERACTIONS} in-catalog interactions "
          f"({len(pairs):,} pairs)")

    # Sample-cap users if the active set is still too large for a laptop CF
    # matrix. Seeded RNG → reproducible. Caps the *users*, not the pairs.
    unique_users = pairs["user_id"].unique()
    if len(unique_users) > USER_CAP:
        rng = np.random.default_rng(SEED)
        sampled = rng.choice(unique_users, size=USER_CAP, replace=False)
        pairs = pairs[pairs["user_id"].isin(sampled)]
        print(f"      sampled down to {USER_CAP:,} users (seed={SEED})")

    # Build the sparse matrix. Rows = users, cols = games (catalog order).
    # We use the catalog's own ordering for the game axis so col i in the
    # matrix corresponds to catalog row i — this keeps Phase 3 simple.
    user_index = pairs["user_id"].drop_duplicates().tolist()  # row -> user_id
    game_index = [int(x) for x in catalog_ids]                # col -> game_id

    user_to_row = {u: r for r, u in enumerate(user_index)}
    game_to_col = {g: c for c, g in enumerate(game_index)}

    rows = pairs["user_id"].map(user_to_row).to_numpy()
    cols = pairs["app_id"].map(game_to_col).to_numpy()
    data = np.ones(len(pairs), dtype=np.float32)

    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_index), len(game_index)),
        dtype=np.float32,
    )
    print(f"      matrix shape: {matrix.shape}, nnz: {matrix.nnz:,}")
    return matrix, user_index, game_index


# ---------- step 4: export artifacts --------------------------------------

def export(catalog: pd.DataFrame, matrix, user_index, game_index) -> dict:
    """Persist everything Phases 2-4 will load."""
    print("[4/5] writing artifacts...")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # games.json — the catalog. One record per game with the fields the
    # frontend + filters will need later. `genres` is included as an empty
    # list for now: this Steam dataset has no separate genres column; the
    # tag list is the de-facto genre signal. (Recorded as a Phase 1 note.)
    games_records = []
    for _, r in catalog.iterrows():
        games_records.append({
            "game_id": int(r["app_id"]),     # game_id == app_id (locked decision)
            "name": str(r["title"]),
            "genres": [],                    # see comment above
            "tags": list(r["tags"]),
            "price": float(r["price_final"]),
            "description": str(r["description"]),
            "soup": str(r["soup"]),          # convenience for Phase 2 (TF-IDF input)
            "n_interactions": int(r["n_interactions"]),
            "positive_ratio": (
                float(r["positive_ratio"]) if pd.notna(r["positive_ratio"]) else None
            ),
            "date_release": (
                str(r["date_release"]) if pd.notna(r["date_release"]) else None
            ),
        })
    (ARTIFACTS / "games.json").write_text(json.dumps(games_records, ensure_ascii=False))
    print(f"      wrote artifacts/games.json ({len(games_records):,} games)")

    # The interaction matrix as scipy .npz — CSR (cheap row slicing for the
    # item-item CF in Phase 3, which actually wants column slicing — so
    # Phase 3 will .tocsc() once at load time).
    sparse.save_npz(ARTIFACTS / "interactions.npz", matrix)
    print(f"      wrote artifacts/interactions.npz "
          f"(shape={matrix.shape}, nnz={matrix.nnz:,})")

    # Index maps so a row/col in the matrix can be mapped back to its real id.
    (ARTIFACTS / "user_index.json").write_text(
        json.dumps([int(u) for u in user_index])
    )
    (ARTIFACTS / "game_index.json").write_text(
        json.dumps([int(g) for g in game_index])
    )
    print("      wrote artifacts/user_index.json + artifacts/game_index.json")

    # Tiny id <-> name lookup. Two-way maps stored as JSON (string keys
    # because JSON object keys are strings). game_id -> name and name -> id.
    id_to_name = {int(r["app_id"]): str(r["title"]) for _, r in catalog.iterrows()}
    name_to_id: dict[str, int] = {}
    for gid, name in id_to_name.items():
        # If two games share a title, keep the first (most-interacted, since
        # the catalog is already sorted by interaction count). Phase 2 adds
        # fuzzy matching on top of this for the voice/LLM path.
        name_to_id.setdefault(name, gid)
    lookup = {
        "id_to_name": {str(k): v for k, v in id_to_name.items()},
        "name_to_id": name_to_id,
    }
    (ARTIFACTS / "id_name_lookup.json").write_text(
        json.dumps(lookup, ensure_ascii=False)
    )
    print("      wrote artifacts/id_name_lookup.json")

    # Dataset stats — keep for the Phase 9 report.
    n_users, n_games = matrix.shape
    density = matrix.nnz / (n_users * n_games)
    stats = {
        "catalog_size": int(n_games),
        "n_users": int(n_users),
        "n_interactions": int(matrix.nnz),
        "density": float(density),
        "sparsity": float(1.0 - density),
        "seed": SEED,
        "config": {
            "CATALOG_SIZE": CATALOG_SIZE,
            "MIN_USER_INTERACTIONS": MIN_USER_INTERACTIONS,
            "USER_CAP": USER_CAP,
        },
    }
    (ARTIFACTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print("      wrote artifacts/dataset_stats.json")
    return stats


# ---------- driver ---------------------------------------------------------

def main() -> None:
    print(f"raw data:  {RAW}")
    print(f"artifacts: {ARTIFACTS}\n")

    interaction_counts = count_interactions_per_app()
    catalog = build_catalog(interaction_counts)
    matrix, user_index, game_index = build_interaction_matrix(
        catalog["app_id"].to_numpy()
    )
    stats = export(catalog, matrix, user_index, game_index)

    print("\n[5/5] summary:")
    print(f"  catalog size:       {stats['catalog_size']:,} games")
    print(f"  sampled users:      {stats['n_users']:,}")
    print(f"  interactions (nnz): {stats['n_interactions']:,}")
    print(f"  matrix sparsity:    {stats['sparsity'] * 100:.4f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
