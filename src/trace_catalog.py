"""
Diagnostic only — trace each probe app_id through data_prep's REAL pipeline
and report the first stage at which it gets eliminated from the catalog.

We import `count_interactions_per_app` directly from data_prep, and we replay
its `build_catalog` logic line-for-line (tag filter, popularity sort, head(N))
so any bug in data_prep is reproduced here — not bypassed.

Writes nothing. Touches nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import (
    CATALOG_SIZE,
    RAW,
    count_interactions_per_app,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

PROBES: list[tuple[int, str]] = [
    (413150, "Stardew Valley"),
    (105600, "Terraria"),
    (367520, "Hollow Knight"),
    (427520, "Factorio"),
    (1145360, "Hades"),
    (620, "Portal 2 (base)"),
    (292030, "The Witcher 3: Wild Hunt (base)"),
]


def main() -> None:
    # ------------------------------------------------------------------
    # Replay data_prep stages exactly.
    # ------------------------------------------------------------------
    print("=" * 78)
    print("Replaying data_prep stages")
    print("=" * 78)

    # Stage A — load games.csv with the same dtypes data_prep uses.
    games = pd.read_csv(
        RAW / "games.csv",
        usecols=["app_id", "title", "price_final", "positive_ratio",
                 "user_reviews", "date_release"],
        dtype={
            "app_id": np.int32, "title": "string",
            "price_final": np.float32, "positive_ratio": np.float32,
            "user_reviews": np.int32,
        },
    )
    games_set = set(games["app_id"].tolist())

    # Stage B — load games_metadata.json the same way (JSON-lines, then int32).
    metadata = pd.read_json(RAW / "games_metadata.json", lines=True)
    metadata["app_id"] = metadata["app_id"].astype(np.int32)
    metadata_by_id = {int(r["app_id"]): r for _, r in metadata.iterrows()}

    # Stage C — inner-join games <-> metadata on app_id (data_prep's join).
    df = games.merge(metadata, on="app_id", how="inner")

    # Stage D — data_prep's tag normalisation + ">=1 tag" predicate.
    df["tags"] = df["tags"].apply(lambda t: t if isinstance(t, list) else [])
    has_tags_mask = df["tags"].apply(len) > 0
    df_after_tag_filter = df[has_tags_mask].copy()

    # Stage E — interaction counts (REAL data_prep counter, streams the 41M file).
    interaction_counts = count_interactions_per_app()
    df_after_tag_filter["n_interactions"] = (
        df_after_tag_filter["app_id"].map(interaction_counts).fillna(0).astype(np.int64)
    )

    # Stage F — sort by popularity, take head(CATALOG_SIZE) — the catalog.
    catalog_df = (
        df_after_tag_filter
        .sort_values("n_interactions", ascending=False)
        .head(CATALOG_SIZE)
        .reset_index(drop=True)
    )
    catalog_app_ids = set(int(a) for a in catalog_df["app_id"].tolist())

    # Rank within the post-tag-filter table — i.e. where data_prep ranks each
    # qualifying game in the sort step.
    ranked = (
        df_after_tag_filter
        .sort_values("n_interactions", ascending=False)
        .reset_index(drop=True)
    )
    rank_by_app: dict[int, int] = {
        int(a): i for i, a in enumerate(ranked["app_id"].tolist())
    }

    # ------------------------------------------------------------------
    # dtype audit — make any int/str/float mismatch visible.
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("Dtype audit")
    print("=" * 78)
    print(f"  games.csv         app_id dtype : {games['app_id'].dtype}  "
          f"(sample value type: {type(games['app_id'].iloc[0]).__name__})")
    print(f"  games_metadata    app_id dtype : {metadata['app_id'].dtype}  "
          f"(sample value type: {type(metadata['app_id'].iloc[0]).__name__})")
    # recommendations.csv chunk dtype — read one tiny chunk.
    one_chunk = next(pd.read_csv(
        RAW / "recommendations.csv",
        usecols=["app_id"], dtype={"app_id": np.int32}, chunksize=1000,
    ))
    print(f"  recommendations   app_id dtype : {one_chunk['app_id'].dtype}  "
          f"(sample value type: {type(one_chunk['app_id'].iloc[0]).__name__})")
    # interaction_counts (Counter) key type.
    a_key = next(iter(interaction_counts))
    print(f"  interaction_counts key type    : {type(a_key).__name__}")
    # game_index.json key type (it's a JSON list of ids, so element type).
    game_index = json.loads((ART / "game_index.json").read_text())
    print(f"  game_index.json   element type : {type(game_index[0]).__name__}  "
          f"(len={len(game_index)})")
    # metadata_by_id keys (we built it ourselves from int casts).
    mk = next(iter(metadata_by_id))
    print(f"  metadata_by_id    key type     : {type(mk).__name__}  "
          f"(populated via int(r['app_id']) — int)")

    # ------------------------------------------------------------------
    # Per-probe trace.
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("Per-probe stage trace")
    print("=" * 78)

    game_index_set = set(int(g) for g in game_index)

    for app_id, label in PROBES:
        print()
        print(f"--- app_id={app_id}  ({label}) ---")

        # Step 1: in games.csv?
        in_games_csv = app_id in games_set
        if in_games_csv:
            row = games[games["app_id"] == app_id].iloc[0]
            print(f"  [1] games.csv:        IN   "
                  f"(stored app_id={row['app_id']} dtype={type(row['app_id']).__name__}, "
                  f"title={row['title']!r})")
        else:
            print(f"  [1] games.csv:        NOT IN")

        # Step 2: games_metadata.json — raw tags + description presence + key dtype.
        meta_row = metadata_by_id.get(int(app_id))
        in_metadata = meta_row is not None
        if in_metadata:
            raw_tags = meta_row["tags"]
            desc = meta_row.get("description", "") if isinstance(meta_row, pd.Series) else ""
            desc_nonempty = isinstance(desc, str) and bool(desc.strip())
            print(f"  [2] metadata.json:    IN   "
                  f"tags={raw_tags!r}  desc_nonempty={desc_nonempty}  "
                  f"(lookup uses int(app_id) against metadata_by_id whose keys are int)")
        else:
            print(f"  [2] metadata.json:    NOT IN")
            raw_tags = None

        # Step 3: data_prep's own tag predicate, applied to THIS app_id.
        # That predicate runs *after* the games<->metadata inner-join, so a row
        # with no metadata row never even reaches it.
        merged_row = df[df["app_id"] == app_id]
        if len(merged_row) == 0:
            print(f"  [3] tag filter:       N/A  (eliminated at inner-join step C; "
                  f"not present in both games + metadata)")
            tag_filter_pass = False
        else:
            tags_after_norm = merged_row["tags"].iloc[0]
            tag_filter_pass = (
                isinstance(tags_after_norm, list) and len(tags_after_norm) > 0
            )
            print(f"  [3] tag filter:       "
                  f"{'PASS' if tag_filter_pass else 'FAIL'}  "
                  f"normalised_tags={tags_after_norm!r}  predicate=(len(tags)>0)")

        # Step 4: data_prep's interaction count for this app_id vs an independent
        # full re-count. We already streamed once; do the cross-check by reading
        # recommendations.csv a second time but counting ONLY this app_id (cheap
        # mask — keeps memory tiny). Doing it per probe would be 7x slow; do it
        # once for all probes below in a single second pass.
        dp_count = int(interaction_counts.get(int(app_id), 0))
        independent = INDEPENDENT_COUNTS.get(int(app_id), 0)
        eq = dp_count == independent
        print(f"  [4] interaction count: data_prep={dp_count:,}  "
              f"independent_recount={independent:,}  equal={eq}")

        # Step 5: rank by data_prep's count within the post-tag-filter table.
        rank = rank_by_app.get(int(app_id))
        if rank is None:
            in_ranked_table = False
            print(f"  [5] rank in tag-filtered table: NOT PRESENT in the table "
                  f"(eliminated before sort)")
        else:
            in_ranked_table = True
            print(f"  [5] rank in tag-filtered table: #{rank:,}  "
                  f"(top-{CATALOG_SIZE:,} cut → "
                  f"{'WITHIN' if rank < CATALOG_SIZE else 'OUTSIDE'})")

        in_catalog = int(app_id) in catalog_app_ids
        in_index_file = int(app_id) in game_index_set
        print(f"  [6] in catalog (replay): {'YES' if in_catalog else 'NO'}   "
              f"in game_index.json: {'YES' if in_index_file else 'NO'}")

        # First-stage-of-elimination verdict.
        if not in_games_csv:
            first_stage = "games.csv (not present)"
        elif not in_metadata:
            first_stage = "metadata.json (no row → dropped at inner-join)"
        elif len(merged_row) == 0:
            first_stage = "inner-join (key mismatch between games and metadata)"
        elif not tag_filter_pass:
            first_stage = "tag filter (>=1 tag predicate failed)"
        elif not in_ranked_table:
            first_stage = "post-tag-filter table (unexpected — not in ranking)"
        elif rank is not None and rank >= CATALOG_SIZE:
            first_stage = f"head({CATALOG_SIZE}) cut (ranked #{rank})"
        else:
            first_stage = "survives → should be in catalog"
        print(f"  >>> FIRST ELIM STAGE: {first_stage}")


# Independent full re-count of probe app_ids, computed once before the per-
# probe loop runs.
INDEPENDENT_COUNTS: dict[int, int] = {}


def _independent_recount() -> None:
    """Stream recommendations.csv ONCE more and count only the probe app_ids,
    independent of data_prep's Counter. Lets us spot any miscount."""
    probe_ids = {a for a, _ in PROBES}
    counts = {a: 0 for a in probe_ids}
    rows = 0
    for chunk in pd.read_csv(
        RAW / "recommendations.csv",
        usecols=["app_id"], dtype={"app_id": np.int32},
        chunksize=2_000_000,
    ):
        masked = chunk["app_id"][chunk["app_id"].isin(probe_ids)]
        for a in masked.tolist():
            counts[int(a)] += 1
        rows += len(chunk)
    for a, c in counts.items():
        INDEPENDENT_COUNTS[a] = c
    print(f"[independent recount] scanned {rows:,} rows for probe app_ids")
    for a, c in counts.items():
        print(f"   app_id={a:<10}  count={c:,}")


if __name__ == "__main__":
    _independent_recount()
    main()
