"""
Diagnostic only — read-only sanity check on the Phase-1 catalog.

Asks: are flagship games like Stardew Valley / Terraria / Witcher 3 silently
absent from artifacts/game_index.json? For each probe we want to distinguish:
  - IN_CATALOG               — the game is present (Phase 1 picked it up)
  - BELOW_THRESHOLD          — found in raw data but with too few interactions
                               to clear the top-6000 popularity cut
  - MISSING_BUT_QUALIFIES    — found in raw data, interaction count >= the
                               catalog's lowest, yet absent — a real bug
  - NAME_FIELD_BROKEN        — caught only by step 4: catalog names are blank
                               (would make name-based matching impossible)
  - NOT_IN_RAW               — no substring match in the raw 50,872-row catalog
                               at all (so this isn't a sampling issue)

Writes nothing. Touches nothing.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ART = ROOT / "artifacts"

GAMES_CSV = RAW / "games.csv"
RECS_CSV = RAW / "recommendations.csv"
GAME_INDEX_JSON = ART / "game_index.json"
GAMES_JSON = ART / "games.json"

PROBES = [
    "Stardew Valley",
    "Terraria",
    "The Witcher 3",
    "Hades",
    "Hollow Knight",
    "Portal 2",
    "Factorio",
]


def main() -> None:
    print("=" * 78)
    print("STEP 1 — loading full raw games.csv")
    print("=" * 78)
    games_raw = pd.read_csv(GAMES_CSV, usecols=["app_id", "title"])
    print(f"  rows: {len(games_raw):,}  cols: {list(games_raw.columns)}")
    # Title normalised once for case-insensitive substring matching.
    games_raw["title_lc"] = games_raw["title"].astype(str).str.lower()

    print()
    print("=" * 78)
    print("STEP 2 — substring matches for each probe (case-insensitive)")
    print("=" * 78)
    matches: dict[str, list[tuple[int, str]]] = {}
    for probe in PROBES:
        q = probe.lower()
        hits = games_raw[games_raw["title_lc"].str.contains(q, regex=False, na=False)]
        matches[probe] = list(zip(hits["app_id"].astype(int), hits["title"]))
        print(f"\n  {probe!r}  -> {len(matches[probe])} match(es)")
        for app_id, title in matches[probe][:12]:
            print(f"      app_id={app_id:<10}  title={title!r}")
        if len(matches[probe]) > 12:
            print(f"      ... and {len(matches[probe]) - 12} more")

    # Collect every app_id we care about counting interactions for: probe hits
    # + every catalog app_id (to derive the catalog threshold).
    print()
    print("=" * 78)
    print("STEP 3a — loading catalog (artifacts/game_index.json)")
    print("=" * 78)
    catalog_ids = set(json.loads(GAME_INDEX_JSON.read_text()))
    print(f"  catalog size: {len(catalog_ids):,}")

    probe_app_ids: set[int] = set()
    for probe, hits in matches.items():
        for app_id, _ in hits:
            probe_app_ids.add(app_id)
    ids_to_count = catalog_ids | probe_app_ids
    print(f"  app_ids to stream-count: {len(ids_to_count):,}  "
          f"(catalog {len(catalog_ids):,} + extra probe hits {len(probe_app_ids - catalog_ids):,})")

    print()
    print("=" * 78)
    print("STEP 3b — streaming recommendations.csv to count interactions")
    print("=" * 78)
    counts: Counter[int] = Counter()
    chunk_size = 2_000_000
    rows_seen = 0
    # We only need the app_id column; that keeps memory tiny even on the 41M-row file.
    for chunk in pd.read_csv(RECS_CSV, usecols=["app_id"], chunksize=chunk_size):
        ids = chunk["app_id"].astype(int)
        ids = ids[ids.isin(ids_to_count)]
        counts.update(ids.tolist())
        rows_seen += len(chunk)
        print(f"  ...{rows_seen:>12,} rows scanned  (running unique tracked: {len(counts):,})")
    print(f"  done. total rows: {rows_seen:,}")

    # Catalog threshold = the lowest interaction count among catalog games. A
    # game absent from the catalog but with count >= threshold is a real bug.
    catalog_counts = {gid: counts.get(gid, 0) for gid in catalog_ids}
    cat_count_values = sorted(catalog_counts.values())
    threshold = cat_count_values[0]
    print()
    print(f"  catalog interaction stats: "
          f"min={cat_count_values[0]:,}  "
          f"p25={cat_count_values[len(cat_count_values)//4]:,}  "
          f"median={cat_count_values[len(cat_count_values)//2]:,}  "
          f"max={cat_count_values[-1]:,}")
    n_zero_in_catalog = sum(1 for v in cat_count_values if v == 0)
    print(f"  catalog games with 0 counted interactions: {n_zero_in_catalog}")
    print(f"  >>> threshold (lowest catalog count) = {threshold:,}")

    print()
    print("=" * 78)
    print("STEP 4 — name-field integrity on 10 random catalog ids")
    print("=" * 78)
    games_json = json.loads(GAMES_JSON.read_text())
    by_id = {g["game_id"]: g for g in games_json}
    rng = random.Random(42)
    sample_ids = rng.sample(sorted(catalog_ids), 10)
    name_field_ok_total = 0
    for gid in sample_ids:
        rec = by_id.get(gid)
        name = (rec or {}).get("name", "")
        ok = bool(name and name.strip())
        name_field_ok_total += int(ok)
        flag = "OK " if ok else "BAD"
        print(f"  [{flag}] game_id={gid:<10}  name={name!r}")
    name_field_broken = name_field_ok_total < 10

    print()
    print("=" * 78)
    print("STEP 5 — verdict per probe")
    print("=" * 78)
    # For each probe, pick the "best" matching app_id: prefer one already in the
    # catalog (if any); otherwise the one with the highest interaction count.
    for probe in PROBES:
        hits = matches[probe]
        if not hits:
            print(f"  {probe:<22} NOT_IN_RAW")
            continue

        in_catalog_hits = [(aid, t) for aid, t in hits if aid in catalog_ids]
        if in_catalog_hits:
            aid, title = max(in_catalog_hits, key=lambda x: counts.get(x[0], 0))
            cnt = counts.get(aid, 0)
            verdict = "NAME_FIELD_BROKEN" if name_field_broken else "IN_CATALOG"
            print(f"  {probe:<22} {verdict:<22} "
                  f"app_id={aid:<10}  title={title!r}  interactions={cnt:,}  "
                  f"threshold={threshold:,}")
            continue

        # Not in catalog — among the raw matches, pick the highest-count one.
        aid, title = max(hits, key=lambda x: counts.get(x[0], 0))
        cnt = counts.get(aid, 0)
        if cnt >= threshold:
            verdict = "MISSING_BUT_QUALIFIES"
        else:
            verdict = "BELOW_THRESHOLD"
        print(f"  {probe:<22} {verdict:<22} "
              f"app_id={aid:<10}  title={title!r}  interactions={cnt:,}  "
              f"threshold={threshold:,}")

    print()
    print("Done. No artifacts were modified.")


if __name__ == "__main__":
    main()
