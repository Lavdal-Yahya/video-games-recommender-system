"""
Diagnostic only — three checks to decide how to fix the Phase-1 tag-filter
bug found in src/trace_catalog.py.

  CHECK 1: are the raw games_metadata.json lines really empty for the probes,
           or is pandas/data_prep manufacturing the emptiness?
  CHECK 2: how many catalog slots does the empty-tag bug actually corrupt
           (i.e. how many empty-tag games belong in the top-6000)?
  CHECK 3: does games.csv carry any column we could use as a soup fallback?

Reads only. Writes nothing.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

GAMES_CSV = RAW / "games.csv"
METADATA_JSONL = RAW / "games_metadata.json"
RECS_CSV = RAW / "recommendations.csv"

PROBES: list[tuple[int, str]] = [
    (292030, "The Witcher 3"),
    (105600, "Terraria"),
    (413150, "Stardew Valley"),
    (620, "Portal 2"),
    (367520, "Hollow Knight"),
    (427520, "Factorio"),
    (1145360, "Hades"),
]
PROBE_IDS = {a for a, _ in PROBES}


# ----------------------------------------------------------------------------
# CHECK 1 — raw JSONL truth.
# ----------------------------------------------------------------------------
def check1_raw_jsonl() -> dict[int, list[dict]]:
    """Stream games_metadata.json line-by-line with stdlib json.

    No pandas, no data_prep loader, no dtype coercion — we want the literal
    contents of the file, including duplicates.
    """
    print("=" * 78)
    print("CHECK 1 — raw games_metadata.json (line-by-line, stdlib json only)")
    print("=" * 78)

    found: dict[int, list[dict]] = defaultdict(list)
    raw_lines: dict[int, list[str]] = defaultdict(list)
    total_lines = 0
    with METADATA_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # Cheap pre-filter so we don't json.loads all 50k lines we don't need:
            # only parse lines that contain one of our probe ids as a substring.
            if not any(str(pid) in line for pid in PROBE_IDS):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN: line {total_lines} failed to parse: {e}")
                continue
            aid = obj.get("app_id")
            if isinstance(aid, int) and aid in PROBE_IDS:
                found[aid].append(obj)
                raw_lines[aid].append(line)

    print(f"  scanned {total_lines:,} lines total")
    print()
    for app_id, label in PROBES:
        rows = found.get(app_id, [])
        print(f"  app_id={app_id}  ({label})  -> {len(rows)} line(s) in raw JSONL")
        if not rows:
            print(f"    (absent from raw metadata file)")
            continue
        for i, obj in enumerate(rows):
            tags = obj.get("tags", "<missing key>")
            desc = obj.get("description", "<missing key>")
            desc_len = len(desc) if isinstance(desc, str) else None
            # Print key type as the parser produced it.
            tags_type = type(tags).__name__
            desc_type = type(desc).__name__
            print(f"    [#{i}] tags ({tags_type}, n={len(tags) if hasattr(tags, '__len__') else '?'}): {tags!r}")
            if isinstance(desc, str):
                snippet = desc[:160] + ("..." if len(desc) > 160 else "")
                print(f"         description ({desc_type}, len={desc_len}): {snippet!r}")
            else:
                print(f"         description ({desc_type}): {desc!r}")
            # Print the verbatim line (truncated if huge) so nothing is hidden.
            line = raw_lines[app_id][i]
            if len(line) > 300:
                line = line[:300] + f"... [truncated, full len={len(line)}]"
            print(f"         RAW LINE: {line}")
    return found


# ----------------------------------------------------------------------------
# CHECK 2 — blast radius.
# ----------------------------------------------------------------------------
def check2_blast_radius() -> None:
    print()
    print("=" * 78)
    print("CHECK 2 — blast radius: empty-tag games × interaction count")
    print("=" * 78)

    # Step a — collect the set of empty-tag app_ids from raw metadata JSONL.
    empty_tag_ids: set[int] = set()
    nonempty_tag_ids: set[int] = set()
    n_lines = 0
    with METADATA_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            aid = obj.get("app_id")
            if not isinstance(aid, int):
                continue
            tags = obj.get("tags", [])
            if isinstance(tags, list) and len(tags) > 0:
                nonempty_tag_ids.add(aid)
            else:
                empty_tag_ids.add(aid)
    print(f"  raw metadata lines: {n_lines:,}")
    print(f"  app_ids with empty tags: {len(empty_tag_ids):,}")
    print(f"  app_ids with >=1 tag   : {len(nonempty_tag_ids):,}")

    # Step b — stream recommendations.csv ONCE, counting per-app_id but only for
    # the union (empty + non-empty). That's still ~50k unique ids — Counter handles it.
    needed_ids = empty_tag_ids | nonempty_tag_ids
    print(f"  counting interactions for {len(needed_ids):,} ids (stream pass)...")
    counts: Counter[int] = Counter()
    rows_seen = 0
    # Use csv.reader to avoid a pandas dep — slower but simpler and fine for a
    # one-off diagnostic. Read app_id column only.
    with RECS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = header.index("app_id")
        for row in reader:
            rows_seen += 1
            try:
                aid = int(row[idx])
            except ValueError:
                continue
            if aid in needed_ids:
                counts[aid] += 1
            if rows_seen % 5_000_000 == 0:
                print(f"    ...{rows_seen:,} rows scanned")
    print(f"  done. total rows: {rows_seen:,}")

    # Step c — rank all (empty + non-empty) by count, then ask:
    # of the top-6000 by interaction count, how many have empty tags?
    ranked = sorted(needed_ids, key=lambda a: counts.get(a, 0), reverse=True)
    top_6000 = ranked[:6000]
    corrupted = sum(1 for a in top_6000 if a in empty_tag_ids)
    print()
    print(f"  TOP-6000 (by interaction count) breakdown:")
    print(f"    with    tags: {6000 - corrupted:,}")
    print(f"    with NO tags: {corrupted:,}   <-- catalog slots the bug burns")

    # Step d — names lookup so the top-20 list is readable.
    titles: dict[int, str] = {}
    with GAMES_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        aid_idx = header.index("app_id")
        title_idx = header.index("title")
        for row in reader:
            try:
                aid = int(row[aid_idx])
            except ValueError:
                continue
            titles[aid] = row[title_idx]

    # Step e — print the top-20 empty-tag games by interaction count.
    empty_ranked = sorted(empty_tag_ids, key=lambda a: counts.get(a, 0), reverse=True)
    print()
    print(f"  TOP 20 empty-tag games by interaction count:")
    print(f"  {'rank':>4}  {'app_id':>8}  {'count':>12}  title")
    for i, aid in enumerate(empty_ranked[:20], start=1):
        print(f"  {i:>4}  {aid:>8}  {counts.get(aid, 0):>12,}  {titles.get(aid, '<unknown>')!r}")


# ----------------------------------------------------------------------------
# CHECK 3 — recovery field in games.csv.
# ----------------------------------------------------------------------------
def check3_recovery() -> None:
    print()
    print("=" * 78)
    print("CHECK 3 — recovery field availability in games.csv")
    print("=" * 78)
    with GAMES_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"  games.csv columns ({len(header)}): {header}")
        target_row = None
        aid_idx = header.index("app_id")
        for row in reader:
            try:
                if int(row[aid_idx]) == 292030:
                    target_row = row
                    break
            except ValueError:
                continue
    if target_row is None:
        print("  app_id=292030 not found in games.csv")
        return
    print()
    print("  full row for app_id=292030 (The Witcher 3):")
    for col, val in zip(header, target_row):
        print(f"    {col:<18} = {val!r}")
    print()
    # State the verdict explicitly.
    informative_text_cols = [c for c in header if c.lower() in
                             {"description", "short_description", "about_the_game",
                              "categories", "genres", "tags", "developer", "publisher"}]
    print(f"  text-bearing columns spotted in games.csv: {informative_text_cols or '(none)'}")


def main() -> None:
    check1_raw_jsonl()
    check2_blast_radius()
    check3_recovery()


if __name__ == "__main__":
    main()
