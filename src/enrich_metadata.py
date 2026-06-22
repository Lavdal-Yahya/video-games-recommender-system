"""Phase 1.5 — enrich empty-tag catalog games from Steam's public appdetails API.

Background: ~1,244 rows in `data/raw/games_metadata.json` ship with empty
`tags` + empty `description` (anything Steam doesn't expose in the dataset
dump). Among the top-6,000 catalog, **565** of those are flagship titles
(Witcher 3, GTA V, CS:GO, TF2, ...). With Phase 1 stripped of its tag gate,
they're now in the catalog — but with no tags, the content arm can't say
anything about them. We hit Steam's public storefront API to recover
genres/categories/short_description for each, and map the response into the
exact same "soup" shape a natively-tagged game has.

The hard part is **rate limiting**, not parsing:
  * Steam throttles around ~200 requests / 5 min per IP, and when it does,
    it returns `{"<app_id>": {"success": false}}` or `null`. That looks like
    "no data" but is actually "slow down". So we treat success==False as a
    *retryable failure*, not as "empty tags".
  * We sleep ~1.5s between calls (well under the limit), set a real
    User-Agent, and cache **every** raw response under
    `data/raw/steam_cache/<app_id>.json`. Re-runs skip anything already
    cached → the script is resumable if it dies or gets throttled.
  * Failed app_ids are retried in up to 3 passes with a longer backoff
    (60s) between passes.

This script never touches `data/raw/games_metadata.json` (raw stays raw).
It updates `artifacts/games.json` in place: for each enriched app_id, we
fill `tags`, `description`, and rebuild `soup` to the same recipe Phase 1
uses (`tag_part + " " + description` or tags-only if description is empty).

Run with: `conda run -n ds python -m src.enrich_metadata`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
CACHE = ROOT / "data" / "raw" / "steam_cache"

# Polite pacing: 1.5s/req = 40 req/min — well under Steam's ~200 req / 5min cap.
REQUEST_DELAY_S = 1.5
RETRY_PASSES = 3            # extra passes over still-failing app_ids
BETWEEN_PASS_SLEEP_S = 60.0  # cooldown between retry passes (cap-reset window)
REQUEST_TIMEOUT_S = 15.0

# A real User-Agent — Steam serves blank responses to generic Python UA strings.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

URL = "https://store.steampowered.com/api/appdetails"


# ---------- cache ----------------------------------------------------------

def cache_path(app_id: int) -> Path:
    return CACHE / f"{app_id}.json"


def load_cached(app_id: int) -> dict | None:
    """Return the cached raw JSON for an app_id, or None if not cached."""
    p = cache_path(app_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        # Corrupt cache file → treat as missing so the next pass refetches.
        return None


def write_cache(app_id: int, payload: dict) -> None:
    cache_path(app_id).write_text(json.dumps(payload, ensure_ascii=False))


# ---------- fetch ----------------------------------------------------------

def fetch_one(session: requests.Session, app_id: int) -> dict | None:
    """One call to Steam appdetails. Returns the raw JSON dict or None on error.

    We don't interpret success/failure here — that's done in extract(). This
    function just guards against network/HTTP errors so the outer loop can
    record the app_id as still-failed and continue.
    """
    try:
        r = session.get(
            URL,
            params={"appids": app_id, "l": "english"},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT_S,
        )
        if r.status_code != 200:
            return None
        return r.json()
    except (requests.RequestException, json.JSONDecodeError):
        return None


# ---------- extract --------------------------------------------------------

def extract(raw: dict | None, app_id: int) -> tuple[list[str], str] | None:
    """Map Steam's appdetails JSON to (tags, description) in our format.

    Steam's payload shape: { "<app_id>": { "success": bool, "data": {...} } }.
    On throttle / unknown app, success is False or data is missing — return
    None to mean "retryable failure".

    Tag source (post-2026-06-20 soup-quality fix): **`genres[].description`
    only**. Steam's `categories[]` field is where the platform plumbing lives
    ("Single-player", "Steam Achievements", "Steam Cloud", "Full controller
    support", ...) — those describe infrastructure, not content, and they
    flood the TF-IDF vocabulary with terms that are noise for similarity.
    Native rows in `games_metadata.json` never carried that plumbing, so
    leaving categories in created an enriched-vs-native asymmetry. Genres
    alone (Action, RPG, Indie, ...) match the native flavor. A small
    `SOUP_STOPLIST` in `content.py` is a symmetric belt-and-braces filter
    in case any plumbing tokens still slip through on either side.

    Dropping categories can leave an enriched game with very few or zero
    tags (some Steam pages have empty `genres`). That's accepted — the
    soup then leans on description + CF, same as the 5 F2P MMOs that
    failed to fetch at all.
    """
    if not raw:
        return None
    entry = raw.get(str(app_id)) or raw.get(app_id)
    if not entry or not entry.get("success"):
        return None
    data = entry.get("data") or {}

    tags: list[str] = []
    seen: set[str] = set()
    for item in data.get("genres") or []:
        d = (item or {}).get("description")
        if isinstance(d, str) and d.strip() and d not in seen:
            seen.add(d)
            tags.append(d.strip())

    desc = data.get("short_description") or ""
    if not isinstance(desc, str):
        desc = ""
    desc = desc.strip()

    if not tags and not desc:
        # Steam returned a successful but empty payload — extremely rare,
        # treated as a permanent miss (nothing more to retry).
        return [], ""
    return tags, desc


# ---------- soup (must match data_prep.make_soup) --------------------------

def make_soup(tags: list[str], description: str) -> str:
    """Exact same recipe data_prep.py uses, so enriched rows feed Phase 2
    identically to natively-tagged rows."""
    tag_part = " ".join(tags)
    if description:
        return f"{tag_part} {description}".strip()
    return tag_part


# ---------- main pipeline --------------------------------------------------

def run() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)

    targets: list[int] = json.loads((ART / "enrich_targets.json").read_text())
    print(f"[enrich] {len(targets)} target app_ids loaded")

    # Pass 0 = first pass over all targets. Then up to RETRY_PASSES passes
    # over still-failing ids.
    session = requests.Session()
    pending = list(targets)
    pass_index = 0

    # Tracks the best raw payload we have for each app_id (so the final
    # write step can reread from cache and trust extract()).
    while pending and pass_index <= RETRY_PASSES:
        label = "initial" if pass_index == 0 else f"retry {pass_index}"
        print(f"\n[enrich] === pass {pass_index} ({label}) over {len(pending)} ids ===")
        still_failing: list[int] = []
        n_skipped_cached = 0
        n_ok = 0
        n_fail = 0

        for i, app_id in enumerate(pending, 1):
            cached = load_cached(app_id)
            if cached is not None and extract(cached, app_id) is not None:
                # Already have a non-failed cached response → nothing to do.
                n_skipped_cached += 1
                continue

            raw = fetch_one(session, app_id)
            time.sleep(REQUEST_DELAY_S)  # rate-limit even on errors

            if raw is None:
                still_failing.append(app_id)
                n_fail += 1
                if i % 25 == 0:
                    print(f"  [{i}/{len(pending)}] no-response (HTTP/JSON error)")
                continue

            # Always cache the raw response — even if success==False, so we
            # can see *why* a retry is needed without hitting the network.
            write_cache(app_id, raw)
            parsed = extract(raw, app_id)
            if parsed is None:
                still_failing.append(app_id)
                n_fail += 1
            else:
                n_ok += 1
            if i % 25 == 0:
                print(f"  [{i}/{len(pending)}] ok={n_ok} fail={n_fail} "
                      f"cached-skip={n_skipped_cached}")

        print(f"[enrich] pass {pass_index} done: ok={n_ok}, fail={n_fail}, "
              f"already-cached={n_skipped_cached}, still-failing={len(still_failing)}")
        pending = still_failing
        pass_index += 1
        if pending and pass_index <= RETRY_PASSES:
            print(f"[enrich] sleeping {BETWEEN_PASS_SLEEP_S}s before retry pass...")
            time.sleep(BETWEEN_PASS_SLEEP_S)

    # ---------- apply enrichment to games.json ---------------------------
    print("\n[enrich] applying enrichment to artifacts/games.json...")
    games_path = ART / "games.json"
    games = json.loads(games_path.read_text())

    # Snapshot "before" tag counts for the report.
    before_tag_count = {g["game_id"]: len(g["tags"]) for g in games}

    enriched_ids: set[int] = set()
    still_empty_ids: list[int] = []

    for g in games:
        gid = g["game_id"]
        if gid not in set(targets):
            continue
        cached = load_cached(gid)
        parsed = extract(cached, gid) if cached is not None else None
        if parsed is None:
            still_empty_ids.append(gid)
            continue
        tags, desc = parsed
        g["tags"] = tags
        g["description"] = desc
        g["soup"] = make_soup(tags, desc)
        if tags:
            enriched_ids.add(gid)
        else:
            # Cached, successful, but Steam genuinely had nothing — keep
            # the row recorded as a permanent miss so the report is honest.
            still_empty_ids.append(gid)

    games_path.write_text(json.dumps(games, ensure_ascii=False))
    print(f"[enrich] wrote {games_path} (in-place update)")

    # ---------- report ---------------------------------------------------
    targets_set = set(targets)
    by_id = {g["game_id"]: g for g in games}
    still_empty_ids = sorted({gid for gid in still_empty_ids if gid in targets_set})
    successfully_enriched = sorted(enriched_ids)
    recovery_rate = (
        len(successfully_enriched) / len(targets) if targets else 0.0
    )

    # Five before/after samples (prefer ones that actually changed).
    sample_changed = [
        gid for gid in successfully_enriched if gid in by_id
    ][:5]
    samples = []
    for gid in sample_changed:
        g = by_id[gid]
        samples.append({
            "app_id": gid,
            "name": g["name"],
            "tags_before": before_tag_count.get(gid, 0),
            "tags_after": len(g["tags"]),
        })

    report = {
        "targets_total": len(targets),
        "successfully_enriched": len(successfully_enriched),
        "still_empty": len(still_empty_ids),
        "recovery_rate": recovery_rate,
        "still_empty_app_ids": [
            {"app_id": gid, "name": by_id[gid]["name"] if gid in by_id else "?"}
            for gid in still_empty_ids
        ],
        "sample_before_after": samples,
    }
    (ART / "enrich_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n========== enrichment report ==========")
    print(f"targets total:          {report['targets_total']}")
    print(f"successfully enriched:  {report['successfully_enriched']}")
    print(f"still empty:            {report['still_empty']}")
    print(f"recovery rate:          {recovery_rate * 100:.2f}%")
    print(f"\nstill-empty app_ids ({len(still_empty_ids)}):")
    for entry in report["still_empty_app_ids"][:50]:
        print(f"  {entry['app_id']:>10}  {entry['name']}")
    if len(still_empty_ids) > 50:
        print(f"  ... and {len(still_empty_ids) - 50} more (see enrich_report.json)")
    print("\nsample before/after:")
    for s in samples:
        print(f"  {s['app_id']:>10}  {s['name'][:60]:<60} "
              f"tags {s['tags_before']} -> {s['tags_after']}")
    print("=======================================")
    return report


if __name__ == "__main__":
    run()
