"""
Phase 7.5 (rebuild) — decisive test.

Exercises the three-tier vibe-seed derivation end-to-end on the locked anchor,
decline, and sanity probes. Mirrors what /ask does: parse -> resolve -> on
null-seed, derive_vibe_seed -> hybrid.recommend. Prints tier provenance + the
top-5 recommendations + per-call latency, so the report can be read directly
from the raw output without any post-processing.

Run from the project root:
    python -m src.phase75_decisive_test
"""

from __future__ import annotations

import time

from src import hybrid, llm_query
from src.content import resolve_game_id
from src.vibe import derive_vibe_seed


# ASK_SEED_THRESHOLD lives in src/api.py; duplicated as a literal here to keep
# this probe importable without spinning the FastAPI app. Same value, same gate.
ASK_SEED_THRESHOLD = 0.80


# Probes from the phase brief. The "expected" string is documentation for the
# reader — it isn't matched programmatically; the human (and the LLM logs)
# read the seed name and the top-5 cluster and judge thematic correctness.
ANCHORS: list[tuple[str, str]] = [
    ("a relaxing farming game",
     "cozy farm/life sim (Stardew / Coral Island / My Time at Portia class), "
     "NOT Farming Simulator / a vehicle sim"),
    ("an open-world rpg with a big story",
     "story-driven RPG (Witcher 3 / Skyrim / Cyberpunk / Fallout class), "
     "NOT a grindy looter-MMO (Warframe / Path of Exile / Black Desert)"),
    ("a hard fast roguelike",
     "real roguelike/roguelite (Hades / Dead Cells / Slay the Spire class)"),
    ("a cozy game to relax after work",
     "cozy/wholesome title, NOT competitive or twitch"),
]

# Filter-split probe — vibe-only seed AND price filter together. We FORCE the
# vibe path even if the LLM would have named a seed directly, by calling the
# vibe orchestrator on the raw text and passing the parsed price filter to
# hybrid.recommend(). This exercises the locked "filters CONSTRAIN, vibe SEEDS"
# split end-to-end in a single call.
FILTER_PROBE: tuple[str, str] = (
    "a relaxing farming game under $20",
    "cozy-sim anchor AND every result <= $20",
)

DECLINES: list[str] = ["asdkfjqwer", "", "the"]

NAMED_SANITY: str = "something chill like Stardew under $20"


def _ask_pipeline(text: str, force_vibe_for_filter_split: bool = False) -> dict:
    """
    Reproduce /ask's pipeline without HTTP. Returns a dict the printer below
    can read; carries tier provenance and timings.
    """
    out: dict = {"text": text}

    # --- LLM parse (always run) ---
    t0 = time.perf_counter()
    parsed, llm_raw = llm_query.parse_query(text)
    parse_ms = (time.perf_counter() - t0) * 1000.0
    out["parsed"] = parsed
    out["llm_raw_parse"] = llm_raw
    out["parse_ms"] = parse_ms

    seed_text = parsed.get("seed")
    filters = parsed.get("filters") or {}
    out["filters"] = filters

    # Named-seed sanity: resolve the LLM-named title and run /recommend
    # directly. The vibe orchestrator is NOT consulted on this branch — same
    # as /ask. The filter-probe override below skips this branch.
    if seed_text and not force_vibe_for_filter_split:
        gid, score, matched_name = resolve_game_id(seed_text)
        out["resolver_score"] = float(score)
        out["resolver_name"] = matched_name
        if gid is not None and score >= ASK_SEED_THRESHOLD:
            out["path_taken"] = "named_seed"
            t1 = time.perf_counter()
            rec = hybrid.recommend(seed=int(gid), k=5, filters=filters or None)
            out["recommend_ms"] = (time.perf_counter() - t1) * 1000.0
            out["seed_id"] = int(gid)
            out["seed_name"] = matched_name
            out["results"] = rec["results"]
            return out

    # Vibe path — either because parse returned seed=null or because the
    # filter-split probe forced it.
    out["path_taken"] = "vibe"
    t1 = time.perf_counter()
    vibe = derive_vibe_seed(text)
    vibe_ms = (time.perf_counter() - t1) * 1000.0
    out["vibe_ms"] = vibe_ms
    out["vibe"] = vibe

    if vibe["seed_id"] is None:
        out["result"] = "no_seed"
        return out

    t2 = time.perf_counter()
    rec = hybrid.recommend(
        seed=int(vibe["seed_id"]), k=5, filters=filters or None,
    )
    out["recommend_ms"] = (time.perf_counter() - t2) * 1000.0
    out["seed_id"] = int(vibe["seed_id"])
    out["seed_name"] = (
        vibe["resolver_name"] if vibe["seed_source"] == "llm"
        else vibe["tfidf_candidate"]
    )
    out["results"] = rec["results"]
    return out


def _print(o: dict, header: str) -> None:
    print("=" * 78)
    print(header)
    print("-" * 78)
    print(f"input:           {o['text']!r}")
    print(f"parsed.seed:     {o['parsed'].get('seed')!r}")
    print(f"parsed.filters:  {o['parsed'].get('filters')!r}")
    print(f"parse_ms:        {o['parse_ms']:.0f}")
    print(f"path_taken:      {o.get('path_taken')!r}")
    if o.get("path_taken") == "named_seed":
        print(f"resolver_score:  {o.get('resolver_score'):.4f}")
        print(f"resolver_name:   {o.get('resolver_name')!r}")
    else:
        v = o.get("vibe", {})
        print(f"vibe_ms (3-tier total): {o.get('vibe_ms', 0.0):.0f}")
        print(f"  tier1 llm_proposal:  {v.get('llm_proposal')!r}")
        print(f"  tier1 llm_raw:       {v.get('llm_raw')!r}")
        print(f"  tier2 resolver_score: {v.get('resolver_score')}")
        print(f"  tier2 resolver_name:  {v.get('resolver_name')!r}")
        print(f"  tier3 tfidf_score:    {v.get('tfidf_score')}")
        print(f"  tier3 tfidf_candidate:{v.get('tfidf_candidate')!r}")
        print(f"  -> seed_source:       {v.get('seed_source')!r}")
    if o.get("result") == "no_seed":
        print("FINAL: no_seed")
        return
    print(f"seed_id:         {o.get('seed_id')}")
    print(f"seed_name:       {o.get('seed_name')!r}")
    print(f"recommend_ms:    {o.get('recommend_ms', 0.0):.0f}")
    print("top-5 results:")
    for i, r in enumerate(o.get("results", []), 1):
        # The /recommend rows carry price when present; print it if there.
        price = r.get("price")
        price_s = f"  price=${price:.2f}" if isinstance(price, (int, float)) else ""
        print(f"  {i}. {r['name']!r}  (gid={r['game_id']}, score={r['score']:.4f}){price_s}")


def main() -> None:
    print("\n###### ANCHOR PROBES ######\n")
    for text, expected in ANCHORS:
        o = _ask_pipeline(text)
        _print(o, header=f"ANCHOR: {text!r}\nexpected class: {expected}")

    print("\n###### FILTER-SPLIT PROBE (force vibe path) ######\n")
    text, expected = FILTER_PROBE
    o = _ask_pipeline(text, force_vibe_for_filter_split=True)
    _print(o, header=f"FILTER-SPLIT: {text!r}\nexpected: {expected}")
    # Explicitly check the $20 cap on every result.
    if "results" in o:
        prices = [r.get("price") for r in o["results"]]
        print(f"  prices: {prices}")
        violations = [p for p in prices if isinstance(p, (int, float)) and p > 20.0]
        print(f"  $20 violations: {violations} (must be empty)")

    print("\n###### DECLINE PROBES ######\n")
    for text in DECLINES:
        o = _ask_pipeline(text)
        _print(o, header=f"DECLINE: {text!r}\nexpected: no_seed")

    print("\n###### SANITY: NAMED SEED (must NOT take vibe path) ######\n")
    o = _ask_pipeline(NAMED_SANITY)
    _print(o, header=f"NAMED: {NAMED_SANITY!r}\nexpected: path_taken='named_seed', NOT 'vibe'")

    print("\n###### DETERMINISM ######\n")
    a = _ask_pipeline("a relaxing farming game")
    b = _ask_pipeline("a relaxing farming game")
    same_proposal = (
        a["vibe"]["llm_proposal"] == b["vibe"]["llm_proposal"]
    )
    same_seed = a.get("seed_id") == b.get("seed_id")
    print(f"same llm_proposal: {same_proposal}  ({a['vibe']['llm_proposal']!r})")
    print(f"same seed_id:      {same_seed}  ({a.get('seed_id')!r})")


if __name__ == "__main__":
    main()
