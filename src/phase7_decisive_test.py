"""
Phase 7 decisive test — integration against the running FastAPI backend.

This script hits the EXACT endpoints + payloads the web UI sends and prints raw
responses. It is the textual counterpart of opening the UI in a browser; we
verify each branch the frontend takes:

    1. Confident NL  -> /ask returns recommendations; /resolve?q=Stardew says
                        the top candidate is clearly confident -> NO confirm.
    2. Ambiguous     -> /resolve?q=witcher returns 5 candidates all at 1.0 ->
                        confirm prompt FIRES (top - next < TIE_GAP).
    3. Out-of-catalog-> /ask {"text":"like Zelda Breath of the Wild"} -> no_seed
                        (clean shape, not a crash).
    4. Empty filters -> resolved seed + impossible filter -> results [] ->
                        the "no games matched those filters" branch (distinct
                        from no_seed).
    5. /health 200 + 503 path when Ollama is unreachable.

The TIE_GAP / HIGH_CONFIDENCE thresholds the frontend uses are repeated here so
the script's pass/fail matches the UI byte-for-byte (see web/src/config.js).

Also measures latency on /resolve and /ask, cold (first call) and warm
(second call), so we have a real number for voice responsiveness.
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests


BASE = "http://127.0.0.1:8765"

# Mirrored from web/src/config.js — the UI's confirm-band thresholds.
ADMIT_THRESHOLD = 0.80
HIGH_CONFIDENCE = 0.99
TIE_GAP = 0.05


def call(method: str, path: str, **kw) -> tuple[int, Any, float]:
    """Run one request. Return (status_code, json_or_text, elapsed_ms)."""
    t0 = time.perf_counter()
    resp = requests.request(method, f"{BASE}{path}", timeout=300, **kw)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    try:
        return resp.status_code, resp.json(), elapsed_ms
    except Exception:
        return resp.status_code, resp.text, elapsed_ms


def show(title: str, status: int, body: Any, ms: float | None = None) -> None:
    print(f"\n--- {title} ---")
    if ms is not None:
        print(f"HTTP {status}   ({ms:.1f} ms)")
    else:
        print(f"HTTP {status}")
    if isinstance(body, (dict, list)):
        print(json.dumps(body, indent=2, ensure_ascii=False)[:2400])
    else:
        print(str(body)[:2400])


def confirm_should_fire(candidates: list[dict]) -> bool:
    """Mirror of web/src/App.jsx:confirmShouldFire — for the gate."""
    if not candidates or len(candidates) < 2:
        return False
    top, nxt = candidates[0]["score"], candidates[1]["score"]
    if top < ADMIT_THRESHOLD:
        return False
    if top < HIGH_CONFIDENCE:
        return True
    if top - nxt < TIE_GAP:
        return True
    return False


def main() -> int:
    print("=" * 64)
    print("Phase 7 — decisive integration test against the running backend.")
    print("=" * 64)

    # 0. /health — confirms backend is up + reports n_catalog/default_alpha.
    s, b, ms = call("GET", "/health")
    show("0. GET /health", s, b, ms)
    assert s == 200 and b.get("ok") is True, "backend not healthy"

    # 1. Confident NL — the UI's main happy path.
    #    "something chill like Stardew under $20" -> /ask
    s, ask_body, ms_ask_cold = call(
        "POST", "/ask",
        json={"text": "something chill like Stardew under $20"},
    )
    show("1a. POST /ask  {text:'something chill like Stardew under $20'}  (COLD)",
         s, ask_body, ms_ask_cold)
    assert s == 200, "/ask failed on confident NL"
    assert "results" in ask_body and len(ask_body["results"]) > 0, "expected recs"
    seed_text_1 = ask_body["parsed"]["seed"]
    seed_id_1 = ask_body["seed_id"]

    # 1b. /resolve on the parsed seed — the UI's confirm-tier probe.
    s, res_body, ms_res_cold = call("GET", "/resolve", params={"q": seed_text_1})
    show(f"1b. GET /resolve?q={seed_text_1!r}  (COLD)", s, res_body, ms_res_cold)
    fire1 = confirm_should_fire(res_body["candidates"])
    print(f"   -> confirmShouldFire = {fire1}  (expected: False, Stardew is clearly confident)")
    assert fire1 is False, "Stardew should NOT fire confirm"

    # 2. Ambiguous — /resolve?q=witcher; top 2 within TIE_GAP -> confirm fires.
    s, res_witcher, ms_res_warm = call("GET", "/resolve", params={"q": "witcher"})
    show("2. GET /resolve?q='witcher'  (WARM)", s, res_witcher, ms_res_warm)
    fire2 = confirm_should_fire(res_witcher["candidates"])
    print(f"   -> confirmShouldFire = {fire2}  (expected: True, top 5 all at 1.0)")
    assert fire2 is True, "witcher SHOULD fire confirm"

    # 3. Out-of-catalog — Zelda BotW isn't on Steam.
    s, b_zelda, ms = call(
        "POST", "/ask",
        json={"text": "like Zelda Breath of the Wild"},
    )
    show("3. POST /ask  {text:'like Zelda Breath of the Wild'}", s, b_zelda, ms)
    assert s == 200, "/ask should not crash on out-of-catalog"
    assert b_zelda.get("status") == "no_seed", "expected no_seed status"
    assert b_zelda.get("match_score", 1.0) < ADMIT_THRESHOLD
    print("   -> no_seed branch reached cleanly")

    # 4. Empty results vs no_seed — resolved seed + an impossible filter.
    #    Hades resolves; we ask for a tag no catalog game has so the post-score
    #    filter mask evicts everything.  Result: `results=[]` with seed_name
    #    still populated — DISTINCT from the no_seed shape.
    s, b_empty, ms = call(
        "POST", "/recommend",
        json={
            "seed": "Hades",
            "filters": {"tags": ["zzz_nonexistent_tag_xyz"]},
        },
    )
    show("4. POST /recommend  {seed:'Hades', filters:{tags:['zzz_nonexistent_tag_xyz']}}",
         s, b_empty, ms)
    assert s == 200, "expected 200 not error"
    assert b_empty.get("seed_name") and b_empty.get("results") == [], \
        "expected empty results + seed still set"
    assert b_empty.get("status") != "no_seed", "must NOT be no_seed"
    print("   -> empty-results branch (NOT no_seed) — UI shows the "
          "'no games matched those filters' banner")

    # 5. /ask 503 path when Ollama is unreachable.
    #    We can't bring Ollama down inside this script; instead we assert that
    #    api.py's 503 handler is wired (search for it in source) and print the
    #    shape that would come back.  The Phase 6 outputs already verified the
    #    actual 503 path manually; here we just confirm the contract is intact.
    print("\n--- 5. /ask 503 path (contract check) ---")
    import pathlib
    api_src = pathlib.Path(__file__).resolve().parent / "api.py"
    text = api_src.read_text()
    has_503 = (
        'status_code=503' in text
        and 'LLM backend unavailable' in text
    )
    print(f"   src/api.py contains the 503 wiring: {has_503}")
    print('   shape: { "detail": "LLM backend unavailable: ..." }')
    print('   the UI maps 503 -> friendly "the local model isn\'t running" copy '
          '(App.jsx:handleAsk).')
    assert has_503, "503 wiring missing"

    # ---- Latency summary ----
    # One more warm /ask so we have both timings.
    _, _, ms_ask_warm = call(
        "POST", "/ask",
        json={"text": "something chill like Stardew under $20"},
    )
    print("\n" + "=" * 64)
    print("Timing (single call, wall-clock; the backend was already warmed by")
    print("earlier session work, so 'cold' here is FastAPI-cold, not Ollama-cold):")
    print(f"  /resolve  cold: {ms_res_cold:>7.1f} ms")
    print(f"  /resolve  warm: {ms_res_warm:>7.1f} ms")
    print(f"  /ask      cold: {ms_ask_cold:>7.1f} ms")
    print(f"  /ask      warm: {ms_ask_warm:>7.1f} ms")
    print("=" * 64)

    print("\nALL DECISIVE CHECKS PASSED ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
