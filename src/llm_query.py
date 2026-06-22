"""
Phase 6 — natural language -> {"seed": str|null, "filters": {...}}.

The LLM PARSES ONLY. It maps free-form English ("something chill like Stardew
under $20") into a structured query the hybrid recommender understands; it
NEVER generates, ranks, or describes recommendations. The hybrid (Phase 4)
does that, period. If you ever find yourself tempted to let the model write
game suggestions here, stop — that's the line the project design is built on.

Model: llama3.1:8b via Ollama (the local instruct model already pulled in
Phase 0). Deterministic settings:
  - temperature = 0
  - seed       = 42
  - format     = "json"   (Ollama constrains output to syntactically valid JSON)
so a given input always yields the same parse — important for the report and
for the deterministic-parse sanity check.

Allowed filter keys (EXACTLY what hybrid.recommend() applies; anything else is
silently dropped from the parse — a filter the pipeline ignores would be a lie
to the user):
    - max_price : number
    - tags      : list[str]
    - genres    : list[str]

Mood / vibe words ("chill", "relaxing", "hard", "cozy", "dark") get mapped into
tags, because the hybrid only has tag intersection — there's no separate mood
filter to honour.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests


# --- Ollama wiring -----------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
# 120s covers the cold model-load on the first call (loading 4.9 GB of weights
# off disk can take ~30-60 s) AND a slow CPU-bound generation. Warm calls return
# in ~1-3 s. A malformed parse falls back without raising anyway, so a slow
# first call won't crash the API.
OLLAMA_TIMEOUT = 120.0

# What the model is told it may emit. Mirrors the hybrid filter contract exactly
# so the prompt and the runtime can never drift apart.
ALLOWED_FILTER_KEYS: tuple[str, ...] = ("max_price", "tags", "genres")


# Few-shot prompt. We bake the contract + two NL->JSON examples + the
# "don't guess a title" rule directly into a single prompt. Keeping system and
# user merged simplifies the request; format:"json" + temperature=0 do the
# heavy lifting on output discipline.
SYSTEM_PROMPT = """You are a strict JSON parser for a video-game recommender.
Given a user's natural-language request, output ONE JSON object and nothing else.

Schema (exactly these top-level keys):
{
  "seed":    string OR null,
  "filters": object
}

Rules for "seed":
- "seed" is the NAME of a specific video game the user references (a real,
  named game like "Stardew Valley", "Hades", "The Witcher 3").
- If the user does NOT name a specific game, OR you do not recognize the title
  they named, set "seed" to null. DO NOT invent or guess a title. Returning
  null is the correct, honest answer when there is no recognizable named game.
- "Something like X" / "similar to X" / "in the style of X" -> seed is X.

Rules for "filters" (ONLY these keys are allowed; omit any you do not need):
- "max_price": a NUMBER (the price ceiling in dollars). Example: "under $20" -> 20.
- "tags":      a LIST of short lowercase strings describing the gameplay style,
               theme, or mood the user wants. Mood words ("chill", "relaxing",
               "hard", "cozy", "dark", "scary") go HERE as tags. Genre/style
               words ("roguelike", "rpg", "open world", "fps") also go HERE.
- "genres":    a LIST of short lowercase strings for genre labels if the user
               clearly named a genre (e.g. "rpg", "shooter", "platformer").
               You may put the same value in both "tags" and "genres".

Never output any other top-level key. Never output keys like "mood", "vibe",
"platform", "developer", "year" — they are NOT supported by the recommender
and will be silently dropped.

Output ONLY the JSON object. No prose, no markdown, no code fences.
"""


# Two or three short examples — small enough that an 8B instruct model
# consistently follows the schema, large enough to anchor the mood->tag rule
# and the "null seed when no game named" rule.
FEW_SHOT_EXAMPLES = """Examples:

Input: something chill like Stardew under $20
Output: {"seed": "Stardew Valley", "filters": {"max_price": 20, "tags": ["chill", "cozy"]}}

Input: a hard roguelike like Hades
Output: {"seed": "Hades", "filters": {"tags": ["hard", "difficult", "roguelike"]}}

Input: open-world rpg like the witcher
Output: {"seed": "The Witcher 3", "filters": {"tags": ["open world", "rpg"], "genres": ["rpg"]}}

Input: just recommend me something fun
Output: {"seed": null, "filters": {"tags": ["fun"]}}

Input: recommend something like Glorbax Quest 9
Output: {"seed": null, "filters": {}}
"""


def _build_prompt(user_text: str) -> str:
    """Glue system rules + few-shot examples + the current request."""
    return (
        SYSTEM_PROMPT
        + "\n"
        + FEW_SHOT_EXAMPLES
        + "\nInput: "
        + user_text.strip()
        + "\nOutput: "
    )


# --- low-level Ollama call ---------------------------------------------------
def _call_ollama(prompt: str) -> str:
    """
    Hit Ollama's /api/generate with the deterministic-parse options.

    Raises requests.RequestException on a network/HTTP failure — the /ask
    handler turns that into a clean 503 instead of a stack trace. parse_query
    itself catches malformed-JSON responses (different failure mode) and falls
    back to an empty parse.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,        # one-shot, simpler to consume
        "format": "json",       # constrain output to syntactically valid JSON
        "options": {
            "temperature": 0,   # deterministic decoding
            "seed": 42,         # locked seed for reproducible parses
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    body = r.json()
    # Ollama returns the generated text under "response".
    return str(body.get("response", "")).strip()


# --- JSON cleanup + validation ----------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    """
    Defensive: even with format='json' a chatty model can occasionally wrap
    its output in ```json ... ``` fences. Strip them before json.loads.
    """
    return _FENCE_RE.sub("", text).strip()


def _coerce_filters(raw: Any) -> dict[str, Any]:
    """
    Keep only the allowed filter keys, coerce their values, drop everything else.

    Anything outside ALLOWED_FILTER_KEYS is silently discarded — the hybrid
    pipeline ignores unknown keys, so emitting them in the response would
    promise the user a filter that never actually fires.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}

    # max_price -> number (accept "20" as well as 20)
    if "max_price" in raw and raw["max_price"] is not None:
        try:
            out["max_price"] = float(raw["max_price"])
        except (TypeError, ValueError):
            pass  # malformed -> drop, don't crash

    # tags / genres -> list[str]; accept a single string too, in case the model
    # forgets the list wrapper.
    for key in ("tags", "genres"):
        if key in raw and raw[key]:
            val = raw[key]
            if isinstance(val, str):
                val = [val]
            if isinstance(val, list):
                cleaned = [str(x).strip().lower() for x in val if str(x).strip()]
                if cleaned:
                    out[key] = cleaned

    return out


def _empty_parse() -> dict[str, Any]:
    """The honest no-info fallback. Used whenever the model output is unusable."""
    return {"seed": None, "filters": {}}


# --- public API --------------------------------------------------------------
def parse_query(text: str) -> tuple[dict[str, Any], str]:
    """
    Parse a free-text request into the structured {seed, filters} dict.

    Returns (parsed, llm_raw) where:
      - parsed:  validated {"seed": str|null, "filters": {max_price?, tags?, genres?}}
      - llm_raw: the model's exact response string, included so /ask can show
                 the defense WHY a given seed was picked (or rejected).

    Robustness contract:
      - Empty / whitespace-only input -> ({"seed": null, "filters": {}}, "").
      - Unparseable JSON              -> ({"seed": null, "filters": {}}, raw).
      - Unknown filter keys           -> dropped (NOT echoed back).
      - Never raises on a bad model output. Network / HTTP errors DO propagate;
        the /ask handler maps them to a 503-style JSON.
    """
    if not text or not text.strip():
        return _empty_parse(), ""

    prompt = _build_prompt(text)
    llm_raw = _call_ollama(prompt)

    # Step 1: strip stray markdown fences if the model added them.
    cleaned = _strip_code_fences(llm_raw)

    # Step 2: try to load it as JSON. Honest fall back on failure instead of
    # raising — the user's request still deserves a clean response.
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return _empty_parse(), llm_raw

    if not isinstance(data, dict):
        return _empty_parse(), llm_raw

    # Step 3: pull out seed; coerce to str|None, never invent.
    seed = data.get("seed")
    if seed is not None:
        seed = str(seed).strip()
        if not seed or seed.lower() in {"null", "none"}:
            seed = None

    # Step 4: keep only the allowed filter keys.
    filters = _coerce_filters(data.get("filters"))

    return {"seed": seed, "filters": filters}, llm_raw


# --- Phase 7.5 tier 1: vibe-seed proposal ------------------------------------
# The parser above sets seed=null whenever the user did not NAME a game. For
# the vibe path we then ask the SAME model a different, narrower question:
# "given this vibe, what's the single most representative well-known game?".
# This is a SECOND, separate call (different prompt, same deterministic Ollama
# settings). Its output is NEVER trusted directly — it is always grounded
# through resolve_game_id before becoming a seed. So the LLM is still only
# allowed to propose a NAME; hybrid.recommend() still writes every pick.
VIBE_PROPOSAL_PROMPT = """You name a representative SEED game for a vibe.

STEP 1 — classify the input first. Decide ONE of:
  (A) The input describes a video-game vibe (a mood, theme, gameplay style, or
      reference to a kind of game). Examples: "a relaxing farming game",
      "atmospheric horror", "fast-paced platformer", "cozy life sim".
  (B) The input is NOT a video-game vibe. This includes:
        - empty input, punctuation only, or a single function word
          ("the", "a", "and", "of", "is");
        - gibberish or random keystrokes ("asdkfjqwer", "qwerty", "xxx");
        - greetings, math, random nouns, or anything else that doesn't
          describe how a game should FEEL or PLAY.

STEP 2 — output ONE JSON object and nothing else.
  If (A): {"game": "<the single most representative, well-known game title>"}
  If (B): {"game": null}

ABSOLUTE RULES — read carefully, they override everything else:
- Class (B) ALWAYS gets null. NEVER pick a game for a (B) input. Do not
  default to a popular game. Do not guess. Do not "be helpful". null is the
  correct, required answer.
- If you are unsure whether the input is (A) or (B), it is (B) — output null.
- Output ONLY the JSON. No prose, no markdown, no code fences.
- For class (A): pick ONE famous, recognizable game most people would name
  first for that vibe. Use its common English title.

Examples (read all of them — the (B) examples are as important as the (A) ones):

Vibe: asdkfjqwer
Output: {"game": null}

Vibe: the
Output: {"game": null}

Vibe: qwerty
Output: {"game": null}

Vibe: hello
Output: {"game": null}

Vibe: 2+2
Output: {"game": null}

Vibe: xkcd
Output: {"game": null}

Vibe: a relaxing farming game
Output: {"game": "Stardew Valley"}

Vibe: a hard fast roguelike
Output: {"game": "Hades"}

Vibe: an open-world rpg with a big story
Output: {"game": "The Witcher 3"}

Vibe: atmospheric horror
Output: {"game": "Silent Hill 2"}

Vibe: a cozy game to relax after work
Output: {"game": "Animal Crossing: New Horizons"}
"""


def _build_vibe_proposal_prompt(vibe_text: str) -> str:
    return (
        VIBE_PROPOSAL_PROMPT
        + "\nVibe: "
        + vibe_text.strip()
        + "\nOutput: "
    )


def propose_vibe_seed_title(vibe_text: str) -> tuple[str | None, str]:
    """
    Tier 1 of Phase 7.5's three-tier vibe-seed derivation.

    Ask the LLM to propose the single most representative, well-known game for
    a vibe description. Returns ``(title, llm_raw)`` where ``title`` is either
    the proposed title (a string the resolver still has to ground) or ``None``
    if the model declined or its output was unusable.

    The LLM's role here is to supply SEMANTICS that pure lexical TF-IDF
    cannot: it knows *Stardew Valley* is the canonical "relaxing farming"
    anchor, not *Farming Simulator*. The proposal is NEVER used directly —
    ``resolve_game_id`` grounds it against the catalog with the same strict
    >= 0.80 admit gate used everywhere else.

    Deterministic Ollama settings (temperature=0, seed=42, format="json") so
    the same vibe text always produces the same proposal — important for the
    decisive test and for reproducibility.
    """
    if not vibe_text or not vibe_text.strip():
        return None, ""

    prompt = _build_vibe_proposal_prompt(vibe_text)
    llm_raw = _call_ollama(prompt)
    cleaned = _strip_code_fences(llm_raw)

    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return None, llm_raw

    if not isinstance(data, dict):
        return None, llm_raw

    title = data.get("game")
    if title is None:
        return None, llm_raw
    title = str(title).strip()
    if not title or title.lower() in {"null", "none", "n/a", "unknown"}:
        return None, llm_raw
    return title, llm_raw
