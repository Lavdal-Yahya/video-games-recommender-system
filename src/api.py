"""
Phase 5 — FastAPI HTTP wrapper around the hybrid recommender.

This module is a THIN WRAPPER. All recommendation logic lives in src/hybrid.py
(routing, blending, filters, top-k); the API resolves the request, calls
hybrid.recommend(), and serialises the result. It must not re-score, re-rank,
or reshape — keeping one source of truth keeps the API from drifting away from
the module after later phases tune α.

Two routes (contract frozen — Phase 6 /ask and Phase 7 web app bind to it):

    POST /recommend
        request:  { "seed": str | int,
                    "k": int = 10,
                    "alpha": float | null = null,   # null -> hybrid.DEFAULT_ALPHA
                    "filters": { ... } | null = null }
        response: { "seed_id": int,
                    "seed_name": str,
                    "path": "blend" | "content_only" | "cf_only" | "popularity",
                    "alpha_effective": float,
                    "results": [ { "game_id": int, "name": str, "score": float }, ... ] }

    GET /games/search?q=<text>&limit=<int>
        response: { "query": str,
                    "results": [ { "game_id": int, "name": str }, ... ] }

Artifacts (games.json, id_name_lookup.json, TF-IDF matrix, CF cosine matrix)
are loaded ONCE at startup via the lifespan handler, so per-request cost is
just the cosine row reads + blend + top-k.
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import wave
from contextlib import asynccontextmanager
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

import requests

from src import hybrid, llm_query
from src.content import find_game_id, resolve_candidates, resolve_game_id
from src.vibe import derive_vibe_seed


# --- LLM-path seed-resolution threshold --------------------------------------
# The LLM may hallucinate a game title that isn't on Steam ("Zelda Breath of
# the Wild"). Phase-2's find_game_id() uses difflib cutoff 0.6 — fine for voice
# typos against a name the user actually said, but two lenient layers (the LLM
# guessing + 0.6 snap) compound into "confident-but-wrong" snaps.
#
# /ask gates on content.resolve_game_id's score >= 0.80. That resolver scans
# all 6000 catalog names and admits on max(char_ratio, token_containment_ratio)
# with deterministic number normalisation (1..20, word/roman/digit unify) and a
# popularity tiebreak. The containment ratio is what does the work — a real
# token-level overlap scores ~1.0 cleanly, while incidental letter coincidence
# ("splatoon 3" vs "pla_toon", char-ratio 0.778) and single-token overlap with
# unrelated games ("Zelda Breath of the Wild" -> shares only {wild} with
# Witcher 3, ratio 1/3) both fall well below the bar. 0.80 was validated by
# full re-probe (see Phase 6 Outputs in tasks.md for the margin numbers and the
# two named residuals: abbreviations -> no_seed; single-token typo ~0.02 margin
# -> Phase 7 confirmation tier).
ASK_SEED_THRESHOLD: float = 0.80


# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ID_NAME_JSON = ART / "id_name_lookup.json"

# Voice model locations (Phase 7.6). The files live under ./models/ (gitignored,
# they are NOT artifacts) and are fetched by src.fetch_voice_models. Both paths
# are checked at startup; if a file is missing the API still boots and only the
# voice endpoints fail (with a clean 503), so text mode keeps working.
MODELS_DIR = ROOT / "models"
STT_MODEL_DIR = MODELS_DIR / "whisper"
# Default to base.en. We measured base.en vs small.en on the cold path (see the
# Phase 7.6 Outputs note in tasks.md): base.en transcribes a 4 s clip in ~0.6 s
# on this laptop with no observed errors on common game-title vocabulary, while
# small.en is ~3x slower for no measurable accuracy gain on the same clips.
STT_MODEL_NAME = "base.en"
# Piper voice — en_US-lessac-medium is the default-recommended American English
# voice. Audibly cleaner than the browser SpeechSynthesis output that the
# Phase-7 stopgap shipped (the original complaint).
TTS_VOICE_PATH = MODELS_DIR / "piper" / "en_US-lessac-medium.onnx"


# --- typeahead state (loaded at startup) -------------------------------------
# Three parallel lists kept in catalog order; the search loop walks them once.
# Pre-computing the lowercase + alphanum-stripped forms is the only optimisation
# — search is otherwise a plain linear scan over 6000 names per query.
_NAMES: list[str] = []
_NAMES_LOWER: list[str] = []
_NAMES_ALNUM: list[str] = []
_GAME_IDS: list[int] = []

_ALNUM_STRIP_RE = re.compile(r"[^a-z0-9]")


def _load_typeahead_index() -> None:
    """Read the id_name lookup into the parallel lists above."""
    global _NAMES, _NAMES_LOWER, _NAMES_ALNUM, _GAME_IDS
    lookup = json.loads(ID_NAME_JSON.read_text())
    pairs = [(int(gid), name) for gid, name in lookup["id_to_name"].items()]
    _GAME_IDS = [gid for gid, _ in pairs]
    _NAMES = [name for _, name in pairs]
    _NAMES_LOWER = [name.lower() for name in _NAMES]
    _NAMES_ALNUM = [_ALNUM_STRIP_RE.sub("", n) for n in _NAMES_LOWER]


def _search_games(q: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Typeahead-style ranked match against the catalog names.

    Tiers (highest priority first):
        4 — exact case-insensitive match
        3 — name starts with the query
        2 — substring match
        1 — alphanumeric-only substring match ("witcher3" -> "The Witcher 3")
    If after the substring pass we still don't have `limit` results, top up
    with difflib.get_close_matches as a fuzzy fallback — same matcher Phase 2
    uses for find_game_id, just multi-result. No new index, no new dependency.
    """
    if not q or not q.strip():
        return []
    qn = q.strip().lower()
    qn_alnum = _ALNUM_STRIP_RE.sub("", qn)

    scored: list[tuple[int, int, int]] = []  # (-priority, name_length, row)
    for i, (name_l, name_a) in enumerate(zip(_NAMES_LOWER, _NAMES_ALNUM)):
        if name_l == qn:
            pri = 4
        elif name_l.startswith(qn):
            pri = 3
        elif qn in name_l:
            pri = 2
        elif qn_alnum and qn_alnum in name_a:
            pri = 1
        else:
            continue
        scored.append((-pri, len(name_l), i))
    scored.sort()
    rows = [i for _, _, i in scored[:limit]]

    if len(rows) < limit:
        # Fuzzy top-up — cutoff matches Phase 2's find_game_id default.
        already = set(rows)
        fuzzy = get_close_matches(qn, _NAMES_LOWER, n=limit * 2, cutoff=0.6)
        # _NAMES_LOWER can contain duplicates in principle; map via index.
        seen_lower: set[str] = set()
        for fl in fuzzy:
            if fl in seen_lower:
                continue
            seen_lower.add(fl)
            i = _NAMES_LOWER.index(fl)
            if i in already:
                continue
            rows.append(i)
            already.add(i)
            if len(rows) >= limit:
                break

    return [{"game_id": _GAME_IDS[i], "name": _NAMES[i]} for i in rows[:limit]]


# --- request / response models ----------------------------------------------
class RecommendRequest(BaseModel):
    """Frozen contract — mirror this in /recommend's docstring if it ever changes."""

    seed: str | int = Field(..., description="game name (fuzzy-matched) or game_id")
    k: int = Field(10, ge=1, le=100, description="number of results")
    alpha: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="blend weight; null uses hybrid.DEFAULT_ALPHA",
    )
    filters: dict[str, Any] | None = Field(
        None,
        description="optional {max_price, tags|genres}; see hybrid._passes_filters",
    )


class RecommendResultRow(BaseModel):
    game_id: int
    name: str
    score: float


class RecommendResponse(BaseModel):
    seed_id: int
    seed_name: str
    path: str  # 'blend' | 'content_only' | 'cf_only' | 'popularity'
    alpha_effective: float
    results: list[RecommendResultRow]


class SearchResultRow(BaseModel):
    game_id: int
    name: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultRow]


class ResolveCandidate(BaseModel):
    game_id: int
    name: str
    score: float


class ResolveResponse(BaseModel):
    admit: bool
    threshold: float
    candidates: list[ResolveCandidate]


class AskRequest(BaseModel):
    """Frozen contract — mirror in /ask's docstring if it ever changes."""

    text: str = Field(..., description="free-text request, e.g. 'chill like Stardew under $20'")
    k: int = Field(10, ge=1, le=100, description="number of results")
    alpha: float | None = Field(
        None, ge=0.0, le=1.0,
        description="blend weight; null uses hybrid.DEFAULT_ALPHA",
    )


# --- voice singletons (loaded once at startup) -------------------------------
# Loaded by the lifespan handler so the cost is paid at boot, not on the first
# /stt or /tts call. Each may be None if the underlying model file is missing
# on disk; the endpoints check that and return a clean 503 instead of crashing.
# This is the same "instance-down -> 503" pattern /ask already uses for Ollama.
_STT_MODEL: Any = None  # faster_whisper.WhisperModel
_TTS_VOICE: Any = None  # piper.PiperVoice


# --- app ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm everything up at boot so per-request latency is just compute, not I/O:
    typeahead index, the hybrid module's catalog + games.json, the TF-IDF
    matrix (via content._ensure_loaded), the CF cosine matrix, the per-row
    content norms used for routing, AND the voice singletons (Phase 7.6).

    Voice-model failures are non-fatal — text mode is the contract; voice is an
    additive enhancement. If a model file is missing we log it and leave the
    corresponding singleton as None; the /stt or /tts endpoint then 503s.
    """
    _load_typeahead_index()
    # Forces hybrid + content + collaborative singletons to load now.
    hybrid._ensure_loaded()
    hybrid._content_norms()
    # Touch a known seed so the CF cosine matrix actually deserialises.
    from src.collaborative import _ensure_loaded as cf_ensure_loaded
    cf_ensure_loaded()

    # Voice — load once, reuse on every /stt and /tts. Catch broadly so a
    # missing model file or a runtime mismatch never blocks text mode.
    global _STT_MODEL, _TTS_VOICE
    try:
        from faster_whisper import WhisperModel
        _STT_MODEL = WhisperModel(
            STT_MODEL_NAME,
            device="cpu",
            compute_type="int8",
            download_root=str(STT_MODEL_DIR),
        )
        print(f"[voice] STT loaded: faster-whisper {STT_MODEL_NAME}")
    except Exception as e:
        _STT_MODEL = None
        print(f"[voice] STT NOT loaded ({type(e).__name__}: {e}) — /stt will 503")
    try:
        if not TTS_VOICE_PATH.exists():
            raise FileNotFoundError(TTS_VOICE_PATH)
        from piper import PiperVoice
        _TTS_VOICE = PiperVoice.load(str(TTS_VOICE_PATH))
        print(f"[voice] TTS loaded: piper {TTS_VOICE_PATH.name}")
    except Exception as e:
        _TTS_VOICE = None
        print(f"[voice] TTS NOT loaded ({type(e).__name__}: {e}) — /tts will 503")

    yield


app = FastAPI(
    title="Game Recommender API",
    description="HTTP wrapper around the Phase 4 hybrid recommender.",
    version="0.5.0",
    lifespan=lifespan,
)

# CORS for the local Vite dev origin (Phase 7 frontend).
#
# We allow BOTH 127.0.0.1 and localhost because browsers treat them as distinct
# origins, and Vite's --host binds to 127.0.0.1 while the npm UX often shows
# "localhost" — accepting both removes a class of "works for me, broken for
# you" CORS failures that depend on which URL the user types.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --- routes ------------------------------------------------------------------
@app.post("/recommend", response_model=RecommendResponse)
def post_recommend(req: RecommendRequest) -> dict[str, Any]:
    """
    Hybrid recommendation. Frozen contract — do NOT change after Phase 5.

    Request:
        { "seed":    str | int,                # game name (fuzzy) or game_id
          "k":       int   (1..100, default 10),
          "alpha":   float | null (0..1, default null -> hybrid.DEFAULT_ALPHA),
          "filters": { "max_price": number,    # optional
                       "tags":    [str, ...],  # optional
                       "genres":  [str, ...] } # optional, treated like tags
                     | null (default null) }

    Response:
        { "seed_id":         int,
          "seed_name":       str,
          "path":            "blend" | "content_only" | "cf_only" | "popularity",
          "alpha_effective": float,
          "results": [ { "game_id": int, "name": str, "score": float }, ... ] }

    Unresolved seed -> 404 with a JSON error body, NOT a 500 stack trace.

    Implementation note: this endpoint is a thin wrapper. It calls
    hybrid.recommend(...) and returns the result verbatim. No re-ranking,
    no re-filtering, no post-processing happens here — if the module's
    behaviour ever drifts from the endpoint's, the endpoint is wrong.
    """
    try:
        return hybrid.recommend(
            seed=req.seed,
            k=req.k,
            alpha=req.alpha,
            filters=req.filters,
        )
    except KeyError as e:
        # find_game_id miss or game_id not in catalog. Pull the raw message off
        # e.args[0] so we don't ship KeyError's repr-quoted wrapper to clients.
        msg = e.args[0] if e.args else "unresolved seed"
        raise HTTPException(status_code=404, detail=str(msg))


@app.get("/games/search", response_model=SearchResponse)
def get_games_search(
    q: str = Query(..., description="search text"),
    limit: int = Query(10, ge=1, le=50, description="max results"),
) -> dict[str, Any]:
    """
    Typeahead search over catalog names. Frozen contract — do NOT change
    after Phase 5.

    Request (query string):
        q:     str               # search text (required)
        limit: int (1..50, default 10)

    Response:
        { "query":   str,
          "results": [ { "game_id": int, "name": str }, ... ] }

    Ranking is substring-priority (exact > startswith > substring > alphanumeric
    substring) with a difflib fuzzy top-up if substring matches don't fill the
    limit. Built from id_name_lookup.json only — no separate search index.
    """
    return {"query": q, "results": _search_games(q, limit=limit)}


@app.post("/ask")
def post_ask(req: AskRequest) -> dict[str, Any]:
    """
    Natural-language entry point. Phase 6's user-facing route.

    Pipeline (thin, by design — no recommendation logic here):
        text  --llm_query.parse_query-->  {seed, filters}, llm_raw
              --content.resolve_game_id-->  (game_id, score, matched_name)
              --[score >= ASK_SEED_THRESHOLD]--> hybrid.recommend(...)
                          OR
                                       --> { status: "no_seed", ... }

    Request:
        { "text":  str,                       # the free-text request
          "k":     int  (1..100, default 10),
          "alpha": float | null (0..1, default null -> hybrid.DEFAULT_ALPHA) }

    Response (resolved seed — the /recommend shape PLUS parse provenance):
        { "seed_id":         int,
          "seed_name":       str,
          "path":            "blend" | "content_only" | "cf_only" | "popularity",
          "alpha_effective": float,
          "results":         [ { game_id, name, score }, ... ],
          "parsed":          { "seed": str,        # what the LLM gave us
                               "filters": {...} }, # validated, allowed keys only
          "llm_raw":         str,                  # the model's exact output
          "match_score":     float }               # resolver confidence (>= threshold) }

    Response (no confident seed — clean, speakable, NOT a recommendation):
        { "status":  "no_seed",
          "message": str,                          # short, speakable
          "parsed":  { "seed": str|null, "filters": {...} },
          "llm_raw": str,
          "match_score": float,                    # 0.0 or the low score
          "match_candidate": str | null }          # the catalog name we WOULD have
                                                   # snapped to, for debugging

    Vibe-seeding fallback (Phase 7.5 rebuild): when ``parsed.seed is None`` the
    request is treated as a vibe and run through ``vibe.derive_vibe_seed``,
    which orchestrates three tiers in order:
        1. The LLM proposes the single most representative, well-known game
           for the vibe (a second deterministic Ollama call).
        2. The resolver grounds the proposed title (>= 0.80 admit gate) —
           hallucinated / out-of-catalog titles are filtered here.
        3. The original TF-IDF vibe path (``vibe_seed``) is the fallback if
           tiers 1-2 declined.
    If any tier produces a confident seed we run ``hybrid.recommend()`` from
    it and stamp the response with ``vibe_seeded: true`` and a
    ``seed_source`` of ``"llm"`` or ``"tfidf"`` for the defense. If all three
    decline we return the standard no_seed shape. The LLM is allowed to
    PROPOSE a seed name (always resolver-grounded); ``hybrid.recommend()``
    still writes every recommendation.

    Errors:
        - Ollama unreachable / network failure -> HTTP 503 with a JSON body.
        - All other failure modes (bad JSON from the model, weird filters, low
          match score) are handled INSIDE this route as the no_seed response.
    """
    # 1) Parse free text into {seed, filters}. parse_query() never raises on
    #    bad model output; it does raise on network failure (intentional —
    #    cleaner error than a degraded fallback that pretends Ollama is up).
    try:
        parsed, llm_raw = llm_query.parse_query(req.text)
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM backend unavailable: {type(e).__name__}: {e}",
        )

    seed_text = parsed.get("seed")
    filters = parsed.get("filters") or {}

    # 2) Resolve the LLM's seed string to a catalog game_id WITH CONFIDENCE.
    #    Empty seed (model honestly said null) -> match_score = 0.0; the vibe
    #    fallback below decides whether we can still anchor on the raw text.
    if seed_text:
        gid, score, matched_name = resolve_game_id(seed_text)
    else:
        gid, score, matched_name = None, 0.0, None

    # 3) Two honest gates: the LLM said null OR the resolver's confidence
    #    is below ASK_SEED_THRESHOLD. Two different next moves, though:
    #      - LLM said null -> the user described a vibe, not a title. Try
    #        the vibe-seeding fallback (Phase 7.5) before giving up.
    #      - LLM named a title but the resolver doesn't recognise it (e.g.
    #        "Zelda Breath of the Wild") -> the user DID name a specific
    #        game and we can't pretend a thematic vibe match is what they
    #        asked for. Go straight to no_seed.
    if gid is None or score < ASK_SEED_THRESHOLD:
        if not seed_text:
            # Vibe-seeding path: three-tier orchestrator. The LLM proposes a
            # seed name; resolve_game_id grounds it (>= 0.80); TF-IDF cosine
            # is the fallback (>= VIBE_THRESHOLD). See src/vibe.py.
            vibe = derive_vibe_seed(req.text)
            vibe_gid = vibe["seed_id"]
            if vibe_gid is not None:
                try:
                    rec = hybrid.recommend(
                        seed=int(vibe_gid),
                        k=req.k,
                        alpha=req.alpha,
                        filters=filters or None,
                    )
                except KeyError as e:
                    msg = e.args[0] if e.args else "unresolved seed"
                    raise HTTPException(status_code=404, detail=str(msg))
                # Provenance: which TIER produced the seed, plus the
                # tier-specific numbers (LLM proposal + resolver score, or
                # TF-IDF cosine + candidate). match_score keeps carrying the
                # PARSE-path resolver score (0.0 here — the LLM said null)
                # so the frozen Phase-6 response shape is preserved.
                rec["parsed"] = parsed
                rec["llm_raw"] = llm_raw
                rec["match_score"] = float(score)
                rec["vibe_seeded"] = True
                rec["seed_source"] = vibe["seed_source"]
                rec["llm_proposal"] = vibe["llm_proposal"]
                rec["llm_proposal_raw"] = vibe["llm_raw"]
                rec["resolver_score"] = vibe["resolver_score"]
                rec["resolver_name"] = vibe["resolver_name"]
                rec["tfidf_score"] = vibe["tfidf_score"]
                rec["tfidf_candidate"] = vibe["tfidf_candidate"]
                # Back-compat: keep the older vibe_score / vibe_candidate
                # surface populated with whichever tier actually fired so
                # any caller reading the prior shape still works.
                if vibe["seed_source"] == "llm":
                    rec["vibe_score"] = vibe["resolver_score"]
                    rec["vibe_candidate"] = vibe["resolver_name"]
                else:
                    rec["vibe_score"] = vibe["tfidf_score"]
                    rec["vibe_candidate"] = vibe["tfidf_candidate"]
                return rec

            # All three tiers declined: LLM had no usable proposal OR its
            # proposal didn't ground AND TF-IDF cosine was below threshold.
            msg = (
                "I recommend games similar to a game you like — try naming one, "
                "e.g. 'something chill like Stardew'."
            )
            return {
                "status": "no_seed",
                "message": msg,
                "parsed": parsed,
                "llm_raw": llm_raw,
                "match_score": float(score),
                "match_candidate": vibe["tfidf_candidate"],
                "vibe_seeded": False,
                "seed_source": None,
                "llm_proposal": vibe["llm_proposal"],
                "llm_proposal_raw": vibe["llm_raw"],
                "resolver_score": vibe["resolver_score"],
                "resolver_name": vibe["resolver_name"],
                "tfidf_score": vibe["tfidf_score"],
                "tfidf_candidate": vibe["tfidf_candidate"],
                "vibe_score": vibe["tfidf_score"] or 0.0,
            }
        else:
            msg = (
                f"I don't recognize '{seed_text}' in the catalog. "
                "Try naming a game that's on Steam."
            )
            return {
                "status": "no_seed",
                "message": msg,
                "parsed": parsed,
                "llm_raw": llm_raw,
                "match_score": float(score),
                "match_candidate": matched_name,
            }

    # 4) We have a confident LLM-named seed — hand the structured query to
    #    the hybrid recommender. /recommend's logic stays the one source of
    #    truth; the vibe path is never consulted on this branch.
    try:
        rec = hybrid.recommend(
            seed=int(gid),
            k=req.k,
            alpha=req.alpha,
            filters=filters or None,
        )
    except KeyError as e:
        # Shouldn't happen post-threshold (gid came from the catalog), but
        # surface cleanly if it ever does.
        msg = e.args[0] if e.args else "unresolved seed"
        raise HTTPException(status_code=404, detail=str(msg))

    # 5) /ask response = /recommend response + parse provenance.
    rec["parsed"] = parsed
    rec["llm_raw"] = llm_raw
    rec["match_score"] = float(score)
    return rec


@app.get("/resolve", response_model=ResolveResponse)
def get_resolve(
    q: str = Query(..., description="free-text seed (e.g. 'witcher') to resolve"),
) -> dict[str, Any]:
    """
    Top-5 resolver candidates with scores — the read-only sibling of /ask's
    internal seed-resolution step. Phase 7 reads this to detect franchise
    ambiguity (e.g. 'witcher' → Witcher 3 / Witcher 2 are near-tied) so the UI
    can surface a 'I picked X — did you mean Y?' confirmation without having
    to re-parse the original NL request.

    Request (query string):
        q: str   # free-text seed (required)

    Response:
        { "admit":     bool,    # true iff top candidate's score >= threshold
          "threshold": float,   # the /ask admit threshold (currently 0.80)
          "candidates": [ { "game_id": int, "name": str, "score": float }, ... top 5 ] }

    Implementation: thin wrapper around content.resolve_candidates(), which
    runs the EXACT scoring loop /ask's resolve_game_id uses (same
    max(char_ratio, token_containment), same (score DESC, n_interactions DESC)
    tiebreak). No recommendation logic, no filtering — read-only.
    """
    candidates = resolve_candidates(q, n=5)
    admit = bool(candidates and candidates[0][1] >= ASK_SEED_THRESHOLD)
    return {
        "admit": admit,
        "threshold": ASK_SEED_THRESHOLD,
        "candidates": [
            {"game_id": int(gid), "name": name, "score": float(score)}
            for gid, score, name in candidates
        ],
    }


@app.get("/health")
def get_health() -> dict[str, Any]:
    """Minimal liveness check — confirms artifacts are loaded into memory."""
    return {
        "ok": True,
        "n_catalog": len(_GAME_IDS),
        "default_alpha": hybrid.DEFAULT_ALPHA,
        # Voice readiness (Phase 7.6) — UI uses this to decide whether to show
        # the mic button. False here means /stt or /tts will 503; the frontend
        # falls back to typed input.
        "stt_ready": _STT_MODEL is not None,
        "tts_ready": _TTS_VOICE is not None,
    }


# --- voice routes (Phase 7.6, frozen contract) -------------------------------
@app.post("/stt")
async def post_stt(audio: UploadFile = File(...)) -> dict[str, Any]:
    """
    Speech-to-text. Frozen contract — do NOT change after Phase 7.6.

    Request (multipart/form-data):
        audio: file field — a recorded audio clip from the browser. We accept
               whatever the MediaRecorder produces (webm/opus on Chromium,
               ogg/opus on Firefox); faster-whisper decodes via libav so the
               container choice does not matter here.

    Response (application/json):
        { "text":       str,    # the transcript (may be empty if silence)
          "duration_s": float,  # length of the clip the model saw
          "model":      str }   # which whisper model produced the transcript

    Errors:
        503 — STT model not loaded (model file missing / load failed at boot).
        400 — empty upload.
        500 — decode / transcription failure (logged with the exception type).

    The frontend is a CAPTURE device only: it records the mic and POSTs the
    clip here. All transcription happens server-side — the browser never sees
    a recognizer.
    """
    if _STT_MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "STT model not loaded. Run "
                "`python -m src.fetch_voice_models` and restart the server."
            ),
        )

    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty audio upload")

    # faster-whisper accepts a file path or a numpy array; the simplest reliable
    # path is to spill the upload to a temp file (libav reads the container off
    # the bytes regardless of suffix).
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as tmp:
        tmp.write(payload)
        tmp.flush()
        try:
            segments, info = _STT_MODEL.transcribe(
                tmp.name,
                language="en",
                vad_filter=True,  # drop leading/trailing silence — shorter clip
                beam_size=1,      # greedy is enough for short prompts and ~3x faster
            )
            # segments is a generator — consume it to actually run the decode.
            text = " ".join(seg.text.strip() for seg in segments).strip()
            duration = float(getattr(info, "duration", 0.0))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"transcription failed: {type(e).__name__}: {e}",
            )

    return {
        "text": text,
        "duration_s": duration,
        "model": f"faster-whisper:{STT_MODEL_NAME}",
    }


class TTSRequest(BaseModel):
    """Frozen contract — mirror in /tts's docstring if it ever changes."""

    text: str = Field(..., min_length=1, max_length=2000,
                      description="the text to synthesise (1..2000 chars)")


@app.post("/tts")
def post_tts(req: TTSRequest) -> Response:
    """
    Text-to-speech. Frozen contract — do NOT change after Phase 7.6.

    Request (application/json):
        { "text": str   # 1..2000 chars; the line to speak }

    Response (audio/wav):
        a single WAV file the browser plays via an <audio> element. We return
        WAV (not opus/mp3) because Piper synthesises PCM samples and WAV is the
        zero-dependency container for PCM — every browser plays it.

    Errors:
        503 — TTS voice not loaded (.onnx missing / load failed at boot).
        500 — synthesis failure.

    The frontend is a PLAYBACK device only: it POSTs text here and plays the
    returned audio. No browser speechSynthesis is involved.
    """
    if _TTS_VOICE is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "TTS voice not loaded. Run "
                "`python -m src.fetch_voice_models` and restart the server."
            ),
        )

    # Synthesise into an in-memory WAV. Piper writes PCM frames into a
    # wave.Wave_write; we hand it a BytesIO buffer so we never touch disk.
    buf = io.BytesIO()
    try:
        with wave.open(buf, "wb") as wav_out:
            _TTS_VOICE.synthesize_wav(req.text, wav_out)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"synthesis failed: {type(e).__name__}: {e}",
        )

    return Response(content=buf.getvalue(), media_type="audio/wav")
