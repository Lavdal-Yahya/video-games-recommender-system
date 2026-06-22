"""
Phase 7.5 — vibe-based seeding (three-tier derivation).

When the LLM parser returns ``seed=null`` because the user described a vibe
instead of naming a game ("a relaxing farming game", "a hard fast roguelike"),
we still want to recommend something — but the recommender is fundamentally
seed-based (k-NN of a seed game). This module derives a seed from the vibe
text itself in three tiers; ``hybrid.recommend()`` then runs from that derived
seed exactly as it would for an LLM-named seed.

Mechanism (locked — see project.md / HANDOFF.md). ``derive_vibe_seed(text)`` is
the public orchestrator; each tier only fires when the previous one declines.

  TIER 1 — LLM proposes a seed name (``llm_query.propose_vibe_seed_title``).
      A second, separate Ollama call asks the same model for the single most
      representative, well-known game for the vibe. The LLM supplies the
      semantics that pure lexical TF-IDF lacks (it knows *Stardew Valley* is
      the relaxing-farming anchor, not *Farming Simulator*). The proposal is
      NEVER trusted directly.

  TIER 2 — Resolver grounds the proposal (``content.resolve_game_id``).
      The proposed title is run through the same strict >= 0.80 admit gate
      used everywhere else. This blocks hallucinated / out-of-catalog titles.
      It does NOT check thematic quality — a real-but-thematically-wrong
      title would still admit; the decisive test carries the quality bar.

  TIER 3 — TF-IDF fallback (``vibe_seed``).
      If the LLM declined or its proposal didn't ground, fall back to the
      original TF-IDF vibe path: ``.transform()`` through the fitted Phase-2
      vectorizer, cosine vs catalog, popularity tiebreak among the top-K,
      gated by ``VIBE_THRESHOLD``. The TF-IDF tier is RETAINED — not deleted
      — as the safety net for vibes the LLM has no useful guess for.

  If all three decline -> ``derive_vibe_seed`` returns ``seed_id=None`` and
  the caller emits the no_seed response. "Confidently anchor or confidently
  decline" — same principle as the resolver's 0.80 admit gate.

Design contract (DO NOT loosen):
- The LLM PROPOSES a name (tier 1) — the resolver grounds it before use.
  ``hybrid.recommend()`` still produces every recommendation. The LLM never
  writes the picks themselves.
- NO keyword -> tag-name string matching anywhere in the pipeline. Tier 3 is
  pure TF-IDF cosine in the same space as the catalog.
- vibe text -> SEED; parsed filters (``max_price`` / ``tags`` / ``genres``)
  -> CONSTRAIN downstream at the existing /recommend filter stage. Filters
  are NOT folded into the vibe vector and the LLM's seed proposal is NOT
  asked to honour filters — that would distort the seed choice.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

GAMES_JSON = ART / "games.json"
GAME_INDEX_JSON = ART / "game_index.json"
ID_NAME_JSON = ART / "id_name_lookup.json"
TFIDF_NPZ = ART / "content_tfidf.npz"
VECTORIZER_PKL = ART / "content_vectorizer.joblib"


# --- tunables ----------------------------------------------------------------
# Confidence gate on the best vibe cosine. Below this we return None
# (-> /ask returns the no_seed shape). Picked by probing both sides:
#   * lowest legitimate-vibe cosine seen in the sanity set: 0.191
#     ("a hard fast roguelike" — short, low-IDF text, the worst case)
#   * highest pure-noise cosine seen:                       0.136
#     ("the" — single function word, leaks weakly through descriptions)
# 0.15 sits cleanly inside the gap. Inputs with no vocabulary overlap
# (``asdkfjqwer``) hit the natural ``nnz == 0`` short-circuit before the
# gate fires. Empty text is rejected by /ask before this is called.
VIBE_THRESHOLD: float = 0.15

# Top-K candidate window for the popularity tiebreak. The brief specifies
# "popularity tiebreak ... so a recognized anchor beats a thematically-
# similar obscurity *when cosines are close*". K is what "close" means here.
#
# Picked by probe on the sanity set (cozy farming, hard fast roguelike,
# open-world rpg, atmospheric horror, competitive shooter, cozy life sim):
#   * K=5  -> all six anchors are thematically correct (e.g. "hard fast
#             roguelike" -> bit Dungeon II, a real Action Roguelike).
#   * K=10/20 -> "hard fast roguelike" leaks to Shatterline (a popular
#             competitive FPS that matches "hard"+"fast" descriptor tokens
#             but not the "roguelike" tag) because the popularity gap
#             between Shatterline (pop=12660) and any catalog roguelike
#             inside the top-10 (pop ~500-1200) is wider than the cosine
#             tier those candidates share, so the tiebreak drags theme.
# K=5 keeps the popularity tiebreak honest to its qualifier — "close" —
# and lets a niche-but-correct roguelike win when the recognized
# alternative isn't actually in the same vibe pocket. Small cost: the
# "competitive shooter" anchor is Natural Selection 2 (pop=5464) rather
# than Counter-Strike: Source (pop=85502); both are competitive FPS, so
# the vibe is preserved even though the brand-recognition drops.
VIBE_TOPK: int = 5


# --- module singletons (loaded once on first use) ----------------------------
_VEC: TfidfVectorizer | None = None
_X: sparse.csr_matrix | None = None
_GAME_INDEX: list[int] | None = None
_POP: np.ndarray | None = None       # per-row n_interactions, aligned to game_index
_ID_TO_NAME: dict[int, str] | None = None


def _ensure_loaded() -> None:
    """
    Load the fitted vectorizer, the TF-IDF matrix, and the per-row popularity
    vector once. We reuse the EXACT artifacts Phase 2 wrote — no re-fit, no
    parallel space — so cosine values here are directly comparable to the
    ones the rest of the system already produces.
    """
    global _VEC, _X, _GAME_INDEX, _POP, _ID_TO_NAME
    if _VEC is not None:
        return
    _VEC = joblib.load(VECTORIZER_PKL)
    _X = sparse.load_npz(TFIDF_NPZ).tocsr()
    _GAME_INDEX = json.loads(GAME_INDEX_JSON.read_text())
    games = json.loads(GAMES_JSON.read_text())
    by_id = {g["game_id"]: g for g in games}
    # Per-row n_interactions (float32, aligned to game_index) — the popularity
    # tiebreak key. Loaded here so vibe_seed() is a self-contained lookup.
    _POP = np.array(
        [float(by_id[gid].get("n_interactions", 0)) for gid in _GAME_INDEX],
        dtype=np.float32,
    )
    lookup = json.loads(ID_NAME_JSON.read_text())
    _ID_TO_NAME = {int(k): v for k, v in lookup["id_to_name"].items()}


# --- public API --------------------------------------------------------------
def vibe_seed(text: str) -> tuple[int | None, float, str | None]:
    """
    Derive a seed game_id from free vibe text.

    Returns ``(game_id, vibe_score, candidate_name)`` where ``vibe_score`` is
    the *best* catalog cosine for this text (the confidence the gate is
    applied to), and ``candidate_name`` is the popularity-tiebroken pick from
    the top-K candidates.

    Returns ``(None, 0.0, None)`` when:
      - the text is empty / whitespace only
      - no token in the text overlaps the fitted vocabulary (``nnz == 0``)
      - the best vibe cosine is below ``VIBE_THRESHOLD``

    These three short-circuits are how the no_seed shape stays honest: the
    function does NOT fall back to "best of a bad lot" when the signal is
    too weak to anchor on.
    """
    if not text or not text.strip():
        return None, 0.0, None

    _ensure_loaded()
    assert _VEC is not None and _X is not None and _GAME_INDEX is not None
    assert _POP is not None and _ID_TO_NAME is not None

    # Vectorize the vibe text in the SAME space as the catalog rows.
    v = _VEC.transform([text])
    if v.nnz == 0:
        # No vocabulary overlap (e.g. "asdkfjqwer"). Honest decline — there is
        # literally no measurable signal to anchor on.
        return None, 0.0, None

    # X is L2-normalised; v is L2-normalised by transform; so X @ v.T IS cosine.
    sims = (_X @ v.T).toarray().ravel().astype(np.float32, copy=False)
    best_score = float(sims.max())

    # Confidence gate — below this the signal is indistinguishable from a
    # single-function-word leak ("the" ~ 0.136). Decline cleanly.
    if best_score < VIBE_THRESHOLD:
        # Surface the would-have-been candidate so the no_seed response can
        # still report what the vibe weakly resembled, for debugging.
        top_idx = int(np.argmax(sims))
        return None, best_score, _ID_TO_NAME.get(int(_GAME_INDEX[top_idx]))

    # Popularity tiebreak: take the top-K candidates by cosine, then pick the
    # one with the highest n_interactions among them. This is what makes
    # "competitive shooter" land on Counter-Strike: Source rather than a
    # similarly-themed obscurity at a marginally higher cosine.
    k = min(VIBE_TOPK, sims.shape[0])
    top_idx = np.argpartition(-sims, k - 1)[:k]
    pop_window = _POP[top_idx]
    chosen_local = int(np.argmax(pop_window))
    chosen_row = int(top_idx[chosen_local])
    chosen_gid = int(_GAME_INDEX[chosen_row])
    chosen_name = _ID_TO_NAME.get(chosen_gid)

    # We return the BEST cosine as the confidence score (what the gate is
    # applied to). The actual chosen game's cosine may be slightly lower —
    # that's expected because popularity is the tiebreak; the score returned
    # describes the strength of the vibe match, not the rank of the chosen
    # game inside the top-K.
    return chosen_gid, best_score, chosen_name


# --- three-tier orchestrator (Phase 7.5 rebuild) -----------------------------
def derive_vibe_seed(text: str) -> dict:
    """
    Three-tier seed derivation for a vibe description. Returns a dict with
    full provenance so the API layer can stamp the response with WHICH tier
    produced the seed (or why none did).

    Return shape:
        {
            "seed_id":           int | None,        # final seed, or None
            "seed_source":       "llm" | "tfidf" | None,
            # tier-1 provenance (always populated; the LLM is always asked
            # unless the input was empty)
            "llm_proposal":      str | None,        # title the LLM named
            "llm_raw":           str,               # raw model output
            # tier-2 provenance (populated iff the LLM proposed a title)
            "resolver_score":    float | None,      # 0..1, >=0.80 to admit
            "resolver_name":     str | None,        # catalog name it grounded to
            # tier-3 provenance (populated iff tier 3 was reached)
            "tfidf_score":       float | None,      # best catalog cosine
            "tfidf_candidate":   str | None,        # popularity-tiebroken pick
        }

    Tiers fire strictly in order: LLM proposes -> resolver grounds at >= 0.80
    -> TF-IDF fallback gated by VIBE_THRESHOLD. The first one to confidently
    anchor wins; if all decline, ``seed_id`` is ``None`` and the caller emits
    the no_seed shape.
    """
    # Local import keeps the dependency one-way (api -> vibe -> llm_query,
    # content) and avoids a vibe<->llm_query circular at import time.
    from src.llm_query import propose_vibe_seed_title
    from src.content import resolve_game_id

    out: dict = {
        "seed_id": None,
        "seed_source": None,
        "llm_proposal": None,
        "llm_raw": "",
        "resolver_score": None,
        "resolver_name": None,
        "tfidf_score": None,
        "tfidf_candidate": None,
    }

    if not text or not text.strip():
        return out

    # TIER 1 — ask the LLM to name a representative game for the vibe.
    proposal, llm_raw = propose_vibe_seed_title(text)
    out["llm_proposal"] = proposal
    out["llm_raw"] = llm_raw

    # TIER 2 — ground the proposal through the strict resolver. Same >= 0.80
    # admit gate used everywhere else (see api.ASK_SEED_THRESHOLD); imported
    # locally as a literal to avoid an api<->vibe import cycle.
    if proposal:
        gid, score, matched_name = resolve_game_id(proposal)
        out["resolver_score"] = float(score)
        out["resolver_name"] = matched_name
        if gid is not None and score >= 0.80:
            out["seed_id"] = int(gid)
            out["seed_source"] = "llm"
            return out

    # TIER 3 — TF-IDF fallback. Only reached if the LLM declined or its
    # proposal didn't ground. The fallback has its own confidence gate
    # (VIBE_THRESHOLD) and returns (None, score, candidate) if nothing
    # clears it. The retained TF-IDF tier is the safety net for vibes the
    # LLM has no useful guess for; it is NOT a thematic-quality net.
    tfidf_gid, tfidf_score, tfidf_candidate = vibe_seed(text)
    out["tfidf_score"] = float(tfidf_score)
    out["tfidf_candidate"] = tfidf_candidate
    if tfidf_gid is not None:
        out["seed_id"] = int(tfidf_gid)
        out["seed_source"] = "tfidf"
        return out

    # All three tiers declined.
    return out
