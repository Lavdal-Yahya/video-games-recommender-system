// Single source of truth for the backend URL.
//
// Phase 5's FastAPI uvicorn runs at http://127.0.0.1:8765 (see src/api.py
// CORSMiddleware allow_origins=['http://localhost:5173'], and the Vite dev
// server defaults to that origin).  ANY change to this constant has to be
// matched in src/api.py's CORS allow_origins AND the project README — those
// three values are the port/CORS triple the Phase 7 brief calls out.
export const API_BASE_URL = 'http://127.0.0.1:8765'

// Confirmation-tier thresholds — used ONLY to decide whether to display the
// "I picked X — did you mean Y?" banner; never to re-rank or re-score the
// backend's results.  Justification (mirrors the resolver semantics in
// src/api.py:ASK_SEED_THRESHOLD and src/content.py:resolve_game_id):
//
//   ADMIT_THRESHOLD (0.80)   — same gate /ask uses.  Below this the backend
//                              returns no_seed and we never get here.
//   HIGH_CONFIDENCE (0.99)   — only a score == 1.0 means the normalized query
//                              EQUALS the catalog name (SequenceMatcher.ratio
//                              of identical strings is 1.0, and containment of
//                              a fully-covering token set is 1.0).  Anything
//                              strictly below 1.0 is either a char-typo
//                              ("hadez" -> 0.80) or partial containment, both
//                              of which are exactly what the confirm tier is
//                              for.  We treat [0.80, 0.99) as "good enough to
//                              proceed AND worth a single confirmation prompt".
//   TIE_GAP (0.05)           — if the top two candidates differ by less than
//                              this, treat it as a franchise tie ("witcher"
//                              gives 5 candidates all at 1.0) and fire the
//                              confirm prompt even though the top is at 1.0.
//                              0.05 is wider than any float-noise drift but
//                              tight enough that "stardew" (1.0 vs 0.80, gap
//                              0.20) is NOT classified as a tie.
export const ADMIT_THRESHOLD = 0.80
export const HIGH_CONFIDENCE = 0.99
export const TIE_GAP = 0.05
