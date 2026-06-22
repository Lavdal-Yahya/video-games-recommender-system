// THE ONLY PLACE fetch() LIVES IN THIS APP.
//
// Phase 7 brief: the frontend is render-only.  All scoring, ranking, filtering,
// and the alpha blend stay server-side.  This module enforces that boundary by
// owning every network call — UI components consume its return values and
// display them verbatim in the order the backend chose.  If a component starts
// importing fetch directly, that's a scope violation; route it through here.
//
// Endpoint coverage:
//   health()                        — GET /health      (warmup ping)
//   searchGames(q, limit=10)        — GET /games/search
//   recommend(seed, opts)           — POST /recommend
//   ask(text, opts)                 — POST /ask        (NL -> recs, single round trip)
//   resolve(q)                      — GET /resolve     (top-5 candidates + admit flag)

import { API_BASE_URL } from './config.js'

// --- response error classes ---------------------------------------------------
// We DO NOT recover from HTTP errors inside this layer — the UI needs to know
// the difference between "backend is down", "LLM is warming up", and "you sent
// a bad seed".  These classes carry the status code so callers can branch on it
// without parsing strings.
export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

// Build a typed error from a fetch Response.  Tries to read `detail` (FastAPI's
// HTTPException JSON shape) and falls back to statusText.
async function asError(resp) {
  let detail
  try {
    const body = await resp.json()
    detail = body.detail
  } catch {
    detail = resp.statusText
  }
  return new ApiError(detail || `HTTP ${resp.status}`, resp.status, detail)
}

async function jsonGet(path) {
  const resp = await fetch(`${API_BASE_URL}${path}`)
  if (!resp.ok) throw await asError(resp)
  return resp.json()
}

async function jsonPost(path, body) {
  const resp = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) throw await asError(resp)
  return resp.json()
}

// --- endpoint wrappers --------------------------------------------------------
// Health check.  Used on mount to warm the FastAPI process / catalog cache.
// `/ask`'s cold-start cost is the Ollama model load (30-60s); /health is cheap
// and lets us trigger any backend-side lazy loading before the user types.
export function health() {
  return jsonGet('/health')
}

// Typeahead search.  Backend ranks results (substring tiers + difflib fuzzy
// top-up); we display them in that exact order.
export function searchGames(q, limit = 10) {
  const qs = new URLSearchParams({ q, limit: String(limit) })
  return jsonGet(`/games/search?${qs.toString()}`)
}

// Direct recommendation by seed (name OR int game_id).
// opts: { k, alpha, filters }  — any can be omitted; backend uses defaults.
export function recommend(seed, opts = {}) {
  const body = { seed }
  if (opts.k != null) body.k = opts.k
  if (opts.alpha != null) body.alpha = opts.alpha
  if (opts.filters != null) body.filters = opts.filters
  return jsonPost('/recommend', body)
}

// Natural-language entry point.  ONE round trip — backend parses, resolves,
// and runs the hybrid in a single call.  Returns either the /recommend shape
// (with parsed/llm_raw/match_score) OR { status: 'no_seed', ... }.
export function ask(text, opts = {}) {
  const body = { text }
  if (opts.k != null) body.k = opts.k
  if (opts.alpha != null) body.alpha = opts.alpha
  return jsonPost('/ask', body)
}

// Resolver candidates.  Returns the top 5 with scores so the confirmation
// tier can decide whether to show "did you mean X?".  Uses the SAME scoring
// loop /ask gates on — alternates and main pick are guaranteed consistent.
export function resolve(q) {
  const qs = new URLSearchParams({ q })
  return jsonGet(`/resolve?${qs.toString()}`)
}
