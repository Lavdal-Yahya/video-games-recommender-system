import { useEffect, useRef, useState } from 'react'
import { ApiError, ask, health, recommend, resolve } from './api.js'
import { speak } from './speech.js'
import { ADMIT_THRESHOLD, HIGH_CONFIDENCE, TIE_GAP } from './config.js'

// speak() returns a Promise (network round-trip to /tts + audio playback).
// Most callers are fire-and-forget; we swallow rejections so a TTS failure
// (model not loaded, audio decode error) never breaks the visible UI — the
// text on screen is the authoritative channel.
function speakSafe(text) {
  speak(text).catch(() => { /* TTS is best-effort */ })
}

import SearchBox from './components/SearchBox.jsx'
import AskBar from './components/AskBar.jsx'
import ResultsList from './components/ResultsList.jsx'
import Provenance from './components/Provenance.jsx'
import ConfirmBanner from './components/ConfirmBanner.jsx'
import StatusBanner from './components/StatusBanner.jsx'

import './App.css'

// --- confirm-band classifier --------------------------------------------------
// One small piece of UI logic that operates on the RESOLVER OUTPUT only (not on
// recommendation scores).  Returns true iff we should ask "did you mean Y?".
// Rules (see config.js for the thresholds + justification):
//   1) The top candidate must be above ADMIT (else /ask already returned
//      no_seed and we wouldn't be in this branch).
//   2) Fire if top < HIGH_CONFIDENCE  (fuzzy / char-typo / partial match),
//      OR if (top - runnerUp) < TIE_GAP  (franchise tie like "witcher").
function confirmShouldFire(candidates) {
  if (!candidates || candidates.length < 2) return false
  const top = candidates[0].score
  const next = candidates[1].score
  if (top < ADMIT_THRESHOLD) return false
  if (top < HIGH_CONFIDENCE) return true
  if (top - next < TIE_GAP) return true
  return false
}

// --- TTS phrasing -------------------------------------------------------------
// The brief: TTS must never read out an empty list as silence; empty-results
// and no_seed have DISTINCT spoken messages.  We build the spoken line here so
// the wording lives in one place and can't drift between visible and spoken.
function announceRecommendation(rec) {
  const seedName = rec.seed_name
  if (!rec.results || rec.results.length === 0) {
    return `I found ${seedName} but no games matched those filters.`
  }
  const topNames = rec.results.slice(0, 3).map((r) => r.name).join(', ')
  return `Here are games like ${seedName}. ${topNames}.`
}

function announceNoSeed() {
  return "I didn't catch a game in the catalog — which one did you mean?"
}

// --- main app ---------------------------------------------------------------
export default function App() {
  // Whole-app state.  Kept here so the search path, the /ask path, and the
  // confirmation tier share one place to mutate "what is on screen now".
  const [rec, setRec] = useState(null)          // /recommend or /ask success body
  const [noSeed, setNoSeed] = useState(null)    // /ask no_seed body
  const [busy, setBusy] = useState(false)       // a network call is in flight
  const [warming, setWarming] = useState(true)  // cold-start ping
  const [errorMsg, setErrorMsg] = useState(null)
  const [confirmCandidates, setConfirmCandidates] = useState(null)
  // Voice readiness is reported by /health (Phase 7.6). Default `null` =
  // "we haven't asked yet"; AskBar treats anything but `false` as ready, so
  // the mic stays enabled while we're still warming.
  const [sttReady, setSttReady] = useState(null)
  const askedTextRef = useRef('')               // for retry/debug only

  // --- cold-start warmup ------------------------------------------------------
  // The brief: "fire one cheap warmup request (GET /health or a tiny /ask) so
  // the model loads while the user is still reading the page".  /health is
  // enough to wake the FastAPI process + warm the catalog; the LLM itself only
  // pre-loads on the FIRST /ask, so we keep the "warming up" hint visible
  // briefly even after /health returns.
  useEffect(() => {
    let cancelled = false
    health()
      .then((info) => {
        if (cancelled) return
        setWarming(false)
        // Surface STT readiness so AskBar can disable the mic when the model
        // file is missing on the server. TTS readiness is checked lazily by
        // speak() — a failed /tts just drops the spoken line, the UI text is
        // already correct.
        setSttReady(Boolean(info?.stt_ready))
      })
      .catch((err) => {
        if (cancelled) return
        setWarming(false)
        setErrorMsg(formatErr(err, 'Backend is not reachable.'))
      })
    return () => { cancelled = true }
  }, [])

  // --- direct-pick path (typed search box) -----------------------------------
  // Click a typeahead result → POST /recommend with that game_id.  No NL parse,
  // no resolver, no confirm tier — the user picked the exact game.
  const handlePick = async (game) => {
    setErrorMsg(null)
    setNoSeed(null)
    setConfirmCandidates(null)
    setBusy(true)
    try {
      const data = await recommend(game.game_id)
      setRec(data)
      speakSafe(announceRecommendation(data))
    } catch (err) {
      setRec(null)
      setErrorMsg(formatErr(err, 'Could not load recommendations.'))
    } finally {
      setBusy(false)
    }
  }

  // --- NL/voice path (free text -> /ask) -------------------------------------
  // Single round trip.  After a confident pick, fire /resolve to detect
  // franchise ambiguity and surface the confirm banner if needed (non-blocking;
  // the recommendations are already on screen by the time we check).
  const handleAsk = async (text) => {
    askedTextRef.current = text
    setErrorMsg(null)
    setNoSeed(null)
    setConfirmCandidates(null)
    setBusy(true)
    try {
      const data = await ask(text)
      if (data.status === 'no_seed') {
        setRec(null)
        setNoSeed(data)
        speakSafe(announceNoSeed())
        return
      }
      setRec(data)
      speakSafe(announceRecommendation(data))

      // Confirmation tier — fire /resolve on the parsed seed.  Done after we
      // already showed/spoke the result so the user isn't kept waiting.
      const seedText = data.parsed?.seed
      if (seedText) {
        try {
          const resv = await resolve(seedText)
          if (confirmShouldFire(resv.candidates)) {
            // Exclude the game we already picked from the alternates list.
            const alts = resv.candidates.filter(
              (c) => c.game_id !== data.seed_id,
            ).slice(0, 3)
            if (alts.length > 0) {
              setConfirmCandidates(alts)
              speakSafe(`I picked ${data.seed_name}. Did you mean ${alts[0].name}?`)
            }
          }
        } catch {
          // /resolve failure is non-fatal — the user already has results.
        }
      }
    } catch (err) {
      setRec(null)
      // 503 from /ask -> Ollama not running.  Phrase friendlier than
      // "HTTP 503"; this is the cold-start failure mode the brief calls out.
      if (err instanceof ApiError && err.status === 503) {
        setErrorMsg(
          "The local model isn't running. Start Ollama with `ollama serve` and " +
          'make sure llama3.1:8b is pulled.',
        )
      } else {
        setErrorMsg(formatErr(err, 'The ask request failed.'))
      }
    } finally {
      setBusy(false)
    }
  }

  // --- confirm-banner switch --------------------------------------------------
  // The brief: "Switching calls POST /recommend directly with the chosen
  // game_id (no re-parse)."  Reuses the parsed.filters from the original /ask
  // so the user keeps their constraints across the switch.
  const handleConfirmPick = async (alt) => {
    setConfirmCandidates(null)
    setBusy(true)
    try {
      const filters = rec?.parsed?.filters || null
      const data = await recommend(alt.game_id, { filters: hasKeys(filters) ? filters : null })
      // Re-graft the original NL provenance so the user can still see what
      // the LLM parsed — only the seed flipped.
      const next = {
        ...data,
        parsed: rec?.parsed ?? null,
        llm_raw: rec?.llm_raw ?? null,
      }
      setRec(next)
      speakSafe(`Switched to ${data.seed_name}. ${announceRecommendation(data)}`)
    } catch (err) {
      setErrorMsg(formatErr(err, 'Could not switch to that game.'))
    } finally {
      setBusy(false)
    }
  }

  // --- view -----------------------------------------------------------------
  const showResults = rec && rec.results
  const emptyFilters = showResults && rec.results.length === 0
  const banner = computeBanner({ warming, errorMsg, noSeed, emptyFilters, rec })

  return (
    <div className="app">
      <header className="app-header">
        <h1>game recommender</h1>
        <p className="tagline">
          weighted hybrid (content + collaborative), with a local LLM and voice I/O
        </p>
      </header>

      <StatusBanner kind={banner.kind} message={banner.message} />

      <section className="entry">
        <SearchBox onPick={handlePick} disabled={warming || busy} />
        <AskBar onAsk={handleAsk} disabled={warming} busy={busy} sttReady={sttReady} />
      </section>

      {confirmCandidates && rec && (
        <ConfirmBanner
          pickedName={rec.seed_name}
          alternates={confirmCandidates}
          onPick={handleConfirmPick}
          onDismiss={() => setConfirmCandidates(null)}
        />
      )}

      {showResults && (
        <section className="results-section">
          <h2 className="results-heading">
            recommendations for <em>{rec.seed_name}</em>
          </h2>
          <ResultsList results={rec.results} />
          <Provenance rec={rec} />
        </section>
      )}

      <footer className="app-footer">
        <span>backend at <code>http://127.0.0.1:8765</code></span>
      </footer>
    </div>
  )
}

// --- helpers ----------------------------------------------------------------
function hasKeys(obj) { return obj && Object.keys(obj).length > 0 }

function formatErr(err, fallback) {
  if (err instanceof ApiError) {
    return `${fallback} (${err.status}: ${err.detail || err.message})`
  }
  return `${fallback} (${err?.message || err})`
}

// Picks ONE message + kind to surface in the top banner.  Priority order
// (most-actionable first):
//   1. warming      — cold-start hint
//   2. errorMsg     — backend down / 503 / bad request
//   3. no_seed      — distinct spoken/visible message
//   4. emptyFilters — distinct from no_seed (seed resolved, but no games match)
function computeBanner({ warming, errorMsg, noSeed, emptyFilters, rec }) {
  if (warming) return { kind: 'info', message: 'warming up the backend...' }
  if (errorMsg) return { kind: 'error', message: errorMsg }
  if (noSeed) return { kind: 'warn', message: noSeed.message }
  if (emptyFilters) {
    return {
      kind: 'warn',
      message: `I found ${rec.seed_name} but no games matched those filters.`,
    }
  }
  return { kind: null, message: null }
}
