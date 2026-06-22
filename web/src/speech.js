// Voice I/O — local FastAPI stack (Phase 7.6).
//
// The frontend is a CAPTURE + PLAYBACK device only: it records the mic via
// MediaRecorder, POSTs the clip to /stt, and plays the audio returned by /tts.
// No transcription or synthesis happens in the browser; all of that lives in
// src/api.py (faster-whisper for STT, Piper for TTS). This removes the
// browser-side Web Speech API entirely — that path was Chrome-only and the
// `mic: network` failures came from its Google round-trip.

import { API_BASE_URL } from './config.js'

// --- feature detection ------------------------------------------------------
// Mic capture needs both MediaRecorder and getUserMedia. Both are universal
// on modern Chromium / Firefox / Safari (the Web Speech API was the
// Chrome-only piece — we're free of it now). On localhost / 127.0.0.1 the mic
// permission works even without HTTPS, which is what the dev setup uses.
export function hasMic() {
  return (
    typeof window !== 'undefined' &&
    typeof window.MediaRecorder !== 'undefined' &&
    !!navigator.mediaDevices?.getUserMedia
  )
}

// --- recording --------------------------------------------------------------
// startRecording() -> { stop(): Promise<Blob>, cancel(): void }
//
// Requests mic permission, opens a MediaRecorder, and returns a controller.
// `stop()` resolves with the recorded audio Blob (the browser picks the
// container — webm/opus on Chromium, ogg/opus on Firefox); faster-whisper
// decodes both via libav so we don't normalise on the client.
export async function startRecording() {
  if (!hasMic()) {
    throw new Error('microphone is unavailable in this browser')
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  const recorder = new MediaRecorder(stream)
  const chunks = []
  recorder.addEventListener('dataavailable', (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data)
  })
  recorder.start()

  const stopAllTracks = () => stream.getTracks().forEach((t) => t.stop())

  return {
    // Resolves once the recorder fires its final `stop` event so we know all
    // chunks have flushed; otherwise we'd race the dataavailable callback.
    stop() {
      return new Promise((resolve, reject) => {
        recorder.addEventListener('stop', () => {
          stopAllTracks()
          const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })
          resolve(blob)
        }, { once: true })
        recorder.addEventListener('error', (e) => {
          stopAllTracks()
          reject(e?.error || new Error('recorder error'))
        }, { once: true })
        try { recorder.stop() } catch (err) { stopAllTracks(); reject(err) }
      })
    },
    cancel() {
      try { recorder.stop() } catch { /* ignore */ }
      stopAllTracks()
    },
  }
}

// --- transcription ----------------------------------------------------------
// Sends the recorded clip to POST /stt and returns the transcript string.
// Throws on non-2xx so the caller can surface the error in the UI (the brief:
// voice failure must NEVER block text input — the AskBar falls back to typed).
export async function transcribe(blob) {
  const form = new FormData()
  // The backend reads multipart by field name `audio` (see /stt's signature).
  // The filename is arbitrary — faster-whisper sniffs the container itself.
  form.append('audio', blob, 'clip.webm')
  const resp = await fetch(`${API_BASE_URL}/stt`, {
    method: 'POST',
    body: form,
  })
  if (!resp.ok) {
    const detail = await safeDetail(resp)
    throw new Error(detail || `stt failed (${resp.status})`)
  }
  const body = await resp.json()
  return body.text || ''
}

// --- playback ---------------------------------------------------------------
// Single shared <audio> element so a new utterance interrupts the previous one
// (matches the old speechSynthesis.cancel() semantics — the brief calls out
// not stacking the previous query's spoken line on top of the new one).
let currentAudio = null

// speak(text) — POST text to /tts, play the returned WAV, resolve when audio
// finishes (or rejects on network / playback error). The caller can `await` it
// when they want strict ordering (announce no_seed THEN start the next ask).
export async function speak(text) {
  if (!text) return
  // Cancel anything currently playing before we kick off the new request, so
  // a fast back-to-back call doesn't end up double-speaking.
  if (currentAudio) {
    try { currentAudio.pause() } catch { /* ignore */ }
    currentAudio = null
  }
  const resp = await fetch(`${API_BASE_URL}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!resp.ok) {
    const detail = await safeDetail(resp)
    throw new Error(detail || `tts failed (${resp.status})`)
  }
  const blob = await resp.blob()
  const url = URL.createObjectURL(blob)
  const audio = new Audio(url)
  currentAudio = audio
  return new Promise((resolve, reject) => {
    audio.addEventListener('ended', () => {
      URL.revokeObjectURL(url)
      if (currentAudio === audio) currentAudio = null
      resolve()
    })
    audio.addEventListener('error', (e) => {
      URL.revokeObjectURL(url)
      if (currentAudio === audio) currentAudio = null
      reject(e?.error || new Error('audio playback failed'))
    })
    audio.play().catch(reject)
  })
}

async function safeDetail(resp) {
  try {
    const body = await resp.json()
    return body.detail
  } catch {
    return null
  }
}
