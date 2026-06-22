import { useRef, useState } from 'react'
import { hasMic, startRecording, transcribe } from '../speech.js'

// Natural-language entry point: mic + text input (Phase 7.6 — local stack).
//
// The mic button is a press-to-toggle: first click starts recording, second
// click stops it, the clip is POSTed to /stt, the transcript is dropped into
// the text input AND submitted (same path the typed flow drives). All of the
// transcription happens server-side; the browser just captures audio and
// uploads it. If anything fails the user can still TYPE — voice failure must
// never block text input (the brief's #1 invariant for this phase).

export default function AskBar({ onAsk, disabled, busy, sttReady }) {
  const [text, setText] = useState('')
  const [voiceState, setVoiceState] = useState('idle') // idle | recording | transcribing
  const [voiceError, setVoiceError] = useState(null)
  const recorderRef = useRef(null)
  const micAvailable = hasMic()
  // The mic button is enabled when the browser supports recording AND the
  // server reports the STT model is loaded. /health surfaces stt_ready; when
  // the model file is missing we keep the mic disabled rather than letting
  // the user record a clip we know will fail.
  const micEnabled = micAvailable && sttReady !== false

  const submit = (value) => {
    const trimmed = (value ?? text).trim()
    if (!trimmed) return
    setText(trimmed)
    onAsk?.(trimmed)
  }

  const handleMic = async () => {
    if (disabled || busy) return
    setVoiceError(null)

    // Toggle stop: we're already recording, so finish the clip and transcribe.
    if (voiceState === 'recording' && recorderRef.current) {
      const rec = recorderRef.current
      recorderRef.current = null
      setVoiceState('transcribing')
      try {
        const blob = await rec.stop()
        const transcript = await transcribe(blob)
        if (!transcript.trim()) {
          setVoiceError("I didn't catch that — try again or type your request.")
          setVoiceState('idle')
          return
        }
        setText(transcript)
        setVoiceState('idle')
        submit(transcript)
      } catch (err) {
        setVoiceError(err?.message || 'voice input failed')
        setVoiceState('idle')
      }
      return
    }

    // Toggle start: open the mic and begin recording.
    try {
      recorderRef.current = await startRecording()
      setVoiceState('recording')
    } catch (err) {
      setVoiceError(err?.message || 'could not access microphone')
      setVoiceState('idle')
    }
  }

  // Hint shown under the input. Priority: error > recording > transcribing >
  // mic-disabled. We do NOT hide the mic button when the server reports the
  // STT model is missing — we disable it and explain in the hint, so the user
  // can still type and see a clear reason.
  let hint = null
  let hintKind = 'hint'
  if (voiceError) { hint = `voice: ${voiceError}`; hintKind = 'hint error' }
  else if (voiceState === 'recording') hint = 'recording — click the mic again to stop'
  else if (voiceState === 'transcribing') hint = 'transcribing your clip...'
  else if (!micAvailable) hint = 'mic capture is unavailable in this browser — typed input works'
  else if (sttReady === false) hint = 'voice model is not loaded on the server — typed input works'

  const micLabel = voiceState === 'recording'
    ? 'stop'
    : voiceState === 'transcribing'
      ? '...'
      : 'mic'

  return (
    <div className="ask-bar">
      <label className="field-label">Or describe what you want</label>
      <form
        className="ask-form"
        onSubmit={(e) => { e.preventDefault(); submit() }}
      >
        <input
          type="text"
          className="ask-input"
          placeholder='e.g. "something chill like Stardew under $20"'
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={disabled || busy}
        />
        {micAvailable && (
          <button
            type="button"
            className={`mic-btn${voiceState === 'recording' ? ' listening' : ''}`}
            onClick={handleMic}
            disabled={disabled || busy || !micEnabled || voiceState === 'transcribing'}
            aria-label={voiceState === 'recording' ? 'Stop recording' : 'Speak your request'}
            title={voiceState === 'recording' ? 'Stop recording' : 'Speak your request'}
          >
            {micLabel}
          </button>
        )}
        <button
          type="submit"
          className="ask-btn"
          disabled={disabled || busy || !text.trim()}
        >
          {busy ? 'asking...' : 'ask'}
        </button>
      </form>
      {hint && <div className={hintKind}>{hint}</div>}
    </div>
  )
}
