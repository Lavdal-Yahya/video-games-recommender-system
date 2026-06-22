# web — Vite + React frontend (Phase 7)

Frontend for the hybrid game recommender. Talks to the FastAPI backend over
HTTP; no recommendation logic lives here (render-only invariant).

## Run

```bash
# 0) One-time — fetch the local voice models (Phase 7.6, ~200 MB total):
conda run -n ds python -m src.fetch_voice_models

# 1) Backend (in the repo root, NOT in web/):
conda run -n ds uvicorn src.api:app --host 127.0.0.1 --port 8765

# 2) Dev server (in web/):
npm install              # first time only
npm run dev              # -> http://127.0.0.1:5173/
```

The backend loads the STT (faster-whisper `base.en`, ~145 MB on disk) and TTS
(Piper `en_US-lessac-medium`, ~61 MB) models once at startup; expect a ~3 s
boot pause before `/health` returns. `GET /health` reports `stt_ready` and
`tts_ready` so the UI can decide whether to enable the mic button.

## Port / CORS contract (must agree across three places)

| place                                            | value                             |
| ------------------------------------------------ | --------------------------------- |
| `src/api.py` — uvicorn host/port                 | `127.0.0.1:8765`                  |
| `src/api.py` — `CORSMiddleware.allow_origins`    | `["http://localhost:5173", "http://127.0.0.1:5173"]` |
| `web/src/config.js` — `API_BASE_URL`             | `"http://127.0.0.1:8765"`         |
| Vite dev server (default + npm script flag)      | `127.0.0.1:5173`                  |

If you change any of these, change them all. CORS accepts BOTH `localhost`
and `127.0.0.1` for the Vite origin — browsers treat them as distinct origins,
and accepting both removes a confusing class of CORS failures that depends on
which spelling the user types into the URL bar.

## Endpoints used

- `GET /health`               — cold-start warmup ping on mount; also reports
                                `stt_ready` / `tts_ready`
- `GET /games/search?q=…`     — typeahead in the search box
- `POST /recommend`           — direct-pick path (search box) and the
                                confirm-banner switch
- `POST /ask`                 — natural language → seed + filters → hybrid
- `GET /resolve?q=…`          — top-5 resolver candidates with scores; powers
                                the "did you mean X?" confirmation tier
- `POST /stt`                 — Phase 7.6: multipart audio in, transcript out
                                (faster-whisper). Browser only records the clip.
- `POST /tts`                 — Phase 7.6: `{text}` in, WAV out (Piper).
                                Browser only plays the returned audio.

## Render-only invariant

The frontend renders, it never recommends. Every scoring, ranking, filtering,
and α-blend decision stays server-side. `web/src/api.js` is the SOLE place
`fetch()` is called; every other file consumes its results in the backend's
order without re-sorting or filtering on score.

## Voice support (Phase 7.6 — fully local stack)

The browser is a capture / playback device only. The mic clip is POSTed to
`/stt`; the response text is POSTed to `/tts` and the returned WAV plays in
an `<audio>` element. No browser `SpeechRecognition` or `speechSynthesis` is
involved — that path was the Phase-7 stopgap (Chrome-only, with a Google
round-trip behind intermittent `mic: network` errors). The new stack works
in Chrome, Firefox, and Safari.

- **STT** — `MediaRecorder` + `getUserMedia` on the browser (universal on
  modern desktop browsers); transcription happens on the server via
  faster-whisper.
- **TTS** — server-side Piper synthesises a short WAV that the browser
  plays. Audibly cleaner than the old browser-default voice.
- **Cold start** — the first `/health` response carries `stt_ready` /
  `tts_ready`. The two heavy models load at server boot (~3 s), so per-request
  STT cost is just decode (~0.4 s on a 2 s clip) and TTS is ~0.07 s.
- **Failure mode** — if mic capture, `/stt`, or `/tts` fails, the user can
  still type their request. Voice failure never blocks text input.
