# Video Game Recommender — Project Overview

## Goal
Build a video game recommendation system that takes a natural-language request
(typed or spoken) and returns relevant games, read back aloud. The recommender
is a **weighted hybrid**: it blends content-based similarity (game metadata) with
collaborative similarity (player behavior). A small local LLM turns free-form
requests into a structured query, and a **local voice stack** (faster-whisper for
STT, Piper for TTS, both served by FastAPI) handles speech in and speech out.

This is a learning-focused academic project. Clarity and correctness matter more
than scale or production polish. Every component should be simple enough to
explain in a defense.

## What makes it interesting (the "wow" without the weight)
- A genuine **hybrid** recommender (Burke's *weighted* hybrid), not a single method.
- An **LLM front-end** that parses "something chill like Stardew under $20" into
  `{ "seed": "Stardew Valley", "filters": { "mood": "relaxing", "max_price": 20 } }`.
- **Voice I/O** — ask out loud, hear the picks back.
- A clean **ablation** (content-only vs collaborative-only vs hybrid) that proves
  the blend earns its place.

## Core idea in one line
There are two ways to measure how similar two games are — by their *metadata* and
by *who plays them* — and we blend the two, then recommend the nearest neighbors
of a seed game.

```
score(a, b) = α · content_similarity(a, b) + (1 − α) · collaborative_similarity(a, b)
```

## Architecture (request flow)
1. **Query** — typed, or spoken: the frontend RECORDS an audio clip and POSTs it
   to `/stt` (faster-whisper, local) which returns the transcript. The browser
   does no transcription itself — it is a capture device.
2. **LLM query understanding** — a local model parses the request into
   `{ seed, filters }`. If `seed` is null, the **vibe-seeding** path (see below)
   derives a seed from the request text before recommendation.
3. **Hybrid scorer** — content cosine + item-item CF cosine, blended by α.
4. **Filter + rank** — apply parsed filters, return the Top-N neighbors of the seed.
5. **Output** — results shown in the UI and read aloud: the frontend POSTs the
   spoken text to `/tts` (Piper, local), which returns audio the browser plays.
   The browser does no synthesis itself — it is a playback device.

## Tech stack
- **Data / ML:** Python, pandas, scikit-learn (TF-IDF, cosine, nearest neighbors).
- **Backend:** FastAPI — serves recommendations and calls the local LLM.
- **LLM:** local model via Ollama (`llama3.1:8b`), used **only** to PARSE
  requests into `{ seed, filters }` JSON AND, when no seed is named, to PROPOSE
  the single most representative game for the requested vibe — that proposal is
  grounded by the resolver before use. The LLM **never** generates the
  recommendation list itself: `hybrid.recommend()` produces every pick.
- **Frontend:** Vite + React. The frontend captures microphone audio and plays
  back returned audio — it does NOT transcribe or synthesize anything itself.
- **Voice (local, served by FastAPI):**
  - **STT:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) running
    locally; the frontend POSTs the recorded clip to `POST /stt` and gets the
    transcript back.
  - **TTS:** [Piper](https://github.com/rhasspy/piper) running locally; the
    frontend POSTs text to `POST /tts` and gets a short audio file back to play.
  - Both endpoints are new and will be built in the voice phase. They sit in
    `src/api.py` next to `/recommend`, `/ask`, `/resolve`, and `/games/search`.
  - Rationale: removes the Google round-trip that caused intermittent
    `mic: network` failures, replaces low-quality browser TTS, and keeps the
    whole stack local (no cloud dependency) — consistent with the local-LLM
    design choice for `/ask`.
- **Data:** a Steam dataset that contains **both** game metadata (genres, tags,
  description) and user–game interactions — e.g. the "Game Recommendations on
  Steam" dataset on Kaggle. The hybrid needs both signals.

## Repository structure
```
game-recommender/
├── CLAUDE.md            # agent operating manual (read every session)
├── project.md           # this file — the design
├── tasks.md             # progress tracker (source of truth)
├── README.md            # run instructions
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/             # downloaded dataset (gitignored, large)
│   └── processed/       # cleaned + sampled data
├── artifacts/           # generated similarity artifacts, games.json
├── src/
│   ├── data_prep.py     # Phase 1
│   ├── content.py       # Phase 2 — content similarity
│   ├── collaborative.py # Phase 3 — item-item CF
│   ├── hybrid.py        # Phase 4 — weighted blend + recommend()
│   ├── llm_query.py     # Phase 6 — NL → {seed, filters}
│   └── api.py           # Phase 5 — FastAPI (+ /ask added in Phase 6)
├── notebooks/
│   └── eval.ipynb       # Phase 8 — evaluation + ablation
├── web/                 # Phase 7 — Vite + React (mic capture + audio playback only)
└── report/              # Phase 9 — report + slides
```

## Dataset notes
- Two signals are required: game **metadata** (for the content arm) and
  **user–game interactions** (for the collaborative arm).
- The full interaction set is large; we work with a **sampled** matrix (popular
  games + a subset of users) so item-item CF stays tractable on a laptop.

## Vibe-based seeding (handles seed=null)
The recommender is fundamentally **seed-based**: it returns nearest neighbours of
a given game. The LLM parser can return `seed=null` when the user's request
doesn't name a recognized title (e.g. *"something chill and atmospheric for a
rainy night"*). The original design dead-ended here with a `no_seed` response.

**New behavior:** when `seed=null` AND the request text isn't empty/garbage, we
derive a seed from the *vibe text* itself, then run the normal hybrid from that
derived seed. The LLM still only PARSES the request and — for the vibe branch —
PROPOSES a representative seed name; it never writes the recommendations
themselves. The vibe path only supplies a SEED; everything downstream
(α-blend, routing, filters, top-k) is unchanged.

### Mechanism (the design, locked) — three-tier seed derivation
When the LLM parse returns `seed=null` and the text is non-empty, derive a seed
in this strict order. Each tier only fires when the previous one declines.

1. **LLM proposes a seed.** Re-prompt the LLM to name the single most
   representative, well-known game for the requested vibe — its best semantic
   guess (e.g. *"a relaxing farming game"* → *"Stardew Valley"*, *"an
   open-world rpg with a big story"* → *"The Witcher 3"*). This proposal is
   **never** trusted directly: the LLM has the world knowledge to pick a
   thematically right anchor that pure lexical TF-IDF cannot, but it can also
   hallucinate or name an out-of-catalog title.
2. **Resolver grounds the proposal.** Run the proposed title through
   `resolve_game_id` (the same strict ≥ 0.80 admit gate used everywhere else).
   If it resolves to a real catalog `game_id`, seed from that. The resolver is
   the **safety layer**: it blocks hallucinations and titles we don't have on
   Steam, but it is purely a string-match gate — it will admit a real-but-
   thematically-wrong title if the LLM names one. The phase's decisive test
   therefore carries the **thematic-quality bar**; the resolver carries only
   the existence bar.
3. **TF-IDF vibe fallback.** If the LLM's proposal does NOT ground (out-of-
   catalog, hallucinated, or the LLM declined to name a game), fall back to
   the existing TF-IDF vibe path: `.transform()` the vibe text through the
   already-fitted Phase-2 `TfidfVectorizer` (NOT re-fit — so vocabulary, IDF
   weights, and L2 normalisation match the catalog rows exactly), cosine vs
   the catalog, popularity-tiebreak from the top-K candidates, gated by its
   own confidence threshold (TBD in the phase). If TF-IDF also declines
   (cosine below threshold or no vocabulary overlap), return `no_seed`.

The TF-IDF tier is retained — not deleted — as the safety net for the cases
where the LLM either has no useful guess or names something the catalog
doesn't contain.

### Division of labour (locked)
- **Vibe text → chooses the SEED** (via the LLM's proposal, grounded by the
  resolver; TF-IDF cosine as the fallback).
- **Parsed filters (`max_price` / `tags` / `genres`) → CONSTRAIN the results
  AFTER seeding.** Filters are NOT folded into the vibe vector, and the LLM's
  seed proposal is NOT asked to honour filters — they apply at the same
  post-scoring / pre-top-k stage as they already do in `/recommend`.
- **The LLM never writes the recommendation list.** Its only roles are
  parsing the request into `{ seed, filters }` and — when no seed is named —
  proposing a single representative seed title. `hybrid.recommend()` produces
  all recommendations.

This keeps the two concerns separated and means a user can say *"something
roguelike under $15"* and get a roguelike seed (from the LLM's anchor or the
TF-IDF fallback) with a $15 cap (from `max_price`), without the cap distorting
either tier.

### Honest-failure requirement (locked)
If neither tier produces a confident anchor — the LLM's proposal doesn't ground
through the resolver AND the TF-IDF fallback's best cosine is below its
confidence threshold (exact value TBD in the phase that builds this) — the
system MUST return `no_seed` instead of force-anchoring on a near-zero cosine.
Same "confidently anchor or confidently decline" principle that already governs
the resolver's 0.80 admit gate. Garbage in → honest "I didn't catch that" out.

With vibe-seeding in place, `no_seed` only fires on inputs that are genuinely
unusable: empty text, a single function word, or a vibe that has no measurable
overlap with the catalog *and* no LLM-proposed anchor the catalog contains.
Anything with a usable signal gets a seed.

### Known prerequisite — OPEN QUESTION
This whole mechanism requires the **fitted `TfidfVectorizer` object** to be
loadable at runtime, not just the document-term matrix. Phase 2's Outputs note
says `artifacts/content_vectorizer.joblib` exists, but the vibe phase MUST
verify this before building anything else. If only the matrix
(`content_tfidf.npz`) persisted and the vectorizer didn't, the vibe phase's
**first step** is to (a) reload the catalog soup, (b) re-fit the vectorizer
with the exact Phase-2 recipe (token_pattern, min_df, sublinear_tf, soup
stoplist — all locked in the Phase-2 Outputs note), and (c) persist it as
`artifacts/content_vectorizer.joblib`. Only then proceed.

## UX: the no_seed message (intended behavior)
The `no_seed` response message must **guide the user**, not dead-end them.

Because the recommender's whole shape is "give me a game you like, I'll find
games like it," a `no_seed` reply should say so plainly. Suggested wording
(the spirit, not the exact string — exact copy is the voice-phase or web-phase
deliverable):

> *"I recommend games similar to a game you already like. Try naming one —
> for example, 'something chill like Stardew' or 'a hard roguelike like
> Hades'."*

With vibe-seeding live, `no_seed` is reserved for the small set of inputs that
are genuinely unusable (empty, garbage, or a vibe with no catalog overlap), so
this guidance is appropriate for everything that still triggers it.

## Build model
The project is built **one phase at a time.** A single session covers exactly
one phase all the way to a clean, documented finish — including any probes,
fixes, and re-runs that finish requires — and a fresh session only starts once
that phase is closed and `tasks.md` is updated. `tasks.md` is the source of
truth for progress and `CLAUDE.md` tells the agent how to operate (and is
authoritative on the session-boundary rule). Read both before starting any
phase.
