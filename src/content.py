"""
Phase 2 — content-based similarity.

Two games are "similar" here if their *metadata* looks alike: same tags, similar
words in the description. We measure that with TF-IDF + cosine similarity.

Pipeline (high level):
    games.json + game_index.json
      -> per-game "soup" string (tags repeated ~3x, then description words)
      -> TfidfVectorizer.fit_transform -> sparse TF-IDF matrix X (n_games x n_terms)
      -> cosine(a, b) = X[a] . X[b]  (rows are L2-normalised by TfidfVectorizer,
         so the dot product *is* the cosine; no extra normalisation needed)

A few decisions worth remembering (locked, see tasks.md / project.md):
- Row order of X is the SAME as artifacts/game_index.json. Row i of the content
  matrix matches column i of the Phase-3 interaction matrix. The hybrid arm in
  Phase 4 relies on that alignment.
- Tags become atomic tokens ("Open World" -> "open_world") and are repeated ~3x
  so they dominate the noisy description. min_df=2 + sublinear_tf=True let the
  IDF naturally down-weight ubiquitous tags ("indie", "action") without us
  hand-curating a stopword list.
- Cosine on non-negative TF-IDF lives in [0, 1], which matches the CF arm's
  cosine. Do NOT rescale here — Phase 4 blends the two with one α.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from difflib import SequenceMatcher, get_close_matches
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
NEIGHBORS_JSON = ART / "content_neighbors.json"

# How many times each tag token is repeated in the soup. Tags are short, high-
# precision genre/mood signals; descriptions are noisier prose, so we push tags
# up by simple repetition (a poor man's term weighting that TF-IDF understands).
TAG_REPEAT = 3

# Drop punctuation -> spaces. We keep the regex simple and global; anything that
# isn't a word char or whitespace becomes a space.
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)

# Platform / plumbing tokens to strip from EVERY game's soup before TF-IDF.
# Native Phase-1 tags rarely include these, but the Phase-1.5 Steam appdetails
# enrichment originally surfaced them through `categories[]` (now disabled in
# enrich_metadata.extract). This stoplist is the symmetric belt-and-braces: any
# of these tokens — whether they reach us via Steam categories, a future
# tag-source change, or the description text — never enter the soup. It only
# covers infrastructure ("can you play it online?", "does it have achievements?",
# "what controller does it support?"), never gameplay/genre/theme tokens like
# "rpg", "roguelike", "open_world", "cozy", "souls_like", "farming_sim".
SOUP_STOPLIST: frozenset[str] = frozenset({
    # multiplayer modes / capabilities — they describe HOW you play, not WHAT
    "single_player", "singleplayer",
    "multi_player", "multiplayer",
    "co_op", "online_co_op", "lan_co_op", "shared_split_screen_co_op",
    "pvp", "online_pvp", "lan_pvp", "shared_split_screen_pvp",
    "pve", "online_pve",
    "cross_platform_multiplayer",
    "mmo",  # NB: keep "mmorpg" — it's a genre signal; "mmo" alone is just "online"
    # Steam features
    "steam_achievements",
    "steam_cloud",
    "steam_trading_cards",
    "steam_workshop",
    "steam_leaderboards",
    "steamvr_collectibles",
    "remote_play_together",
    "remote_play_on_phone",
    "remote_play_on_tablet",
    "remote_play_on_tv",
    # controller / accessibility plumbing
    "full_controller_support",
    "partial_controller_support",
    "controller",
    "tracked_controller_support",
    "captions_available",
    "commentary_available",
    "in_app_purchases",
    "stats",
    "includes_level_editor",
    # VR plumbing (genre-bearing tokens like "vr" alone are kept;
    # the *_supported variants are catalog/configuration descriptors)
    "vr_supported",
    "vr_only",
    "valve_anti_cheat_enabled",
    "anti_cheat",
})


# --- soup builder ------------------------------------------------------------
def _tag_token(tag: str) -> str:
    """'Open World' -> 'open_world'. Keeps multi-word tags as one TF-IDF term."""
    return tag.strip().lower().replace(" ", "_")


def _clean_description(desc: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not desc:
        return ""
    desc = desc.lower()
    desc = _PUNCT_RE.sub(" ", desc)
    return " ".join(desc.split())


def build_soup(tags: Iterable[str], description: str) -> str:
    """
    Build the per-game text we feed to TF-IDF.

    Format:
        tag1 tag2 ... (repeated TAG_REPEAT times)  description words

    Tags first + repeated so they dominate; description appended as plain words.
    When the description is empty, the soup is just the repeated tags. Tokens
    in SOUP_STOPLIST are dropped from BOTH the tag block and the description
    word stream — applied symmetrically to native + enriched games so the two
    populations share one clean vocabulary.
    """
    tag_tokens = [
        _tag_token(t) for t in tags
        if t and t.strip() and _tag_token(t) not in SOUP_STOPLIST
    ]
    tag_block = " ".join(tag_tokens)
    repeated = " ".join([tag_block] * TAG_REPEAT) if tag_block else ""
    desc_block = _clean_description(description)
    if desc_block:
        desc_block = " ".join(
            w for w in desc_block.split() if w not in SOUP_STOPLIST
        )
    if repeated and desc_block:
        return repeated + " " + desc_block
    return repeated or desc_block


# --- loading + fitting -------------------------------------------------------
def _load_games_in_index_order() -> tuple[list[dict], list[int]]:
    """
    Load games.json and reorder rows to match game_index.json.

    games.json from Phase 1 is *already* aligned with game_index.json, but we
    re-key on game_id and rebuild the list anyway so this stays correct even
    if the Phase-1 export order ever changes. Row i of the returned list is
    the game with game_id == game_index[i].
    """
    games = json.loads(GAMES_JSON.read_text())
    game_index = json.loads(GAME_INDEX_JSON.read_text())
    by_id = {g["game_id"]: g for g in games}
    ordered = [by_id[gid] for gid in game_index]
    return ordered, game_index


def fit_content_matrix(
    games_ordered: list[dict],
) -> tuple[sparse.csr_matrix, TfidfVectorizer]:
    """
    Build the soup for each game (in index order) and TF-IDF fit/transform it.

    token_pattern=r"[^\\s]+" treats anything between whitespace as one token, so
    our underscored tags ("open_world") survive intact.
    min_df=2 drops tokens that appear in only one game (mostly proper nouns from
    descriptions); sublinear_tf=True uses 1+log(tf) so a tag repeated 3x doesn't
    swamp the soup linearly.
    """
    soups = [build_soup(g.get("tags", []), g.get("description", "") or "") for g in games_ordered]
    vec = TfidfVectorizer(
        token_pattern=r"[^\s]+",
        min_df=2,
        sublinear_tf=True,
    )
    X = vec.fit_transform(soups)
    # TfidfVectorizer L2-normalises rows by default, so X @ X.T is already cosine.
    return X.tocsr(), vec


# --- module-level singletons (loaded on first use) ---------------------------
_X: sparse.csr_matrix | None = None
_VEC: TfidfVectorizer | None = None
_GAME_INDEX: list[int] | None = None
_ID_TO_ROW: dict[int, int] | None = None
_ID_TO_NAME: dict[int, str] | None = None
_NAME_TO_ID: dict[str, int] | None = None  # lowercased name -> game_id


def _ensure_loaded() -> None:
    """
    Lazy-load the fitted matrix + vectorizer. Prefers the saved artifacts so
    callers don't pay the fit cost; falls back to re-fitting if they're missing
    (useful during development before the first `python -m src.content` run).
    """
    global _X, _VEC, _GAME_INDEX, _ID_TO_ROW, _ID_TO_NAME, _NAME_TO_ID
    if _X is not None:
        return

    _GAME_INDEX = json.loads(GAME_INDEX_JSON.read_text())
    _ID_TO_ROW = {gid: i for i, gid in enumerate(_GAME_INDEX)}

    lookup = json.loads(ID_NAME_JSON.read_text())
    _ID_TO_NAME = {int(k): v for k, v in lookup["id_to_name"].items()}
    # Lowercased keys so fuzzy/exact lookups are case-insensitive.
    _NAME_TO_ID = {name.lower(): int(gid) for name, gid in lookup["name_to_id"].items()}

    if TFIDF_NPZ.exists() and VECTORIZER_PKL.exists():
        _X = sparse.load_npz(TFIDF_NPZ).tocsr()
        _VEC = joblib.load(VECTORIZER_PKL)
    else:
        ordered, _ = _load_games_in_index_order()
        _X, _VEC = fit_content_matrix(ordered)


# --- public API --------------------------------------------------------------
def content_scores(game_id: int) -> np.ndarray:
    """
    Return cosine similarity of `game_id` to every catalog game, as a dense
    float32 array of length n_games aligned to artifacts/game_index.json.

    Self-similarity is forced to 0 so it never wins its own top-k. Phase 4 needs
    this full-vector form to blend with the CF arm row-for-row.
    """
    _ensure_loaded()
    assert _X is not None and _ID_TO_ROW is not None
    if game_id not in _ID_TO_ROW:
        raise KeyError(f"game_id {game_id} is not in the catalog")
    row = _ID_TO_ROW[game_id]
    # X is L2-normalised, so (X @ X[row].T) IS cosine. Densify to a 1-D vector.
    sims = (_X @ _X[row].T).toarray().ravel().astype(np.float32, copy=False)
    sims[row] = 0.0
    return sims


def content_similarity(game_id: int, k: int = 10) -> list[tuple[int, float]]:
    """
    Top-k neighbours of `game_id` by content cosine, as (game_id, score) pairs
    sorted descending. Built on top of content_scores so there is one source of
    truth for the actual similarity numbers.
    """
    _ensure_loaded()
    assert _GAME_INDEX is not None
    sims = content_scores(game_id)
    # argpartition gives us the k largest indices without sorting the whole
    # 6000-long vector; we then sort just those k.
    k = min(k, len(sims) - 1)
    top_idx = np.argpartition(-sims, k)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(int(_GAME_INDEX[i]), float(sims[i])) for i in top_idx]


def find_game_id(query: str) -> int | None:
    """
    Fuzzy name -> game_id. Case-insensitive exact match first; if that misses,
    fall back to difflib's closest match. Returns None if nothing close enough.

    We reuse this in Phase 6 to resolve seeds parsed from voice / NL queries,
    where users won't type the exact catalog spelling.
    """
    _ensure_loaded()
    assert _NAME_TO_ID is not None
    if not query:
        return None
    q = query.strip().lower()
    if q in _NAME_TO_ID:
        return _NAME_TO_ID[q]
    # difflib uses SequenceMatcher; cutoff=0.6 is its default and is forgiving
    # enough for "stardew" -> "Stardew Valley" without being noisy.
    matches = get_close_matches(q, list(_NAME_TO_ID.keys()), n=1, cutoff=0.6)
    if matches:
        return _NAME_TO_ID[matches[0]]
    return None


# Match-only stopwords (kept tiny — these are the connective words that catalog
# names share with each other and so carry no discriminative signal).
_MATCH_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "at",
})

# Strip the registered-trademark family + any non-alphanumeric. We keep this
# narrow on purpose — "The Witcher® 3: Wild Hunt" -> "the witcher 3 wild hunt"
# is the asymmetry we need to dissolve; we are NOT trying to lemmatise / stem.
_TRADEMARK_RE = re.compile(r"[®™©]")
_PUNCT_FOR_MATCH_RE = re.compile(r"[^\w\s]")


def _normalize_for_match(s: str) -> str:
    """Lowercase, drop trademark glyphs, replace punctuation with spaces."""
    s = s.lower()
    s = _TRADEMARK_RE.sub("", s)
    s = _PUNCT_FOR_MATCH_RE.sub(" ", s)
    return " ".join(s.split())


# --- number normalization (1..20) -------------------------------------------
# Deterministic, mechanical, both directions. Applied to tokens on BOTH sides
# so number-words, roman numerals, and digits all unify on a single canonical
# form (the digit). This handles real spoken/LLM variants like "the witcher
# three" -> Witcher 3, "civilization 6" -> Civilization VI, "dota two" -> Dota 2.
# Range 1-20 covers the universe of game numbering in practice; extending the
# range further would let unrelated digit collisions sneak in for no real gain.
_NUM_WORD_TO_INT: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_ROMAN_TO_INT: dict[str, int] = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8,
    "ix": 9, "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15,
    "xvi": 16, "xvii": 17, "xviii": 18, "xix": 19, "xx": 20,
}


def _canon_num_token(tok: str) -> str:
    """Canonicalise a single token: number-word or roman numeral -> digit form."""
    if tok in _NUM_WORD_TO_INT:
        return str(_NUM_WORD_TO_INT[tok])
    if tok in _ROMAN_TO_INT:
        return str(_ROMAN_TO_INT[tok])
    return tok


def _canon_match_tokens(s: str) -> set[str]:
    """Significant tokens for containment scoring, with number normalisation.

    Stopwords are dropped, then each surviving token is canonicalised so that
    "three" / "iii" / "3" all collapse to "3".
    """
    return {
        _canon_num_token(t)
        for t in _normalize_for_match(s).split()
        if t not in _MATCH_STOPWORDS
    }


def _token_containment(q_tokens: set[str], c_tokens: set[str]) -> float:
    """
    Containment ratio: fraction of the query's meaningful tokens that the
    candidate covers. Returns a value in [0, 1].

    CRITICAL anti-leak rule: a numeric (post-canonicalisation, digit-only) token
    counts toward coverage ONLY if at least one NON-numeric query token is also
    covered. A bare number cannot establish a match on its own — otherwise
    "Splatoon 3" would leak via the "3" token alone into any catalog game whose
    name ends in 3, and "Pikmin 4" would leak via "4". Numbers only sharpen a
    match that has independent textual evidence; they never start one.
    """
    if not q_tokens:
        return 0.0
    covered = q_tokens & c_tokens
    if not covered:
        return 0.0
    # If only numeric tokens overlapped, the match is incidental — discard it.
    if not any(not t.isdigit() for t in covered):
        return 0.0
    return len(covered) / len(q_tokens)


# --- resolver index (built once, on first resolve_game_id call) -------------
# Tuple per catalog game: (game_id, lower_name, norm, canon_tokens, n_interactions).
# Built once and cached so each resolver call is just the per-candidate scoring
# loop. n_interactions is loaded from games.json and used as the deterministic
# tiebreak when two candidates score the same max — bare franchise queries like
# "witcher" then resolve to the most-played entry.
_RESOLVER_INDEX: list[tuple[int, str, str, frozenset[str], int]] | None = None


def _ensure_resolver_index() -> None:
    global _RESOLVER_INDEX
    if _RESOLVER_INDEX is not None:
        return
    _ensure_loaded()
    assert _ID_TO_NAME is not None
    games = json.loads(GAMES_JSON.read_text())
    pop_by_id = {int(g["game_id"]): int(g.get("n_interactions", 0)) for g in games}
    index: list[tuple[int, str, str, frozenset[str], int]] = []
    for gid, name in _ID_TO_NAME.items():
        norm = _normalize_for_match(name)
        toks = frozenset(_canon_match_tokens(name))
        n_int = int(pop_by_id.get(gid, 0))
        index.append((gid, name.lower(), norm, toks, n_int))
    _RESOLVER_INDEX = index


def resolve_game_id(query: str) -> tuple[int | None, float, str | None]:
    """
    Resolve a free-text query to a catalog game with an exposed confidence.

    Returns (game_id, score, matched_name):
      - empty / whitespace query -> (None, 0.0, None)
      - exact lowercased name hit -> (gid, 1.0, name)
      - otherwise: FULL SCAN over every catalog game, pick the best by
            score = max(char_ratio, token_containment)
        with deterministic popularity tiebreak (highest n_interactions wins).

    The two-component score:
      * char_ratio       — SequenceMatcher on normalised full strings. Catches
                           short typos ("hadez" -> "hades", ~0.80) and rewards
                           near-exact spellings.
      * token_containment — fraction of the query's meaningful tokens (after
                           stopwords + number normalisation) that the candidate
                           covers. Numeric/roman tokens count toward coverage
                           ONLY when at least one non-numeric query token also
                           overlaps (see `_token_containment` — without this
                           rule "splatoon 3" would leak via the bare "3").

    Number normalisation (1-20, both directions) unifies "three" / "iii" / "3"
    in BOTH query and candidate, so "the witcher three" cleanly matches
    "The Witcher 3: Wild Hunt" and "civilization 6" matches "Sid Meier's
    Civilization VI" without any hand-maintained alias dict.

    True abbreviations ("gta 5", "civ 6") share no surface tokens with their
    targets and fall to a low score by design — they're the named no_seed
    residual, deferred to Phase 7's confirmation tier. The single-token char-typo
    band (hadez 0.80 vs splatoon 0.778, ~0.02 margin) is the other named
    residual: both currently resolve correctly, but the margin is thin and the
    "did you mean X?" voice tier in Phase 7 is the proper place to disambiguate.

    Phase 6's /ask gates on `score >= ASK_SEED_THRESHOLD` (0.80) and returns the
    no_seed shape otherwise. find_game_id (used by /recommend) is untouched.
    """
    _ensure_resolver_index()
    assert _RESOLVER_INDEX is not None and _NAME_TO_ID is not None and _ID_TO_NAME is not None
    if not query:
        return None, 0.0, None
    q = query.strip().lower()
    if not q:
        return None, 0.0, None
    if q in _NAME_TO_ID:
        gid = _NAME_TO_ID[q]
        return gid, 1.0, _ID_TO_NAME.get(gid)

    q_norm = _normalize_for_match(q)
    q_toks = _canon_match_tokens(q)

    best_gid: int | None = None
    best_score: float = -1.0
    best_pop: int = -1
    # Full scan — 6000 SequenceMatcher.ratio() calls on short strings is
    # cheap (<100 ms), and avoiding the get_close_matches(n=5) shortlist is the
    # whole point of this rewrite (the shortlist filtered the right candidate
    # out before containment could rescue it, e.g. "stardew" -> "StarMade").
    for gid, _key, c_norm, c_toks, n_int in _RESOLVER_INDEX:
        char = SequenceMatcher(a=q_norm, b=c_norm).ratio()
        contain = _token_containment(q_toks, c_toks)
        score = char if char > contain else contain
        if score > best_score or (score == best_score and n_int > best_pop):
            best_score = score
            best_pop = n_int
            best_gid = gid

    if best_gid is None:
        return None, 0.0, None
    return best_gid, float(best_score), _ID_TO_NAME.get(best_gid)


def resolve_candidates(query: str, n: int = 5) -> list[tuple[int, float, str]]:
    """
    Return the top-n candidate (game_id, score, name) tuples for the same
    scoring resolve_game_id picks the single best by. Phase 7's confirmation
    tier reads this so the UI's alternates+scores never diverge from /ask's
    binary admit/reject — the frontend sees what the resolver sees.

    Scoring is identical: score = max(char_ratio, token_containment), and the
    sort is (score DESC, n_interactions DESC) — same tiebreak as resolve_game_id.
    """
    _ensure_resolver_index()
    assert _RESOLVER_INDEX is not None and _ID_TO_NAME is not None
    if not query or not query.strip():
        return []
    q = query.strip().lower()
    q_norm = _normalize_for_match(q)
    q_toks = _canon_match_tokens(q)

    scored: list[tuple[float, int, int]] = []  # (score, n_interactions, gid)
    for gid, _key, c_norm, c_toks, n_int in _RESOLVER_INDEX:
        char = SequenceMatcher(a=q_norm, b=c_norm).ratio()
        contain = _token_containment(q_toks, c_toks)
        score = char if char > contain else contain
        scored.append((score, n_int, gid))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [
        (gid, float(s), _ID_TO_NAME.get(gid, ""))
        for s, _ni, gid in scored[:n]
    ]


# --- artifact build + sanity check (run as a script) -------------------------
def _compute_all_top_neighbors(
    X: sparse.csr_matrix, game_index: list[int], k: int = 20
) -> dict[str, list[dict]]:
    """
    Top-k neighbours for every game, returned as
        { str(game_id): [{"game_id": int, "score": float, "name": str}, ...] }

    We compute the full 6000x6000 cosine in one go (~144 MB float32) — well
    within laptop RAM at this catalog size, and far simpler than a chunked loop.
    """
    sims = (X @ X.T).toarray().astype(np.float32, copy=False)
    np.fill_diagonal(sims, 0.0)

    lookup = json.loads(ID_NAME_JSON.read_text())
    id_to_name = {int(gid): name for gid, name in lookup["id_to_name"].items()}

    out: dict[str, list[dict]] = {}
    n = len(game_index)
    for i in range(n):
        row = sims[i]
        kk = min(k, n - 1)
        top_idx = np.argpartition(-row, kk)[:kk]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        gid_i = int(game_index[i])
        out[str(gid_i)] = [
            {
                "game_id": int(game_index[j]),
                "name": id_to_name.get(int(game_index[j]), ""),
                "score": float(row[j]),
            }
            for j in top_idx
        ]
    return out


def _print_neighbors(name: str, k: int = 8) -> None:
    """Resolve a name with the fuzzy matcher and print its top-k content neighbours."""
    gid = find_game_id(name)
    if gid is None:
        print(f"  [{name}] not found")
        return
    assert _ID_TO_NAME is not None
    seed_name = _ID_TO_NAME.get(gid, str(gid))
    print(f"  Seed: {seed_name}  (game_id={gid})")
    for nid, score in content_similarity(gid, k=k):
        print(f"    {score:.3f}  {_ID_TO_NAME.get(nid, str(nid))}")


def main() -> None:
    print("Loading catalog and fitting TF-IDF ...")
    ordered, game_index = _load_games_in_index_order()
    X, vec = fit_content_matrix(ordered)
    print(f"  matrix shape: {X.shape}  (n_games x vocab_size)")
    print(f"  vocab size  : {len(vec.vocabulary_)}")
    print(f"  nnz         : {X.nnz}")

    print("Saving artifacts ...")
    sparse.save_npz(TFIDF_NPZ, X)
    joblib.dump(vec, VECTORIZER_PKL)

    print("Computing top-20 neighbours for every game ...")
    neighbours = _compute_all_top_neighbors(X, game_index, k=20)
    NEIGHBORS_JSON.write_text(json.dumps(neighbours))
    print(f"  wrote {NEIGHBORS_JSON.name}  ({len(neighbours)} games)")

    # Force-reload from the freshly written artifacts so the sanity check uses
    # them (and exercises the load path).
    global _X, _VEC
    _X, _VEC = None, None
    _ensure_loaded()

    # How many catalog games ended up with an all-zero soup vector? After the
    # Phase-1.5 enrichment, this should be ~5 (the F2P MMOs Steam appdetails
    # refuses to return — Vindictus / Mabinogi / MapleStory / Conqueror's Blade
    # / Lost Ark). Zero vectors are a cold-start signal for Phase 4.
    row_nnz = X.getnnz(axis=1)
    zero_rows = int((row_nnz == 0).sum())
    print(f"\nZero-vector catalog games: {zero_rows}")
    if zero_rows:
        zero_ids = [int(game_index[i]) for i, n in enumerate(row_nnz) if n == 0]
        lookup = json.loads(ID_NAME_JSON.read_text())
        id_to_name = {int(k): v for k, v in lookup["id_to_name"].items()}
        for gid in zero_ids:
            print(f"  {gid:>10}  {id_to_name.get(gid, '?')}")

    print("\nSanity check — neighbours of well-known seeds:")
    for seed in [
        "Stardew Valley",
        "Hades",
        "The Witcher 3: Wild Hunt",
        "Counter-Strike",
        "Grand Theft Auto V",
        "Team Fortress 2",
        "DOOM",
    ]:
        resolved_gid = find_game_id(seed)
        assert _ID_TO_NAME is not None
        resolved_name = _ID_TO_NAME.get(resolved_gid) if resolved_gid else None
        match_flag = "OK" if resolved_name and seed.lower() in resolved_name.lower() else "??"
        print(f"  [resolve {match_flag}] {seed!r} -> {resolved_name!r}")
        _print_neighbors(seed, k=10)
        print()


if __name__ == "__main__":
    main()
