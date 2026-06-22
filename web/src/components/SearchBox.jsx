import { useEffect, useRef, useState } from 'react'
import { searchGames } from '../api.js'

// Typeahead search input.
//
// CRITICAL: this component does NOT rank or filter games.  It sends the user's
// text to GET /games/search and renders the results in the EXACT order the
// backend returned (substring-priority + difflib fallback live in api.py).  The
// onPick callback gets the raw {game_id, name} the backend chose for us.

export default function SearchBox({ onPick, disabled }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const debounceRef = useRef(null)

  // Debounce so we don't fire a request per keystroke.  150 ms is short enough
  // to feel live, long enough that "stardew" doesn't fire 7 requests in a row.
  useEffect(() => {
    if (!query.trim()) {
      setResults([])
      setOpen(false)
      return
    }
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      setLoading(true)
      try {
        const data = await searchGames(query, 10)
        // Render in backend order — no sort, no rerank.
        setResults(data.results)
        setOpen(true)
      } catch {
        setResults([])
        setOpen(false)
      } finally {
        setLoading(false)
      }
    }, 150)
    return () => clearTimeout(debounceRef.current)
  }, [query])

  const handlePick = (game) => {
    setQuery(game.name)
    setOpen(false)
    onPick?.(game)
  }

  return (
    <div className="search-box">
      <label className="field-label">Pick a game you like</label>
      <input
        type="text"
        className="search-input"
        placeholder="e.g. Stardew Valley"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => results.length > 0 && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        disabled={disabled}
        autoComplete="off"
      />
      {open && results.length > 0 && (
        <ul className="suggestions">
          {results.map((g) => (
            <li key={g.game_id}>
              <button
                type="button"
                className="suggestion"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => handlePick(g)}
              >
                {g.name}
              </button>
            </li>
          ))}
        </ul>
      )}
      {open && results.length === 0 && !loading && query.trim() && (
        <div className="suggestions empty">no matches</div>
      )}
    </div>
  )
}
