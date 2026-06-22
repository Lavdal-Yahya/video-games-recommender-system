// Render the recommendation results EXACTLY as the backend returned them.
//
// The Phase 7 brief calls this out explicitly:
//   "the frontend renders, it never recommends ... web/ may sort by nothing of
//    its own — it displays what the backend returns, in the backend's order."
//
// So this component:
//   - iterates results in the order the API gave them (no .sort())
//   - shows the backend's score field unchanged (for transparency in the demo)
//   - does NOT apply any client-side filter, threshold, or reranking
//
// If you ever feel the urge to add a .sort() here, stop — that's the
// scope-violation tripwire the brief warned about.

export default function ResultsList({ results }) {
  if (!results || results.length === 0) return null
  return (
    <ol className="results">
      {results.map((row, idx) => (
        <li key={row.game_id} className="result-row">
          <span className="rank">{idx + 1}</span>
          <span className="result-name">{row.name}</span>
          <span className="result-score" title="backend score (display-only)">
            {row.score.toFixed(3)}
          </span>
        </li>
      ))}
    </ol>
  )
}
