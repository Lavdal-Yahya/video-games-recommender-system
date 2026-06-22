// "Why this seed?" — the defense-mode provenance block.
//
// Shows the routing path, effective alpha, the LLM's parsed {seed, filters},
// the LLM's raw output, and the resolver's match_score.  All values come
// straight from /ask's response — no client-side derivation.

export default function Provenance({ rec }) {
  if (!rec) return null
  const { path, alpha_effective, parsed, llm_raw, match_score, seed_name } = rec
  return (
    <details className="provenance">
      <summary>why this seed?</summary>
      <div className="prov-grid">
        <div className="prov-row">
          <span className="prov-key">seed</span>
          <span className="prov-val">{seed_name}</span>
        </div>
        <div className="prov-row">
          <span className="prov-key">path</span>
          <span className="prov-val">{path}</span>
        </div>
        <div className="prov-row">
          <span className="prov-key">alpha_effective</span>
          <span className="prov-val">{alpha_effective?.toFixed?.(3) ?? alpha_effective}</span>
        </div>
        {match_score != null && (
          <div className="prov-row">
            <span className="prov-key">resolver score</span>
            <span className="prov-val">{match_score.toFixed(3)}</span>
          </div>
        )}
        {parsed && (
          <div className="prov-row">
            <span className="prov-key">parsed</span>
            <pre className="prov-pre">{JSON.stringify(parsed, null, 2)}</pre>
          </div>
        )}
        {llm_raw && (
          <div className="prov-row">
            <span className="prov-key">llm_raw</span>
            <pre className="prov-pre">{llm_raw}</pre>
          </div>
        )}
      </div>
    </details>
  )
}
