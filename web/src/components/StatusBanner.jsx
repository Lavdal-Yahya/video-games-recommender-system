// Top-of-page status strip: warmup, errors, no_seed messages, "empty
// filters" messages.  Single line, color-coded.  All copy comes from the
// parent — this component just renders.

export default function StatusBanner({ kind, message }) {
  if (!message) return null
  return <div className={`status-banner ${kind || 'info'}`}>{message}</div>
}
