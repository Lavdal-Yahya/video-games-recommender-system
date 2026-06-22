// "I picked X — did you mean Y?" banner.
//
// Triggered ONLY when the parent App has already decided the resolve response
// landed in the confirm band (see App.jsx:confirmShouldFire).  This component
// has no scoring logic of its own — it just renders the alternates and calls
// the parent back with the picked game_id.

export default function ConfirmBanner({ pickedName, alternates, onPick, onDismiss }) {
  if (!alternates || alternates.length === 0) return null
  return (
    <div className="confirm-banner" role="alert">
      <div className="confirm-text">
        I picked <strong>{pickedName}</strong> — did you mean...
      </div>
      <div className="confirm-options">
        {alternates.map((c) => (
          <button
            key={c.game_id}
            type="button"
            className="confirm-alt"
            onClick={() => onPick?.(c)}
            title={`switch to ${c.name}`}
          >
            {c.name}
            <span className="confirm-score">{c.score.toFixed(2)}</span>
          </button>
        ))}
        <button
          type="button"
          className="confirm-dismiss"
          onClick={onDismiss}
        >
          keep current
        </button>
      </div>
    </div>
  )
}
