import { useTiming } from '../hooks/useTiming'
import { usePause } from '../hooks/usePause'

export default function TimingOverlay() {
  const timing = useTiming()
  const { paused, toggle } = usePause()

  const entries = [...timing.entries()]
  const totalMs = entries.reduce((sum, [, { ms }]) => sum + ms, 0)

  return (
    <>
      {entries.length > 0 && (
        <div className="timing-overlay">
          {entries.map(([nodeId, { opName, ms }]) => (
            <div key={nodeId} className="timing-overlay__row">
              <span className="timing-overlay__name">{opName}</span>
              <span className="timing-overlay__ms">{ms.toFixed(1)} ms</span>
            </div>
          ))}
          <div className="timing-overlay__separator" />
          <div className="timing-overlay__row">
            <span className="timing-overlay__name">Total</span>
            <span className="timing-overlay__ms">{totalMs.toFixed(1)} ms</span>
          </div>
        </div>
      )}
      <button className="pause-btn" onClick={toggle}>
        {paused ? '▶ 再開 / Resume' : '⏸ 一時停止 / Pause'}
      </button>
    </>
  )
}
