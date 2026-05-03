import { useState, useRef, useEffect } from 'react'
import { matchesAPI } from '../../services/api'

const CORNER_LABELS = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
const CORNER_COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#ec4899']

export default function CourtCalibrationModal({ matchId, videoId, onClose }) {
  const [corners, setCorners]   = useState([])
  const [applying, setApplying] = useState(false)
  const [error, setError]       = useState(null)
  const videoRef   = useRef(null)
  const overlayRef = useRef(null)

  const token = localStorage.getItem('access_token')
  const streamUrl = videoId
    ? `/api/videos/${videoId}/stream${token ? `?token=${token}` : ''}`
    : null

  useEffect(() => {
    const h = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [onClose])

  const handleLoadedMetadata = (e) => {
    e.target.currentTime = 2
  }

  const handleVideoClick = (e) => {
    if (corners.length >= 4) return
    const video = videoRef.current
    if (!video) return
    const rect   = video.getBoundingClientRect()
    const scaleX = video.videoWidth  / rect.width
    const scaleY = video.videoHeight / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top)  * scaleY
    setCorners(prev => [...prev, { x, y }])
  }

  const handleApply = async () => {
    if (corners.length < 4) return
    setApplying(true)
    setError(null)
    try {
      const court_corners = corners.map(c => [c.x, c.y])
      await matchesAPI.setHomography(matchId, court_corners)
      onClose()
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to apply calibration')
      setApplying(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="bg-[#0d1117] rounded-xl overflow-hidden w-full max-w-2xl border border-white/10 shadow-2xl">

        {/* Header */}
        <div className="flex justify-between items-start p-4 border-b border-white/10">
          <div>
            <h3 className="text-white font-semibold">Fix Court Map</h3>
            <p className="text-white/40 text-sm mt-0.5">Click the 4 court corners in order</p>
          </div>
          <button onClick={onClose} className="text-white/40 hover:text-white/80 text-xl">✕</button>
        </div>

        {/* Corner pills */}
        <div className="flex gap-2 px-4 pt-3 flex-wrap">
          {CORNER_LABELS.map((label, i) => {
            const placed  = i < corners.length
            const current = i === corners.length
            return (
              <span
                key={i}
                className="px-3 py-1 rounded-full text-xs font-semibold text-white transition-all"
                style={{
                  background: placed
                    ? CORNER_COLORS[i]
                    : current
                    ? `${CORNER_COLORS[i]}55`
                    : 'rgba(255,255,255,0.08)',
                  border: current ? `1px solid ${CORNER_COLORS[i]}` : '1px solid transparent',
                }}
              >
                {i + 1} {label} {placed ? '✓' : current ? '←' : ''}
              </span>
            )
          })}
        </div>

        {/* Video with click overlay */}
        <div className="p-4">
          <div
            className="relative w-full"
            style={{ cursor: corners.length < 4 ? 'crosshair' : 'default' }}
          >
            {streamUrl ? (
              <>
                <video
                  ref={videoRef}
                  src={streamUrl}
                  className="w-full rounded-lg border border-white/10"
                  muted
                  preload="metadata"
                  onLoadedMetadata={handleLoadedMetadata}
                  onClick={handleVideoClick}
                />
                <div ref={overlayRef} className="absolute inset-0 pointer-events-none rounded-lg overflow-hidden">
                  {corners.map((c, i) => {
                    const video = videoRef.current
                    if (!video || !video.videoWidth) return null
                    const rect       = video.getBoundingClientRect()
                    const parentRect = video.parentElement.getBoundingClientRect()
                    const scaleX = rect.width  / video.videoWidth
                    const scaleY = rect.height / video.videoHeight
                    const px = c.x * scaleX + (rect.left - parentRect.left)
                    const py = c.y * scaleY + (rect.top  - parentRect.top)
                    return (
                      <div
                        key={i}
                        className="absolute w-3 h-3 rounded-full"
                        style={{
                          left:       px - 6,
                          top:        py - 6,
                          background: CORNER_COLORS[i],
                          boxShadow:  `0 0 8px ${CORNER_COLORS[i]}`,
                        }}
                      />
                    )
                  })}
                </div>
              </>
            ) : (
              <div className="w-full aspect-video bg-white/5 rounded-lg flex items-center justify-center text-white/30">
                No video
              </div>
            )}
          </div>
        </div>

        {error && (
          <p className="px-4 pb-2 text-red-400 text-sm">{error}</p>
        )}

        {/* Actions */}
        <div className="flex justify-between p-4 border-t border-white/10">
          <button
            onClick={() => setCorners([])}
            className="px-4 py-2 text-sm text-white/50 hover:text-white/80 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
          >
            Reset
          </button>
          <button
            onClick={handleApply}
            disabled={corners.length < 4 || applying}
            className="px-4 py-2 text-sm text-blue-300 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-600/30 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {applying
              ? 'Applying...'
              : corners.length < 4
              ? `Apply (${4 - corners.length} more needed)`
              : 'Apply'}
          </button>
        </div>
      </div>
    </div>
  )
}
