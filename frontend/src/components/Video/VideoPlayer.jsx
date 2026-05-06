import React, { useRef, useState, useEffect, useCallback } from 'react'
import {
  Play, Pause, Volume2, VolumeX, Maximize, Minimize,
  SkipBack, SkipForward, Settings
} from 'lucide-react'
import { videosAPI } from '../../services/api'
import PlayerProfileModal from './PlayerProfileModal'

function formatTime(s) {
  if (!s || isNaN(s)) return '0:00'
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = Math.floor(s % 60)
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
    : `${m}:${String(sec).padStart(2, '0')}`
}

/**
 * VideoPlayer
 * ───────────
 * Props:
 *   videoId      – UUID of the video
 *   trackingData – live { players: [...], ball: {...} } from useTrackingData
 *   onTimeUpdate – callback(timestamp_seconds)
 *   showOverlay  – boolean, toggle tracking overlay
 */
export default function VideoPlayer({ videoId, matchId, trackingData = null, onTimeUpdate, showOverlay = true }) {
  const videoRef   = useRef(null)
  const canvasRef  = useRef(null)
  const containerRef = useRef(null)
  const [playing,    setPlaying]    = useState(false)
  const [selectedPlayerId, setSelectedPlayerId] = useState(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration,   setDuration]   = useState(0)
  const [volume,     setVolume]     = useState(1)
  const [muted,      setMuted]      = useState(false)
  const [fullscreen, setFullscreen] = useState(false)
  const [showControls, setShowControls] = useState(true)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [showRates,  setShowRates]  = useState(false)
  const hideTimer = useRef(null)

  const videoSrc = videosAPI.streamUrl(videoId)

  // Draw tracking overlays on canvas
  const drawOverlays = useCallback(() => {
    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas || !video || !trackingData) return

    const ctx = canvas.getContext('2d')
    canvas.width  = video.videoWidth  || video.clientWidth
    canvas.height = video.videoHeight || video.clientHeight
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const speedColor = (kmh) => {
      if (kmh == null)    return 'rgba(200,200,200,0.8)'
      if (kmh < 40)       return 'rgba(80,200,80,0.9)'
      if (kmh < 70)       return 'rgba(255,200,0,0.9)'
      return 'rgba(255,60,60,0.9)'
    }

    const scaleX = canvas.width  / video.videoWidth
    const scaleY = canvas.height / video.videoHeight

    // Draw player bounding boxes
    if (trackingData.players) {
      trackingData.players.forEach(p => {
        const x = p.bbox_x * scaleX
        const y = p.bbox_y * scaleY
        const w = p.bbox_w * scaleX
        const h = p.bbox_h * scaleY

        // Box color: blue=TeamA, red=TeamB, grey=unknown
        const boxColor = p.team === 'A' ? '#3b82f6' : p.team === 'B' ? '#ef4444' : '#94a3b8'
        ctx.strokeStyle = boxColor
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, w, h)

        // Label background
        const labelText = `#${p.display_number ?? p.player_track_id ?? '?'}`
        ctx.font = 'bold 11px Inter'
        const labelW = ctx.measureText(labelText).width + 6
        ctx.fillStyle = boxColor + 'cc'
        ctx.fillRect(x, y - 20, labelW, 20)

        // Label text
        ctx.fillStyle = '#ffffff'
        ctx.fillText(labelText, x + 3, y - 5)
      })
    }

    // Draw ball trajectory arc + speed badge
    if (trackingData.ball) {
      const b = trackingData.ball
      const vt = videoRef.current?.currentTime ?? 0
      const isRecent = b.timestamp == null || Math.abs(b.timestamp - vt) <= 2.0

      // Trajectory arc from ball.trajectory (array of {x, y, speed_kmh} objects from DB)
      const traj = b.trajectory || []
      if (traj.length > 1) {
        for (let i = 1; i < traj.length; i++) {
          const alpha = i / traj.length   // fade: oldest=0.15, newest=1.0
          const opacity = 0.15 + alpha * 0.85
          ctx.globalAlpha = opacity
          ctx.strokeStyle = speedColor(traj[i].speed_kmh ?? b.speed_kmh)
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.moveTo(traj[i - 1].x * scaleX, traj[i - 1].y * scaleY)
          ctx.lineTo(traj[i].x * scaleX, traj[i].y * scaleY)
          ctx.stroke()
        }
        ctx.globalAlpha = 1.0
      }

      if (isRecent) {
        const bx = b.x * scaleX
        const by = b.y * scaleY
        // Ball circle
        ctx.beginPath()
        ctx.arc(bx, by, 8, 0, Math.PI * 2)
        ctx.fillStyle = '#facc15cc'
        ctx.fill()
        ctx.strokeStyle = '#fbbf24'
        ctx.lineWidth = 2
        ctx.stroke()

        // Speed badge above ball
        if (b.speed_kmh != null) {
          const label = `${b.speed_kmh} km/h`
          ctx.font = 'bold 10px Inter'
          const lw = ctx.measureText(label).width + 6
          ctx.fillStyle = speedColor(b.speed_kmh)
          ctx.fillRect(bx - lw / 2, by - 26, lw, 16)
          ctx.fillStyle = '#fff'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillText(label, bx, by - 18)
          ctx.textAlign = 'left'
          ctx.textBaseline = 'alphabetic'
        }
      }
    }

    // Draw homography mini-map (bottom-right)
    drawMiniMap(ctx, canvas.width, canvas.height, trackingData)
  }, [trackingData])

  const drawMiniMap = (ctx, cw, ch, data) => {
    const mapW = 260
    const mapH = 162
    const mapX = cw - mapW - 14
    const mapY = ch - mapH - 14

    // Background
    ctx.fillStyle = 'rgba(10,15,25,0.85)'
    ctx.beginPath()
    ctx.roundRect(mapX - 2, mapY - 2, mapW + 4, mapH + 4, 6)
    ctx.fill()

    // Court outline
    ctx.strokeStyle = '#334155'
    ctx.lineWidth = 1.5
    ctx.strokeRect(mapX, mapY, mapW, mapH)

    // Net (center line)
    ctx.strokeStyle = '#475569'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(mapX, mapY + mapH / 2)
    ctx.lineTo(mapX + mapW, mapY + mapH / 2)
    ctx.stroke()

    // 3m attack lines
    ctx.strokeStyle = '#1e3a5f'
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    ctx.moveTo(mapX, mapY + mapH * 0.25)
    ctx.lineTo(mapX + mapW, mapY + mapH * 0.25)
    ctx.moveTo(mapX, mapY + mapH * 0.75)
    ctx.lineTo(mapX + mapW, mapY + mapH * 0.75)
    ctx.stroke()
    ctx.setLineDash([])

    // Players on mini-map
    if (data.players) {
      const players = data.players
      const nullCourtCount = players.filter(p => p.court_x == null).length
      const totalPlayers   = players.length

      players.forEach(p => {
        if (p.court_x == null || p.court_y == null) return
        const cx = Math.max(0, Math.min(1, p.court_x))
        const cy = Math.max(0, Math.min(1, p.court_y))
        const px = mapX + cx * mapW
        const py = mapY + cy * mapH
        const dotR = 6

        // Team color
        ctx.fillStyle = p.team === 'A' ? '#3b82f6' : p.team === 'B' ? '#ef4444' : '#9ca3af'
        ctx.beginPath()
        ctx.arc(px, py, dotR, 0, Math.PI * 2)
        ctx.fill()

        // Display number inside dot
        const label = p.display_number != null ? String(p.display_number) : '?'
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 6px sans-serif'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(label, px, py)
      })

      // Warning badge if >30% of players have null court coords
      if (totalPlayers > 0 && nullCourtCount / totalPlayers > 0.3) {
        ctx.fillStyle = 'rgba(251,191,36,0.9)'
        ctx.font = 'bold 9px sans-serif'
        ctx.textAlign = 'right'
        ctx.textBaseline = 'top'
        ctx.fillText('⚠', mapX + mapW - 2, mapY + 2)
      }

      // Reset text alignment for subsequent drawing code
      ctx.textAlign = 'left'
      ctx.textBaseline = 'alphabetic'
    }

    // Ball trajectory trail on mini-map
    if (data.ball?.court_x != null) {
      const vt = videoRef.current?.currentTime ?? 0
      const isRecent = data.ball.timestamp == null || Math.abs(data.ball.timestamp - vt) <= 2.0

      // Trajectory: draw last 15 pixel positions as approximated court dots
      const traj = data.ball.trajectory || []
      const trailLen = Math.min(15, traj.length)
      const video = videoRef.current
      if (video && trailLen > 1) {
        const vw = video.videoWidth || video.clientWidth || 1
        const vh = video.videoHeight || video.clientHeight || 1
        for (let i = traj.length - trailLen; i < traj.length; i++) {
          const alpha = (i - (traj.length - trailLen)) / trailLen
          const dotR = 1 + alpha * 2   // 1px oldest → 3px newest
          // Use court coords if available, fall back to pixel projection
          const tx = traj[i].court_x != null
            ? mapX + traj[i].court_x * mapW
            : mapX + (traj[i].x / vw) * mapW
          const ty = traj[i].court_y != null
            ? mapY + traj[i].court_y * mapH
            : mapY + (traj[i].y / vh) * mapH
          ctx.beginPath()
          ctx.arc(tx, ty, dotR, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(250,204,21,${0.3 + alpha * 0.7})`
          ctx.fill()
        }
      }

      // Current ball position (bright dot)
      if (isRecent) {
        const bx = mapX + data.ball.court_x * mapW
        const by = mapY + data.ball.court_y * mapH
        ctx.beginPath()
        ctx.arc(bx, by, 4, 0, Math.PI * 2)
        ctx.fillStyle = '#facc15'
        ctx.fill()
      }
    }

    // Label
    ctx.fillStyle = '#64748b'
    ctx.font = '9px Inter'
    ctx.fillText('COURT VIEW', mapX + 5, mapY + 11)
  }

  // Animate overlays in sync with video
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const handler = () => {
      setCurrentTime(video.currentTime)
      onTimeUpdate?.(video.currentTime)
      if (trackingData) drawOverlays()
    }
    video.addEventListener('timeupdate', handler)
    video.addEventListener('loadedmetadata', () => setDuration(video.duration))
    return () => video.removeEventListener('timeupdate', handler)
  }, [trackingData, drawOverlays, onTimeUpdate])

  const togglePlay = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) {
      v.play().catch(() => {})
      setPlaying(true)
    } else {
      v.pause()
      setPlaying(false)
    }
  }

  const seek = (e) => {
    const v = videoRef.current
    if (!v) return
    const rect = e.currentTarget.getBoundingClientRect()
    const pct  = (e.clientX - rect.left) / rect.width
    v.currentTime = pct * duration
  }

  const skip = (secs) => {
    const v = videoRef.current
    if (!v) return
    v.currentTime = Math.min(Math.max(v.currentTime + secs, 0), duration)
  }

  const toggleMute = () => {
    const v = videoRef.current
    if (!v) return
    v.muted = !v.muted
    setMuted(v.muted)
  }

  const changeVolume = (e) => {
    const v = videoRef.current
    if (!v) return
    v.volume = e.target.value
    setVolume(e.target.value)
    setMuted(v.muted)
  }

  const toggleFullscreen = () => {
    const el = containerRef.current
    if (!document.fullscreenElement) {
      el.requestFullscreen()
      setFullscreen(true)
    } else {
      document.exitFullscreen()
      setFullscreen(false)
    }
  }

  const changeRate = (rate) => {
    if (videoRef.current) videoRef.current.playbackRate = rate
    setPlaybackRate(rate)
    setShowRates(false)
  }

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas || !video || !trackingData?.players?.length) {
      togglePlay()
      return
    }
    const rect   = canvas.getBoundingClientRect()
    const clickX = (e.clientX - rect.left) * (video.videoWidth  / canvas.clientWidth)
    const clickY = (e.clientY - rect.top)  * (video.videoHeight / canvas.clientHeight)

    for (const p of trackingData.players) {
      const { bbox_x: x, bbox_y: y, bbox_w: w, bbox_h: h } = p
      if (clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h) {
        setSelectedPlayerId(p.player_id)
        return
      }
    }
    togglePlay()
  }

  const showControlsTemporarily = () => {
    setShowControls(true)
    clearTimeout(hideTimer.current)
    hideTimer.current = setTimeout(() => playing && setShowControls(false), 3000)
  }

  const progress = duration ? (currentTime / duration) * 100 : 0

  return (
    <div
      ref={containerRef}
      className="relative bg-black rounded-xl overflow-hidden group"
      onMouseMove={showControlsTemporarily}
      onMouseLeave={() => playing && setShowControls(false)}
    >
      {/* Video element */}
      <video
        ref={videoRef}
        src={videoSrc}
        className="w-full aspect-video"
        preload="auto"
        playsInline
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onError={(e) => console.error('Video error:', e.target.error)}
      />

      {/* Tracking overlay canvas — always rendered so clicks toggle play when no overlay */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full cursor-pointer"
        onClick={handleCanvasClick}
      />

      {/* Player profile modal */}
      {selectedPlayerId && matchId && (
        <PlayerProfileModal
          matchId={matchId}
          playerId={selectedPlayerId}
          onClose={() => setSelectedPlayerId(null)}
          onSeek={(ts) => {
            const v = videoRef.current
            if (v) v.currentTime = ts
            setSelectedPlayerId(null)
          }}
        />
      )}

      {/* Controls */}
      <div className={`absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent transition-opacity duration-200 ${showControls ? 'opacity-100' : 'opacity-0'}`}>
        {/* Progress bar */}
        <div
          className="mx-3 mb-1 h-1 bg-white/20 rounded-full cursor-pointer hover:h-1.5 transition-all group/bar"
          onClick={seek}
        >
          <div
            className="h-full bg-blue-500 rounded-full relative"
            style={{ width: `${progress}%` }}
          >
            <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full opacity-0 group-hover/bar:opacity-100 transition-opacity" />
          </div>
        </div>

        {/* Buttons row */}
        <div className="flex items-center gap-2 px-3 pb-2">
          <button onClick={() => skip(-10)} className="text-white/70 hover:text-white transition-colors p-1">
            <SkipBack className="w-4 h-4" />
          </button>
          <button onClick={togglePlay} className="text-white hover:text-blue-300 transition-colors p-1">
            {playing ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          </button>
          <button onClick={() => skip(10)} className="text-white/70 hover:text-white transition-colors p-1">
            <SkipForward className="w-4 h-4" />
          </button>

          {/* Volume */}
          <button onClick={toggleMute} className="text-white/70 hover:text-white transition-colors p-1">
            {muted || volume == 0 ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>
          <input
            type="range" min="0" max="1" step="0.05"
            value={muted ? 0 : volume}
            onChange={changeVolume}
            className="w-16 h-1 accent-blue-500"
          />

          {/* Time */}
          <span className="text-xs text-white/70 ml-1 tabular-nums">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>

          <div className="flex-1" />

          {/* Playback rate */}
          <div className="relative">
            <button
              onClick={() => setShowRates(r => !r)}
              className="flex items-center gap-1 text-xs text-white/70 hover:text-white transition-colors px-2 py-1 rounded bg-white/10"
            >
              <Settings className="w-3 h-3" />
              {playbackRate}x
            </button>
            {showRates && (
              <div className="absolute bottom-full right-0 mb-1 bg-court-panel border border-court-border rounded-lg overflow-hidden shadow-xl">
                {[0.25, 0.5, 0.75, 1, 1.25, 1.5, 2].map(r => (
                  <button
                    key={r}
                    onClick={() => changeRate(r)}
                    className={`block w-full text-left px-4 py-1.5 text-xs hover:bg-court-border transition-colors ${playbackRate === r ? 'text-blue-400' : 'text-slate-300'}`}
                  >
                    {r}x
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Fullscreen */}
          <button onClick={toggleFullscreen} className="text-white/70 hover:text-white transition-colors p-1">
            {fullscreen ? <Minimize className="w-4 h-4" /> : <Maximize className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </div>
  )
}
