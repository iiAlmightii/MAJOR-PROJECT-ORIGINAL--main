import React, { useRef, useState, useEffect, useCallback } from 'react'
import {
  Play, Pause, Volume2, VolumeX, Maximize, Minimize,
  SkipBack, SkipForward, Settings
} from 'lucide-react'
import { videosAPI } from '../../services/api'

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
export default function VideoPlayer({ videoId, trackingData = null, onTimeUpdate, showOverlay = true }) {
  const videoRef   = useRef(null)
  const canvasRef  = useRef(null)
  const containerRef = useRef(null)
  const [playing,    setPlaying]    = useState(false)
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
        const labelText = `#${p.player_track_id ?? '?'}`
        ctx.font = 'bold 11px Inter'
        const labelW = ctx.measureText(labelText).width + 6
        ctx.fillStyle = boxColor + 'cc'
        ctx.fillRect(x, y - 20, labelW, 20)

        // Label text
        ctx.fillStyle = '#ffffff'
        ctx.fillText(labelText, x + 3, y - 5)
      })
    }

    // Draw ball
    if (trackingData.ball) {
      const b = trackingData.ball
      ctx.beginPath()
      ctx.arc(b.x * scaleX, b.y * scaleY, 8, 0, Math.PI * 2)
      ctx.fillStyle = '#facc15cc'
      ctx.fill()
      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 2
      ctx.stroke()
    }

    // Draw homography mini-map (bottom-right)
    drawMiniMap(ctx, canvas.width, canvas.height, trackingData)
  }, [trackingData])

  const drawMiniMap = (ctx, cw, ch, data) => {
    const mapW = 160
    const mapH = 100
    const mapX = cw - mapW - 12
    const mapY = ch - mapH - 12

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
      data.players.forEach(p => {
        if (p.court_x == null || p.court_y == null) return
        const cx = Math.max(0, Math.min(1, p.court_x))
        const cy = Math.max(0, Math.min(1, p.court_y))
        const px = mapX + cx * mapW
        const py = mapY + cy * mapH
        ctx.beginPath()
        ctx.arc(px, py, 4, 0, Math.PI * 2)
        ctx.fillStyle = p.team === 'A' ? '#3b82f6' : p.team === 'B' ? '#ef4444' : '#94a3b8'
        ctx.fill()
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 7px Inter'
        ctx.fillText(p.player_track_id ?? '?', px - 3, py + 2.5)
      })
    }

    // Ball on mini-map
    if (data.ball?.court_x != null) {
      const bx = mapX + data.ball.court_x * mapW
      const by = mapY + data.ball.court_y * mapH
      ctx.beginPath()
      ctx.arc(bx, by, 3, 0, Math.PI * 2)
      ctx.fillStyle = '#facc15'
      ctx.fill()
    }

    // Label
    ctx.fillStyle = '#64748b'
    ctx.font = '8px Inter'
    ctx.fillText('COURT VIEW', mapX + 4, mapY + 10)
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
    v.paused ? v.play() : v.pause()
    setPlaying(!v.paused)
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
        onClick={togglePlay}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        crossOrigin="use-credentials"
      />

      {/* Tracking overlay canvas */}
      {trackingData && (
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
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
