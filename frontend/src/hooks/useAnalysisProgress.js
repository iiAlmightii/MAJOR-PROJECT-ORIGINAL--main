/**
 * useAnalysisProgress
 * ────────────────────
 * Connects to the WebSocket progress endpoint for a match analysis.
 * Returns { progress, message, connected }.
 */
import { useState, useEffect, useRef } from 'react'

export function useAnalysisProgress(matchId, enabled = false) {
  const [progress,  setProgress]  = useState(0)
  const [message,   setMessage]   = useState('')
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const pingRef = useRef(null)

  useEffect(() => {
    if (!matchId || !enabled) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host     = window.location.host
    const url      = `${protocol}//${host}/api/matches/${matchId}/ws/progress`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      // Send periodic pings to keep alive
      pingRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping')
      }, 20_000)
    }

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        if (data.progress !== undefined) {
          setProgress(data.progress < 0 ? 0 : data.progress)
          setMessage(data.message || '')
        }
      } catch {/* ignore malformed messages */}
    }

    ws.onclose  = () => { setConnected(false); clearInterval(pingRef.current) }
    ws.onerror  = () => setConnected(false)

    return () => {
      clearInterval(pingRef.current)
      ws.close()
    }
  }, [matchId, enabled])

  return { progress, message, connected }
}
