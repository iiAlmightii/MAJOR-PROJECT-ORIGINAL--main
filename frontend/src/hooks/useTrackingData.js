/**
 * useTrackingData
 * ───────────────
 * Fetches player + ball tracking data from the backend as the video plays.
 * Throttled to one request per 100ms to avoid overwhelming the API.
 */
import { useState, useRef, useCallback } from 'react'
import api from '../services/api'

export function useTrackingData(matchId) {
  const [trackingData, setTrackingData] = useState(null)
  const lastFetch = useRef(0)
  const pending   = useRef(false)

  const fetchAtTime = useCallback(async (timestamp) => {
    if (!matchId || pending.current) return
    const now = Date.now()
    if (now - lastFetch.current < 100) return   // throttle

    pending.current = true
    lastFetch.current = now

    try {
      const { data } = await api.get(`/matches/${matchId}/tracking`, {
        params: { timestamp, window: 0.15 },
      })
      setTrackingData(data)
    } catch {
      // Silently ignore — tracking may not be ready yet
    } finally {
      pending.current = false
    }
  }, [matchId])

  return { trackingData, fetchAtTime }
}
