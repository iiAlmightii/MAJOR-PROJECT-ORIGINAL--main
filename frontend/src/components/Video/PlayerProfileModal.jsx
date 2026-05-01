import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { matchesAPI } from '../../services/api'

const TEAM_COLORS = {
  A: { bg: 'from-blue-900 to-blue-800', badge: 'bg-blue-600', dot: '#3b82f6' },
  B: { bg: 'from-red-900 to-red-800',  badge: 'bg-red-600',  dot: '#ef4444' },
}

const RESULT_ICON = { success: '✓', error: '✗', neutral: '—' }
const RESULT_COLOR = { success: 'text-green-400', error: 'text-red-400', neutral: 'text-yellow-400' }

function fmtTime(secs) {
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}

export default function PlayerProfileModal({ matchId, playerId, onClose, onSeek }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['playerStats', matchId, playerId],
    queryFn:  () => matchesAPI.playerStats(matchId, playerId).then(r => r.data),
    staleTime: 60_000,
  })

  // Close on Escape
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const colors = TEAM_COLORS[data?.team] || TEAM_COLORS.A

  const ACTION_LABELS = {
    attack: 'Attacks', block: 'Blocks', dig: 'Digs',
    serve: 'Serves', set: 'Sets', reception: 'Receptions',
  }

  const maxZoneCount = data
    ? Math.max(1, ...Object.values(data.zones).map(Number))
    : 1

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="bg-[#0d1117] rounded-xl overflow-hidden w-full max-w-lg shadow-2xl border border-white/10">

        {/* Header */}
        <div className={`bg-gradient-to-r ${colors.bg} p-5 flex items-center gap-4 relative`}>
          <div className={`w-14 h-14 rounded-full ${colors.badge} flex items-center justify-center text-2xl font-bold text-white`}>
            {isLoading ? '?' : `#${data?.display_number ?? '?'}`}
          </div>
          <div>
            <div className="text-lg font-bold text-white">
              {isLoading ? 'Loading...' : `Player #${data?.display_number ?? '?'}`}
            </div>
            {data && (
              <div className="flex items-center gap-2 mt-1">
                <span className={`${colors.badge} text-white text-xs font-semibold px-2 py-0.5 rounded-full`}>
                  Team {data.team}
                </span>
                <span className="text-white/50 text-sm">
                  {fmtTime(data.presence.time_on_court_seconds)} on court
                </span>
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            className="absolute right-4 top-4 text-white/40 hover:text-white/80 text-xl"
          >✕</button>
        </div>

        {isLoading && (
          <div className="p-8 text-center text-white/40">Loading stats...</div>
        )}

        {isError && (
          <div className="p-8 text-center text-red-400">Failed to load player stats.</div>
        )}

        {data && (
          <>
            {/* Involvement bar */}
            <div className="px-5 pt-4 pb-3 border-b border-white/5">
              <div className="flex justify-between mb-1.5">
                <span className="text-xs text-white/40 uppercase tracking-wider">Match Involvement</span>
                <span className="text-sm font-semibold text-green-400">{data.presence.involvement_pct}%</span>
              </div>
              <div className="h-1.5 bg-white/10 rounded-full">
                <div
                  className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full"
                  style={{ width: `${Math.min(data.presence.involvement_pct, 100)}%` }}
                />
              </div>
            </div>

            {/* Stats grid */}
            <div className="px-5 py-3 grid grid-cols-3 gap-2 border-b border-white/5">
              {Object.entries(ACTION_LABELS).map(([key, label]) => (
                <div key={key} className="bg-white/[0.04] rounded-lg p-2.5 text-center">
                  <div className="text-xl font-bold text-blue-400">
                    {data.actions[key]?.total ?? 0}
                  </div>
                  <div className="text-xs text-white/40 mt-0.5">{label}</div>
                </div>
              ))}
              <div className="bg-white/[0.04] rounded-lg p-2.5 text-center">
                <div className="text-xl font-bold text-orange-400">
                  {data.efficiency.attack_eff.toFixed(2)}
                </div>
                <div className="text-xs text-white/40 mt-0.5">Atk Eff</div>
              </div>
            </div>

            {/* Zone activity */}
            {Object.keys(data.zones).length > 0 && (
              <div className="px-5 py-3 border-b border-white/5">
                <p className="text-xs text-white/40 uppercase tracking-wider mb-2">Court Zone Activity</p>
                <div className="flex items-end gap-1 h-16">
                  {[1, 2, 3, 4, 5, 6].map((z) => {
                    const count = Number(data.zones[String(z)] ?? 0)
                    const height = Math.max(4, (count / maxZoneCount) * 56)
                    return (
                      <div key={z} className="flex-1 flex flex-col items-center gap-1">
                        <div
                          className="w-full bg-blue-500 rounded-t"
                          style={{ height: `${height}px`, opacity: 0.3 + (count / maxZoneCount) * 0.7 }}
                        />
                        <span className="text-xs text-white/30">Z{z}</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Recent actions */}
            {data.recent_actions.length > 0 && (
              <div className="px-5 py-3">
                <p className="text-xs text-white/40 uppercase tracking-wider mb-2">Recent Actions</p>
                <div className="space-y-1.5 max-h-36 overflow-y-auto">
                  {data.recent_actions.map((a, i) => (
                    <button
                      key={i}
                      onClick={() => onSeek(a.timestamp)}
                      className="w-full flex justify-between items-center text-sm
                                 hover:bg-white/5 rounded px-2 py-1 transition-colors text-left"
                    >
                      <span className="text-white/70 capitalize">{a.action_type}</span>
                      <span className={RESULT_COLOR[a.result]}>
                        {RESULT_ICON[a.result]} {a.result}
                      </span>
                      <span className="text-white/30">{fmtTime(a.timestamp)}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
