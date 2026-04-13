// frontend/src/pages/PlayerComparePage.jsx
import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip, Legend,
} from 'recharts'
import { Users, ChevronUp, ChevronDown, Minus, Loader2 } from 'lucide-react'
import { analyticsAPI } from '../services/api'

const CHART_STYLE = {
  contentStyle: { background: '#232b3e', border: '1px solid #2e3a52', borderRadius: 8, color: '#f1f5f9' },
}

const STAT_ROWS = [
  { key: 'attacks',      label: 'Attacks'          },
  { key: 'kills',        label: 'Kills'             },
  { key: 'attack_eff',   label: 'Attack Eff %'      },
  { key: 'serves',       label: 'Serves'            },
  { key: 'aces',         label: 'Aces'              },
  { key: 'serve_eff',    label: 'Serve Eff %'       },
  { key: 'blocks',       label: 'Blocks'            },
  { key: 'block_pts',    label: 'Block Points'      },
  { key: 'digs',         label: 'Digs'              },
  { key: 'receptions',   label: 'Receptions'        },
  { key: 'reception_eff',label: 'Reception Eff %'   },
]

function DeltaBadge({ a, b }) {
  if (a === b) return <Minus className="w-3 h-3 text-slate-500 mx-auto" />
  if (a > b) return <ChevronUp className="w-3 h-3 text-green-400 mx-auto" />
  return <ChevronDown className="w-3 h-3 text-red-400 mx-auto" />
}

export default function PlayerComparePage() {
  const [playerAId, setPlayerAId] = useState('')
  const [playerBId, setPlayerBId] = useState('')

  const { data: players = [], isLoading: loadingPlayers } = useQuery({
    queryKey: ['players-list'],
    queryFn: () => analyticsAPI.playersList().then(r => r.data),
  })

  const enabled = !!(playerAId && playerBId && playerAId !== playerBId)

  const { data: comparison, isLoading: loadingCompare } = useQuery({
    queryKey: ['player-compare', playerAId, playerBId],
    queryFn: () => analyticsAPI.playerCompare(playerAId, playerBId).then(r => r.data),
    enabled,
  })

  const pa = comparison?.player_a
  const pb = comparison?.player_b

  const radarData = pa && pb ? [
    { stat: 'Attacking',  a: Math.min(Math.abs(pa.attack_eff), 100),     b: Math.min(Math.abs(pb.attack_eff), 100) },
    { stat: 'Serving',    a: Math.min(Math.abs(pa.serve_eff), 100),      b: Math.min(Math.abs(pb.serve_eff), 100) },
    { stat: 'Blocking',   a: Math.min((pa.blocks / Math.max(pa.blocks + pb.blocks, 1)) * 100, 100),
                          b: Math.min((pb.blocks / Math.max(pa.blocks + pb.blocks, 1)) * 100, 100) },
    { stat: 'Digging',    a: Math.min((pa.digs / Math.max(pa.digs + pb.digs, 1)) * 100, 100),
                          b: Math.min((pb.digs / Math.max(pa.digs + pb.digs, 1)) * 100, 100) },
    { stat: 'Reception',  a: Math.min(Math.abs(pa.reception_eff), 100),  b: Math.min(Math.abs(pb.reception_eff), 100) },
  ] : []

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-white">Player Comparison</h2>
        <p className="text-slate-400 text-sm mt-1">Select two players to compare their stats</p>
      </div>

      {/* Player selectors */}
      <div className="card grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          { label: 'Player A', value: playerAId, set: setPlayerAId, color: 'text-blue-400' },
          { label: 'Player B', value: playerBId, set: setPlayerBId, color: 'text-orange-400' },
        ].map(({ label, value, set, color }) => (
          <div key={label}>
            <label className={`text-xs font-semibold ${color} mb-1 block`}>{label}</label>
            <select
              value={value}
              onChange={e => set(e.target.value)}
              className="w-full bg-[#1a1f2e] text-slate-300 border border-[#2e3a52] rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-500"
            >
              <option value="">— Select player —</option>
              {players.map(p => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.team}) — {p.match_title}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>

      {loadingCompare && (
        <div className="flex items-center justify-center h-32">
          <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
        </div>
      )}

      {pa && pb && !loadingCompare && (
        <>
          {/* Player name headers */}
          <div className="grid grid-cols-2 gap-4">
            <div className="card text-center">
              <Users className="w-8 h-8 text-blue-400 mx-auto mb-1" />
              <div className="text-white font-bold">{pa.name}</div>
              <div className="text-xs text-slate-400">Team {pa.team}</div>
            </div>
            <div className="card text-center">
              <Users className="w-8 h-8 text-orange-400 mx-auto mb-1" />
              <div className="text-white font-bold">{pb.name}</div>
              <div className="text-xs text-slate-400">Team {pb.team}</div>
            </div>
          </div>

          {/* Radar chart */}
          <div className="card">
            <h3 className="text-sm font-semibold text-white mb-3">Skill Comparison</h3>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#2e3a52" />
                <PolarAngleAxis dataKey="stat" tick={{ fill: '#64748b', fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 9 }} />
                <Radar name={pa.name} dataKey="a" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
                <Radar name={pb.name} dataKey="b" stroke="#f97316" fill="#f97316" fillOpacity={0.25} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
                <Tooltip contentStyle={CHART_STYLE.contentStyle} formatter={v => `${v.toFixed(1)}`} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Stat diff table */}
          <div className="card overflow-x-auto">
            <h3 className="text-sm font-semibold text-white mb-3">Stat Breakdown</h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-500 border-b border-[#2e3a52] text-xs">
                  <th className="text-right py-2 pr-4 text-blue-400">{pa.name}</th>
                  <th className="text-center py-2 px-3">Stat</th>
                  <th className="text-center py-2 px-2">Δ</th>
                  <th className="text-left py-2 pl-4 text-orange-400">{pb.name}</th>
                </tr>
              </thead>
              <tbody>
                {STAT_ROWS.map(({ key, label }) => (
                  <tr key={key} className="border-b border-[#2e3a52]/50 last:border-0">
                    <td className="py-2 pr-4 text-right text-blue-300 font-mono">{pa[key]}</td>
                    <td className="py-2 px-3 text-center text-slate-400 text-xs">{label}</td>
                    <td className="py-2 px-2 text-center">
                      <DeltaBadge a={pa[key]} b={pb[key]} />
                    </td>
                    <td className="py-2 pl-4 text-left text-orange-300 font-mono">{pb[key]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {!enabled && !loadingCompare && (
        <div className="card text-center py-12 text-slate-500">
          Select two different players above to compare their stats.
        </div>
      )}
    </div>
  )
}
