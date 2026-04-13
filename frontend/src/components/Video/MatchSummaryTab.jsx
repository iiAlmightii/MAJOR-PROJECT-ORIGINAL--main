/**
 * MatchSummaryTab
 * ───────────────
 * 5th tab on MatchDetailPage showing:
 *   1. Ball trajectory heat map on court SVG
 *   2. Ball zone bar chart (court split into 6 volleyball zones)
 *   3. Key moments timeline from match.summary.key_moments
 *
 * Props:
 *   match        – match object (needs match.summary for key_moments)
 *   heatmapData  – response from GET /matches/{id}/tracking/ball-heatmap
 *   matchId      – string UUID (used to build annotate link)
 */

import React, { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'
import { Activity, Zap, Award, Clock } from 'lucide-react'

const CHART_STYLE = {
  contentStyle: {
    background: '#232b3e', border: '1px solid #2e3a52',
    borderRadius: 8, color: '#f1f5f9',
  },
  labelStyle: { color: '#94a3b8' },
}

// Map a heat count → interpolated colour between cold (blue) and hot (orange→red)
function heatColor(count, maxCount) {
  if (!maxCount || count === 0) return 'rgba(59, 130, 246, 0.08)'
  const t = Math.min(count / maxCount, 1)
  if (t < 0.33) {
    const k = t / 0.33
    return `rgba(59, 130, 246, ${0.15 + k * 0.35})`
  } else if (t < 0.66) {
    const k = (t - 0.33) / 0.33
    return `rgba(249, 115, 22, ${0.4 + k * 0.3})`
  } else {
    const k = (t - 0.66) / 0.34
    return `rgba(239, 68, 68, ${0.6 + k * 0.4})`
  }
}

// Volleyball zones 1-6 (standard FIVB layout):
//   Back row:  zone 1 (right), zone 6 (center), zone 5 (left)
//   Front row: zone 2 (right), zone 3 (center), zone 4 (left)
const ZONE_BOUNDS = {
  1: { xMin: 0.67, xMax: 1.0,  yMin: 0.5, yMax: 1.0  },
  6: { xMin: 0.33, xMax: 0.67, yMin: 0.5, yMax: 1.0  },
  5: { xMin: 0.0,  xMax: 0.33, yMin: 0.5, yMax: 1.0  },
  2: { xMin: 0.67, xMax: 1.0,  yMin: 0.0, yMax: 0.5  },
  3: { xMin: 0.33, xMax: 0.67, yMin: 0.0, yMax: 0.5  },
  4: { xMin: 0.0,  xMax: 0.33, yMin: 0.0, yMax: 0.5  },
}

const ZONE_COLORS = {
  1: '#3b82f6', 2: '#f97316', 3: '#22c55e',
  4: '#a855f7', 5: '#06b6d4', 6: '#eab308',
}

function fmtTime(secs) {
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

function seekVideo(ts) {
  const v = document.querySelector('video')
  if (v) v.currentTime = ts
}

// ── Heat map SVG ────────────────────────────────────────────────────────────

function BallHeatmap({ heatmapData }) {
  const SVG_W = 320
  const SVG_H = 200
  const PAD   = 10

  if (!heatmapData || heatmapData.total_points === 0) {
    return (
      <div className="flex items-center justify-center h-44 rounded-lg bg-court-bg border border-court-border">
        <p className="text-slate-500 text-xs">No ball tracking data available</p>
      </div>
    )
  }

  const { cols, rows, max_count, cells } = heatmapData
  const cellW = (SVG_W - PAD * 2) / cols
  const cellH = (SVG_H - PAD * 2) / rows

  return (
    <div className="space-y-1">
      <span className="text-xs text-slate-500 font-medium px-1">Ball trajectory heat map</span>
      <svg
        viewBox={`0 0 ${SVG_W + 20} ${SVG_H + PAD}`}
        width="100%"
        className="rounded-lg bg-[#0f1724] border border-court-border"
        style={{ maxHeight: 220 }}
      >
        {/* Court outline */}
        <rect
          x={PAD} y={PAD} width={SVG_W - PAD * 2} height={SVG_H - PAD * 2}
          fill="none" stroke="#2e3a52" strokeWidth={1.5} rx={3}
        />
        {/* Net */}
        <line
          x1={PAD} y1={PAD + (SVG_H - PAD * 2) / 2}
          x2={SVG_W - PAD} y2={PAD + (SVG_H - PAD * 2) / 2}
          stroke="#3b82f6" strokeWidth={1} strokeDasharray="5 4" opacity={0.5}
        />

        {/* Heat cells */}
        {cells.map(({ col, row, count }) => (
          <rect
            key={`${col}-${row}`}
            x={PAD + col * cellW}
            y={PAD + row * cellH}
            width={cellW}
            height={cellH}
            fill={heatColor(count, max_count)}
            rx={1}
          />
        ))}

        {/* NET label */}
        <text
          x={SVG_W - PAD + 3}
          y={PAD + (SVG_H - PAD * 2) / 2 + 3}
          fontSize="7" fill="#3b82f6" fontWeight="600"
        >
          NET
        </text>

        {/* Color scale legend */}
        {[0.1, 0.3, 0.6, 0.9].map((t, i) => (
          <rect
            key={i}
            x={PAD + i * 28} y={SVG_H - 4}
            width={26} height={5}
            rx={2}
            fill={heatColor(Math.round(t * max_count), max_count)}
          />
        ))}
        <text x={PAD} y={SVG_H + 8} fontSize="7" fill="#475569">low</text>
        <text x={PAD + 4 * 28 - 14} y={SVG_H + 8} fontSize="7" fill="#475569">high</text>
      </svg>
      <p className="text-[10px] text-slate-600 px-1">
        {heatmapData.total_points.toLocaleString()} ball detections
      </p>
    </div>
  )
}

// ── Zone bar chart ──────────────────────────────────────────────────────────

function ZoneBarChart({ heatmapData }) {
  const zoneData = useMemo(() => {
    if (!heatmapData || heatmapData.total_points === 0) return []

    const counts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 }
    const { cols, rows, cells } = heatmapData

    cells.forEach(({ col, row, count }) => {
      const cx = (col + 0.5) / cols
      const cy = (row + 0.5) / rows
      for (const [zone, { xMin, xMax, yMin, yMax }] of Object.entries(ZONE_BOUNDS)) {
        if (cx >= xMin && cx < xMax && cy >= yMin && cy < yMax) {
          counts[Number(zone)] += count
          break
        }
      }
    })

    return Object.entries(counts).map(([zone, count]) => ({
      zone: `Zone ${zone}`,
      count,
      fill: ZONE_COLORS[Number(zone)],
    }))
  }, [heatmapData])

  if (!zoneData.length) return null

  return (
    <div className="space-y-1">
      <span className="text-xs text-slate-500 font-medium px-1">Ball touches by court zone</span>
      <div className="bg-court-bg rounded-lg border border-court-border p-3" style={{ height: 180 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={zoneData} barSize={28}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2e3a52" />
            <XAxis dataKey="zone" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
            <Tooltip
              contentStyle={CHART_STYLE.contentStyle}
              labelStyle={CHART_STYLE.labelStyle}
              formatter={(v) => [v, 'ball detections']}
            />
            <Bar dataKey="count" radius={[3, 3, 0, 0]}>
              {zoneData.map((d, i) => (
                <Cell key={i} fill={d.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ── Key moments timeline ────────────────────────────────────────────────────

function KeyMomentsTimeline({ moments }) {
  if (!moments || moments.length === 0) {
    return (
      <div className="card text-center py-6">
        <Clock className="w-7 h-7 text-slate-600 mx-auto mb-2" />
        <p className="text-slate-500 text-xs">No key moments detected</p>
      </div>
    )
  }

  const MOMENT_ICON = {
    kill:        { icon: Zap,      color: 'text-orange-400', bg: 'bg-orange-900/30' },
    ace:         { icon: Award,    color: 'text-green-400',  bg: 'bg-green-900/30' },
    block:       { icon: Activity, color: 'text-cyan-400',   bg: 'bg-cyan-900/30' },
    serve_error: { icon: Activity, color: 'text-red-400',    bg: 'bg-red-900/30' },
    attack_error:{ icon: Activity, color: 'text-red-400',    bg: 'bg-red-900/30' },
  }

  return (
    <div className="space-y-1">
      <span className="text-xs text-slate-500 font-medium px-1">Key moments</span>
      <div className="space-y-1.5">
        {moments.map((m, idx) => {
          const meta = MOMENT_ICON[m.type] || { icon: Activity, color: 'text-slate-400', bg: 'bg-slate-800' }
          const Icon = meta.icon
          return (
            <div
              key={idx}
              className="card flex items-center gap-3 py-2 cursor-pointer hover:border-blue-600/40 transition-colors"
              onClick={() => m.timestamp != null && seekVideo(m.timestamp)}
            >
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${meta.bg}`}>
                <Icon className={`w-4 h-4 ${meta.color}`} />
              </div>
              <div className="flex-1 min-w-0">
                <div className={`text-xs font-semibold capitalize ${meta.color}`}>
                  {m.type?.replace(/_/g, ' ')}
                </div>
                {m.description && (
                  <div className="text-[10px] text-slate-500 truncate">{m.description}</div>
                )}
              </div>
              {m.team && (
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                  m.team === 'A' ? 'bg-blue-900/30 text-blue-400' : 'bg-red-900/30 text-red-400'
                }`}>
                  Team {m.team}
                </span>
              )}
              {m.timestamp != null && (
                <span className="text-xs text-slate-500 flex-shrink-0">{fmtTime(m.timestamp)}</span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Main export ─────────────────────────────────────────────────────────────

export default function MatchSummaryTab({ match, heatmapData }) {
  if (!match || match.status !== 'completed') {
    return (
      <div className="card text-center py-10">
        <Activity className="w-9 h-9 text-slate-600 mx-auto mb-2" />
        <p className="text-slate-400 text-sm">Run analysis to generate match summary</p>
      </div>
    )
  }

  const keyMoments = match.summary?.key_moments || []

  return (
    <div className="space-y-5">
      {/* Heat map + Zone chart side by side on wide screens */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <BallHeatmap heatmapData={heatmapData} />
        <ZoneBarChart heatmapData={heatmapData} />
      </div>

      {/* Key moments */}
      <KeyMomentsTimeline moments={keyMoments} />
    </div>
  )
}
