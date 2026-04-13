/**
 * RotationPanel
 * ─────────────
 * Renders an SVG top-down volleyball court diagram with player markers
 * in their detected 6-slot rotation positions.
 *
 * Slot layout (front = near net):
 *   ┌─────────────────────────────────┐
 *   │  4 (FL)  │  5 (FC)  │  6 (FR)  │  ← front row (near net)
 *   ├──────────┼──────────┼───────────┤
 *   │  1 (BL)  │  2 (BC)  │  3 (BR)  │  ← back row
 *   └─────────────────────────────────┘
 *
 * Props:
 *   rotation  – object from GET /matches/{id}/rotations, or null
 *   players   – array of { id, display_name, team } for labels
 */

import React from 'react'

const COURT_W = 300
const COURT_H = 180
const PAD     = 16

// Slot centers as fractions of court (col: 0=left, 1=mid, 2=right; row: 0=front, 1=back)
const SLOT_CENTERS = {
  4: { cx: 0.17, cy: 0.25 },  // front-left
  5: { cx: 0.50, cy: 0.25 },  // front-center
  6: { cx: 0.83, cy: 0.25 },  // front-right
  1: { cx: 0.17, cy: 0.75 },  // back-left
  2: { cx: 0.50, cy: 0.75 },  // back-center
  3: { cx: 0.83, cy: 0.75 },  // back-right
}

const TEAM_COLORS = {
  A: { fill: '#1e40af', stroke: '#3b82f6', text: '#bfdbfe' },
  B: { fill: '#7f1d1d', stroke: '#ef4444', text: '#fecaca' },
  default: { fill: '#1e3a5f', stroke: '#60a5fa', text: '#e2e8f0' },
}

function getInitials(name) {
  if (!name) return '?'
  const parts = name.replace(/Player\s+#?/i, '').trim().split(/\s+/)
  if (parts.length === 1) return parts[0].slice(0, 3).toUpperCase()
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
}

export default function RotationPanel({ rotation, players = [] }) {
  if (!rotation) {
    return (
      <div className="flex items-center justify-center h-36 rounded-lg bg-court-bg border border-court-border">
        <p className="text-slate-500 text-xs">No rotation data for this rally</p>
      </div>
    )
  }

  const playerMap = {}
  players.forEach(p => { playerMap[p.id] = p })

  const slots = rotation.slots || {}

  function renderSlot(slotNum) {
    const { cx, cy } = SLOT_CENTERS[slotNum]
    const x = PAD + cx * COURT_W
    const y = PAD + cy * COURT_H
    const playerId = slots[String(slotNum)]
    const player   = playerId ? playerMap[playerId] : null
    const colors   = player
      ? (TEAM_COLORS[player.team] || TEAM_COLORS.default)
      : null

    return (
      <g key={slotNum}>
        {/* Slot background circle */}
        <circle
          cx={x} cy={y} r={18}
          fill={colors ? colors.fill : '#1a1f2e'}
          stroke={colors ? colors.stroke : '#2e3a52'}
          strokeWidth={1.5}
          opacity={colors ? 1 : 0.5}
        />

        {/* Slot number label (small, top-right of circle) */}
        <text
          x={x + 12} y={y - 12}
          fontSize="8" fill="#64748b"
          textAnchor="middle" fontWeight="600"
        >
          {slotNum}
        </text>

        {/* Player initials or empty dash */}
        <text
          x={x} y={y + 4}
          fontSize={colors ? 9 : 8}
          fill={colors ? colors.text : '#475569'}
          textAnchor="middle"
          fontWeight="700"
        >
          {colors ? getInitials(player?.display_name) : '—'}
        </text>
      </g>
    )
  }

  const svgW = COURT_W + PAD * 2
  const svgH = COURT_H + PAD * 2

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between px-1">
        <span className="text-xs text-slate-500 font-medium">Rotation snapshot</span>
        <span className="text-[10px] text-slate-600">
          {rotation.timestamp != null ? `t = ${rotation.timestamp.toFixed(1)}s` : ''}
          {rotation.team && rotation.team !== 'unknown' ? ` · ${rotation.team}` : ''}
        </span>
      </div>

      <svg
        viewBox={`0 0 ${svgW} ${svgH}`}
        width="100%"
        className="rounded-lg bg-[#0f1724] border border-court-border"
        style={{ maxHeight: 200 }}
      >
        {/* Court outline */}
        <rect
          x={PAD} y={PAD} width={COURT_W} height={COURT_H}
          fill="none" stroke="#2e3a52" strokeWidth={1.5} rx={4}
        />

        {/* Net line */}
        <line
          x1={PAD} y1={PAD + COURT_H / 2}
          x2={PAD + COURT_W} y2={PAD + COURT_H / 2}
          stroke="#3b82f6" strokeWidth={1} strokeDasharray="6 4" opacity={0.6}
        />

        {/* Attack lines (3m from net) */}
        <line
          x1={PAD} y1={PAD + COURT_H * 0.37}
          x2={PAD + COURT_W} y2={PAD + COURT_H * 0.37}
          stroke="#2e3a52" strokeWidth={0.75} strokeDasharray="3 3" opacity={0.5}
        />
        <line
          x1={PAD} y1={PAD + COURT_H * 0.63}
          x2={PAD + COURT_W} y2={PAD + COURT_H * 0.63}
          stroke="#2e3a52" strokeWidth={0.75} strokeDasharray="3 3" opacity={0.5}
        />

        {/* Column dividers */}
        {[0.33, 0.67].map(f => (
          <line
            key={f}
            x1={PAD + f * COURT_W} y1={PAD}
            x2={PAD + f * COURT_W} y2={PAD + COURT_H}
            stroke="#1e2d42" strokeWidth={0.75}
          />
        ))}

        {/* Net label */}
        <text
          x={PAD + COURT_W + 5} y={PAD + COURT_H / 2 + 3}
          fontSize="7" fill="#3b82f6" fontWeight="600"
        >
          NET
        </text>

        {/* Slot markers */}
        {[1, 2, 3, 4, 5, 6].map(renderSlot)}
      </svg>

      {/* Legend */}
      <div className="flex items-center gap-3 px-1">
        <span className="flex items-center gap-1 text-[10px] text-slate-500">
          <span className="w-2.5 h-2.5 rounded-full bg-[#1e40af] border border-[#3b82f6] inline-block" />
          Team A
        </span>
        <span className="flex items-center gap-1 text-[10px] text-slate-500">
          <span className="w-2.5 h-2.5 rounded-full bg-[#7f1d1d] border border-[#ef4444] inline-block" />
          Team B
        </span>
        <span className="flex items-center gap-1 text-[10px] text-slate-500">
          <span className="w-2.5 h-2.5 rounded-full bg-[#1a1f2e] border border-[#2e3a52] inline-block" />
          Empty
        </span>
      </div>
    </div>
  )
}
