import React, { useState } from 'react'
import { X, Filter, ChevronDown, ChevronUp, RotateCcw } from 'lucide-react'
import clsx from 'clsx'

const ACTION_TYPES = [
  { id: 'serve',             label: 'Serve',            color: 'blue'   },
  { id: 'reception',         label: 'Reception',        color: 'green'  },
  { id: 'set',               label: 'Set',              color: 'yellow' },
  { id: 'attack',            label: 'Attack / Spike',   color: 'red'    },
  { id: 'block',             label: 'Block',            color: 'purple' },
  { id: 'dig',               label: 'Dig',              color: 'teal'   },
  { id: 'free_ball_sent',    label: 'Free Ball Sent',   color: 'slate'  },
  { id: 'free_ball_received',label: 'Free Ball Recv.',  color: 'slate'  },
]

const POSITIONS = ['Setter', 'Libero', 'Outside Hitter', 'Opposite Hitter', 'Middle Blocker']
const ZONES     = [1, 2, 3, 4, 5, 6]
const TIME_OPTS = [
  { id: '+1', label: '+1 sec' },
  { id: '+2', label: '+2 sec' },
  { id: '+5', label: '+5 sec' },
  { id: 'end', label: 'End of Rally' },
]

const ACTION_COLOR = {
  blue:   'border-blue-600/40   bg-blue-900/20   text-blue-400',
  green:  'border-green-600/40  bg-green-900/20  text-green-400',
  yellow: 'border-yellow-600/40 bg-yellow-900/20 text-yellow-400',
  red:    'border-red-600/40    bg-red-900/20    text-red-400',
  purple: 'border-purple-600/40 bg-purple-900/20 text-purple-400',
  teal:   'border-teal-600/40   bg-teal-900/20   text-teal-400',
  slate:  'border-slate-600/40  bg-slate-700/20  text-slate-400',
}

function Section({ title, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border-b border-court-border last:border-0">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-court-border/30 transition-colors"
      >
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</span>
        {open ? <ChevronUp className="w-3.5 h-3.5 text-slate-500" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-500" />}
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  )
}

function ToggleChip({ label, active, color = 'blue', onClick }) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'text-xs px-2.5 py-1 rounded-full border transition-all duration-150 font-medium',
        active ? ACTION_COLOR[color] : 'border-court-border text-slate-500 hover:border-slate-500 hover:text-slate-400'
      )}
    >
      {label}
    </button>
  )
}

export default function FilterPanel({ open, onClose, players = [], filters, onChange }) {
  const toggle = (key, value) => {
    const current = filters[key] || []
    const next = current.includes(value)
      ? current.filter(v => v !== value)
      : [...current, value]
    onChange({ ...filters, [key]: next })
  }

  const toggleSingle = (key, value) => {
    onChange({ ...filters, [key]: filters[key] === value ? null : value })
  }

  const reset = () => onChange({
    players: [], actions: [], positions: [], zones: [], timeOffset: null, labels: [],
  })

  const activeCount = [
    ...(filters.players   || []),
    ...(filters.actions   || []),
    ...(filters.positions || []),
    ...(filters.zones     || []),
    filters.timeOffset ? [filters.timeOffset] : [],
  ].flat().length

  return (
    <>
      {/* Backdrop */}
      {open && (
        <div
          className="fixed inset-0 bg-black/30 z-20 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Panel */}
      <aside
        className={clsx(
          'absolute right-0 top-0 h-full w-72 bg-court-panel border-l border-court-border',
          'flex flex-col z-30 transition-transform duration-300 shadow-2xl',
          open ? 'translate-x-0' : 'translate-x-full'
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-court-border flex-shrink-0">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-blue-400" />
            <span className="font-semibold text-white text-sm">Filters</span>
            {activeCount > 0 && (
              <span className="w-5 h-5 bg-blue-600 rounded-full text-xs flex items-center justify-center text-white font-bold">
                {activeCount}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {activeCount > 0 && (
              <button onClick={reset} className="text-xs text-slate-500 hover:text-red-400 transition-colors flex items-center gap-1">
                <RotateCcw className="w-3 h-3" /> Reset
              </button>
            )}
            <button onClick={onClose} className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-court-border">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto">

          {/* Players */}
          <Section title="Players">
            {players.length === 0 ? (
              <p className="text-xs text-slate-600">No players detected yet</p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {players.map(p => (
                  <ToggleChip
                    key={p.id}
                    label={`#${p.player_track_id}`}
                    active={(filters.players || []).includes(p.id)}
                    color="blue"
                    onClick={() => toggle('players', p.id)}
                  />
                ))}
              </div>
            )}
          </Section>

          {/* Actions */}
          <Section title="Actions">
            <div className="flex flex-wrap gap-2">
              {ACTION_TYPES.map(a => (
                <ToggleChip
                  key={a.id}
                  label={a.label}
                  active={(filters.actions || []).includes(a.id)}
                  color={a.color}
                  onClick={() => toggle('actions', a.id)}
                />
              ))}
            </div>
          </Section>

          {/* Positions */}
          <Section title="Positions">
            <div className="flex flex-wrap gap-2">
              {POSITIONS.map(p => (
                <ToggleChip
                  key={p}
                  label={p}
                  active={(filters.positions || []).includes(p)}
                  color="purple"
                  onClick={() => toggle('positions', p)}
                />
              ))}
            </div>
          </Section>

          {/* Court Zones */}
          <Section title="Court Zones">
            <div className="grid grid-cols-3 gap-2">
              {/* Zone layout (volleyball zones 1-6) */}
              {[4, 3, 2, 5, 6, 1].map(z => (
                <button
                  key={z}
                  onClick={() => toggle('zones', z)}
                  className={clsx(
                    'aspect-square rounded-lg border text-sm font-bold transition-all',
                    (filters.zones || []).includes(z)
                      ? 'border-blue-500 bg-blue-900/30 text-blue-400'
                      : 'border-court-border text-slate-500 hover:border-slate-500 hover:text-slate-400'
                  )}
                >
                  Z{z}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-slate-600 mt-2">Back row: 1(right), 6(mid), 5(left)</p>
          </Section>

          {/* Action Time */}
          <Section title="Action Time">
            <div className="flex flex-wrap gap-2">
              {TIME_OPTS.map(t => (
                <ToggleChip
                  key={t.id}
                  label={t.label}
                  active={filters.timeOffset === t.id}
                  color="yellow"
                  onClick={() => toggleSingle('timeOffset', t.id)}
                />
              ))}
            </div>
          </Section>

          {/* Labels */}
          <Section title="Labels" defaultOpen={false}>
            <div className="flex flex-wrap gap-2">
              {['Starred', 'Key Moment', 'Error', 'Best Play'].map(l => (
                <ToggleChip
                  key={l}
                  label={l}
                  active={(filters.labels || []).includes(l)}
                  color="yellow"
                  onClick={() => toggle('labels', l)}
                />
              ))}
            </div>
          </Section>
        </div>

        {/* Apply indicator */}
        {activeCount > 0 && (
          <div className="px-4 py-3 border-t border-court-border bg-blue-900/20 flex-shrink-0">
            <p className="text-xs text-blue-400 text-center">
              {activeCount} filter{activeCount !== 1 ? 's' : ''} active — video highlights updated
            </p>
          </div>
        )}
      </aside>
    </>
  )
}
