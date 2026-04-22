import React, { useEffect, useState, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ChevronLeft, SlidersHorizontal, BarChart2, Loader2,
  Play, Zap, Users, AlertCircle, Wifi, WifiOff, Eye, EyeOff,
  Activity, Shield, ArrowUp, Radio, Minus, Tag
} from 'lucide-react'
import toast from 'react-hot-toast'
import { matchesAPI } from '../services/api'
import api from '../services/api'
import VideoPlayer from '../components/Video/VideoPlayer'
import FilterPanel from '../components/Video/FilterPanel'
import RotationPanel from '../components/Video/RotationPanel'
import MatchSummaryTab from '../components/Video/MatchSummaryTab'
import SpeechTab from '../components/Video/SpeechTab'
import useAuthStore from '../store/authStore'
import { useTrackingData } from '../hooks/useTrackingData'
import { useAnalysisProgress } from '../hooks/useAnalysisProgress'
import clsx from 'clsx'

const DEFAULT_FILTERS = {
  players: [], actions: [], positions: [], zones: [], timeOffset: null, labels: [],
}

// ── Action type meta ─────────────────────────────────────────────────────────
const ACTION_META = {
  serve:             { label: 'Serve',        icon: Radio,    color: 'text-blue-400',   bg: 'bg-blue-900/40' },
  spike:             { label: 'Spike',        icon: ArrowUp,  color: 'text-orange-400', bg: 'bg-orange-900/40' },
  attack:            { label: 'Attack',       icon: ArrowUp,  color: 'text-orange-400', bg: 'bg-orange-900/40' },
  block:             { label: 'Block',        icon: Shield,   color: 'text-green-400',  bg: 'bg-green-900/40' },
  reception:         { label: 'Reception',    icon: Activity, color: 'text-purple-400', bg: 'bg-purple-900/40' },
  dig:               { label: 'Dig',          icon: Activity, color: 'text-cyan-400',   bg: 'bg-cyan-900/40' },
  set:               { label: 'Set',          icon: Minus,    color: 'text-yellow-400', bg: 'bg-yellow-900/40' },
  free_ball_sent:    { label: 'Free Ball',    icon: Minus,    color: 'text-slate-400',  bg: 'bg-slate-800' },
  free_ball_received:{ label: 'Free Recv.',   icon: Minus,    color: 'text-slate-400',  bg: 'bg-slate-800' },
  unknown:           { label: 'Unknown',      icon: Activity, color: 'text-slate-500',  bg: 'bg-slate-800' },
}

function ConfidenceDot({ value }) {
  if (value == null) return null
  const pct = Math.round(value * 100)
  const color = pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <span className="flex items-center gap-1 text-xs text-slate-400">
      <span className={`w-2 h-2 rounded-full ${color}`} />
      {pct}%
    </span>
  )
}

function seekVideo(timestamp) {
  const video = document.querySelector('video')
  if (video) video.currentTime = timestamp
}

function fmtTime(secs) {
  const h = Math.floor(secs / 3600)
  const m = Math.floor((secs % 3600) / 60)
  const s = Math.floor(secs % 60)
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`
}

const ALL_ACTION_TYPES = Object.keys(ACTION_META)

function ActionTab({ match, actionsData, actionFilters, setActionFilters }) {
  const actions = actionsData?.items || []

  // Group by action type for summary row — hook must be before any early returns
  const summary = useMemo(() => {
    const counts = {}
    actions.forEach(a => {
      const k = a.action_type
      if (!counts[k]) counts[k] = { total: 0, success: 0 }
      counts[k].total++
      if (a.result === 'success') counts[k].success++
    })
    return counts
  }, [actions])

  if (match.status !== 'completed') {
    return (
      <div className="card text-center py-8">
        <Activity className="w-8 h-8 text-slate-600 mx-auto mb-2" />
        <p className="text-slate-400 text-sm">
          {match.status === 'processing'
            ? 'Actions appear here when analysis completes'
            : 'Run analysis to detect player actions'}
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Filter bar */}
      <div className="card flex items-center gap-3 flex-wrap">
        <span className="text-xs text-slate-500 font-medium">Filter:</span>
        <select
          value={actionFilters.type}
          onChange={e => setActionFilters(f => ({ ...f, type: e.target.value }))}
          className="text-xs bg-court-bg border border-court-border text-slate-300 rounded px-2 py-1 focus:outline-none"
        >
          <option value="">All actions</option>
          {ALL_ACTION_TYPES.map(t => (
            <option key={t} value={t}>{ACTION_META[t]?.label ?? t}</option>
          ))}
        </select>
        <label className="flex items-center gap-1.5 text-xs text-slate-400">
          Min confidence:
          <input
            type="range" min="0" max="0.9" step="0.05"
            value={actionFilters.minConf}
            onChange={e => setActionFilters(f => ({ ...f, minConf: parseFloat(e.target.value) }))}
            className="w-20 accent-blue-500"
          />
          <span className="w-8 text-right">{Math.round(actionFilters.minConf * 100)}%</span>
        </label>
        {actionsData && (
          <span className="ml-auto text-xs text-slate-500">
            {actionsData.total} action{actionsData.total !== 1 ? 's' : ''} detected
          </span>
        )}
      </div>

      {/* Summary pills */}
      {Object.keys(summary).length > 0 && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(summary).map(([type, { total, success }]) => {
            const meta = ACTION_META[type] || ACTION_META.unknown
            const Icon = meta.icon
            return (
              <button
                key={type}
                onClick={() => setActionFilters(f => ({ ...f, type: f.type === type ? '' : type }))}
                className={clsx(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors',
                  actionFilters.type === type
                    ? `${meta.bg} ${meta.color} border-current`
                    : 'bg-court-bg border-court-border text-slate-400 hover:border-slate-500'
                )}
              >
                <Icon className="w-3 h-3" />
                {meta.label} <span className="font-bold">{total}</span>
                {success > 0 && <span className="text-green-400">({success}✓)</span>}
              </button>
            )
          })}
        </div>
      )}

      {/* Action timeline */}
      {actions.length === 0 ? (
        <div className="card text-center py-8">
          <p className="text-slate-400 text-sm">No actions match current filters</p>
        </div>
      ) : (
        <div className="space-y-1.5">
          {actions.map(action => {
            const meta = ACTION_META[action.action_type] || ACTION_META.unknown
            const Icon = meta.icon
            const isError = action.result === 'error'
            return (
              <div
                key={action.id}
                onClick={() => seekVideo(action.timestamp)}
                className="card flex items-center gap-3 hover:border-blue-600/40 transition-colors cursor-pointer py-2.5"
              >
                {/* Icon */}
                <div className={clsx('w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0', meta.bg)}>
                  <Icon className={clsx('w-4 h-4', meta.color)} />
                </div>

                {/* Timestamp + type */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className={clsx('text-xs font-semibold', meta.color)}>{meta.label}</span>
                    {isError && (
                      <span className="text-[10px] bg-red-900/40 text-red-400 px-1.5 py-0.5 rounded">error</span>
                    )}
                    {action.result === 'success' && (
                      <span className="text-[10px] bg-green-900/40 text-green-400 px-1.5 py-0.5 rounded">✓</span>
                    )}
                    {action.team && (
                      <span className={clsx(
                        'text-[10px] px-1.5 py-0.5 rounded',
                        action.team === 'A' ? 'bg-blue-900/30 text-blue-400' : 'bg-red-900/30 text-red-400'
                      )}>
                        Team {action.team}
                      </span>
                    )}
                  </div>
                  {action.player_track_id != null && (
                    <div className="text-[10px] text-slate-600 mt-0.5">
                      Player #{action.player_track_id}
                      {action.zone && ` · Zone ${action.zone}`}
                    </div>
                  )}
                </div>

                {/* Confidence */}
                <ConfidenceDot value={action.confidence} />

                {/* Timestamp button */}
                <button className="flex items-center gap-1 px-2 py-1 rounded text-xs text-slate-400 hover:text-blue-400 hover:bg-blue-900/20 transition-colors flex-shrink-0">
                  <Play className="w-3 h-3" />
                  {fmtTime(action.timestamp)}
                </button>
              </div>
            )
          })}
          {actionsData?.total > actions.length && (
            <p className="text-center text-xs text-slate-500 py-2">
              Showing {actions.length} of {actionsData.total} — raise confidence threshold to narrow results
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function StatPill({ label, value, color = 'blue' }) {
  const clr = { blue:'text-blue-400', green:'text-green-400', red:'text-red-400', yellow:'text-yellow-400', purple:'text-purple-400' }
  return (
    <div className="flex flex-col items-center gap-0.5 px-3 py-2 bg-court-bg rounded-lg border border-court-border">
      <span className={`text-xl font-bold ${clr[color]}`}>{value ?? '—'}</span>
      <span className="text-[10px] text-slate-500 text-center leading-tight">{label}</span>
    </div>
  )
}

export default function MatchDetailPage() {
  const { id }  = useParams()
  const userRole = useAuthStore((s) => s.user?.role)
  const canAnalyze = userRole === 'coach' || userRole === 'admin'
  const qc      = useQueryClient()
  const [filterOpen,      setFilterOpen]      = useState(false)
  const [filters,         setFilters]         = useState(DEFAULT_FILTERS)
  const [currentTime,     setCurrentTime]     = useState(0)
  const [activeTab,       setActiveTab]       = useState('overview')
  const [showOverlay,     setShowOverlay]     = useState(true)
  const [courtCorners,    setCourtCorners]    = useState(null)
  const [selectedRallyId, setSelectedRallyId] = useState(null)

  // ── Data fetching ──────────────────────────────────────────────────────────
  const { data: match, isLoading: matchLoading } = useQuery({
    queryKey: ['match', id],
    queryFn:  () => matchesAPI.get(id).then(r => r.data),
    refetchInterval: (query) => query?.state?.data?.status === 'processing' ? 2000 : false,
  })

  const { data: rallies = [] } = useQuery({
    queryKey: ['rallies', id],
    queryFn:  () => matchesAPI.rallies(id).then(r => r.data),
    enabled: !!match && match.status === 'completed',
  })

  const { data: analytics } = useQuery({
    queryKey: ['analytics', id],
    queryFn:  () => matchesAPI.analytics(id).then(r => r.data),
    enabled: !!match && match.status === 'completed',
  })

  const [actionFilters, setActionFilters] = useState({ type: '', minConf: 0.0 })
  const { data: actionsData } = useQuery({
    queryKey: ['actions', id, actionFilters],
    queryFn:  () => matchesAPI.actions(id, {
      action_type:    actionFilters.type    || undefined,
      min_confidence: actionFilters.minConf || undefined,
      limit: 200,
    }).then(r => r.data),
    enabled: !!match && match.status === 'completed' && activeTab === 'actions',
  })

  const { data: heatmapData } = useQuery({
    queryKey: ['ball-heatmap', id],
    queryFn:  () => matchesAPI.ballHeatmap(id).then(r => r.data),
    enabled: !!match && match.status === 'completed' && activeTab === 'summary',
  })

  const { data: rotationsData } = useQuery({
    queryKey: ['rotations', id],
    queryFn:  () => matchesAPI.rotations(id).then(r => r.data),
    enabled: !!match && match.status === 'completed' && activeTab === 'rallies',
  })

  // Map rally_id → rotation for quick lookup
  const rotationByRallyId = useMemo(() => {
    const map = {}
    ;(rotationsData?.rotations || []).forEach(rot => {
      if (rot.rally_id) map[rot.rally_id] = rot
    })
    return map
  }, [rotationsData])

  // ── Live tracking overlay ──────────────────────────────────────────────────
  const { trackingData, fetchAtTime } = useTrackingData(
    match?.status === 'completed' ? id : null
  )

  const handleTimeUpdate = (ts) => {
    setCurrentTime(ts)
    if (showOverlay && match?.status === 'completed') fetchAtTime(ts)
  }

  // ── WebSocket progress ─────────────────────────────────────────────────────
  const isProcessing = match?.status === 'processing'
  const { progress: wsProgress, message: wsMsg, connected: wsConnected, failed: analysisFailed } =
    useAnalysisProgress(id, isProcessing)

  // ── Mutations ──────────────────────────────────────────────────────────────
  const analyzeMut = useMutation({
    mutationFn: () => api.post(`/matches/${id}/analyze`, { court_corners: courtCorners }),
    onSuccess:  () => { toast.success('Analysis started!'); qc.invalidateQueries(['match', id]) },
    onError:    (e) => toast.error(e.response?.data?.detail || 'Failed'),
  })

  // ── Filter rallies ─────────────────────────────────────────────────────────
  const filteredRallies = useMemo(() => {
    if (!rallies.length) return []
    if (!filters.players.length && !filters.actions.length && !filters.zones.length) return rallies
    return rallies.filter(r =>
      r.events?.some(ev =>
        (!filters.players.length || filters.players.includes(ev.player_id)) &&
        (!filters.actions.length || filters.actions.includes(ev.action))
      )
    )
  }, [rallies, filters])

  const availableTabs = useMemo(() => (
    match?.status === 'completed'
      ? ['overview', 'rallies', 'actions', 'analytics', 'summary', 'speech']
      : ['overview', 'rallies', 'speech']
  ), [match?.status])

  useEffect(() => {
    if (!availableTabs.includes(activeTab)) {
      setActiveTab('overview')
    }
  }, [activeTab, availableTabs])

  if (matchLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
    </div>
  )
  if (!match) return <div className="text-center py-20 text-slate-400">Match not found</div>

  const displayProgress = isProcessing
    ? Math.max(match?.processing_progress || 0, wsProgress)
    : (match?.processing_progress || 0)

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm">
        <Link to="/matches" className="text-slate-500 hover:text-slate-300 flex items-center gap-1">
          <ChevronLeft className="w-4 h-4" /> Matches
        </Link>
        <span className="text-slate-600">/</span>
        <span className="text-slate-300 truncate max-w-xs">{match.title}</span>
      </div>

      {/* Match header */}
      <div className="card">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h2 className="text-xl font-bold text-white">{match.title}</h2>
            {match.team_a && match.team_b && (
              <div className="flex items-center gap-3 mt-2">
                <span className="text-lg font-semibold text-blue-400">{match.team_a}</span>
                <div className="flex items-center gap-1">
                  <span className="text-2xl font-bold text-white">{match.team_a_score}</span>
                  <span className="text-slate-600 mx-1">—</span>
                  <span className="text-2xl font-bold text-white">{match.team_b_score}</span>
                </div>
                <span className="text-lg font-semibold text-red-400">{match.team_b}</span>
              </div>
            )}
            <div className="flex items-center gap-4 mt-2 flex-wrap">
              <span className={`status-${match.status}`}>{match.status}</span>

              {/* Progress bar */}
              {isProcessing && (
                <div className="flex items-center gap-2">
                  <div className="w-32 h-1.5 bg-court-bg rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-500"
                      style={{ width: `${displayProgress}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-500">{displayProgress}%</span>
                  {wsConnected
                    ? <Wifi className="w-3 h-3 text-green-500" />
                    : <WifiOff className="w-3 h-3 text-slate-600" />
                  }
                </div>
              )}

              {/* WS message */}
              {isProcessing && wsMsg && (
                <span className="text-xs text-blue-400 truncate max-w-xs">{wsMsg}</span>
              )}

              {match.total_rallies > 0 && (
                <span className="text-xs text-slate-500">{match.total_rallies} rallies</span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            {/* Annotate button */}
            {canAnalyze && (
              <Link
                to={`/matches/${match.id}/annotate`}
                className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-[#2e3a52] hover:bg-[#3e4a62] text-slate-300 rounded-lg"
              >
                <Tag className="w-4 h-4" />
                Annotate
              </Link>
            )}

            {/* Overlay toggle */}
            {match.status === 'completed' && (
              <button
                onClick={() => setShowOverlay(v => !v)}
                className={clsx(
                  'btn-secondary flex items-center gap-2 text-sm',
                  showOverlay && 'bg-green-900/20 border-green-600/40 text-green-400'
                )}
              >
                {showOverlay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                Overlay
              </button>
            )}

            {canAnalyze && !isProcessing && (
              <button
                onClick={() => analyzeMut.mutate()}
                disabled={analyzeMut.isPending}
                className="btn-primary flex items-center gap-2 text-sm"
              >
                {analyzeMut.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                {match.status === 'completed' ? 'Re-Analyze' : 'Analyze'}
              </button>
            )}

            <button
              onClick={() => setFilterOpen(o => !o)}
              className={clsx(
                'btn-secondary flex items-center gap-2 text-sm',
                filterOpen && 'bg-blue-900/20 border-blue-600/50 text-blue-400'
              )}
            >
              <SlidersHorizontal className="w-4 h-4" />
              Filters
              {(filters.players.length + filters.actions.length) > 0 && (
                <span className="w-4 h-4 bg-blue-600 rounded-full text-[10px] flex items-center justify-center text-white">
                  {filters.players.length + filters.actions.length}
                </span>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Video + Filter Panel */}
      <div className="relative overflow-hidden rounded-xl">
        <div className={clsx('transition-all duration-300', filterOpen ? 'mr-72' : 'mr-0')}>
          {match.video_id && (
            <VideoPlayer
              videoId={match.video_id}
              trackingData={showOverlay ? trackingData : null}
              onTimeUpdate={handleTimeUpdate}
              showOverlay={showOverlay}
            />
          )}
        </div>

        <FilterPanel
          open={filterOpen}
          onClose={() => setFilterOpen(false)}
          players={[]}
          filters={filters}
          onChange={setFilters}
        />
      </div>

      {/* Processing status card */}
      {isProcessing && (
        <div className="card border-blue-800/40 bg-blue-900/10">
          <div className="flex items-center gap-3">
            <Loader2 className="w-5 h-5 animate-spin text-blue-400 flex-shrink-0" />
            <div>
              <div className="text-sm font-medium text-blue-300">
                AI Analysis Running — {displayProgress}%
              </div>
              <div className="text-xs text-slate-500 mt-0.5">
                {wsMsg || 'Processing video frames, detecting players and ball...'}
              </div>
            </div>
          </div>
          <div className="mt-3 w-full bg-court-bg rounded-full h-2 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all duration-700"
              style={{ width: `${displayProgress}%` }}
            />
          </div>
        </div>
      )}

      {analysisFailed && (
        <div className="flex items-center gap-3 p-4 bg-red-900/30 border border-red-700 rounded-lg">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <div>
            <div className="text-red-300 font-medium text-sm">Analysis Failed</div>
            <div className="text-red-400/80 text-xs mt-0.5">{wsMsg}</div>
          </div>
        </div>
      )}

      {match.status === 'failed' && !analysisFailed && (
        <div className="flex items-center gap-3 p-4 bg-red-900/30 border border-red-700 rounded-lg">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <div className="text-red-300 text-sm">Analysis failed. Check server logs and re-run analysis.</div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-court-border">
        {availableTabs.map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={clsx(
              'px-4 py-2.5 text-sm font-medium capitalize transition-colors border-b-2 -mb-px',
              activeTab === tab
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-slate-400 hover:text-slate-300'
            )}
          >
            {tab}
            {tab === 'rallies' && filteredRallies.length > 0 && (
              <span className="ml-1.5 text-[10px] bg-court-border text-slate-400 px-1.5 py-0.5 rounded-full">
                {filteredRallies.length}
              </span>
            )}
            {tab === 'actions' && actionsData?.total > 0 && (
              <span className="ml-1.5 text-[10px] bg-court-border text-slate-400 px-1.5 py-0.5 rounded-full">
                {actionsData.total}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab: Overview */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <StatPill label="Total Rallies"  value={match.total_rallies} color="blue" />
          <StatPill label="Team A Score"   value={match.team_a_score}  color="green" />
          <StatPill label="Team B Score"   value={match.team_b_score}  color="red" />
          <StatPill label="Status"         value={match.status}        color="yellow" />

          {match.summary && (
            <>
              <StatPill label="Players Detected" value={match.summary.player_detections} color="purple" />
              <StatPill label="Ball Frames"       value={match.summary.ball_detections}   color="blue" />
            </>
          )}
        </div>
      )}

      {/* Tab: Rallies */}
      {activeTab === 'rallies' && (
        <div className="space-y-2">
          {match.status !== 'completed' ? (
            <div className="card text-center py-8">
              <AlertCircle className="w-8 h-8 text-slate-600 mx-auto mb-2" />
              <p className="text-slate-400 text-sm">
                {isProcessing ? 'Analysis in progress — rallies appear here when complete' : 'Run analysis to detect rallies'}
              </p>
            </div>
          ) : filteredRallies.length === 0 ? (
            <div className="card text-center py-8">
              <p className="text-slate-400 text-sm">No rallies match current filters</p>
            </div>
          ) : (
            filteredRallies.map(r => {
              const isSelected = selectedRallyId === r.id
              const rotation   = rotationByRallyId[r.id] || null
              return (
                <div key={r.id} className="space-y-0">
                  <div
                    className={clsx(
                      'card flex items-center gap-4 transition-colors cursor-pointer',
                      isSelected
                        ? 'border-blue-600/60 bg-blue-900/10'
                        : 'hover:border-blue-600/40'
                    )}
                    onClick={() => {
                      const video = document.querySelector('video')
                      if (video) video.currentTime = r.start_time
                      setSelectedRallyId(prev => prev === r.id ? null : r.id)
                    }}
                  >
                    <div className="w-9 h-9 bg-blue-900/40 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Play className="w-4 h-4 text-blue-400" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-slate-200">Rally #{r.rally_number}</div>
                      <div className="text-xs text-slate-500">
                        {new Date(r.start_time * 1000).toISOString().substr(11, 8)}
                        {' → '}
                        {new Date(r.end_time * 1000).toISOString().substr(11, 8)}
                        {' · '}
                        {(r.end_time - r.start_time).toFixed(1)}s
                      </div>
                    </div>
                    {r.winner_team && (
                      <span className={clsx(
                        'text-xs font-semibold px-2 py-1 rounded',
                        r.winner_team === 'A' ? 'bg-blue-900/40 text-blue-400' : 'bg-red-900/40 text-red-400'
                      )}>
                        Team {r.winner_team}
                      </span>
                    )}
                    {r.point_reason && (
                      <span className="text-xs text-slate-500 hidden lg:block">{r.point_reason}</span>
                    )}
                  </div>

                  {/* Rotation panel — inline below selected rally */}
                  {isSelected && (
                    <div className="ml-4 mr-0 p-3 bg-court-panel border border-t-0 border-court-border rounded-b-lg">
                      <RotationPanel rotation={rotation} players={[]} />
                    </div>
                  )}
                </div>
              )
            })
          )}
        </div>
      )}

      {/* Tab: Actions */}
      {activeTab === 'actions' && (
        <ActionTab
          match={match}
          actionsData={actionsData}
          actionFilters={actionFilters}
          setActionFilters={setActionFilters}
        />
      )}

      {/* Tab: Summary */}
      {activeTab === 'summary' && (
        <MatchSummaryTab match={match} heatmapData={heatmapData} />
      )}

      {/* Tab: Speech */}
      {activeTab === 'speech' && (
        <SpeechTab matchId={id} matchStatus={match.status} />
      )}

      {/* Tab: Analytics */}
      {activeTab === 'analytics' && (
        <div>
          {!analytics || analytics.message ? (
            <div className="card text-center py-8">
              <BarChart2 className="w-8 h-8 text-slate-600 mx-auto mb-2" />
              <p className="text-slate-400 text-sm">
                {match.status !== 'completed'
                  ? 'Run analysis to generate player statistics'
                  : 'No analytics data available yet'}
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {(analytics.players || []).map((p, idx) => (
                <div key={p.player_id || idx} className="card">
                  <h4 className="font-medium text-white mb-3 flex items-center gap-2">
                    <Users className="w-4 h-4 text-blue-400" />
                    Player #{p.player_id?.slice(0, 8)}
                    {p.team && (
                      <span className={clsx(
                        'text-xs px-2 py-0.5 rounded',
                        p.team === 'A' ? 'bg-blue-900/40 text-blue-400' : 'bg-red-900/40 text-red-400'
                      )}>
                        Team {p.team}
                      </span>
                    )}
                  </h4>
                  <div className="grid grid-cols-4 sm:grid-cols-8 gap-2">
                    <StatPill label="Serves"   value={p.serves}   color="blue" />
                    <StatPill label="Aces"     value={p.aces}     color="green" />
                    <StatPill label="Attacks"  value={p.attacks}  color="yellow" />
                    <StatPill label="Kills"    value={p.kills}    color="purple" />
                    <StatPill label="Blocks"   value={p.blocks}   color="blue" />
                    <StatPill label="Digs"     value={p.digs}     color="green" />
                    <StatPill label="Atk Eff." value={p.attack_efficiency?.toFixed(2)} color="yellow" />
                    <StatPill label="Srv Eff." value={p.serve_efficiency?.toFixed(2)}  color="purple" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
