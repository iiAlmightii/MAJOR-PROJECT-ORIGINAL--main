import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts'
import {
  Loader2, TrendingUp, BarChart3, Activity, Users, Video, CheckCircle,
  Trophy, Shield, ArrowUp, Radio, Zap, ChevronUp, ChevronDown, Minus,
  Clock, Target,
} from 'lucide-react'
import { analyticsAPI } from '../services/api'
import useAuthStore from '../store/authStore'

const CHART_STYLE = {
  contentStyle: { background: '#232b3e', border: '1px solid #2e3a52', borderRadius: 8, color: '#f1f5f9' },
  labelStyle: { color: '#94a3b8' },
}

const COLORS = {
  attacks: '#3b82f6',
  kills:   '#f97316',
  serves:  '#22c55e',
  aces:    '#a855f7',
  blocks:  '#06b6d4',
  digs:    '#eab308',
  teamA:   '#3b82f6',
  teamB:   '#f97316',
}

// ─── Shared components ───────────────────────────────────────────────────────

function StatCard({ label, value, color = 'text-blue-400', icon: Icon, sub }) {
  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <div>
          <div className={`text-3xl font-bold ${color}`}>{value}</div>
          <div className="text-sm text-slate-400 mt-1">{label}</div>
          {sub && <div className="text-xs text-slate-500 mt-0.5">{sub}</div>}
        </div>
        {Icon && <Icon className={`w-5 h-5 ${color} opacity-60`} />}
      </div>
    </div>
  )
}

function SectionHeader({ icon: Icon, title, color = 'text-blue-400' }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <Icon className={`w-4 h-4 ${color}`} />
      <h3 className="font-semibold text-white text-sm">{title}</h3>
    </div>
  )
}

// ─── Admin Dashboard ─────────────────────────────────────────────────────────

function AdminDashboard({ data }) {
  const stats = data?.stats || {}
  const logs  = data?.recent_activity || []
  const recentMatches = data?.recent_matches || []

  const statusColor = { completed: 'text-green-400', processing: 'text-yellow-400', pending: 'text-slate-400', failed: 'text-red-400' }
  const actionColor  = (a) => a.includes('create') ? 'bg-green-500/20 text-green-400'
    : a.includes('delete') ? 'bg-red-500/20 text-red-400'
    : a.includes('login')  ? 'bg-blue-500/20 text-blue-400'
    : 'bg-slate-500/20 text-slate-400'

  const matchStatusData = [
    { name: 'Completed',  value: stats.completed_analyses || 0, fill: '#22c55e' },
    { name: 'Processing', value: stats.processing_matches || 0, fill: '#eab308' },
    { name: 'Pending',    value: Math.max(0, (stats.total_matches || 0) - (stats.completed_analyses || 0) - (stats.processing_matches || 0)), fill: '#64748b' },
  ]

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Users"        value={stats.total_users || 0}          color="text-blue-400"   icon={Users}       />
        <StatCard label="Total Matches"      value={stats.total_matches || 0}        color="text-green-400"  icon={Video}       />
        <StatCard label="Completed Analyses" value={stats.completed_analyses || 0}   color="text-purple-400" icon={CheckCircle} />
        <StatCard label="Videos Uploaded"    value={stats.total_videos || 0}         color="text-yellow-400" icon={Video}       />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent matches */}
        <div className="card">
          <SectionHeader icon={Activity} title="Recent Matches" />
          <div className="space-y-2">
            {recentMatches.length === 0 && <p className="text-slate-500 text-sm">No matches yet.</p>}
            {recentMatches.map(m => (
              <div key={m.id} className="flex items-center justify-between py-2 border-b border-[#2e3a52] last:border-0">
                <div className="min-w-0">
                  <div className="text-sm text-white truncate">{m.title}</div>
                  <div className="text-xs text-slate-500">{m.team_a} vs {m.team_b}</div>
                </div>
                <div className="flex items-center gap-3 flex-shrink-0">
                  {m.status === 'completed' && (
                    <span className="text-xs font-mono text-white bg-[#2e3a52] px-2 py-0.5 rounded">
                      {m.team_a_score} – {m.team_b_score}
                    </span>
                  )}
                  <span className={`text-xs font-medium ${statusColor[m.status] || 'text-slate-400'}`}>
                    {m.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Match status breakdown */}
        <div className="card">
          <SectionHeader icon={BarChart3} title="Match Status Breakdown" />
          <div className="space-y-3 mt-2">
            {matchStatusData.map(item => (
              <div key={item.name}>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>{item.name}</span>
                  <span>{item.value}</span>
                </div>
                <div className="h-2 bg-[#2e3a52] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: stats.total_matches
                        ? `${Math.round((item.value / stats.total_matches) * 100)}%`
                        : '0%',
                      background: item.fill,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Activity log */}
      <div className="card">
        <SectionHeader icon={Clock} title="Recent Activity Log" />
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-[#2e3a52]">
                <th className="text-left py-2 pr-4">User</th>
                <th className="text-left py-2 pr-4">Action</th>
                <th className="text-left py-2 pr-4">Resource</th>
                <th className="text-left py-2">Time</th>
              </tr>
            </thead>
            <tbody>
              {logs.map(log => (
                <tr key={log.id} className="border-b border-[#2e3a52]/50 last:border-0">
                  <td className="py-2 pr-4 text-slate-300">{log.username}</td>
                  <td className="py-2 pr-4">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${actionColor(log.action)}`}>
                      {log.action}
                    </span>
                  </td>
                  <td className="py-2 pr-4 text-slate-400">{log.resource_type || '—'}</td>
                  <td className="py-2 text-slate-500">
                    {new Date(log.timestamp).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                  </td>
                </tr>
              ))}
              {logs.length === 0 && (
                <tr><td colSpan={4} className="py-4 text-center text-slate-500">No activity yet.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ─── Coach Dashboard ──────────────────────────────────────────────────────────

function CoachDashboard({ data, leaderboard, leaderMetric, setLeaderMetric }) {
  const stats   = data?.stats || {}
  const matchPerf = data?.match_performance || []
  const recent  = data?.recent_matches || []

  const statusBadge = (s) => ({
    completed:  'bg-green-500/20 text-green-400',
    processing: 'bg-yellow-500/20 text-yellow-400',
    pending:    'bg-slate-500/20 text-slate-400',
    failed:     'bg-red-500/20 text-red-400',
  }[s] || 'bg-slate-500/20 text-slate-400')

  const METRIC_OPTIONS = [
    { value: 'attacks',    label: 'Attacks'   },
    { value: 'kills',      label: 'Kills'     },
    { value: 'blocks',     label: 'Blocks'    },
    { value: 'aces',       label: 'Aces'      },
    { value: 'digs',       label: 'Digs'      },
    { value: 'attack_eff', label: 'Atk Eff %' },
  ]

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Matches"    value={stats.total_matches || 0}     color="text-blue-400"   icon={Video}       />
        <StatCard label="Completed"        value={stats.completed_matches || 0} color="text-green-400"  icon={CheckCircle} />
        <StatCard label="Processing"       value={stats.processing_matches || 0} color="text-yellow-400" icon={Loader2}    />
        <StatCard label="Pending"          value={stats.pending_matches || 0}   color="text-slate-400"  icon={Clock}       />
      </div>

      {/* Match performance bar chart (real data) */}
      {matchPerf.length > 0 && (
        <div className="card">
          <SectionHeader icon={BarChart3} title="Match Performance (Attacks / Kills / Blocks)" />
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={matchPerf} barSize={12}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2e3a52" />
              <XAxis dataKey="label" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={CHART_STYLE.contentStyle} labelStyle={CHART_STYLE.labelStyle} />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              <Bar dataKey="attacks" fill={COLORS.attacks} radius={[3,3,0,0]} name="Attacks" />
              <Bar dataKey="kills"   fill={COLORS.kills}   radius={[3,3,0,0]} name="Kills"   />
              <Bar dataKey="blocks"  fill={COLORS.blocks}  radius={[3,3,0,0]} name="Blocks"  />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Attack efficiency trend */}
      {matchPerf.length > 0 && (
        <div className="card">
          <SectionHeader icon={TrendingUp} title="Attack Efficiency Trend (%)" color="text-green-400" />
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={matchPerf}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2e3a52" />
              <XAxis dataKey="label" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip contentStyle={CHART_STYLE.contentStyle} labelStyle={CHART_STYLE.labelStyle} formatter={(v) => [`${v}%`]} />
              <Line type="monotone" dataKey="attack_eff" stroke="#22c55e" strokeWidth={2} dot={{ fill: '#22c55e', r: 3 }} name="Atk Eff %" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Player leaderboard */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <SectionHeader icon={Trophy} title="Player Leaderboard" color="text-yellow-400" />
            <select
              value={leaderMetric}
              onChange={e => setLeaderMetric(e.target.value)}
              className="text-xs bg-[#2e3a52] text-slate-300 border border-[#3e4a62] rounded px-2 py-1 outline-none"
            >
              {METRIC_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
          {(!leaderboard || leaderboard.length === 0) ? (
            <p className="text-slate-500 text-sm">No player data yet. Run analysis on a match first.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-500 border-b border-[#2e3a52]">
                    <th className="text-left py-2 pr-3">#</th>
                    <th className="text-left py-2 pr-3">Player</th>
                    <th className="text-left py-2 pr-3">Team</th>
                    <th className="text-right py-2 pr-3">Atk</th>
                    <th className="text-right py-2 pr-3">Kills</th>
                    <th className="text-right py-2 pr-3">Blk</th>
                    <th className="text-right py-2">Eff%</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.map((p, i) => (
                    <tr key={i} className="border-b border-[#2e3a52]/50 last:border-0">
                      <td className="py-1.5 pr-3 text-slate-500">{i + 1}</td>
                      <td className="py-1.5 pr-3 text-white font-medium">{p.name}</td>
                      <td className="py-1.5 pr-3">
                        <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${p.team === 'A' ? 'bg-blue-500/20 text-blue-400' : 'bg-orange-500/20 text-orange-400'}`}>
                          {p.team}
                        </span>
                      </td>
                      <td className="py-1.5 pr-3 text-right text-slate-300">{p.attacks}</td>
                      <td className="py-1.5 pr-3 text-right text-orange-400">{p.kills}</td>
                      <td className="py-1.5 pr-3 text-right text-cyan-400">{p.blocks}</td>
                      <td className="py-1.5 text-right text-green-400">{p.attack_eff}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Recent matches */}
        <div className="card">
          <SectionHeader icon={Activity} title="Recent Matches" />
          <div className="space-y-2">
            {recent.length === 0 && <p className="text-slate-500 text-sm">No matches yet.</p>}
            {recent.map(m => (
              <div key={m.id} className="flex items-center justify-between py-2 border-b border-[#2e3a52] last:border-0">
                <div className="min-w-0">
                  <div className="text-sm text-white truncate">{m.title}</div>
                  <div className="text-xs text-slate-500">{m.team_a} vs {m.team_b}</div>
                </div>
                <div className="flex items-center gap-3 flex-shrink-0">
                  {m.status === 'completed' && (
                    <span className="text-xs font-mono text-white bg-[#2e3a52] px-2 py-0.5 rounded">
                      {m.team_a_score} – {m.team_b_score}
                    </span>
                  )}
                  <span className={`text-xs font-medium px-2 py-0.5 rounded ${statusBadge(m.status)}`}>
                    {m.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Player Dashboard ─────────────────────────────────────────────────────────

function PlayerDashboard({ data }) {
  const stats   = data?.stats || {}
  const player  = data?.player || {}
  const history = data?.match_history || []

  const radarData = [
    { stat: 'Serving',    value: Math.min(Math.abs((stats.avg_serve_efficiency || 0)) * 100, 100) },
    { stat: 'Attacking',  value: Math.min(Math.abs((stats.avg_attack_efficiency || 0)) * 100, 100) },
    { stat: 'Blocking',   value: Math.min(((stats.total_blocks || 0) / Math.max(history.length * 5, 1)) * 100, 100) },
    { stat: 'Digging',    value: Math.min(((stats.total_digs || 0) / Math.max(history.length * 10, 1)) * 100, 100) },
    { stat: 'Reception',  value: Math.min(Math.abs((stats.avg_reception_efficiency || 0)) * 100, 100) },
  ]

  if (data?.message && !stats.total_matches) {
    return (
      <div className="card text-center py-12">
        <Activity className="w-10 h-10 text-slate-600 mx-auto mb-3" />
        <p className="text-slate-400">{data.message}</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Profile header */}
      <div className="card flex items-center gap-4">
        <div className="w-12 h-12 rounded-full bg-blue-500/20 flex items-center justify-center">
          <Users className="w-6 h-6 text-blue-400" />
        </div>
        <div>
          <div className="text-white font-semibold">{player.username}</div>
          <div className="text-xs text-slate-400">{player.position || 'No position set'} · {player.team || 'No team'}</div>
        </div>
        <div className="ml-auto text-right">
          <div className="text-2xl font-bold text-blue-400">{stats.total_matches || 0}</div>
          <div className="text-xs text-slate-400">Matches Played</div>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Attacks" value={stats.total_attacks || 0} color="text-blue-400"   icon={ArrowUp}  sub={`${stats.total_kills || 0} kills`}  />
        <StatCard label="Total Serves"  value={stats.total_serves || 0}  color="text-green-400"  icon={Radio}    sub={`${stats.total_aces || 0} aces`}    />
        <StatCard label="Total Blocks"  value={stats.total_blocks || 0}  color="text-cyan-400"   icon={Shield}   sub={`${stats.block_points || 0} pts`}   />
        <StatCard label="Total Digs"    value={stats.total_digs || 0}    color="text-yellow-400" icon={Target}                                             />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Skill radar */}
        <div className="card">
          <SectionHeader icon={Activity} title="Skill Radar" />
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#2e3a52" />
              <PolarAngleAxis dataKey="stat" tick={{ fill: '#64748b', fontSize: 11 }} />
              <PolarRadiusAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 9 }} />
              <Radar name="You" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
              <Tooltip contentStyle={CHART_STYLE.contentStyle} formatter={(v) => [`${v.toFixed(0)}%`]} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Efficiency summary */}
        <div className="card">
          <SectionHeader icon={TrendingUp} title="Efficiency Summary" color="text-green-400" />
          <div className="space-y-4 mt-2">
            {[
              { label: 'Attack Efficiency', value: stats.avg_attack_efficiency || 0, color: '#3b82f6' },
              { label: 'Serve Efficiency',  value: stats.avg_serve_efficiency  || 0, color: '#22c55e' },
              { label: 'Reception Eff.',    value: stats.avg_reception_efficiency || 0, color: '#a855f7' },
            ].map(item => {
              const pct = Math.min(Math.abs(item.value) * 100, 100)
              return (
                <div key={item.label}>
                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>{item.label}</span>
                    <span style={{ color: item.color }}>{(item.value * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-[#2e3a52] rounded-full overflow-hidden">
                    <div className="h-full rounded-full" style={{ width: `${pct}%`, background: item.color }} />
                  </div>
                </div>
              )
            })}
          </div>

          <div className="mt-6 grid grid-cols-3 gap-3 text-center">
            {[
              { label: 'Kills',      value: stats.total_kills || 0,      color: 'text-orange-400' },
              { label: 'Aces',       value: stats.total_aces || 0,       color: 'text-purple-400' },
              { label: 'Receptions', value: stats.total_receptions || 0, color: 'text-cyan-400'   },
            ].map(item => (
              <div key={item.label} className="bg-[#1a1f2e] rounded-lg p-3">
                <div className={`text-xl font-bold ${item.color}`}>{item.value}</div>
                <div className="text-xs text-slate-500 mt-0.5">{item.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Match progression */}
      {history.length > 0 && (
        <div className="card">
          <SectionHeader icon={TrendingUp} title="Match-by-Match Progression" />
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2e3a52" />
              <XAxis dataKey="label" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={CHART_STYLE.contentStyle} labelStyle={CHART_STYLE.labelStyle} />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              <Line type="monotone" dataKey="attacks" stroke={COLORS.attacks} strokeWidth={2} dot={{ r: 3 }} name="Attacks" />
              <Line type="monotone" dataKey="kills"   stroke={COLORS.kills}   strokeWidth={2} dot={{ r: 3 }} name="Kills"   />
              <Line type="monotone" dataKey="blocks"  stroke={COLORS.blocks}  strokeWidth={2} dot={{ r: 3 }} name="Blocks"  />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const { isAdmin, isCoach, user } = useAuthStore()
  const [leaderMetric, setLeaderMetric] = useState('attacks')

  const dashEndpoint = isAdmin()
    ? analyticsAPI.adminDashboard
    : isCoach()
      ? analyticsAPI.coachDashboard
      : analyticsAPI.playerDashboard

  const { data, isLoading } = useQuery({
    queryKey: ['analytics-dashboard', user?.role],
    queryFn:  () => dashEndpoint().then(r => r.data),
    staleTime: 30_000,
  })

  const { data: leaderboard } = useQuery({
    queryKey: ['leaderboard', leaderMetric],
    queryFn:  () => analyticsAPI.leaderboard(leaderMetric).then(r => r.data),
    enabled:  isAdmin() || isCoach(),
    staleTime: 30_000,
  })

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
    </div>
  )

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-white">Analytics</h2>
        <p className="text-slate-400 text-sm mt-1">Performance insights across all matches</p>
      </div>

      {isAdmin() && <AdminDashboard data={data} />}
      {isCoach() && !isAdmin() && (
        <CoachDashboard
          data={data}
          leaderboard={leaderboard}
          leaderMetric={leaderMetric}
          setLeaderMetric={setLeaderMetric}
        />
      )}
      {!isAdmin() && !isCoach() && <PlayerDashboard data={data} />}
    </div>
  )
}
