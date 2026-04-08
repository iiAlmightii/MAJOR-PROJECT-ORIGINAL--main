import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { Video, Users, BarChart3, Clock, TrendingUp, Loader2, Play, ArrowRight } from 'lucide-react'
import { analyticsAPI } from '../services/api'
import useAuthStore from '../store/authStore'
import { format } from 'date-fns'

function StatCard({ icon: Icon, label, value, color = 'blue', sub }) {
  const colors = {
    blue:   'bg-blue-600/20   text-blue-400   border-blue-600/30',
    green:  'bg-green-600/20  text-green-400  border-green-600/30',
    yellow: 'bg-yellow-600/20 text-yellow-400 border-yellow-600/30',
    purple: 'bg-purple-600/20 text-purple-400 border-purple-600/30',
  }
  return (
    <div className="card flex items-center gap-4">
      <div className={`w-12 h-12 rounded-xl border flex items-center justify-center flex-shrink-0 ${colors[color]}`}>
        <Icon className="w-6 h-6" />
      </div>
      <div>
        <div className="text-2xl font-bold text-white">{value ?? '—'}</div>
        <div className="text-sm text-slate-400">{label}</div>
        {sub && <div className="text-xs text-slate-500 mt-0.5">{sub}</div>}
      </div>
    </div>
  )
}

export default function DashboardPage() {
  const { user, isAdmin, isCoach } = useAuthStore()

  const endpoint = isAdmin()
    ? analyticsAPI.adminDashboard
    : isCoach()
      ? analyticsAPI.coachDashboard
      : analyticsAPI.playerDashboard

  const { data, isLoading, error } = useQuery({
    queryKey: ['dashboard', user?.role],
    queryFn:  () => endpoint().then(r => r.data),
  })

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
    </div>
  )

  const stats = data?.stats || {}

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Welcome */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">
            Welcome back, <span className="text-blue-400">{user?.full_name || user?.username}</span>
          </h2>
          <p className="text-slate-400 mt-1 text-sm">
            {format(new Date(), "EEEE, MMMM d yyyy")} · {user?.team_name || 'VolleyVision Platform'}
          </p>
        </div>
        {isCoach() && (
          <Link to="/upload" className="btn-primary flex items-center gap-2">
            <Video className="w-4 h-4" />
            Upload Match
          </Link>
        )}
      </div>

      {/* Stats grid */}
      {isAdmin() && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard icon={Users}    label="Total Users"      value={stats.total_users}        color="blue" />
          <StatCard icon={Video}    label="Total Matches"    value={stats.total_matches}      color="green" />
          <StatCard icon={BarChart3} label="Completed Analyses" value={stats.completed_analyses} color="purple" />
          <StatCard icon={Clock}    label="Videos Uploaded"  value={stats.total_videos}       color="yellow" />
        </div>
      )}

      {isCoach() && !isAdmin() && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard icon={Video}    label="My Matches"     value={stats.total_matches}      color="blue" />
          <StatCard icon={BarChart3} label="Completed"     value={stats.completed_matches}  color="green" />
          <StatCard icon={Loader2}  label="Processing"     value={stats.processing_matches} color="yellow" />
          <StatCard icon={Clock}    label="Pending"        value={stats.pending_matches}    color="purple" />
        </div>
      )}

      {!isCoach() && !isAdmin() && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard icon={Video}    label="Matches Played"   value={data?.matches_played}               color="blue" />
          <StatCard icon={TrendingUp} label="Total Attacks"  value={stats.total_attacks}                color="green" />
          <StatCard icon={BarChart3} label="Total Blocks"    value={stats.total_blocks}                 color="purple" />
          <StatCard icon={Clock}    label="Avg Attack Eff."  value={stats.avg_attack_efficiency?.toFixed(2)} color="yellow" />
        </div>
      )}

      {/* Recent matches (coach/admin) */}
      {(isCoach() || isAdmin()) && data?.recent_matches && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-white">Recent Matches</h3>
            <Link to="/matches" className="text-sm text-blue-400 hover:text-blue-300 flex items-center gap-1">
              View all <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
          <div className="space-y-2">
            {data.recent_matches.length === 0 && (
              <p className="text-slate-500 text-sm text-center py-4">No matches yet. Upload your first match!</p>
            )}
            {data.recent_matches.map(m => (
              <Link
                key={m.id}
                to={`/matches/${m.id}`}
                className="flex items-center gap-4 p-3 rounded-lg bg-court-bg border border-court-border hover:border-blue-600/50 transition-colors group"
              >
                <div className="w-9 h-9 bg-blue-900/40 rounded-lg flex items-center justify-center flex-shrink-0 group-hover:bg-blue-600/30 transition-colors">
                  <Play className="w-4 h-4 text-blue-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-slate-200 truncate">{m.title}</div>
                  <div className="text-xs text-slate-500">
                    {m.team_a} vs {m.team_b} · {format(new Date(m.created_at), 'MMM d')}
                  </div>
                </div>
                <span className={`status-${m.status}`}>{m.status}</span>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Recent activity (admin) */}
      {isAdmin() && data?.recent_activity && (
        <div className="card">
          <h3 className="font-semibold text-white mb-4">System Activity</h3>
          <div className="space-y-2">
            {data.recent_activity.map(log => (
              <div key={log.id} className="flex items-center gap-3 text-sm py-1.5 border-b border-court-border/50 last:border-0">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                <span className="text-slate-400 flex-1">{log.action}</span>
                <span className="text-slate-600 text-xs">{log.resource_type}</span>
                <span className="text-slate-600 text-xs">{format(new Date(log.timestamp), 'HH:mm')}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
