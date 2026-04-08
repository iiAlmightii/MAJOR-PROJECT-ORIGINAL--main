import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Search, Plus, Play, Trash2, BarChart2, Loader2,
  Video, Calendar, Users, ChevronLeft, ChevronRight
} from 'lucide-react'
import { format } from 'date-fns'
import toast from 'react-hot-toast'
import { matchesAPI } from '../services/api'
import useAuthStore from '../store/authStore'
import clsx from 'clsx'

const STATUS_OPTS = ['', 'pending', 'processing', 'completed', 'failed']

export default function MatchesPage() {
  const { isCoach, isAdmin } = useAuthStore()
  const qc = useQueryClient()
  const [page, setPage]     = useState(1)
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState('')
  const [deleting, setDeleting] = useState(null)

  const { data, isLoading } = useQuery({
    queryKey: ['matches', page, search, status],
    queryFn:  () => matchesAPI.list({ page, per_page: 10, search: search || undefined, status_filter: status || undefined }).then(r => r.data),
    placeholderData: prev => prev,
  })

  const deleteMut = useMutation({
    mutationFn: (id) => matchesAPI.delete(id),
    onSuccess:  () => { toast.success('Match deleted'); qc.invalidateQueries(['matches']) },
    onError:    () => toast.error('Delete failed'),
  })

  const analyzeMut = useMutation({
    mutationFn: (id) => matchesAPI.analyze(id),
    onSuccess:  () => { toast.success('Analysis started!'); qc.invalidateQueries(['matches']) },
    onError:    (err) => toast.error(err.response?.data?.detail || 'Failed to start analysis'),
  })

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Matches</h2>
          <p className="text-slate-400 text-sm mt-0.5">
            {data?.total || 0} matches total
          </p>
        </div>
        {(isCoach() || isAdmin()) && (
          <Link to="/upload" className="btn-primary flex items-center gap-2">
            <Plus className="w-4 h-4" /> New Match
          </Link>
        )}
      </div>

      {/* Filters */}
      <div className="flex gap-3 flex-wrap">
        <div className="flex items-center gap-2 bg-court-panel border border-court-border rounded-lg px-3 py-2 flex-1 min-w-52">
          <Search className="w-4 h-4 text-slate-500" />
          <input
            className="bg-transparent text-sm text-slate-300 placeholder-slate-500 outline-none flex-1"
            placeholder="Search matches..."
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(1) }}
          />
        </div>
        <select
          className="input w-40"
          value={status}
          onChange={e => { setStatus(e.target.value); setPage(1) }}
        >
          {STATUS_OPTS.map(s => (
            <option key={s} value={s}>{s || 'All Status'}</option>
          ))}
        </select>
      </div>

      {/* Table */}
      {isLoading ? (
        <div className="flex justify-center py-16">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
        </div>
      ) : data?.matches?.length === 0 ? (
        <div className="card text-center py-16">
          <Video className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <p className="text-slate-400">No matches found</p>
          {(isCoach() || isAdmin()) && (
            <Link to="/upload" className="btn-primary inline-flex items-center gap-2 mt-4">
              <Plus className="w-4 h-4" /> Upload First Match
            </Link>
          )}
        </div>
      ) : (
        <div className="card p-0 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-court-border text-left">
                <th className="px-4 py-3 text-slate-400 font-medium">Match</th>
                <th className="px-4 py-3 text-slate-400 font-medium hidden md:table-cell">Teams</th>
                <th className="px-4 py-3 text-slate-400 font-medium hidden lg:table-cell">Date</th>
                <th className="px-4 py-3 text-slate-400 font-medium">Status</th>
                <th className="px-4 py-3 text-slate-400 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {data.matches.map((m, idx) => (
                <tr
                  key={m.id}
                  className={clsx(
                    'border-b border-court-border/50 hover:bg-court-border/20 transition-colors',
                    idx === data.matches.length - 1 && 'border-0'
                  )}
                >
                  <td className="px-4 py-3">
                    <div className="font-medium text-slate-200">{m.title}</div>
                    {m.venue && <div className="text-xs text-slate-500">{m.venue}</div>}
                  </td>
                  <td className="px-4 py-3 hidden md:table-cell">
                    {m.team_a && m.team_b ? (
                      <span className="text-slate-300">{m.team_a} vs {m.team_b}</span>
                    ) : (
                      <span className="text-slate-600">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 hidden lg:table-cell text-slate-400">
                    {format(new Date(m.created_at), 'MMM d, yyyy')}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`status-${m.status}`}>{m.status}</span>
                    {m.status === 'processing' && (
                      <span className="ml-2 text-xs text-slate-500">{m.processing_progress}%</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-1">
                      <Link
                        to={`/matches/${m.id}`}
                        className="p-1.5 rounded-lg text-slate-400 hover:text-blue-400 hover:bg-blue-900/20 transition-colors"
                        title="View"
                      >
                        <Play className="w-4 h-4" />
                      </Link>
                      {(isCoach() || isAdmin()) && m.status !== 'processing' && (
                        <button
                          onClick={() => analyzeMut.mutate(m.id)}
                          disabled={analyzeMut.isPending}
                          className="p-1.5 rounded-lg text-slate-400 hover:text-green-400 hover:bg-green-900/20 transition-colors"
                          title="Analyze"
                        >
                          <BarChart2 className="w-4 h-4" />
                        </button>
                      )}
                      {(isCoach() || isAdmin()) && (
                        <button
                          onClick={() => {
                            if (confirm('Delete this match?')) deleteMut.mutate(m.id)
                          }}
                          className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-900/20 transition-colors"
                          title="Delete"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {data && data.total > 10 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-500">
            Page {page} of {Math.ceil(data.total / 10)}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage(p => p - 1)}
              disabled={page <= 1}
              className="btn-secondary p-2 disabled:opacity-40"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setPage(p => p + 1)}
              disabled={page >= Math.ceil(data.total / 10)}
              className="btn-secondary p-2 disabled:opacity-40"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
