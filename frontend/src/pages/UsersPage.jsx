import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Search, UserPlus, Trash2, Edit, Shield, Users, Loader2 } from 'lucide-react'
import { format } from 'date-fns'
import toast from 'react-hot-toast'
import { usersAPI } from '../services/api'
import clsx from 'clsx'

const ROLE_COLORS = {
  admin:  'bg-purple-900/40 text-purple-400 border-purple-800/40',
  coach:  'bg-blue-900/40   text-blue-400   border-blue-800/40',
  player: 'bg-green-900/40  text-green-400  border-green-800/40',
}

export default function UsersPage() {
  const qc = useQueryClient()
  const [search, setSearch] = useState('')
  const [role, setRole]     = useState('')

  const { data: stats } = useQuery({
    queryKey: ['user-stats'],
    queryFn: () => usersAPI.stats().then(r => r.data),
  })

  const { data: users = [], isLoading } = useQuery({
    queryKey: ['users', search, role],
    queryFn: () => usersAPI.list({ params: { search: search || undefined, role: role || undefined, limit: 50 } }).then(r => r.data),
  })

  const deleteMut = useMutation({
    mutationFn: (id) => usersAPI.delete(id),
    onSuccess:  () => { toast.success('User deleted'); qc.invalidateQueries(['users']) },
    onError:    () => toast.error('Delete failed'),
  })

  const toggleMut = useMutation({
    mutationFn: ({ id, is_active }) => usersAPI.update(id, { is_active }),
    onSuccess:  () => { toast.success('User updated'); qc.invalidateQueries(['users']) },
    onError:    () => toast.error('Update failed'),
  })

  return (
    <div className="space-y-5 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">User Management</h2>
          <p className="text-slate-400 text-sm mt-0.5">{stats?.total_users || 0} total users</p>
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-3 gap-3">
          {Object.entries(stats.by_role || {}).map(([r, count]) => (
            <div key={r} className="card flex items-center gap-3">
              <Shield className={clsx('w-5 h-5', r === 'admin' ? 'text-purple-400' : r === 'coach' ? 'text-blue-400' : 'text-green-400')} />
              <div>
                <div className="text-lg font-bold text-white">{count}</div>
                <div className="text-xs text-slate-400 capitalize">{r}s</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-3">
        <div className="flex items-center gap-2 bg-court-panel border border-court-border rounded-lg px-3 py-2 flex-1">
          <Search className="w-4 h-4 text-slate-500" />
          <input
            className="bg-transparent text-sm text-slate-300 placeholder-slate-500 outline-none flex-1"
            placeholder="Search by name or email..."
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
        <select className="input w-36" value={role} onChange={e => setRole(e.target.value)}>
          <option value="">All Roles</option>
          <option value="admin">Admin</option>
          <option value="coach">Coach</option>
          <option value="player">Player</option>
        </select>
      </div>

      {/* Table */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="w-7 h-7 animate-spin text-blue-500" />
        </div>
      ) : (
        <div className="card p-0 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-court-border">
                <th className="px-4 py-3 text-left text-slate-400 font-medium">User</th>
                <th className="px-4 py-3 text-left text-slate-400 font-medium">Role</th>
                <th className="px-4 py-3 text-left text-slate-400 font-medium hidden md:table-cell">Team</th>
                <th className="px-4 py-3 text-left text-slate-400 font-medium hidden lg:table-cell">Joined</th>
                <th className="px-4 py-3 text-left text-slate-400 font-medium">Status</th>
                <th className="px-4 py-3 text-right text-slate-400 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.length === 0 && (
                <tr><td colSpan={6} className="px-4 py-8 text-center text-slate-500">No users found</td></tr>
              )}
              {users.map((u, idx) => (
                <tr key={u.id} className={clsx('border-b border-court-border/50 hover:bg-court-border/20 transition-colors', idx === users.length - 1 && 'border-0')}>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-blue-700 flex items-center justify-center text-sm font-bold text-white uppercase flex-shrink-0">
                        {u.username[0]}
                      </div>
                      <div>
                        <div className="font-medium text-slate-200">{u.full_name || u.username}</div>
                        <div className="text-xs text-slate-500">{u.email}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className={clsx('badge border', ROLE_COLORS[u.role])}>{u.role}</span>
                  </td>
                  <td className="px-4 py-3 hidden md:table-cell text-slate-400 text-xs">
                    {u.team_name || '—'}
                  </td>
                  <td className="px-4 py-3 hidden lg:table-cell text-slate-400 text-xs">
                    {format(new Date(u.created_at), 'MMM d, yyyy')}
                  </td>
                  <td className="px-4 py-3">
                    <span className={clsx('badge border', u.is_active ? 'bg-green-900/30 text-green-400 border-green-800/30' : 'bg-red-900/30 text-red-400 border-red-800/30')}>
                      {u.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-1">
                      <button
                        onClick={() => toggleMut.mutate({ id: u.id, is_active: !u.is_active })}
                        className={clsx(
                          'p-1.5 rounded-lg text-xs transition-colors',
                          u.is_active
                            ? 'text-slate-400 hover:text-yellow-400 hover:bg-yellow-900/20'
                            : 'text-slate-400 hover:text-green-400 hover:bg-green-900/20'
                        )}
                        title={u.is_active ? 'Deactivate' : 'Activate'}
                      >
                        <Shield className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => { if (confirm(`Delete ${u.username}?`)) deleteMut.mutate(u.id) }}
                        className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-900/20 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
