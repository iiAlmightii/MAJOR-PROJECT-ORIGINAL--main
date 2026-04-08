import React, { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { User, Lock, Save, Loader2 } from 'lucide-react'
import toast from 'react-hot-toast'
import { usersAPI, authAPI } from '../services/api'
import useAuthStore from '../store/authStore'

const POSITIONS = ['Setter', 'Libero', 'Outside Hitter', 'Opposite Hitter', 'Middle Blocker', 'Defensive Specialist']

export default function ProfilePage() {
  const { user, updateUser } = useAuthStore()
  const [profile, setProfile] = useState({
    full_name: user?.full_name || '',
    team_name: user?.team_name || '',
    position: user?.position || '',
    jersey_number: user?.jersey_number || '',
  })
  const [pw, setPw] = useState({ current_password: '', new_password: '', confirm: '' })

  const updateMut = useMutation({
    mutationFn: (data) => usersAPI.updateMe(data),
    onSuccess: (res) => {
      updateUser(res.data)
      toast.success('Profile updated!')
    },
    onError: () => toast.error('Update failed'),
  })

  const pwMut = useMutation({
    mutationFn: (data) => authAPI.changePassword(data),
    onSuccess: () => {
      toast.success('Password changed!')
      setPw({ current_password: '', new_password: '', confirm: '' })
    },
    onError: (err) => toast.error(err.response?.data?.detail || 'Password change failed'),
  })

  const handlePwSubmit = (e) => {
    e.preventDefault()
    if (pw.new_password !== pw.confirm) { toast.error('Passwords do not match'); return }
    if (pw.new_password.length < 8) { toast.error('Password must be at least 8 characters'); return }
    pwMut.mutate({ current_password: pw.current_password, new_password: pw.new_password })
  }

  const ROLE_BADGE = {
    admin:  'bg-purple-900/40 text-purple-400',
    coach:  'bg-blue-900/40   text-blue-400',
    player: 'bg-green-900/40  text-green-400',
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
      <h2 className="text-2xl font-bold text-white">My Profile</h2>

      {/* Avatar + identity */}
      <div className="card flex items-center gap-5">
        <div className="w-16 h-16 rounded-2xl bg-blue-700 flex items-center justify-center text-2xl font-bold text-white uppercase flex-shrink-0">
          {user?.username?.[0]}
        </div>
        <div>
          <div className="text-xl font-bold text-white">{user?.full_name || user?.username}</div>
          <div className="text-slate-400 text-sm">{user?.email}</div>
          <span className={`badge mt-1 ${ROLE_BADGE[user?.role]}`}>{user?.role}</span>
        </div>
      </div>

      {/* Profile form */}
      <form
        onSubmit={(e) => { e.preventDefault(); updateMut.mutate(profile) }}
        className="card space-y-4"
      >
        <div className="flex items-center gap-2 mb-1">
          <User className="w-4 h-4 text-blue-400" />
          <h3 className="font-semibold text-white">Personal Info</h3>
        </div>

        <div>
          <label className="label">Full Name</label>
          <input className="input" value={profile.full_name}
            onChange={e => setProfile(p => ({ ...p, full_name: e.target.value }))} />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Team Name</label>
            <input className="input" value={profile.team_name}
              onChange={e => setProfile(p => ({ ...p, team_name: e.target.value }))} />
          </div>
          <div>
            <label className="label">Jersey #</label>
            <input className="input" value={profile.jersey_number}
              onChange={e => setProfile(p => ({ ...p, jersey_number: e.target.value }))} />
          </div>
        </div>

        <div>
          <label className="label">Position</label>
          <select className="input" value={profile.position}
            onChange={e => setProfile(p => ({ ...p, position: e.target.value }))}>
            <option value="">Select position</option>
            {POSITIONS.map(pos => <option key={pos} value={pos}>{pos}</option>)}
          </select>
        </div>

        <button type="submit" className="btn-primary flex items-center gap-2" disabled={updateMut.isPending}>
          {updateMut.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
          Save Changes
        </button>
      </form>

      {/* Password change */}
      <form onSubmit={handlePwSubmit} className="card space-y-4">
        <div className="flex items-center gap-2 mb-1">
          <Lock className="w-4 h-4 text-blue-400" />
          <h3 className="font-semibold text-white">Change Password</h3>
        </div>

        <div>
          <label className="label">Current Password</label>
          <input type="password" className="input" value={pw.current_password}
            onChange={e => setPw(p => ({ ...p, current_password: e.target.value }))} required />
        </div>
        <div>
          <label className="label">New Password</label>
          <input type="password" className="input" value={pw.new_password} minLength={8}
            onChange={e => setPw(p => ({ ...p, new_password: e.target.value }))} required />
        </div>
        <div>
          <label className="label">Confirm New Password</label>
          <input type="password" className="input" value={pw.confirm}
            onChange={e => setPw(p => ({ ...p, confirm: e.target.value }))} required />
        </div>

        <button type="submit" className="btn-primary flex items-center gap-2" disabled={pwMut.isPending}>
          {pwMut.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Lock className="w-4 h-4" />}
          Change Password
        </button>
      </form>
    </div>
  )
}
