import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Activity, Eye, EyeOff, Loader2 } from 'lucide-react'
import toast from 'react-hot-toast'
import { authAPI } from '../services/api'
import useAuthStore from '../store/authStore'

const POSITIONS = ['Setter', 'Libero', 'Outside Hitter', 'Opposite Hitter', 'Middle Blocker', 'Defensive Specialist']

export default function RegisterPage() {
  const navigate = useNavigate()
  const { setAuth } = useAuthStore()
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [form, setForm] = useState({
    email: '', username: '', full_name: '', password: '',
    role: 'player', team_name: '', position: '', jersey_number: '',
  })

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (form.password.length < 8) {
      toast.error('Password must be at least 8 characters')
      return
    }
    setLoading(true)
    try {
      const { data } = await authAPI.register(form)
      setAuth(data.user, data.access_token, data.refresh_token)
      toast.success('Account created! Welcome to VolleyVision.')
      navigate('/dashboard')
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-court-bg flex items-center justify-center p-6">
      <div className="w-full max-w-lg">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-bold text-white">VolleyVision</span>
        </div>

        <h2 className="text-2xl font-bold text-white mb-2">Create account</h2>
        <p className="text-slate-400 mb-8">
          Already have an account?{' '}
          <Link to="/login" className="text-blue-400 hover:text-blue-300 font-medium">Sign in</Link>
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Email *</label>
              <input type="email" className="input" placeholder="you@team.com"
                value={form.email} onChange={e => set('email', e.target.value)} required />
            </div>
            <div>
              <label className="label">Username *</label>
              <input type="text" className="input" placeholder="player123"
                value={form.username} onChange={e => set('username', e.target.value)} required minLength={3} />
            </div>
          </div>

          <div>
            <label className="label">Full Name</label>
            <input type="text" className="input" placeholder="John Doe"
              value={form.full_name} onChange={e => set('full_name', e.target.value)} />
          </div>

          <div>
            <label className="label">Password *</label>
            <div className="relative">
              <input type={showPw ? 'text' : 'password'} className="input pr-10"
                placeholder="Min. 8 characters"
                value={form.password} onChange={e => set('password', e.target.value)} required minLength={8} />
              <button type="button" onClick={() => setShowPw(v => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300">
                {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div>
            <label className="label">Role *</label>
            <select className="input" value={form.role} onChange={e => set('role', e.target.value)}>
              <option value="player">Player</option>
              <option value="coach">Coach</option>
            </select>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="col-span-2">
              <label className="label">Team Name</label>
              <input type="text" className="input" placeholder="Team Alpha"
                value={form.team_name} onChange={e => set('team_name', e.target.value)} />
            </div>
            <div>
              <label className="label">Jersey #</label>
              <input type="text" className="input" placeholder="7"
                value={form.jersey_number} onChange={e => set('jersey_number', e.target.value)} />
            </div>
          </div>

          <div>
            <label className="label">Position</label>
            <select className="input" value={form.position} onChange={e => set('position', e.target.value)}>
              <option value="">Select position</option>
              {POSITIONS.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>

          <button type="submit" className="btn-primary w-full flex items-center justify-center gap-2 py-3 mt-2" disabled={loading}>
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            {loading ? 'Creating account...' : 'Create account'}
          </button>
        </form>
      </div>
    </div>
  )
}
