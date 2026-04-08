import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Activity, Eye, EyeOff, Loader2 } from 'lucide-react'
import toast from 'react-hot-toast'
import { authAPI } from '../services/api'
import useAuthStore from '../store/authStore'

export default function LoginPage() {
  const navigate = useNavigate()
  const { setAuth } = useAuthStore()
  const [form, setForm] = useState({ email: '', password: '' })
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const { data } = await authAPI.login(form)
      setAuth(data.user, data.access_token, data.refresh_token)
      toast.success(`Welcome back, ${data.user.username}!`)
      navigate('/dashboard')
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-court-bg flex">
      {/* Left panel — branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-blue-900/50 to-court-bg relative overflow-hidden items-center justify-center p-12">
        <div className="absolute inset-0 opacity-10">
          {/* Court lines decoration */}
          <svg viewBox="0 0 600 600" className="w-full h-full">
            <rect x="50"  y="50"  width="500" height="500" fill="none" stroke="#3b82f6" strokeWidth="2"/>
            <line x1="300" y1="50"  x2="300" y2="550" stroke="#3b82f6" strokeWidth="1"/>
            <line x1="50"  y1="300" x2="550" y2="300" stroke="#3b82f6" strokeWidth="2"/>
            <rect x="150" y="50"  width="300" height="120" fill="none" stroke="#3b82f6" strokeWidth="1"/>
            <rect x="150" y="430" width="300" height="120" fill="none" stroke="#3b82f6" strokeWidth="1"/>
            <circle cx="300" cy="300" r="60" fill="none" stroke="#3b82f6" strokeWidth="1"/>
          </svg>
        </div>
        <div className="relative z-10 text-center">
          <div className="w-20 h-20 bg-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-2xl shadow-blue-600/40">
            <Activity className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-3">VolleyVision</h1>
          <p className="text-slate-400 text-lg mb-8">AI-Powered Volleyball Analytics</p>
          <div className="grid grid-cols-2 gap-4 text-left">
            {[
              { n: 'Player Tracking', d: 'YOLOv8 + ByteTrack' },
              { n: 'Ball Trajectory', d: 'Frame-by-frame detection' },
              { n: 'Rally Detection', d: 'Auto video segmentation' },
              { n: 'Action Recognition', d: 'Serve · Spike · Block' },
            ].map(f => (
              <div key={f.n} className="bg-court-panel/50 border border-court-border rounded-xl p-4">
                <div className="text-blue-400 font-semibold text-sm">{f.n}</div>
                <div className="text-slate-500 text-xs mt-1">{f.d}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right panel — form */}
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-md">
          <div className="flex items-center gap-3 mb-8 lg:hidden">
            <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-white">VolleyVision</span>
          </div>

          <h2 className="text-2xl font-bold text-white mb-2">Sign in</h2>
          <p className="text-slate-400 mb-8">
            Don't have an account?{' '}
            <Link to="/register" className="text-blue-400 hover:text-blue-300 font-medium">
              Create one
            </Link>
          </p>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="label">Email address</label>
              <input
                type="email"
                className="input"
                placeholder="you@team.com"
                value={form.email}
                onChange={e => setForm(f => ({ ...f, email: e.target.value }))}
                required
              />
            </div>

            <div>
              <label className="label">Password</label>
              <div className="relative">
                <input
                  type={showPw ? 'text' : 'password'}
                  className="input pr-10"
                  placeholder="••••••••"
                  value={form.password}
                  onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPw(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                >
                  {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <button type="submit" className="btn-primary w-full flex items-center justify-center gap-2 py-3" disabled={loading}>
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              {loading ? 'Signing in...' : 'Sign in'}
            </button>
          </form>

          {/* Demo credentials */}
          <div className="mt-6 p-4 bg-court-panel border border-court-border rounded-xl">
            <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-3">Demo Credentials</p>
            <div className="space-y-2">
              {[
                { role: 'Admin',  email: 'admin@volleyball.com', pw: 'Admin@123456' },
              ].map(c => (
                <button
                  key={c.role}
                  onClick={() => setForm({ email: c.email, password: c.pw })}
                  className="w-full text-left px-3 py-2 rounded-lg bg-court-bg border border-court-border hover:border-blue-600/50 transition-colors"
                >
                  <span className="text-xs font-semibold text-blue-400">{c.role}</span>
                  <span className="text-xs text-slate-500 ml-2">{c.email}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
