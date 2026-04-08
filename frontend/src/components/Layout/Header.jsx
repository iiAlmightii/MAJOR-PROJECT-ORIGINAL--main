import React from 'react'
import { useLocation, Link } from 'react-router-dom'
import { Menu, Bell, Search } from 'lucide-react'
import useAuthStore from '../../store/authStore'

const ROUTE_LABELS = {
  '/dashboard': 'Dashboard',
  '/matches':   'Matches',
  '/upload':    'Upload Match',
  '/analytics': 'Analytics',
  '/users':     'User Management',
  '/profile':   'My Profile',
}

export default function Header({ onMenuClick }) {
  const location = useLocation()
  const { user } = useAuthStore()

  const label = ROUTE_LABELS[location.pathname]
    ?? location.pathname.includes('/matches/')
      ? 'Match Analysis'
      : 'VolleyVision'

  return (
    <header className="h-14 bg-court-panel border-b border-court-border flex items-center justify-between px-4 flex-shrink-0">
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-court-border transition-colors"
        >
          <Menu className="w-5 h-5" />
        </button>
        <h1 className="text-base font-semibold text-slate-200">{label}</h1>
      </div>

      <div className="flex items-center gap-3">
        {/* Search */}
        <div className="hidden md:flex items-center gap-2 bg-court-bg border border-court-border rounded-lg px-3 py-1.5">
          <Search className="w-4 h-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search matches..."
            className="bg-transparent text-sm text-slate-300 placeholder-slate-500 outline-none w-44"
          />
        </div>

        {/* Notifications */}
        <button className="relative p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-court-border transition-colors">
          <Bell className="w-5 h-5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-blue-500 rounded-full" />
        </button>

        {/* User Avatar */}
        <Link to="/profile" className="flex items-center gap-2 group">
          <div className="w-8 h-8 rounded-full bg-blue-700 flex items-center justify-center text-sm font-bold text-white uppercase">
            {user?.username?.[0] || 'U'}
          </div>
        </Link>
      </div>
    </header>
  )
}
