import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard, Video, Upload, BarChart3,
  Users, User, LogOut, ChevronLeft, ChevronRight,
  Activity, Shield,
} from 'lucide-react'
import useAuthStore from '../../store/authStore'
import clsx from 'clsx'

const NAV = [
  { to: '/dashboard',  label: 'Dashboard',  icon: LayoutDashboard, roles: ['admin','coach','player'] },
  { to: '/matches',    label: 'Matches',     icon: Video,           roles: ['admin','coach','player'] },
  { to: '/upload',     label: 'Upload',      icon: Upload,          roles: ['admin','coach'] },
  { to: '/analytics',  label: 'Analytics',   icon: BarChart3,       roles: ['admin','coach','player'] },
  { to: '/users',      label: 'Users',       icon: Users,           roles: ['admin'] },
  { to: '/profile',    label: 'Profile',     icon: User,            roles: ['admin','coach','player'] },
]

export default function Sidebar({ open, onToggle }) {
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  const visibleNav = NAV.filter(n => n.roles.includes(user?.role))

  return (
    <aside
      className={clsx(
        'relative flex flex-col bg-court-panel border-r border-court-border transition-all duration-300',
        open ? 'w-56' : 'w-16'
      )}
    >
      {/* Logo */}
      <div className={clsx(
        'flex items-center gap-3 px-4 py-5 border-b border-court-border',
        !open && 'justify-center px-2'
      )}>
        <div className="flex-shrink-0 w-9 h-9 bg-blue-600 rounded-xl flex items-center justify-center">
          <Activity className="w-5 h-5 text-white" />
        </div>
        {open && (
          <div>
            <div className="font-bold text-sm text-white leading-tight">VolleyVision</div>
            <div className="text-[10px] text-slate-500 uppercase tracking-widest">AI Analytics</div>
          </div>
        )}
      </div>

      {/* Role badge */}
      {open && (
        <div className="px-4 py-3 border-b border-court-border">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-full bg-blue-700 flex items-center justify-center text-xs font-bold text-white uppercase">
              {user?.username?.[0] || 'U'}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-medium text-slate-200 truncate">{user?.username}</div>
              <div className={clsx(
                'text-[10px] uppercase font-semibold tracking-wider',
                user?.role === 'admin'  ? 'text-purple-400' :
                user?.role === 'coach'  ? 'text-blue-400' : 'text-green-400'
              )}>
                {user?.role}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Nav */}
      <nav className="flex-1 py-4 space-y-1 px-2 overflow-y-auto">
        {visibleNav.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => clsx(
              'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-150 group',
              !open && 'justify-center px-2',
              isActive
                ? 'bg-blue-600/20 text-blue-400 border border-blue-600/30'
                : 'text-slate-400 hover:bg-court-border hover:text-slate-200'
            )}
            title={!open ? label : undefined}
          >
            <Icon className="w-5 h-5 flex-shrink-0" />
            {open && <span className="text-sm font-medium">{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Logout */}
      <div className="p-2 border-t border-court-border">
        <button
          onClick={handleLogout}
          className={clsx(
            'flex items-center gap-3 px-3 py-2.5 rounded-lg text-slate-400 hover:bg-red-900/20 hover:text-red-400 transition-colors w-full',
            !open && 'justify-center px-2'
          )}
          title={!open ? 'Logout' : undefined}
        >
          <LogOut className="w-5 h-5 flex-shrink-0" />
          {open && <span className="text-sm font-medium">Logout</span>}
        </button>
      </div>

      {/* Toggle button */}
      <button
        onClick={onToggle}
        className="absolute -right-3 top-20 w-6 h-6 bg-court-panel border border-court-border rounded-full flex items-center justify-center text-slate-400 hover:text-blue-400 transition-colors z-10"
      >
        {open ? <ChevronLeft className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
      </button>
    </aside>
  )
}
