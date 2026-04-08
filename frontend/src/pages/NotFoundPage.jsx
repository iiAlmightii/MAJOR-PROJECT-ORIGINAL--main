import React from 'react'
import { Link } from 'react-router-dom'
import { Activity } from 'lucide-react'

export default function NotFoundPage() {
  return (
    <div className="min-h-screen bg-court-bg flex items-center justify-center text-center p-6">
      <div>
        <div className="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
          <Activity className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-6xl font-bold text-white mb-3">404</h1>
        <p className="text-xl text-slate-400 mb-2">Page not found</p>
        <p className="text-slate-500 mb-8">The page you're looking for doesn't exist.</p>
        <Link to="/dashboard" className="btn-primary inline-flex items-center gap-2">
          Go to Dashboard
        </Link>
      </div>
    </div>
  )
}
