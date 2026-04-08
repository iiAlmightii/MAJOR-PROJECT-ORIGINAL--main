// frontend/src/pages/AnnotatePage.jsx
import React, { useState, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { ChevronLeft, Plus, Trash2, Download, Tag } from 'lucide-react'
import toast from 'react-hot-toast'
import { matchesAPI, annotationsAPI } from '../services/api'
import useAuthStore from '../store/authStore'

const ACTION_TYPES = ['spike', 'serve', 'block', 'dig', 'set', 'reception']
const ACTION_COLORS = {
  spike: 'bg-orange-500/20 text-orange-400 border-orange-600',
  serve: 'bg-blue-500/20 text-blue-400 border-blue-600',
  block: 'bg-green-500/20 text-green-400 border-green-600',
  dig: 'bg-cyan-500/20 text-cyan-400 border-cyan-600',
  set: 'bg-yellow-500/20 text-yellow-400 border-yellow-600',
  reception: 'bg-purple-500/20 text-purple-400 border-purple-600',
}

function fmtTime(secs) {
  const m = Math.floor(secs / 60)
  const s = secs.toFixed(2).padStart(5, '0')
  return `${m}:${s}`
}

export default function AnnotatePage() {
  const { id: matchId } = useParams()
  const { isAdmin }     = useAuthStore()
  const videoRef        = useRef(null)
  const qc              = useQueryClient()
  const [selectedType, setSelectedType] = useState('spike')

  const { data: match } = useQuery({
    queryKey: ['match', matchId],
    queryFn: () => matchesAPI.get(matchId).then(r => r.data),
  })

  const { data: annotations = [] } = useQuery({
    queryKey: ['annotations', matchId],
    queryFn: () => annotationsAPI.list(matchId).then(r => r.data),
  })

  const token = localStorage.getItem('access_token')
  const streamUrl = `/api/videos/${match?.video_id}/stream${token ? `?token=${encodeURIComponent(token)}` : ''}`

  const createMutation = useMutation({
    mutationFn: (data) => annotationsAPI.create(matchId, data),
    onSuccess: () => {
      qc.invalidateQueries(['annotations', matchId])
      toast.success('Tag added')
    },
    onError: () => toast.error('Failed to add tag'),
  })

  const deleteMutation = useMutation({
    mutationFn: (id) => annotationsAPI.delete(id),
    onSuccess: () => {
      qc.invalidateQueries(['annotations', matchId])
      toast.success('Tag removed')
    },
  })

  function handleTagCurrent() {
    if (!videoRef.current) return
    const ts = videoRef.current.currentTime
    createMutation.mutate({ timestamp: ts, action_type: selectedType })
  }

  async function handleExport() {
    const res = await annotationsAPI.export()
    const url = URL.createObjectURL(res.data)
    const a = document.createElement('a')
    a.href = url
    a.download = 'annotations.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Link to={`/matches/${matchId}`} className="text-slate-400 hover:text-white">
          <ChevronLeft className="w-5 h-5" />
        </Link>
        <div>
          <h2 className="text-xl font-bold text-white">Annotate Match</h2>
          <p className="text-slate-400 text-sm">{match?.title}</p>
        </div>
        {isAdmin() && (
          <button
            onClick={handleExport}
            className="ml-auto flex items-center gap-2 px-3 py-1.5 text-sm bg-[#2e3a52] hover:bg-[#3e4a62] text-slate-300 rounded-lg"
          >
            <Download className="w-4 h-4" />
            Export JSON
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Video + tagging */}
        <div className="lg:col-span-2 space-y-3">
          <div className="bg-black rounded-xl overflow-hidden aspect-video">
            {match?.video_id && (
              <video
                ref={videoRef}
                src={streamUrl}
                controls
                className="w-full h-full"
              />
            )}
          </div>

          {/* Tag controls */}
          <div className="card flex items-center gap-3 flex-wrap">
            <Tag className="w-4 h-4 text-slate-400 flex-shrink-0" />
            <span className="text-sm text-slate-400">Tag current position as:</span>
            <div className="flex gap-2 flex-wrap">
              {ACTION_TYPES.map(t => (
                <button
                  key={t}
                  onClick={() => setSelectedType(t)}
                  className={`px-3 py-1 text-xs rounded-full border capitalize font-medium transition-all ${
                    selectedType === t
                      ? ACTION_COLORS[t]
                      : 'bg-transparent text-slate-500 border-slate-700 hover:border-slate-500'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
            <button
              onClick={handleTagCurrent}
              disabled={createMutation.isPending}
              className="ml-auto flex items-center gap-1.5 px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg disabled:opacity-50"
            >
              <Plus className="w-4 h-4" />
              Add Tag
            </button>
          </div>
        </div>

        {/* Tag list */}
        <div className="card overflow-hidden">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white">Tags ({annotations.length})</h3>
          </div>
          <div className="space-y-1 max-h-[500px] overflow-y-auto">
            {annotations.length === 0 && (
              <p className="text-slate-500 text-sm text-center py-8">No tags yet. Play the video and tag actions.</p>
            )}
            {annotations.map(a => (
              <div key={a.id} className="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-[#1a1f2e] group">
                <button
                  onClick={() => {
                    if (videoRef.current) videoRef.current.currentTime = a.timestamp
                  }}
                  className="flex-1 flex items-center gap-2 text-left"
                >
                  <span className={`px-2 py-0.5 rounded text-xs border capitalize font-medium ${ACTION_COLORS[a.action_type] || 'bg-slate-700 text-slate-300 border-slate-600'}`}>
                    {a.action_type}
                  </span>
                  <span className="text-xs text-slate-400 font-mono">{fmtTime(a.timestamp)}</span>
                </button>
                <button
                  onClick={() => deleteMutation.mutate(a.id)}
                  className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition-opacity"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
