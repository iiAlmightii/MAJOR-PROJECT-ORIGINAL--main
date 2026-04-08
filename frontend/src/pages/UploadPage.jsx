import React, { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { Upload, Video, CheckCircle, AlertCircle, Loader2, X, FileVideo } from 'lucide-react'
import toast from 'react-hot-toast'
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar'
import 'react-circular-progressbar/dist/styles.css'
import { videosAPI, matchesAPI } from '../services/api'

const ALLOWED_TYPES = { 'video/mp4': ['.mp4'], 'video/avi': ['.avi'], 'video/quicktime': ['.mov'], 'video/x-matroska': ['.mkv'] }
const MAX_SIZE = 500 * 1024 * 1024 // 500 MB

function formatBytes(bytes) {
  if (bytes < 1024)       return `${bytes} B`
  if (bytes < 1048576)    return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`
  return `${(bytes / 1073741824).toFixed(2)} GB`
}

export default function UploadPage() {
  const navigate = useNavigate()
  const [file, setFile]         = useState(null)
  const [progress, setProgress] = useState(0)
  const [phase, setPhase]       = useState('idle') // idle | uploading | creating | done | error
  const [videoId, setVideoId]   = useState(null)
  const [form, setForm]         = useState({
    title: '', description: '', team_a: '', team_b: '', venue: '',
  })

  const onDrop = useCallback((accepted, rejected) => {
    if (rejected.length) {
      toast.error('Invalid file. Use MP4, AVI, MOV, MKV (max 500 MB)')
      return
    }
    setFile(accepted[0])
    if (!form.title) setForm(f => ({ ...f, title: accepted[0].name.replace(/\.[^.]+$/, '') }))
  }, [form.title])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ALLOWED_TYPES,
    maxSize: MAX_SIZE,
    maxFiles: 1,
  })

  const handleUpload = async (e) => {
    e.preventDefault()
    if (!file) { toast.error('Please select a video file'); return }
    if (!form.title.trim()) { toast.error('Please enter a match title'); return }

    try {
      // Step 1: Upload video
      setPhase('uploading')
      setProgress(0)
      const fd = new FormData()
      fd.append('file', file)
      const { data: videoData } = await videosAPI.upload(fd, setProgress)
      setVideoId(videoData.video_id)
      toast.success('Video uploaded!')

      // Step 2: Create match record
      setPhase('creating')
      const { data: matchData } = await matchesAPI.create({
        ...form,
        video_id: videoData.video_id,
      })
      toast.success('Match created! Ready for analysis.')
      setPhase('done')
      setTimeout(() => navigate(`/matches/${matchData.id}`), 1500)
    } catch (err) {
      setPhase('error')
      toast.error(err.response?.data?.detail || 'Upload failed')
    }
  }

  const reset = () => {
    setFile(null); setProgress(0); setPhase('idle'); setVideoId(null)
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-white">Upload Match Video</h2>
        <p className="text-slate-400 mt-1 text-sm">Upload a full match video for AI-powered analytics</p>
      </div>

      <form onSubmit={handleUpload} className="space-y-6">
        {/* Drop zone */}
        {!file ? (
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-200 ${
              isDragActive
                ? 'border-blue-500 bg-blue-900/20'
                : 'border-court-border hover:border-blue-600/60 hover:bg-court-panel/50'
            }`}
          >
            <input {...getInputProps()} />
            <div className="w-16 h-16 bg-blue-900/40 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <Upload className="w-8 h-8 text-blue-400" />
            </div>
            <p className="text-lg font-medium text-slate-200 mb-2">
              {isDragActive ? 'Drop the video here' : 'Drag & drop video here'}
            </p>
            <p className="text-sm text-slate-500">MP4, AVI, MOV, MKV — max 500 MB</p>
            <button type="button" className="btn-secondary mt-4">Browse files</button>
          </div>
        ) : (
          <div className="card flex items-center gap-4">
            <div className="w-12 h-12 bg-blue-900/40 rounded-xl flex items-center justify-center flex-shrink-0">
              <FileVideo className="w-6 h-6 text-blue-400" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="font-medium text-slate-200 truncate">{file.name}</div>
              <div className="text-sm text-slate-500">{formatBytes(file.size)}</div>

              {/* Progress */}
              {phase === 'uploading' && (
                <div className="mt-2">
                  <div className="flex items-center justify-between text-xs text-slate-500 mb-1">
                    <span>Uploading...</span>
                    <span>{progress}%</span>
                  </div>
                  <div className="w-full bg-court-bg rounded-full h-1.5">
                    <div
                      className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}
              {phase === 'creating' && (
                <div className="flex items-center gap-2 mt-2 text-xs text-blue-400">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Creating match record...
                </div>
              )}
              {phase === 'done' && (
                <div className="flex items-center gap-2 mt-2 text-xs text-green-400">
                  <CheckCircle className="w-3 h-3" />
                  Complete! Redirecting to match...
                </div>
              )}
            </div>
            {/* Circular progress */}
            {phase === 'uploading' && (
              <div className="w-12 h-12 flex-shrink-0">
                <CircularProgressbar
                  value={progress}
                  text={`${progress}%`}
                  styles={buildStyles({
                    textSize: '28px',
                    pathColor: '#3b82f6',
                    textColor: '#f1f5f9',
                    trailColor: '#1a1f2e',
                  })}
                />
              </div>
            )}
            {phase === 'idle' && (
              <button type="button" onClick={reset} className="p-2 text-slate-500 hover:text-red-400 transition-colors">
                <X className="w-4 h-4" />
              </button>
            )}
            {phase === 'done' && <CheckCircle className="w-8 h-8 text-green-500 flex-shrink-0" />}
            {phase === 'error' && <AlertCircle className="w-8 h-8 text-red-500 flex-shrink-0" />}
          </div>
        )}

        {/* Match details */}
        <div className="card space-y-4">
          <h3 className="font-semibold text-white">Match Details</h3>

          <div>
            <label className="label">Match Title *</label>
            <input className="input" placeholder="e.g. Team Alpha vs Team Beta — Finals"
              value={form.title} onChange={e => setForm(f => ({ ...f, title: e.target.value }))} required />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Team A</label>
              <input className="input" placeholder="Team Alpha"
                value={form.team_a} onChange={e => setForm(f => ({ ...f, team_a: e.target.value }))} />
            </div>
            <div>
              <label className="label">Team B</label>
              <input className="input" placeholder="Team Beta"
                value={form.team_b} onChange={e => setForm(f => ({ ...f, team_b: e.target.value }))} />
            </div>
          </div>

          <div>
            <label className="label">Venue</label>
            <input className="input" placeholder="Sports Arena, City"
              value={form.venue} onChange={e => setForm(f => ({ ...f, venue: e.target.value }))} />
          </div>

          <div>
            <label className="label">Description</label>
            <textarea className="input resize-none" rows={3} placeholder="Add notes about this match..."
              value={form.description} onChange={e => setForm(f => ({ ...f, description: e.target.value }))} />
          </div>
        </div>

        <div className="flex gap-3">
          <button
            type="submit"
            disabled={!file || phase === 'uploading' || phase === 'creating' || phase === 'done'}
            className="btn-primary flex items-center gap-2 px-8 py-3"
          >
            {(phase === 'uploading' || phase === 'creating') && <Loader2 className="w-4 h-4 animate-spin" />}
            {phase === 'idle'    && <><Upload className="w-4 h-4" /> Upload & Create Match</>}
            {phase === 'uploading' && 'Uploading Video...'}
            {phase === 'creating'  && 'Creating Match...'}
            {phase === 'done'      && <><CheckCircle className="w-4 h-4" /> Done!</>}
            {phase === 'error'     && <><Upload className="w-4 h-4" /> Retry</>}
          </button>
          {phase === 'error' && (
            <button type="button" onClick={reset} className="btn-secondary">Reset</button>
          )}
        </div>
      </form>
    </div>
  )
}
