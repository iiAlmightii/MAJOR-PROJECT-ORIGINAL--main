/**
 * SpeechTab — Speech-to-Knowledge Module UI
 * ──────────────────────────────────────────
 * Lets the user:
 *  1. Upload a commentary audio file OR transcribe the match video's own audio
 *  2. See transcription status
 *  3. Browse NLP-extracted speech events (with fusion status)
 *  4. Click an event to seek the video to that timestamp
 */

import React, { useRef, useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Mic, Upload, Play, CheckCircle, XCircle, Clock,
  RefreshCw, Loader2, Merge, AlertCircle, Radio,
  ArrowUp, Shield, Activity, Minus, MessageSquare
} from 'lucide-react'
import toast from 'react-hot-toast'
import { matchesAPI } from '../../services/api'
import clsx from 'clsx'

const EVENT_META = {
  serve:   { label: 'Serve',     icon: Radio,    color: 'text-blue-400',   bg: 'bg-blue-900/40' },
  spike:   { label: 'Spike',     icon: ArrowUp,  color: 'text-orange-400', bg: 'bg-orange-900/40' },
  block:   { label: 'Block',     icon: Shield,   color: 'text-green-400',  bg: 'bg-green-900/40' },
  receive: { label: 'Receive',   icon: Activity, color: 'text-purple-400', bg: 'bg-purple-900/40' },
  dig:     { label: 'Dig',       icon: Activity, color: 'text-cyan-400',   bg: 'bg-cyan-900/40' },
  set:     { label: 'Set',       icon: Minus,    color: 'text-yellow-400', bg: 'bg-yellow-900/40' },
  unknown: { label: 'Unknown',   icon: Activity, color: 'text-slate-500',  bg: 'bg-slate-800' },
}

const FUSION_BADGE = {
  fused:      { label: 'Fused',      color: 'bg-green-900/40 text-green-400 border-green-700' },
  standalone: { label: 'Speech only',color: 'bg-blue-900/30 text-blue-400  border-blue-700'   },
  conflict:   { label: 'Conflict',   color: 'bg-red-900/30  text-red-400   border-red-700'    },
}

function fmtTime(secs) {
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

function seekVideo(ts) {
  const video = document.querySelector('video')
  if (video) video.currentTime = ts
}

export default function SpeechTab({ matchId, matchStatus }) {
  const qc          = useQueryClient()
  const fileInputRef = useRef(null)
  const [filterType,   setFilterType]   = useState('')
  const [filterFusion, setFilterFusion] = useState('')

  // ── Transcriptions ──────────────────────────────────────────────────────────
  const { data: transcriptionsData, isLoading: txLoading } = useQuery({
    queryKey: ['speech-transcriptions', matchId],
    queryFn:  () => matchesAPI.speechTranscriptions(matchId).then(r => r.data),
    refetchInterval: (query) => {
      const txs = query?.state?.data?.transcriptions || []
      return txs.some(t => t.status === 'pending' || t.status === 'processing') ? 2000 : false
    },
  })
  const transcriptions = transcriptionsData?.transcriptions || []
  const latestTx = transcriptions[0]

  // ── Speech events ───────────────────────────────────────────────────────────
  const { data: eventsData, isLoading: evLoading } = useQuery({
    queryKey: ['speech-events', matchId, filterType, filterFusion],
    queryFn:  () => matchesAPI.speechEvents(matchId, {
      event_type:    filterType    || undefined,
      fusion_status: filterFusion  || undefined,
      limit: 300,
    }).then(r => r.data),
    enabled: transcriptions.some(t => t.status === 'completed'),
  })
  const events = eventsData?.items || []

  // ── Mutations ───────────────────────────────────────────────────────────────
  const transcribeVideoMut = useMutation({
    mutationFn: () => matchesAPI.transcribeVideo(matchId, 'en'),
    onSuccess:  () => {
      toast.success('Transcription started from video audio!')
      qc.invalidateQueries(['speech-transcriptions', matchId])
    },
    onError: (e) => toast.error(e.response?.data?.detail || 'Transcription failed'),
  })

  const uploadAudioMut = useMutation({
    mutationFn: (file) => {
      const fd = new FormData()
      fd.append('audio_file', file)
      return matchesAPI.transcribeAudio(matchId, fd)
    },
    onSuccess: () => {
      toast.success('Audio uploaded — transcription started!')
      qc.invalidateQueries(['speech-transcriptions', matchId])
    },
    onError: (e) => toast.error(e.response?.data?.detail || 'Upload failed'),
  })

  const fuseMut = useMutation({
    mutationFn: () => matchesAPI.runFusion(matchId),
    onSuccess:  (r) => {
      toast.success(r.data?.message || 'Fusion complete')
      qc.invalidateQueries(['speech-events', matchId])
    },
    onError: (e) => toast.error(e.response?.data?.detail || 'Fusion failed'),
  })

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) uploadAudioMut.mutate(file)
  }

  // ── Event type summary ──────────────────────────────────────────────────────
  const summary = useMemo(() => {
    const counts = {}
    events.forEach(ev => {
      if (!counts[ev.event_type]) counts[ev.event_type] = 0
      counts[ev.event_type]++
    })
    return counts
  }, [events])

  // ── Status icon helper ──────────────────────────────────────────────────────
  function TxStatusIcon({ status }) {
    if (status === 'completed') return <CheckCircle className="w-4 h-4 text-green-400" />
    if (status === 'failed')    return <XCircle     className="w-4 h-4 text-red-400" />
    if (status === 'processing' || status === 'pending')
      return <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
    return <Clock className="w-4 h-4 text-slate-500" />
  }

  const isWorking = transcribeVideoMut.isPending || uploadAudioMut.isPending

  return (
    <div className="space-y-4">

      {/* ── Header + actions ───────────────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center gap-2 mb-3">
          <Mic className="w-5 h-5 text-blue-400" />
          <h3 className="font-semibold text-white text-sm">Speech-to-Knowledge</h3>
          <span className="ml-auto text-xs text-slate-500">
            Powered by Whisper ASR + NLP
          </span>
        </div>

        <p className="text-xs text-slate-400 mb-4">
          Upload a commentary audio file or transcribe the match video's audio track.
          The system will extract volleyball events (serve, spike, block, receive) from
          spoken commentary and fuse them with CV-detected events for higher accuracy.
        </p>

        <div className="flex flex-wrap gap-2">
          {/* Transcribe video audio */}
          <button
            onClick={() => transcribeVideoMut.mutate()}
            disabled={isWorking}
            className="flex items-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm rounded-lg transition-colors"
          >
            {transcribeVideoMut.isPending
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <Mic className="w-4 h-4" />}
            Transcribe Match Audio
          </button>

          {/* Upload separate audio */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isWorking}
            className="flex items-center gap-2 px-3 py-2 bg-court-panel hover:bg-court-border disabled:opacity-50 text-slate-300 text-sm rounded-lg border border-court-border transition-colors"
          >
            {uploadAudioMut.isPending
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <Upload className="w-4 h-4" />}
            Upload Commentary File
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,video/*"
            className="hidden"
            onChange={handleFileChange}
          />

          {/* Manual fusion trigger */}
          {matchStatus === 'completed' && events.length > 0 && (
            <button
              onClick={() => fuseMut.mutate()}
              disabled={fuseMut.isPending}
              className="flex items-center gap-2 px-3 py-2 bg-green-900/40 hover:bg-green-900/60 disabled:opacity-50 text-green-400 text-sm rounded-lg border border-green-700 transition-colors"
            >
              {fuseMut.isPending
                ? <Loader2 className="w-4 h-4 animate-spin" />
                : <Merge className="w-4 h-4" />}
              Re-run Fusion
            </button>
          )}
        </div>
      </div>

      {/* ── Transcription history ───────────────────────────────────────────── */}
      {transcriptions.length > 0 && (
        <div className="card">
          <h4 className="text-xs font-medium text-slate-400 mb-2">Transcription Runs</h4>
          <div className="space-y-2">
            {transcriptions.map(tx => (
              <div key={tx.id} className="flex items-center gap-3 p-2 bg-court-bg rounded-lg border border-court-border">
                <TxStatusIcon status={tx.status} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-slate-300 font-medium capitalize">
                      {tx.audio_source === 'video_audio' ? 'Match video audio' : 'Uploaded file'}
                    </span>
                    <span className="text-slate-600">·</span>
                    <span className="text-slate-500">{tx.whisper_model || 'base'} model</span>
                    {tx.duration_seconds > 0 && (
                      <span className="text-slate-600">· {Math.round(tx.duration_seconds)}s audio</span>
                    )}
                  </div>
                  {tx.status === 'failed' && tx.error_message && (
                    <div className="text-xs text-red-400 mt-0.5 truncate">{tx.error_message}</div>
                  )}
                  {(tx.status === 'pending' || tx.status === 'processing') && (
                    <div className="text-xs text-blue-400 mt-0.5">
                      {tx.status === 'pending' ? 'Queued...' : 'Transcribing with Whisper...'}
                    </div>
                  )}
                </div>
                <span className={clsx(
                  'text-[10px] px-2 py-0.5 rounded-full border font-medium',
                  tx.status === 'completed' ? 'bg-green-900/30 text-green-400 border-green-700'
                    : tx.status === 'failed' ? 'bg-red-900/30 text-red-400 border-red-700'
                    : 'bg-blue-900/30 text-blue-400 border-blue-700'
                )}>
                  {tx.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Whisper not installed notice ────────────────────────────────────── */}
      {txLoading === false && transcriptions.length === 0 && (
        <div className="card border-slate-700">
          <div className="flex items-start gap-3">
            <MessageSquare className="w-5 h-5 text-slate-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-slate-300">No transcriptions yet</p>
              <p className="text-xs text-slate-500 mt-1">
                Click "Transcribe Match Audio" above to extract events from spoken commentary.
                Make sure <code className="bg-court-bg px-1 rounded">openai-whisper</code> is
                installed: <code className="bg-court-bg px-1 rounded">pip install openai-whisper</code>
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── Extracted events ────────────────────────────────────────────────── */}
      {events.length > 0 && (
        <>
          {/* Summary pills */}
          <div className="flex flex-wrap gap-2 items-center">
            <span className="text-xs text-slate-500">Events by type:</span>
            {Object.entries(summary).map(([type, count]) => {
              const meta = EVENT_META[type] || EVENT_META.unknown
              const Icon = meta.icon
              return (
                <button
                  key={type}
                  onClick={() => setFilterType(f => f === type ? '' : type)}
                  className={clsx(
                    'flex items-center gap-1.5 px-3 py-1 rounded-lg border text-xs font-medium transition-colors',
                    filterType === type
                      ? `${meta.bg} ${meta.color} border-current`
                      : 'bg-court-bg border-court-border text-slate-400 hover:border-slate-500'
                  )}
                >
                  <Icon className="w-3 h-3" />
                  {meta.label} <span className="font-bold">{count}</span>
                </button>
              )
            })}

            {/* Fusion status filter */}
            <div className="ml-auto flex items-center gap-1">
              <span className="text-xs text-slate-500">Fusion:</span>
              {['', 'fused', 'standalone', 'conflict'].map(fs => (
                <button
                  key={fs}
                  onClick={() => setFilterFusion(fs)}
                  className={clsx(
                    'text-[10px] px-2 py-0.5 rounded border transition-colors',
                    filterFusion === fs
                      ? 'bg-blue-900/40 text-blue-400 border-blue-700'
                      : 'bg-court-bg text-slate-500 border-court-border hover:border-slate-500'
                  )}
                >
                  {fs || 'All'}
                </button>
              ))}
            </div>
          </div>

          {/* Event list */}
          <div className="space-y-1.5">
            {events.map(ev => {
              const meta   = EVENT_META[ev.event_type] || EVENT_META.unknown
              const Icon   = meta.icon
              const fusion = FUSION_BADGE[ev.fusion_status] || FUSION_BADGE.standalone
              const pct    = Math.round((ev.extraction_confidence || 0) * 100)

              return (
                <div
                  key={ev.id}
                  onClick={() => seekVideo(ev.start_time)}
                  className="card flex items-center gap-3 hover:border-blue-600/40 cursor-pointer py-2.5 transition-colors"
                >
                  {/* Action icon */}
                  <div className={clsx('w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0', meta.bg)}>
                    <Icon className={clsx('w-4 h-4', meta.color)} />
                  </div>

                  {/* Text + meta */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={clsx('text-xs font-semibold', meta.color)}>{meta.label}</span>
                      {ev.result !== 'neutral' && (
                        <span className={clsx(
                          'text-[10px] px-1.5 py-0.5 rounded',
                          ev.result === 'success' ? 'bg-green-900/40 text-green-400' : 'bg-red-900/40 text-red-400'
                        )}>
                          {ev.result}
                        </span>
                      )}
                      {ev.player_number != null && (
                        <span className="text-[10px] bg-court-bg text-slate-400 px-1.5 py-0.5 rounded border border-court-border">
                          Player #{ev.player_number}
                        </span>
                      )}
                      {ev.team && (
                        <span className={clsx(
                          'text-[10px] px-1.5 py-0.5 rounded',
                          ev.team === 'A' ? 'bg-blue-900/30 text-blue-400' : 'bg-red-900/30 text-red-400'
                        )}>
                          Team {ev.team}
                        </span>
                      )}
                      <span className={clsx('text-[10px] px-1.5 py-0.5 rounded border', fusion.color)}>
                        {fusion.label}
                      </span>
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5 truncate italic">
                      "{ev.raw_text}"
                    </div>
                  </div>

                  {/* Confidence */}
                  <span className="flex items-center gap-1 text-xs text-slate-400 flex-shrink-0">
                    <span className={clsx(
                      'w-2 h-2 rounded-full',
                      pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                    )} />
                    {pct}%
                  </span>

                  {/* Seek button */}
                  <button className="flex items-center gap-1 px-2 py-1 rounded text-xs text-slate-400 hover:text-blue-400 hover:bg-blue-900/20 transition-colors flex-shrink-0">
                    <Play className="w-3 h-3" />
                    {fmtTime(ev.start_time)}
                  </button>
                </div>
              )
            })}
          </div>
        </>
      )}

      {/* ── No events yet ───────────────────────────────────────────────────── */}
      {!evLoading && events.length === 0 && transcriptions.some(t => t.status === 'completed') && (
        <div className="card text-center py-8">
          <AlertCircle className="w-8 h-8 text-slate-600 mx-auto mb-2" />
          <p className="text-slate-400 text-sm">
            No volleyball events were extracted from the commentary.
          </p>
          <p className="text-xs text-slate-500 mt-1">
            This can happen if the audio has no volleyball action words or is very noisy.
          </p>
        </div>
      )}
    </div>
  )
}
