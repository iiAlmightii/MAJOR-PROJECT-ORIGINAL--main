import axios from 'axios'
import toast from 'react-hot-toast'
import useAuthStore from '../store/authStore'

const api = axios.create({
  baseURL: '/api',
  timeout: 60_000,
})

// Attach JWT to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Auto-refresh on 401
api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const original = error.config || {}
    const requestUrl = original.url || ''
    const isAuthEndpoint = requestUrl.includes('/auth/login') || requestUrl.includes('/auth/register') || requestUrl.includes('/auth/refresh')

    if (error.response?.status === 401 && !original._retry && !isAuthEndpoint) {
      original._retry = true
      const refresh = localStorage.getItem('refresh_token')
      if (refresh) {
        try {
          const { data } = await axios.post('/api/auth/refresh', { refresh_token: refresh })
          localStorage.setItem('access_token', data.access_token)
          localStorage.setItem('refresh_token', data.refresh_token)
          original.headers.Authorization = `Bearer ${data.access_token}`
          return api(original)
        } catch {
          useAuthStore.getState().logout()
          window.location.replace('/login')
        }
      } else {
        useAuthStore.getState().logout()
        window.location.replace('/login')
      }
    }
    return Promise.reject(error)
  }
)

// ─── Auth ─────────────────────────────────────────────
export const authAPI = {
  login:          (data)  => api.post('/auth/login',           data),
  register:       (data)  => api.post('/auth/register',        data),
  me:             ()      => api.get('/auth/me'),
  refresh:        (data)  => api.post('/auth/refresh',         data),
  changePassword: (data)  => api.post('/auth/change-password', data),
}

// ─── Users ────────────────────────────────────────────
export const usersAPI = {
  list:   (params) => api.get('/users',                params),
  stats:  ()       => api.get('/users/stats'),
  get:    (id)     => api.get(`/users/${id}`),
  updateMe: (data) => api.patch('/users/me',           data),
  update: (id, d)  => api.patch(`/users/${id}`,        d),
  delete: (id)     => api.delete(`/users/${id}`),
  logs:   (id, p)  => api.get(`/users/${id}/logs`,     p),
}

// ─── Videos ───────────────────────────────────────────
export const videosAPI = {
  upload: (formData, onProgress) =>
    api.post('/videos/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 600_000,
      onUploadProgress: (e) => {
        if (onProgress) onProgress(Math.round((e.loaded * 100) / e.total))
      },
    }),
  get:       (id)  => api.get(`/videos/${id}`),
  delete:    (id)  => api.delete(`/videos/${id}`),
  streamUrl: (id)  => {
    const token = localStorage.getItem('access_token')
    return token
      ? `/api/videos/${id}/stream?token=${encodeURIComponent(token)}`
      : `/api/videos/${id}/stream`
  },
  thumbUrl:  (id)  => `/api/videos/${id}/thumbnail`,
}

// ─── Matches ──────────────────────────────────────────
export const matchesAPI = {
  create:   (data)    => api.post('/matches/',                   data),
  list:     (params)  => api.get('/matches',                   { params }),
  get:      (id)      => api.get(`/matches/${id}`),
  update:   (id, d)   => api.patch(`/matches/${id}`,             d),
  delete:   (id)      => api.delete(`/matches/${id}`),
  analyze:  (id)      => api.post(`/matches/${id}/analyze`),
  rallies:  (id)      => api.get(`/matches/${id}/rallies`),
  analytics:(id)      => api.get(`/matches/${id}/analytics`),
  actions:  (id, params) => api.get(`/matches/${id}/actions`, { params }),
}

// ─── Annotations ─────────────────────────────────────────────
export const annotationsAPI = {
  list:   (matchId)         => api.get(`/matches/${matchId}/annotations`),
  create: (matchId, data)   => api.post(`/matches/${matchId}/annotations`, data),
  delete: (id)              => api.delete(`/annotations/${id}`),
  export: ()                => api.get('/annotations/export', { responseType: 'blob' }),
}

// ─── Analytics ────────────────────────────────────────
export const analyticsAPI = {
  adminDashboard:  () => api.get('/analytics/dashboard/admin'),
  coachDashboard:  () => api.get('/analytics/dashboard/coach'),
  playerDashboard: () => api.get('/analytics/dashboard/player'),
  leaderboard: (metric = 'attacks') => api.get('/analytics/leaderboard', { params: { metric } }),
  matchCompare: (matchId)           => api.get(`/analytics/match/${matchId}/compare`),
  systemLogs:   (p)                 => api.get('/analytics/logs', { params: p }),
}

export default api
