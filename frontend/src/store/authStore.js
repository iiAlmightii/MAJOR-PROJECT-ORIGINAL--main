import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const useAuthStore = create(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      refreshToken: null,
      isAuthenticated: false,

      setAuth: (user, accessToken, refreshToken) => {
        localStorage.setItem('access_token', accessToken)
        localStorage.setItem('refresh_token', refreshToken)
        set({ user, accessToken, refreshToken, isAuthenticated: true })
      },

      updateUser: (updates) => set((s) => ({ user: { ...s.user, ...updates } })),

      logout: () => {
        localStorage.removeItem('access_token')
        localStorage.removeItem('refresh_token')
        set({ user: null, accessToken: null, refreshToken: null, isAuthenticated: false })
      },

      isAdmin:  () => get().user?.role === 'admin',
      isCoach:  () => get().user?.role === 'coach' || get().user?.role === 'admin',
      isPlayer: () => !!get().user,
    }),
    {
      name: 'vball-auth',
      partialize: (s) => ({
        user: s.user,
        accessToken: s.accessToken,
        refreshToken: s.refreshToken,
        isAuthenticated: s.isAuthenticated,
      }),
      merge: (persistedState, currentState) => {
        const persisted = persistedState || {}
        return {
          ...currentState,
          user: persisted.user ?? currentState.user,
          accessToken: persisted.accessToken ?? currentState.accessToken,
          refreshToken: persisted.refreshToken ?? currentState.refreshToken,
          isAuthenticated: persisted.isAuthenticated ?? currentState.isAuthenticated,
        }
      },
    }
  )
)

export default useAuthStore
