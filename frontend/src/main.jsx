import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'
import App from './App'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')).render(
  <BrowserRouter>
    <QueryClientProvider client={queryClient}>
      <App />
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: '#232b3e',
            color: '#f1f5f9',
            border: '1px solid #2e3a52',
          },
          success: { iconTheme: { primary: '#22c55e', secondary: '#232b3e' } },
          error:   { iconTheme: { primary: '#ef4444', secondary: '#232b3e' } },
        }}
      />
    </QueryClientProvider>
  </BrowserRouter>,
)
