import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { RouterProvider } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { router } from './router'
import './styles/globals.css'

// Configure React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      gcTime: 1000 * 60 * 10, // 10 minutes (formerly cacheTime)
      refetchOnWindowFocus: false,
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
    mutations: {
      retry: 2,
    },
  },
})

// Platform-specific initialization
const initializePlatform = async () => {
  if (window.__TAURI__) {
    const { appWindow } = await import('@tauri-apps/api/window')
    const { listen } = await import('@tauri-apps/api/event')
    
    // Handle file drops
    await listen('tauri://file-drop', (event) => {
      console.log('Files dropped:', event.payload)
      window.dispatchEvent(new CustomEvent('files-dropped', { detail: event.payload }))
    })
    
    // Handle window close
    const { ask } = await import('@tauri-apps/api/dialog')
    await appWindow.onCloseRequested(async (event) => {
      const confirmed = await ask('Are you sure you want to quit? Any unsaved work will be lost.', {
        title: 'AirImpute Pro',
        type: 'warning'
      })
      if (!confirmed) {
        event.preventDefault()
      }
    })
    
    // Set up keyboard shortcuts
    const { register } = await import('@tauri-apps/api/globalShortcut')
    await register('CommandOrControl+O', () => {
      window.dispatchEvent(new CustomEvent('shortcut:open-file'))
    })
    await register('CommandOrControl+S', () => {
      window.dispatchEvent(new CustomEvent('shortcut:save'))
    })
    await register('CommandOrControl+Shift+S', () => {
      window.dispatchEvent(new CustomEvent('shortcut:save-as'))
    })
    await register('CommandOrControl+R', () => {
      window.dispatchEvent(new CustomEvent('shortcut:run-imputation'))
    })
  }
}

// Initialize platform and render app
initializePlatform().then(() => {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
        <Toaster
          position="bottom-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#1f2937',
              color: '#f9fafb',
              borderRadius: '0.5rem',
              padding: '1rem',
            },
            success: {
              iconTheme: {
                primary: '#10b981',
                secondary: '#f9fafb',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#f9fafb',
              },
            },
          }}
        />
        {process.env.NODE_ENV === 'development' && <ReactQueryDevtools />}
      </QueryClientProvider>
    </React.StrictMode>,
  )
})