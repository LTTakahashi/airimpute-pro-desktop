import { useEffect, useState } from 'react'
import { Outlet } from 'react-router-dom'
import { invoke } from '@tauri-apps/api/tauri'
import { listen } from '@tauri-apps/api/event'
import { useStore } from './store'
import Sidebar from './components/layout/Sidebar'
import Header from './components/layout/Header'
import StatusBar from './components/layout/StatusBar'
import LoadingScreen from './components/common/LoadingScreen'
import { ErrorBoundary } from './components/feedback/ErrorBoundary'
import useKeyboardShortcuts from './hooks/useKeyboardShortcuts'
import useAutoSave from './hooks/useAutoSave'
import { cn } from './utils/cn'

export function App(): JSX.Element {
  const [isInitialized, setIsInitialized] = useState(false)
  const [initError, setInitError] = useState<string | null>(null)
  const { theme, setPythonReady } = useStore()
  
  // Initialize keyboard shortcuts
  useKeyboardShortcuts()
  
  // Initialize auto-save
  useAutoSave()
  
  // Initialize application
  useEffect(() => {
    const cleanupFns: Array<() => void> = []
    
    const initialize = async (): Promise<void> => {
      try {
        // Check Python runtime
        try {
          const pythonStatus = await invoke<{ healthy: boolean; version: string }>('check_python_runtime')
          setPythonReady(pythonStatus.healthy)
          if (!pythonStatus.healthy) {
            setInitError('Python runtime is not available. Some features may not work.')
          }
        } catch (err) {
          console.error('Python check failed:', err)
          setPythonReady(false)
        }
        
        // Get system info
        try {
          const systemInfo = await invoke<{ platform: string; version: string; memory: number }>('get_system_info')
          console.log('System info:', systemInfo)
        } catch (err) {
          console.error('Failed to get system info:', err)
        }
        
        // Load user preferences
        try {
          const preferences = await invoke<Record<string, unknown>>('get_user_preferences')
          useStore.getState().updatePreferences(preferences)
        } catch (err) {
          console.error('Failed to load preferences:', err)
        }
        
        // Set up event listeners
        const unlistenLowMemory = await listen('low-memory-warning', (event) => {
          console.warn('Low memory warning:', event.payload)
          // Show user notification
        })
        
        const unlistenProgress = await listen('imputation-progress', (event) => {
          console.log('Imputation progress:', event.payload)
          // Update progress in UI
        })
        
        cleanupFns.push(unlistenLowMemory, unlistenProgress)
        setIsInitialized(true)
      } catch (error) {
        console.error('Initialization error:', error)
        setInitError(error instanceof Error ? error.message : 'Unknown error occurred')
        setIsInitialized(true) // Allow app to load with limited functionality
      }
    }
    
    initialize()
    
    return () => {
      cleanupFns.forEach(fn => fn())
    }
  }, [setPythonReady])
  
  if (!isInitialized) {
    return <LoadingScreen message="Initializing AirImpute Pro..." />
  }
  
  return (
    <ErrorBoundary>
      <div className={cn('flex h-screen overflow-hidden', theme === 'dark' && 'dark')}>
        {/* Sidebar */}
        <Sidebar />
        
        {/* Main Content */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Header */}
          <Header />
          
          {/* Page Content */}
          <main className="flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-900">
            {initError && (
              <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-400 p-4 m-4">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm text-yellow-700 dark:text-yellow-300">
                      {initError}
                    </p>
                  </div>
                </div>
              </div>
            )}
            <Outlet />
          </main>
          
          {/* Status Bar */}
          <StatusBar />
        </div>
      </div>
    </ErrorBoundary>
  )
}