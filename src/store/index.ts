import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'

export interface DatasetInfo {
  name: string;
  rows: number;
  columns: number;
  missing_count: number;
  missing_percentage: number;
  file_size: number;
  column_names: string[];
}

export interface Dataset {
  id: string
  name: string
  path: string
  info: DatasetInfo
  size?: number
  rows?: number
  columns?: number
  missingPercentage?: number
  lastModified?: Date
  metadata?: Record<string, any>
}

export interface ImputationResult {
  id: string;
  method: string;
  imputedCount?: number;
}

export interface ImputationJob {
  id: string
  datasetId: string
  method: string
  parameters: Record<string, any>
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  startTime: Date
  endTime?: Date
  error?: string
  results?: {
    imputedDataPath: string
    metrics: Record<string, number>
    confidenceIntervals?: any
  }
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system'
  autoSave: boolean
  autoSaveInterval: number // minutes
  defaultImputationMethod: string
  enableNotifications: boolean
  enableSounds: boolean
  defaultExportFormat: 'csv' | 'excel' | 'netcdf' | 'hdf5'
  workingDirectory: string
}

interface AppState {
  // UI State
  theme: 'light' | 'dark' | 'system'
  sidebarCollapsed: boolean
  activeView: string
  
  // Data State
  datasets: Dataset[]
  activeDatasetId: string | null
  currentDataset: Dataset | null
  imputationResults: ImputationResult | null
  recentProjects: string[]
  
  // Imputation State
  imputationJobs: ImputationJob[]
  activeJobId: string | null
  
  // User Preferences
  preferences: UserPreferences
  
  // System State
  pythonReady: boolean
  memoryUsage: {
    used: number
    total: number
    percentage: number
  }
  
  // Actions
  setTheme: (theme: 'light' | 'dark' | 'system') => void
  toggleSidebar: () => void
  setSidebarCollapsed: (collapsed: boolean) => void
  setActiveView: (view: string) => void
  
  // Dataset Actions
  addDataset: (dataset: Dataset) => void
  removeDataset: (id: string) => void
  setActiveDataset: (id: string | null) => void
  setCurrentDataset: (dataset: Dataset | null) => void
  updateDataset: (id: string, updates: Partial<Dataset>) => void
  
  // Imputation Actions
  addImputationJob: (job: ImputationJob) => void
  updateImputationJob: (id: string, updates: Partial<ImputationJob>) => void
  removeImputationJob: (id: string) => void
  setActiveJob: (id: string | null) => void
  setImputationResults: (results: ImputationResult | null) => void
  
  // Preference Actions
  updatePreferences: (preferences: Partial<UserPreferences>) => void
  
  // System Actions
  setPythonReady: (ready: boolean) => void
  updateMemoryUsage: (usage: AppState['memoryUsage']) => void
  
  // Project Actions
  addRecentProject: (path: string) => void
  clearRecentProjects: () => void
}

const defaultPreferences: UserPreferences = {
  theme: 'system',
  autoSave: true,
  autoSaveInterval: 5,
  defaultImputationMethod: 'rah',
  enableNotifications: true,
  enableSounds: false,
  defaultExportFormat: 'csv',
  workingDirectory: '',
}

export type RootState = AppState

export const useStore = create<AppState>()(
  devtools(
    persist(
      immer((set) => ({
        // Initial UI State
        theme: 'light',
        sidebarCollapsed: false,
        activeView: 'dashboard',
        
        // Initial Data State
        datasets: [],
        activeDatasetId: null,
        currentDataset: null,
        imputationResults: null,
        recentProjects: [],
        
        // Initial Imputation State
        imputationJobs: [],
        activeJobId: null,
        
        // Initial Preferences
        preferences: defaultPreferences,
        
        // Initial System State
        pythonReady: false,
        memoryUsage: {
          used: 0,
          total: 0,
          percentage: 0,
        },
        
        // UI Actions
        setTheme: (theme) =>
          set((state) => {
            state.theme = theme
          }),
          
        toggleSidebar: () =>
          set((state) => {
            state.sidebarCollapsed = !state.sidebarCollapsed
          }),
          
        setSidebarCollapsed: (collapsed) =>
          set((state) => {
            state.sidebarCollapsed = collapsed
          }),
          
        setActiveView: (view) =>
          set((state) => {
            state.activeView = view
          }),
        
        // Dataset Actions
        addDataset: (dataset) =>
          set((state) => {
            state.datasets.push(dataset)
          }),
          
        removeDataset: (id) =>
          set((state) => {
            state.datasets = state.datasets.filter((d) => d.id !== id)
            if (state.activeDatasetId === id) {
              state.activeDatasetId = null
            }
          }),
          
        setActiveDataset: (id) =>
          set((state) => {
            state.activeDatasetId = id
          }),
          
        setCurrentDataset: (dataset) =>
          set((state) => {
            state.currentDataset = dataset
          }),
          
        updateDataset: (id, updates) =>
          set((state) => {
            const index = state.datasets.findIndex((d) => d.id === id)
            if (index !== -1) {
              state.datasets[index] = { ...state.datasets[index], ...updates }
            }
          }),
        
        // Imputation Actions
        addImputationJob: (job) =>
          set((state) => {
            state.imputationJobs.push(job)
          }),
          
        updateImputationJob: (id, updates) =>
          set((state) => {
            const index = state.imputationJobs.findIndex((j) => j.id === id)
            if (index !== -1) {
              state.imputationJobs[index] = { ...state.imputationJobs[index], ...updates }
            }
          }),
          
        removeImputationJob: (id) =>
          set((state) => {
            state.imputationJobs = state.imputationJobs.filter((j) => j.id !== id)
            if (state.activeJobId === id) {
              state.activeJobId = null
            }
          }),
          
        setActiveJob: (id) =>
          set((state) => {
            state.activeJobId = id
          }),
          
        setImputationResults: (results) =>
          set((state) => {
            state.imputationResults = results
          }),
        
        // Preference Actions
        updatePreferences: (preferences) =>
          set((state) => {
            state.preferences = { ...state.preferences, ...preferences }
          }),
        
        // System Actions
        setPythonReady: (ready) =>
          set((state) => {
            state.pythonReady = ready
          }),
          
        updateMemoryUsage: (usage) =>
          set((state) => {
            state.memoryUsage = usage
          }),
        
        // Project Actions
        addRecentProject: (path) =>
          set((state) => {
            state.recentProjects = [
              path,
              ...state.recentProjects.filter((p) => p !== path),
            ].slice(0, 10)
          }),
          
        clearRecentProjects: () =>
          set((state) => {
            state.recentProjects = []
          }),
      })),
      {
        name: 'airimpute-pro-storage',
        partialize: (state) => ({
          theme: state.theme,
          sidebarCollapsed: state.sidebarCollapsed,
          preferences: state.preferences,
          recentProjects: state.recentProjects,
        }),
      }
    )
  )
)