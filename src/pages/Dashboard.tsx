import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useStore } from '@/store'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Progress } from '@/components/ui/Progress'
import { 
  File as FileIcon, 
  Play as PlayIcon, 
  BarChart3 as ChartBarIcon, 
  Download as CloudDownloadIcon,
  Cpu as CpuChipIcon,
  Clock as ClockIcon,
  CheckCircle as CheckCircleIcon,
  XCircle as XCircleIcon
} from 'lucide-react'
import { formatBytes, formatDuration } from '@/utils/format'
import { DebugPanel } from '@/components/debug/DebugPanel'

export function Dashboard() {
  const navigate = useNavigate()
  const { 
    datasets, 
    imputationJobs, 
    memoryUsage,
    pythonReady,
    activeDatasetId,
  } = useStore()
  
  const [stats, setStats] = useState({
    totalDatasets: 0,
    totalImputations: 0,
    successRate: 0,
    averageTime: 0,
  })
  
  // Calculate statistics
  useEffect(() => {
    const completedJobs = imputationJobs.filter(job => job.status === 'completed')
    const failedJobs = imputationJobs.filter(job => job.status === 'failed')
    
    const totalTime = completedJobs.reduce((acc, job) => {
      if (job.endTime) {
        return acc + (job.endTime.getTime() - job.startTime.getTime())
      }
      return acc
    }, 0)
    
    setStats({
      totalDatasets: datasets.length,
      totalImputations: imputationJobs.length,
      successRate: imputationJobs.length > 0 
        ? (completedJobs.length / (completedJobs.length + failedJobs.length)) * 100 
        : 0,
      averageTime: completedJobs.length > 0 ? totalTime / completedJobs.length : 0,
    })
  }, [datasets, imputationJobs])
  
  // Get recent jobs
  const recentJobs = imputationJobs
    .sort((a, b) => b.startTime.getTime() - a.startTime.getTime())
    .slice(0, 5)
  
  // Get active job
  const activeJob = imputationJobs.find(job => job.status === 'running')
  
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Dashboard
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Welcome to AirImpute Pro - Academic-grade air quality data imputation
          </p>
        </div>
        <div className="flex gap-3">
          <Button
            variant="outline"
            onClick={() => navigate('/import')}
            className="flex items-center gap-2"
          >
            <FileIcon className="h-4 w-4" />
            Import Data
          </Button>
          <Button
            onClick={() => navigate('/imputation')}
            disabled={!activeDatasetId || !pythonReady}
            className="flex items-center gap-2"
          >
            <PlayIcon className="h-4 w-4" />
            Run Imputation
          </Button>
        </div>
      </div>
      
      {/* Python Status Alert */}
      {!pythonReady && (
        <Card className="border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20">
          <div className="flex items-center gap-3 p-4">
            <CpuChipIcon className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />
            <div className="flex-1">
              <h3 className="font-medium text-yellow-900 dark:text-yellow-100">
                Python Runtime Initializing
              </h3>
              <p className="text-sm text-yellow-700 dark:text-yellow-300">
                Some features may be unavailable until the Python runtime is ready.
              </p>
            </div>
          </div>
        </Card>
      )}
      
      {/* Statistics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <div className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Total Datasets
                </p>
                <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">
                  {stats.totalDatasets}
                </p>
              </div>
              <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                <FileIcon className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
            </div>
          </div>
        </Card>
        
        <Card>
          <div className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Total Imputations
                </p>
                <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">
                  {stats.totalImputations}
                </p>
              </div>
              <div className="p-3 bg-green-100 dark:bg-green-900/20 rounded-lg">
                <PlayIcon className="h-6 w-6 text-green-600 dark:text-green-400" />
              </div>
            </div>
          </div>
        </Card>
        
        <Card>
          <div className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Success Rate
                </p>
                <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">
                  {stats.successRate.toFixed(1)}%
                </p>
              </div>
              <div className="p-3 bg-purple-100 dark:bg-purple-900/20 rounded-lg">
                <ChartBarIcon className="h-6 w-6 text-purple-600 dark:text-purple-400" />
              </div>
            </div>
          </div>
        </Card>
        
        <Card>
          <div className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Avg. Time
                </p>
                <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">
                  {formatDuration(stats.averageTime)}
                </p>
              </div>
              <div className="p-3 bg-orange-100 dark:bg-orange-900/20 rounded-lg">
                <ClockIcon className="h-6 w-6 text-orange-600 dark:text-orange-400" />
              </div>
            </div>
          </div>
        </Card>
      </div>
      
      {/* Active Job */}
      {activeJob && (
        <Card>
          <div className="p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Active Imputation
            </h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {activeJob.method.toUpperCase()} Method
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Dataset: {datasets.find(d => d.id === activeJob.datasetId)?.name}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {activeJob.progress}%
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Progress
                  </p>
                </div>
              </div>
              <Progress value={activeJob.progress} className="h-3" />
              <div className="flex justify-between text-sm text-gray-500 dark:text-gray-400">
                <span>Started: {activeJob.startTime.toLocaleTimeString()}</span>
                <span>
                  Elapsed: {formatDuration(Date.now() - activeJob.startTime.getTime())}
                </span>
              </div>
            </div>
          </div>
        </Card>
      )}
      
      {/* Recent Activity and Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Activity */}
        <Card>
          <div className="p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Recent Activity
            </h2>
            <div className="space-y-3">
              {recentJobs.length > 0 ? (
                recentJobs.map((job) => {
                  const dataset = datasets.find(d => d.id === job.datasetId)
                  return (
                    <div
                      key={job.id}
                      className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors cursor-pointer"
                      onClick={() => navigate(`/imputation?job=${job.id}`)}
                    >
                      <div className="flex items-center gap-3">
                        {job.status === 'completed' ? (
                          <CheckCircleIcon className="h-5 w-5 text-green-500" />
                        ) : job.status === 'failed' ? (
                          <XCircleIcon className="h-5 w-5 text-red-500" />
                        ) : job.status === 'running' ? (
                          <div className="h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                        ) : (
                          <ClockIcon className="h-5 w-5 text-gray-400" />
                        )}
                        <div>
                          <p className="font-medium text-gray-900 dark:text-white">
                            {job.method.toUpperCase()}
                          </p>
                          <p className="text-sm text-gray-500 dark:text-gray-400">
                            {dataset?.name || 'Unknown dataset'}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          {job.status === 'running' 
                            ? `${job.progress}%`
                            : job.status.charAt(0).toUpperCase() + job.status.slice(1)
                          }
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {job.startTime.toLocaleString()}
                        </p>
                      </div>
                    </div>
                  )
                })
              ) : (
                <p className="text-center text-gray-500 dark:text-gray-400 py-8">
                  No recent activity
                </p>
              )}
            </div>
          </div>
        </Card>
        
        {/* Quick Actions */}
        <Card>
          <div className="p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Quick Actions
            </h2>
            <div className="grid grid-cols-2 gap-3">
              <Button
                variant="outline"
                className="h-24 flex-col gap-2"
                onClick={() => navigate('/import')}
              >
                <FileIcon className="h-6 w-6" />
                <span>Import Dataset</span>
              </Button>
              <Button
                variant="outline"
                className="h-24 flex-col gap-2"
                onClick={() => navigate('/analysis')}
                disabled={!activeDatasetId}
              >
                <ChartBarIcon className="h-6 w-6" />
                <span>Analyze Data</span>
              </Button>
              <Button
                variant="outline"
                className="h-24 flex-col gap-2"
                onClick={() => navigate('/visualization')}
                disabled={!activeDatasetId}
              >
                <ChartBarIcon className="h-6 w-6" />
                <span>Visualize</span>
              </Button>
              <Button
                variant="outline"
                className="h-24 flex-col gap-2"
                onClick={() => navigate('/export')}
                disabled={!activeDatasetId}
              >
                <CloudDownloadIcon className="h-6 w-6" />
                <span>Export Results</span>
              </Button>
            </div>
          </div>
        </Card>
      </div>
      
      {/* Memory Usage */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              System Resources
            </h2>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {formatBytes(memoryUsage.used)} / {formatBytes(memoryUsage.total)}
            </span>
          </div>
          <Progress value={memoryUsage.percentage} className="h-3" />
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            Memory usage: {memoryUsage.percentage.toFixed(1)}%
          </p>
        </div>
      </Card>
      
      {/* Debug Panel - Only show in development */}
      {process.env.NODE_ENV === 'development' && (
        <DebugPanel />
      )}
    </div>
  )
}