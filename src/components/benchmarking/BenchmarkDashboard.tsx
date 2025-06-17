import React, { useState, useEffect, useMemo } from 'react';
import { listen } from '@tauri-apps/api/event';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription
} from '../ui/Card';
import { Button } from '../ui/Button';
import { Progress } from '../ui/Progress';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '../ui/Select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/Tabs';
import { Badge } from '../ui/Badge';
import { 
  BarChart, 
  ScatterChart,
  RadarChart,
  HeatmapChart,
  BoxPlotChart
} from './BenchmarkCharts';
import { StatisticalTestResults } from './StatisticalTestResults';
import { MetricSelector } from './MetricSelector';
import { DatasetManager } from './DatasetManager';
import { MethodComparison } from './MethodComparison';
import { ReproducibilityReport } from './ReproducibilityReport';
import { ExportPanel } from './ExportPanel';
import { 
  Play, 
  RefreshCw, 
  Cpu,
  Zap,
  Database,
  Award
} from 'lucide-react';
import { motion } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { invoke } from '@tauri-apps/api/tauri';

interface BenchmarkResult {
  methodName: string;
  datasetName: string;
  metrics: Record<string, number>;
  runtime: number;
  memoryUsage: number;
  parameters: Record<string, any>;
  timestamp: string;
  hardwareInfo: Record<string, string>;
}

interface BenchmarkDataset {
  name: string;
  description: string;
  size: number;
  missingRate: number;
  pattern: string;
  metadata: Record<string, any>;
}

interface BenchmarkMethod {
  name: string;
  category: string;
  hasGPUSupport: boolean;
  parameters: Record<string, any>;
  description: string;
}

export const BenchmarkDashboard: React.FC = () => {
  const queryClient = useQueryClient();
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [selectedMethods, setSelectedMethods] = useState<string[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string>('rmse');
  const [runningBenchmark, setRunningBenchmark] = useState(false);
  const [benchmarkProgress, setBenchmarkProgress] = useState(0);
  const [currentTask, setCurrentTask] = useState('');
  const [useGPU, setUseGPU] = useState(true);
  const [viewMode, setViewMode] = useState<'comparison' | 'detailed' | 'timeline'>('comparison');

  // Fetch available datasets
  const { data: datasets = [], isLoading: datasetsLoading } = useQuery({
    queryKey: ['benchmark-datasets'],
    queryFn: async () => {
      const result = await invoke<BenchmarkDataset[]>('get_benchmark_datasets');
      return result;
    }
  });

  // Fetch available methods
  const { data: methods = [], isLoading: methodsLoading } = useQuery({
    queryKey: ['benchmark-methods'],
    queryFn: async () => {
      const result = await invoke<BenchmarkMethod[]>('get_imputation_methods');
      return result;
    }
  });

  // Fetch benchmark results
  const { data: results = [], refetch: refetchResults } = useQuery({
    queryKey: ['benchmark-results', selectedDatasets, selectedMethods],
    queryFn: async () => {
      const result = await invoke<BenchmarkResult[]>('get_benchmark_results', {
        datasets: selectedDatasets,
        methods: selectedMethods
      });
      return result;
    },
    enabled: selectedDatasets.length > 0 && selectedMethods.length > 0
  });

  // Run benchmark mutation
  const runBenchmarkMutation = useMutation({
    mutationFn: async () => {
      setRunningBenchmark(true);
      setBenchmarkProgress(0);
      
      const result = await invoke<BenchmarkResult[]>('run_benchmark', {
        datasets: selectedDatasets,
        methods: selectedMethods,
        useGPU,
        cvSplits: 5,
        savePredictions: true
      });
      
      return result;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['benchmark-results'] });
      setRunningBenchmark(false);
      setBenchmarkProgress(100);
    },
    onError: (error) => {
      console.error('Benchmark failed:', error);
      setRunningBenchmark(false);
    }
  });

  // Listen to benchmark progress
  useEffect(() => {
    const unlisten = listen('benchmark-progress', (event: any) => {
      setBenchmarkProgress(event.payload.progress);
      setCurrentTask(event.payload.task);
    });

    return () => {
      unlisten.then(fn => fn());
    };
  }, []);

  // Calculate summary statistics
  const summaryStats = useMemo(() => {
    if (!results.length) return null;

    const methodStats = selectedMethods.map(method => {
      const methodResults = results.filter(r => r.methodName === method);
      const metrics = methodResults.map(r => r.metrics[selectedMetric] || 0);
      
      return {
        method,
        mean: metrics.reduce((a, b) => a + b, 0) / metrics.length,
        std: Math.sqrt(
          metrics.reduce((a, b) => a + Math.pow(b - metrics.reduce((x, y) => x + y, 0) / metrics.length, 2), 0) / metrics.length
        ),
        min: Math.min(...metrics),
        max: Math.max(...metrics),
        runtime: methodResults.reduce((a, b) => a + b.runtime, 0) / methodResults.length,
        memoryUsage: methodResults.reduce((a, b) => a + b.memoryUsage, 0) / methodResults.length
      };
    });

    return methodStats.sort((a, b) => a.mean - b.mean);
  }, [results, selectedMethods, selectedMetric]);

  // Prepare data for visualizations
  const chartData = useMemo(() => {
    if (!results.length) return null;

    return {
      comparison: {
        labels: selectedMethods,
        datasets: selectedDatasets.map(dataset => ({
          label: dataset,
          data: selectedMethods.map(method => {
            const result = results.find(
              r => r.methodName === method && r.datasetName === dataset
            );
            return result?.metrics[selectedMetric] || 0;
          })
        }))
      },
      timeline: results
        .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
        .map(r => ({
          x: new Date(r.timestamp),
          y: r.metrics[selectedMetric] || 0,
          method: r.methodName,
          dataset: r.datasetName
        })),
      heatmap: {
        methods: selectedMethods,
        datasets: selectedDatasets,
        values: selectedMethods.map(method =>
          selectedDatasets.map(dataset => {
            const result = results.find(
              r => r.methodName === method && r.datasetName === dataset
            );
            return result?.metrics[selectedMetric] || null;
          })
        )
      },
      performance: {
        runtime: selectedMethods.map(method => ({
          method,
          value: results
            .filter(r => r.methodName === method)
            .reduce((acc, r) => acc + r.runtime, 0) / 
            results.filter(r => r.methodName === method).length
        })),
        memory: selectedMethods.map(method => ({
          method,
          value: results
            .filter(r => r.methodName === method)
            .reduce((acc, r) => acc + r.memoryUsage, 0) / 
            results.filter(r => r.methodName === method).length
        }))
      }
    };
  }, [results, selectedMethods, selectedDatasets, selectedMetric]);

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Benchmark Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Compare and analyze imputation method performance
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => refetchResults()}
            disabled={runningBenchmark}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
          <ExportPanel results={results} />
        </div>
      </div>

      {/* Configuration Panel */}
      <Card>
        <CardHeader>
          <CardTitle>Benchmark Configuration</CardTitle>
          <CardDescription>
            Select datasets and methods to compare
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Dataset Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Datasets</label>
              <DatasetManager
                datasets={datasets}
                selectedDatasets={selectedDatasets}
                onSelectionChange={setSelectedDatasets}
                loading={datasetsLoading}
              />
            </div>

            {/* Method Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Methods</label>
              <MethodComparison
                methods={methods}
                selectedMethods={selectedMethods}
                onSelectionChange={setSelectedMethods}
                loading={methodsLoading}
              />
            </div>
          </div>

          {/* Metric Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Primary Metric</label>
            <MetricSelector
              value={selectedMetric}
              onChange={setSelectedMetric}
            />
          </div>

          {/* GPU Acceleration */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4" />
              <span className="text-sm font-medium">GPU Acceleration</span>
            </div>
            <Button
              variant={useGPU ? "default" : "outline"}
              size="sm"
              onClick={() => setUseGPU(!useGPU)}
            >
              {useGPU ? "Enabled" : "Disabled"}
            </Button>
          </div>

          {/* Run Benchmark Button */}
          <Button
            className="w-full"
            onClick={() => runBenchmarkMutation.mutate()}
            disabled={
              runningBenchmark || 
              selectedDatasets.length === 0 || 
              selectedMethods.length === 0
            }
          >
            {runningBenchmark ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Running Benchmark...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Run Benchmark
              </>
            )}
          </Button>

          {/* Progress Bar */}
          {runningBenchmark && (
            <div className="space-y-2">
              <Progress value={benchmarkProgress} />
              <p className="text-sm text-muted-foreground text-center">
                {currentTask || 'Initializing...'}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Results Section */}
      {results.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Summary Statistics */}
          {summaryStats && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Award className="w-5 h-5" />
                  Performance Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {summaryStats.map((stat, index) => (
                    <motion.div
                      key={stat.method}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.1 }}
                      className={`
                        p-4 rounded-lg border
                        ${index === 0 ? 'border-green-500 bg-green-50 dark:bg-green-950' : ''}
                      `}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-semibold">{stat.method}</h4>
                        {index === 0 && (
                          <Badge variant="success">Best</Badge>
                        )}
                      </div>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Mean {selectedMetric.toUpperCase()}:</span>
                          <span className="font-mono">{stat.mean.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Std Dev:</span>
                          <span className="font-mono">{stat.std.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Runtime:</span>
                          <span className="font-mono">{stat.runtime.toFixed(2)}s</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Memory:</span>
                          <span className="font-mono">{stat.memoryUsage.toFixed(0)}MB</span>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Visualization Tabs */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Visualizations</CardTitle>
                <Select value={viewMode} onValueChange={(v: any) => setViewMode(v)}>
                  <SelectTrigger className="w-[180px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="comparison">Comparison View</SelectItem>
                    <SelectItem value="detailed">Detailed Analysis</SelectItem>
                    <SelectItem value="timeline">Timeline View</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="grid w-full grid-cols-5">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="metrics">Metrics</TabsTrigger>
                  <TabsTrigger value="performance">Performance</TabsTrigger>
                  <TabsTrigger value="statistical">Statistical</TabsTrigger>
                  <TabsTrigger value="reproducibility">Reproducibility</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-4 mt-4">
                  {chartData && (
                    <>
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        <BarChart
                          data={chartData.comparison}
                          title="Method Comparison"
                          yAxisLabel={selectedMetric.toUpperCase()}
                        />
                        <HeatmapChart
                          data={chartData.heatmap}
                          title="Performance Heatmap"
                          colorScale="viridis"
                        />
                      </div>
                      <RadarChart
                        data={{
                          labels: ['RMSE', 'MAE', 'R²', 'Runtime', 'Memory'],
                          datasets: selectedMethods.map(method => ({
                            label: method,
                            data: [
                              1 - (summaryStats?.find(s => s.method === method)?.mean || 0) / 10,
                              1 - (results.find(r => r.methodName === method)?.metrics.mae || 0) / 10,
                              results.find(r => r.methodName === method)?.metrics.r2 || 0,
                              1 - (summaryStats?.find(s => s.method === method)?.runtime || 0) / 100,
                              1 - (summaryStats?.find(s => s.method === method)?.memoryUsage || 0) / 1000
                            ]
                          }))
                        }}
                        title="Multi-Metric Comparison"
                      />
                    </>
                  )}
                </TabsContent>

                <TabsContent value="metrics" className="space-y-4 mt-4">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {['rmse', 'mae', 'r2', 'mape'].map(metric => (
                      <BoxPlotChart
                        key={metric}
                        data={{
                          labels: selectedMethods,
                          datasets: [{
                            label: metric.toUpperCase(),
                            data: selectedMethods.map(method => 
                              results
                                .filter(r => r.methodName === method)
                                .map(r => r.metrics[metric] || 0)
                            )
                          }]
                        }}
                        title={`${metric.toUpperCase()} Distribution`}
                      />
                    ))}
                  </div>
                </TabsContent>

                <TabsContent value="performance" className="space-y-4 mt-4">
                  {chartData && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                      <BarChart
                        data={{
                          labels: chartData.performance.runtime.map(d => d.method),
                          datasets: [{
                            label: 'Runtime (seconds)',
                            data: chartData.performance.runtime.map(d => d.value),
                            backgroundColor: 'rgba(59, 130, 246, 0.5)'
                          }]
                        }}
                        title="Runtime Comparison"
                        yAxisLabel="Seconds"
                      />
                      <BarChart
                        data={{
                          labels: chartData.performance.memory.map(d => d.method),
                          datasets: [{
                            label: 'Memory Usage (MB)',
                            data: chartData.performance.memory.map(d => d.value),
                            backgroundColor: 'rgba(236, 72, 153, 0.5)'
                          }]
                        }}
                        title="Memory Usage Comparison"
                        yAxisLabel="MB"
                      />
                    </div>
                  )}
                  <ScatterChart
                    data={{
                      datasets: selectedMethods.map(method => ({
                        label: method,
                        data: results
                          .filter(r => r.methodName === method)
                          .map(r => ({
                            x: r.runtime,
                            y: r.metrics[selectedMetric] || 0
                          }))
                      }))
                    }}
                    title={`Runtime vs ${selectedMetric.toUpperCase()}`}
                    xAxisLabel="Runtime (seconds)"
                    yAxisLabel={selectedMetric.toUpperCase()}
                  />
                </TabsContent>

                <TabsContent value="statistical" className="mt-4">
                  <StatisticalTestResults
                    results={results}
                    methods={selectedMethods}
                    metric={selectedMetric}
                  />
                </TabsContent>

                <TabsContent value="reproducibility" className="mt-4">
                  <ReproducibilityReport
                    results={results}
                    datasets={datasets}
                    methods={methods}
                  />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Hardware Information */}
          {results[0]?.hardwareInfo && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Cpu className="w-5 h-5" />
                  Hardware Configuration
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(results[0].hardwareInfo).map(([key, value]) => (
                    <div key={key} className="space-y-1">
                      <p className="text-sm text-muted-foreground capitalize">
                        {key.replace(/_/g, ' ')}
                      </p>
                      <p className="font-mono text-sm">{value}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </motion.div>
      )}

      {/* Empty State */}
      {results.length === 0 && !runningBenchmark && (
        <Card className="p-12">
          <div className="text-center space-y-4">
            <Database className="w-12 h-12 mx-auto text-muted-foreground" />
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">No Benchmark Results</h3>
              <p className="text-sm text-muted-foreground max-w-md mx-auto">
                Select datasets and methods, then run the benchmark to see performance comparisons.
              </p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};