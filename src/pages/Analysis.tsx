import React, { useState, useEffect } from 'react';
import { BarChart, TrendingUp, FileText, Download } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { ScientificCard } from '@/components/layout/ScientificCard';
import { TimeSeriesChart } from '@/components/scientific/TimeSeriesChart';
import { CorrelationMatrix } from '@/components/scientific/CorrelationMatrix';
import { invoke } from '@tauri-apps/api/tauri';
import { useNavigate } from 'react-router-dom';
import { useStore } from '@/store';

interface AnalysisResult {
  mae: number;
  rmse: number;
  r2: number;
  mape: number;
  coverage: number;
  time_taken: number;
  memory_used: number;
}

interface QualityMetrics {
  variance_preserved: number;
  distribution_similarity: number;
  temporal_consistency: number;
  spatial_coherence: number;
}

const Analysis: React.FC = () => {
  const navigate = useNavigate();
  const { currentDataset, imputationResults } = useStore();
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'timeseries' | 'correlation' | 'quality'>('overview');

  useEffect(() => {
    if (!currentDataset || !imputationResults) {
      navigate('/data-import');
      return;
    }

    loadAnalysis();
  }, [currentDataset, imputationResults, navigate]);

  const loadAnalysis = async () => {
    try {
      setLoading(true);
      
      // Load analysis results
      const result = await invoke<AnalysisResult>('validate_imputation_results', {
        datasetId: currentDataset?.id,
        imputationId: imputationResults?.id
      });
      setAnalysisResult(result);

      // Load quality metrics
      const metrics = await invoke<QualityMetrics>('generate_quality_report', {
        datasetId: currentDataset?.id,
        imputationId: imputationResults?.id
      });
      setQualityMetrics(metrics);
    } catch (err) {
      console.error('Failed to load analysis:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    try {
      await invoke('generate_latex_report', {
        datasetId: currentDataset?.id,
        imputationId: imputationResults?.id,
        outputPath: `${currentDataset?.info.name}_analysis_report.tex`
      });
      // Show success notification
    } catch (err) {
      console.error('Failed to export report:', err);
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto p-6 flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Analyzing results...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Analysis Results</h1>
        <Button onClick={handleExport}>
          <Download className="w-4 h-4 mr-2" />
          Export Report
        </Button>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart },
            { id: 'timeseries', label: 'Time Series', icon: TrendingUp },
            { id: 'correlation', label: 'Correlation', icon: BarChart },
            { id: 'quality', label: 'Quality Metrics', icon: FileText }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`
                py-2 px-1 border-b-2 font-medium text-sm flex items-center
                ${activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}
              `}
            >
              <tab.icon className="w-4 h-4 mr-2" />
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && analysisResult && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <ScientificCard
            title="MAE"
            description="Mean Absolute Error"
          >
            <p className="text-3xl font-bold text-blue-600">
              {analysisResult.mae.toFixed(3)}
            </p>
            <p className="text-sm text-gray-500 mt-1">Lower is better</p>
          </ScientificCard>

          <ScientificCard
            title="RMSE"
            description="Root Mean Square Error"
          >
            <p className="text-3xl font-bold text-blue-600">
              {analysisResult.rmse.toFixed(3)}
            </p>
            <p className="text-sm text-gray-500 mt-1">Lower is better</p>
          </ScientificCard>

          <ScientificCard
            title="R²"
            description="Coefficient of Determination"
          >
            <p className="text-3xl font-bold text-green-600">
              {analysisResult.r2.toFixed(3)}
            </p>
            <p className="text-sm text-gray-500 mt-1">Higher is better</p>
          </ScientificCard>

          <ScientificCard
            title="MAPE"
            description="Mean Absolute Percentage Error"
          >
            <p className="text-3xl font-bold text-blue-600">
              {analysisResult.mape.toFixed(1)}%
            </p>
            <p className="text-sm text-gray-500 mt-1">Lower is better</p>
          </ScientificCard>

          <ScientificCard
            title="Coverage"
            description="Data points successfully imputed"
          >
            <p className="text-3xl font-bold text-green-600">
              {analysisResult.coverage.toFixed(1)}%
            </p>
          </ScientificCard>

          <ScientificCard
            title="Processing Time"
            description="Time taken for imputation"
          >
            <p className="text-3xl font-bold text-gray-600">
              {analysisResult.time_taken.toFixed(2)}s
            </p>
          </ScientificCard>

          <ScientificCard
            title="Memory Used"
            description="Peak memory consumption"
          >
            <p className="text-3xl font-bold text-gray-600">
              {(analysisResult.memory_used / 1024 / 1024).toFixed(1)} MB
            </p>
          </ScientificCard>
        </div>
      )}

      {activeTab === 'timeseries' && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Time Series Comparison</h2>
          <TimeSeriesChart
            data={[]}
            showConfidenceIntervals={true}
            height={400}
          />
        </Card>
      )}

      {activeTab === 'correlation' && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Correlation Analysis</h2>
          <CorrelationMatrix
            data={{
              variables: [],
              values: []
            }}
            colorScale="diverging"
            showValues={true}
          />
        </Card>
      )}

      {activeTab === 'quality' && qualityMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <ScientificCard
            title="Variance Preservation"
            description="How well the imputation preserves data variance"
          >
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Score</span>
                <span className="font-bold">
                  {(qualityMetrics.variance_preserved * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-green-500 h-2 rounded-full"
                  style={{ width: `${qualityMetrics.variance_preserved * 100}%` }}
                />
              </div>
            </div>
          </ScientificCard>

          <ScientificCard
            title="Distribution Similarity"
            description="Statistical similarity to original distribution"
          >
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Score</span>
                <span className="font-bold">
                  {(qualityMetrics.distribution_similarity * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full"
                  style={{ width: `${qualityMetrics.distribution_similarity * 100}%` }}
                />
              </div>
            </div>
          </ScientificCard>

          <ScientificCard
            title="Temporal Consistency"
            description="Preservation of time-based patterns"
          >
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Score</span>
                <span className="font-bold">
                  {(qualityMetrics.temporal_consistency * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-purple-500 h-2 rounded-full"
                  style={{ width: `${qualityMetrics.temporal_consistency * 100}%` }}
                />
              </div>
            </div>
          </ScientificCard>

          <ScientificCard
            title="Spatial Coherence"
            description="Consistency across spatial dimensions"
          >
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Score</span>
                <span className="font-bold">
                  {(qualityMetrics.spatial_coherence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-orange-500 h-2 rounded-full"
                  style={{ width: `${qualityMetrics.spatial_coherence * 100}%` }}
                />
              </div>
            </div>
          </ScientificCard>
        </div>
      )}

      {/* Navigation */}
      <div className="flex justify-between mt-8">
        <Button
          variant="outline"
          onClick={() => navigate('/imputation')}
        >
          Back to Imputation
        </Button>
        <div className="space-x-3">
          <Button
            variant="outline"
            onClick={() => navigate('/visualization')}
          >
            Visualizations
          </Button>
          <Button onClick={() => navigate('/export')}>
            Export Data
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Analysis;