import React, { useState } from 'react';
import { LineChart, BarChart3, ScatterChart, Grid3x3, Download } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { ScientificCard } from '@/components/layout/ScientificCard';
import { TimeSeriesChart } from '@/components/scientific/TimeSeriesChart';
import { CorrelationMatrix } from '@/components/scientific/CorrelationMatrix';
import { invoke } from '@tauri-apps/api/tauri';
import { useNavigate } from 'react-router-dom';
import { useStore } from '@/store';

type VisualizationType = 'timeseries' | 'distribution' | 'correlation' | 'heatmap' | 'uncertainty';

interface VisualizationOption {
  id: VisualizationType;
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}

const visualizationOptions: VisualizationOption[] = [
  {
    id: 'timeseries',
    name: 'Time Series',
    description: 'Compare original and imputed values over time',
    icon: LineChart
  },
  {
    id: 'distribution',
    name: 'Distribution',
    description: 'Statistical distribution comparison',
    icon: BarChart3
  },
  {
    id: 'correlation',
    name: 'Correlation Matrix',
    description: 'Variable correlation analysis',
    icon: ScatterChart
  },
  {
    id: 'heatmap',
    name: 'Missing Pattern Heatmap',
    description: 'Visualize missing data patterns',
    icon: Grid3x3
  },
  {
    id: 'uncertainty',
    name: 'Uncertainty Bands',
    description: 'Confidence intervals for imputed values',
    icon: LineChart
  }
];

const Visualization: React.FC = () => {
  const navigate = useNavigate();
  const { currentDataset, imputationResults } = useStore();
  const [selectedVisualization, setSelectedVisualization] = useState<VisualizationType>('timeseries');
  const [selectedVariables, setSelectedVariables] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const handleExportVisualization = async () => {
    try {
      const format = 'png'; // Could add format selection
      await invoke('export_visualization', {
        type: selectedVisualization,
        format,
        variables: selectedVariables
      });
      // Show success notification
    } catch (err) {
      console.error('Failed to export visualization:', err);
    }
  };

  const renderVisualization = () => {
    switch (selectedVisualization) {
      case 'timeseries':
        return (
          <TimeSeriesChart
            data={[]}
            showConfidenceIntervals={true}
            height={500}
          />
        );
      
      case 'distribution':
        return (
          <div className="h-[500px] flex items-center justify-center bg-gray-50 rounded">
            <p className="text-gray-500">Distribution plot will be rendered here</p>
          </div>
        );
      
      case 'correlation':
        return (
          <CorrelationMatrix
            data={{
              variables: selectedVariables,
              values: []
            }}
            colorScale="diverging"
            showValues={true}
          />
        );
      
      case 'heatmap':
        return (
          <div className="h-[500px] flex items-center justify-center bg-gray-50 rounded">
            <p className="text-gray-500">Missing pattern heatmap will be rendered here</p>
          </div>
        );
      
      case 'uncertainty':
        return (
          <div className="h-[500px] flex items-center justify-center bg-gray-50 rounded">
            <p className="text-gray-500">Uncertainty visualization will be rendered here</p>
          </div>
        );
      
      default:
        return null;
    }
  };

  if (!currentDataset || !imputationResults) {
    return (
      <div className="container mx-auto p-6">
        <Card className="p-8 text-center">
          <p className="text-gray-600 mb-4">No imputation results to visualize</p>
          <Button onClick={() => navigate('/data-import')}>
            Import Data
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Visualizations</h1>
        <Button onClick={handleExportVisualization}>
          <Download className="w-4 h-4 mr-2" />
          Export
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Visualization Type Selection */}
        <div className="lg:col-span-1">
          <ScientificCard
            title="Visualization Type"
            description="Select the type of visualization"
          >
            <div className="space-y-2">
              {visualizationOptions.map(option => (
                <button
                  key={option.id}
                  onClick={() => setSelectedVisualization(option.id)}
                  className={`
                    w-full text-left p-3 rounded-lg transition-colors
                    ${selectedVisualization === option.id
                      ? 'bg-blue-50 border-2 border-blue-500'
                      : 'border-2 border-gray-200 hover:border-gray-300'}
                  `}
                >
                  <div className="flex items-start">
                    <option.icon className="w-5 h-5 mr-3 mt-0.5 text-gray-600" />
                    <div>
                      <p className="font-medium">{option.name}</p>
                      <p className="text-sm text-gray-500">{option.description}</p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </ScientificCard>

          {/* Variable Selection */}
          <ScientificCard
            title="Variables"
            description="Select variables to visualize"
            className="mt-6"
          >
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {currentDataset.info.column_names.map(variable => (
                <label
                  key={variable}
                  className="flex items-center p-2 hover:bg-gray-50 rounded cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={selectedVariables.includes(variable)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedVariables([...selectedVariables, variable]);
                      } else {
                        setSelectedVariables(selectedVariables.filter(v => v !== variable));
                      }
                    }}
                    className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm">{variable}</span>
                </label>
              ))}
            </div>
          </ScientificCard>
        </div>

        {/* Visualization Display */}
        <div className="lg:col-span-3">
          <Card className="p-6">
            {loading ? (
              <div className="h-[500px] flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-gray-600">Generating visualization...</p>
                </div>
              </div>
            ) : (
              renderVisualization()
            )}
          </Card>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex justify-between mt-8">
        <Button
          variant="outline"
          onClick={() => navigate('/analysis')}
        >
          Back to Analysis
        </Button>
        <Button onClick={() => navigate('/export')}>
          Export Data
        </Button>
      </div>
    </div>
  );
};

export default Visualization;