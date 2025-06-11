import React, { useState, useEffect } from 'react';
import { Lightbulb, TrendingUp, Brain, Zap, BarChart } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

interface MethodRecommendation {
  method_id: string;
  score: number;
  reasons: string[];
  pros: string[];
  cons: string[];
  estimated_accuracy: string;
  estimated_time: string;
}

interface DataCharacteristics {
  missing_pattern: 'random' | 'sequential' | 'seasonal' | 'mixed';
  data_size: 'small' | 'medium' | 'large';
  missing_percentage: number;
  has_temporal: boolean;
  has_spatial: boolean;
  is_multivariate: boolean;
  seasonality_detected: boolean;
  trend_detected: boolean;
  column_correlations: number;
}

interface MethodRecommendationProps {
  datasetId: string;
  availableMethods: Array<{
    id: string;
    name: string;
    category: string;
    complexity: string;
  }>;
  onSelectMethod: (methodId: string) => void;
}

export const MethodRecommendation: React.FC<MethodRecommendationProps> = ({
  datasetId,
  availableMethods,
  onSelectMethod,
}) => {
  const [recommendations, setRecommendations] = useState<MethodRecommendation[]>([]);
  const [characteristics, setCharacteristics] = useState<DataCharacteristics | null>(null);
  const [loading, setLoading] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    analyzeDataset();
  }, [datasetId]);

  const analyzeDataset = async () => {
    setLoading(true);
    try {
      // In a real implementation, this would call the backend
      // For now, we'll use mock data based on the dataset
      const mockCharacteristics: DataCharacteristics = {
        missing_pattern: 'random',
        data_size: 'medium',
        missing_percentage: 15,
        has_temporal: true,
        has_spatial: false,
        is_multivariate: true,
        seasonality_detected: true,
        trend_detected: false,
        column_correlations: 0.7,
      };
      
      setCharacteristics(mockCharacteristics);
      generateRecommendations(mockCharacteristics);
    } catch (error) {
      console.error('Failed to analyze dataset:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateRecommendations = (chars: DataCharacteristics) => {
    const recommendations: MethodRecommendation[] = [];

    // Score each method based on characteristics
    availableMethods.forEach(method => {
      let score = 0;
      const reasons: string[] = [];
      const pros: string[] = [];
      const cons: string[] = [];

      // Simple methods for small missing percentages
      if (chars.missing_percentage < 10 && ['mean', 'median', 'forward_fill'].includes(method.id)) {
        score += 30;
        reasons.push('Good for small amount of missing data');
        pros.push('Fast and simple');
        pros.push('Preserves general statistics');
      }

      // Linear interpolation for temporal data with regular gaps
      if (method.id === 'linear' && chars.has_temporal) {
        score += 40;
        reasons.push('Suitable for time series data');
        pros.push('Preserves trends');
        pros.push('No parameters to tune');
        if (chars.missing_percentage > 30) {
          cons.push('May oversimplify complex patterns');
        }
      }

      // Spline for smooth data
      if (method.id === 'spline' && chars.has_temporal && !chars.seasonality_detected) {
        score += 35;
        reasons.push('Good for smooth temporal patterns');
        pros.push('Smooth interpolation');
        cons.push('Can overfit with large gaps');
      }

      // Random Forest for complex patterns
      if (method.id === 'random_forest' && chars.is_multivariate && chars.column_correlations > 0.5) {
        score += 50;
        reasons.push('Captures complex relationships between variables');
        pros.push('Handles non-linear patterns');
        pros.push('Robust to outliers');
        cons.push('Slower than simple methods');
        cons.push('Requires tuning');
      }

      // LSTM for sequential patterns
      if (method.id === 'lstm' && chars.has_temporal && chars.data_size !== 'small') {
        score += 45;
        reasons.push('Excellent for sequential dependencies');
        pros.push('Captures long-term dependencies');
        pros.push('State-of-the-art for time series');
        cons.push('Requires GPU for good performance');
        cons.push('Needs large datasets');
      }

      // Adjust scores based on data size
      if (chars.data_size === 'small' && ['lstm', 'random_forest'].includes(method.id)) {
        score -= 20;
        cons.push('May overfit on small datasets');
      }

      // Estimate accuracy and time
      const estimated_accuracy = getEstimatedAccuracy(method.id, chars);
      const estimated_time = getEstimatedTime(method.id, chars);

      if (score > 0) {
        recommendations.push({
          method_id: method.id,
          score,
          reasons,
          pros,
          cons,
          estimated_accuracy,
          estimated_time,
        });
      }
    });

    // Sort by score and take top 3
    recommendations.sort((a, b) => b.score - a.score);
    setRecommendations(recommendations.slice(0, 3));
  };

  const getEstimatedAccuracy = (methodId: string, chars: DataCharacteristics): string => {
    const baseAccuracy = {
      mean: 70,
      median: 72,
      forward_fill: 75,
      linear: 80,
      spline: 82,
      random_forest: 88,
      lstm: 90,
    };

    let accuracy = baseAccuracy[methodId as keyof typeof baseAccuracy] || 75;

    // Adjust based on characteristics
    if (chars.missing_percentage > 30) accuracy -= 10;
    if (chars.missing_pattern === 'seasonal' && ['lstm', 'random_forest'].includes(methodId)) accuracy += 5;
    if (chars.is_multivariate && ['random_forest', 'lstm'].includes(methodId)) accuracy += 5;

    return `~${Math.min(95, Math.max(60, accuracy))}%`;
  };

  const getEstimatedTime = (methodId: string, chars: DataCharacteristics): string => {
    const timeMultiplier = {
      small: 1,
      medium: 10,
      large: 100,
    };

    const baseTime = {
      mean: 0.1,
      median: 0.1,
      forward_fill: 0.2,
      linear: 0.5,
      spline: 1,
      random_forest: 5,
      lstm: 20,
    };

    const time = (baseTime[methodId as keyof typeof baseTime] || 1) * timeMultiplier[chars.data_size];

    if (time < 1) return '<1s';
    if (time < 60) return `${Math.round(time)}s`;
    return `${Math.round(time / 60)}min`;
  };

  const getMethodIcon = (methodId: string) => {
    switch (methodId) {
      case 'mean':
      case 'median':
        return <BarChart className="w-4 h-4" />;
      case 'linear':
      case 'spline':
        return <TrendingUp className="w-4 h-4" />;
      case 'random_forest':
      case 'lstm':
        return <Brain className="w-4 h-4" />;
      default:
        return <Zap className="w-4 h-4" />;
    }
  };

  const getMethodInfo = (methodId: string) => {
    return availableMethods.find(m => m.id === methodId);
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="flex items-center space-x-2">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500" />
          <span className="text-sm text-gray-500">Analyzing dataset characteristics...</span>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center">
          <Lightbulb className="w-5 h-5 mr-2 text-yellow-500" />
          Method Recommendations
        </h3>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? 'Hide' : 'Show'} Details
        </Button>
      </div>

      {characteristics && (
        <div className="mb-4 p-3 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600 mb-2">Dataset Characteristics:</p>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">
              {characteristics.missing_percentage}% missing
            </Badge>
            <Badge variant="outline">
              {characteristics.missing_pattern} pattern
            </Badge>
            {characteristics.has_temporal && (
              <Badge variant="outline">Time series</Badge>
            )}
            {characteristics.is_multivariate && (
              <Badge variant="outline">Multivariate</Badge>
            )}
            {characteristics.seasonality_detected && (
              <Badge variant="outline">Seasonal</Badge>
            )}
          </div>
        </div>
      )}

      <div className="space-y-3">
        {recommendations.map((rec, index) => {
          const methodInfo = getMethodInfo(rec.method_id);
          if (!methodInfo) return null;

          return (
            <div
              key={rec.method_id}
              className={`p-4 border rounded-lg transition-all ${
                index === 0 ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-1">
                    {getMethodIcon(rec.method_id)}
                    <h4 className="font-medium">{methodInfo.name}</h4>
                    {index === 0 && (
                      <Badge variant="default" size="sm">Best Match</Badge>
                    )}
                  </div>
                  
                  <p className="text-sm text-gray-600 mb-2">
                    {rec.reasons.join('. ')}
                  </p>

                  <div className="flex items-center space-x-4 text-xs">
                    <span className="text-gray-500">
                      Accuracy: <span className="font-medium text-gray-700">{rec.estimated_accuracy}</span>
                    </span>
                    <span className="text-gray-500">
                      Time: <span className="font-medium text-gray-700">{rec.estimated_time}</span>
                    </span>
                    <span className="text-gray-500">
                      Complexity: <span className="font-medium text-gray-700">{methodInfo.complexity}</span>
                    </span>
                  </div>

                  {showDetails && (
                    <div className="mt-3 grid grid-cols-2 gap-3">
                      {rec.pros.length > 0 && (
                        <div>
                          <p className="text-xs font-medium text-green-700 mb-1">Pros:</p>
                          <ul className="text-xs text-green-600 space-y-0.5">
                            {rec.pros.map((pro, i) => (
                              <li key={i}>• {pro}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {rec.cons.length > 0 && (
                        <div>
                          <p className="text-xs font-medium text-orange-700 mb-1">Cons:</p>
                          <ul className="text-xs text-orange-600 space-y-0.5">
                            {rec.cons.map((con, i) => (
                              <li key={i}>• {con}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <Button
                  size="sm"
                  variant={index === 0 ? 'primary' : 'outline'}
                  onClick={() => onSelectMethod(rec.method_id)}
                  className="ml-4"
                >
                  Select
                </Button>
              </div>
            </div>
          );
        })}
      </div>

      {recommendations.length === 0 && (
        <p className="text-sm text-gray-500 text-center py-4">
          No specific recommendations available. Please select a method manually.
        </p>
      )}
    </Card>
  );
};