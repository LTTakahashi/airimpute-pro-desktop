import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/Select';
import { Info } from 'lucide-react';
import { TooltipContent, TooltipProvider, TooltipTrigger, TooltipRoot } from '../ui/Tooltip';

interface MetricSelectorProps {
  value: string;
  onChange: (value: string) => void;
  className?: string;
}

const metrics = [
  {
    value: 'rmse',
    label: 'RMSE',
    description: 'Root Mean Square Error - Standard deviation of residuals',
    formula: '√(Σ(y_pred - y_true)² / n)',
    lowerBetter: true
  },
  {
    value: 'mae',
    label: 'MAE',
    description: 'Mean Absolute Error - Average of absolute differences',
    formula: 'Σ|y_pred - y_true| / n',
    lowerBetter: true
  },
  {
    value: 'r2',
    label: 'R²',
    description: 'Coefficient of Determination - Proportion of variance explained',
    formula: '1 - (SS_res / SS_tot)',
    lowerBetter: false
  },
  {
    value: 'mape',
    label: 'MAPE',
    description: 'Mean Absolute Percentage Error - Average percentage error',
    formula: '100 × Σ|y_pred - y_true| / |y_true| / n',
    lowerBetter: true
  },
  {
    value: 'corr',
    label: 'Correlation',
    description: 'Pearson correlation coefficient',
    formula: 'cov(y_pred, y_true) / (σ_pred × σ_true)',
    lowerBetter: false
  },
  {
    value: 'bias',
    label: 'Bias',
    description: 'Average prediction error',
    formula: 'Σ(y_pred - y_true) / n',
    lowerBetter: true
  },
  {
    value: 'coverage',
    label: 'Coverage',
    description: 'Percentage of values within confidence intervals',
    formula: 'Count(y_true ∈ CI) / n × 100',
    lowerBetter: false
  },
  {
    value: 'sharpness',
    label: 'Sharpness',
    description: 'Average width of prediction intervals',
    formula: 'Σ(upper_bound - lower_bound) / n',
    lowerBetter: true
  }
];

export const MetricSelector: React.FC<MetricSelectorProps> = ({
  value,
  onChange,
  className
}) => {
  const selectedMetric = metrics.find(m => m.value === value);

  return (
    <TooltipProvider>
      <div className={className}>
        <div className="flex items-center gap-2 mb-2">
          <Select value={value} onValueChange={onChange}>
            <SelectTrigger className="flex-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {metrics.map(metric => (
                <SelectItem key={metric.value} value={metric.value}>
                  <div className="flex items-center justify-between w-full">
                    <span>{metric.label}</span>
                    <span className="text-xs text-muted-foreground ml-2">
                      {metric.lowerBetter ? '↓ better' : '↑ better'}
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          {selectedMetric && (
            <TooltipProvider>
              <TooltipRoot>
                <TooltipTrigger asChild>
                  <Info className="w-4 h-4 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent side="right" className="max-w-xs">
                  <div className="space-y-2">
                    <p className="font-semibold">{selectedMetric.label}</p>
                    <p className="text-sm">{selectedMetric.description}</p>
                    <p className="text-xs font-mono bg-muted p-1 rounded">
                    {selectedMetric.formula}
                  </p>
                </div>
              </TooltipContent>
              </TooltipRoot>
            </TooltipProvider>
          )}
        </div>
        
        {selectedMetric && (
          <p className="text-xs text-muted-foreground">
            {selectedMetric.description}
          </p>
        )}
      </div>
    </TooltipProvider>
  );
};