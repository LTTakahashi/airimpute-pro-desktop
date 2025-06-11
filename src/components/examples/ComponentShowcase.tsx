/**
 * Component Showcase
 * Demonstrates usage of all scientific components
 */

import React, { useState } from 'react';
import {
  TimeSeriesChart,
  CorrelationMatrix,
  NumericInput,
  ScientificCard,
  ProgressIndicator,
  ErrorBoundary,
  ThemeProvider,
  useTheme,
} from '@/components';
import type {
  TimeSeriesDataPoint,
  CorrelationMatrix as CorrelationMatrixType,
  ComputationProgress,
  UIMode,
} from '@/types/components';

// Sample data
const generateTimeSeriesData = (): TimeSeriesDataPoint[] => {
  const data: TimeSeriesDataPoint[] = [];
  const startDate = new Date('2024-01-01');
  
  for (let i = 0; i < 100; i++) {
    const date = new Date(startDate);
    date.setHours(date.getHours() + i);
    
    const value = 50 + Math.sin(i / 10) * 20 + Math.random() * 10;
    const isImputed = Math.random() > 0.8;
    
    data.push({
      timestamp: date,
      value,
      confidence: isImputed ? 0.85 : 0.95,
      isImputed,
      imputationMethod: isImputed ? 'RAH' : undefined,
      uncertainty: {
        lower: value - 5,
        upper: value + 5,
      },
    });
  }
  
  return data;
};

const generateCorrelationData = (): CorrelationMatrixType => {
  const variables = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'O3'];
  const n = variables.length;
  
  const values: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));
  const pValues: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        values[i][j] = 1;
        pValues[i][j] = 0;
      } else {
        values[i][j] = Math.random() * 2 - 1;
        pValues[i][j] = Math.random() * 0.1;
      }
    }
  }
  
  return { variables, values, pValues, significanceLevel: 0.05 };
};

export const ComponentShowcase: React.FC = () => {
  const [uiMode, setUiMode] = useState<UIMode>('researcher');
  const [numericValue, setNumericValue] = useState(0.05);
  const [progress, setProgress] = useState<ComputationProgress>({
    phase: 'Data Preprocessing',
    progress: 35,
    estimatedTimeRemaining: 180,
    currentOperation: 'Validating input constraints',
    subProgress: [
      { label: 'Missing value detection', progress: 100 },
      { label: 'Outlier identification', progress: 75 },
      { label: 'Temporal consistency', progress: 20 },
    ],
  });
  
  const timeSeriesData = generateTimeSeriesData();
  const correlationData = generateCorrelationData();
  
  // Simulate progress
  React.useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => ({
        ...prev,
        progress: Math.min(100, prev.progress + 1),
        estimatedTimeRemaining: Math.max(0, prev.estimatedTimeRemaining! - 2),
      }));
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <ThemeProvider>
      <div className="p-8 space-y-8 bg-gray-50 dark:bg-gray-900 min-h-screen">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            AirImpute Pro Component Showcase
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Scientific components following IEEE HCI guidelines
          </p>
        </div>
        
        {/* UI Mode Selector */}
        <ScientificCard title="UI Mode Selection" variant="elevated">
          <div className="flex gap-4">
            {(['student', 'researcher', 'expert'] as UIMode[]).map(mode => (
              <label key={mode} className="flex items-center gap-2">
                <input
                  type="radio"
                  name="uiMode"
                  value={mode}
                  checked={uiMode === mode}
                  onChange={(e) => setUiMode(e.target.value as UIMode)}
                  className="text-blue-600"
                />
                <span className="capitalize">{mode}</span>
              </label>
            ))}
          </div>
        </ScientificCard>
        
        {/* Visualization Components */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ScientificCard
            title="Time Series Visualization"
            subtitle="With confidence intervals and imputed values"
            status="success"
            collapsible
          >
            <ErrorBoundary>
              <TimeSeriesChart
                data={timeSeriesData}
                xAxisLabel="Time"
                yAxisLabel="Concentration (μg/m³)"
                showConfidenceIntervals={true}
                showImputedPoints={true}
                showLegend={true}
                zoomable={true}
                pannable={true}
                highlightMissing={true}
                uiMode={uiMode}
                height={400}
                onPointClick={(point) => {
                  console.log('Clicked point:', point);
                }}
                onRangeSelect={(start, end) => {
                  console.log('Selected range:', start, end);
                }}
              />
            </ErrorBoundary>
          </ScientificCard>
          
          <ScientificCard
            title="Correlation Analysis"
            subtitle="Variable relationships with significance testing"
            status="info"
            collapsible
          >
            <ErrorBoundary>
              <CorrelationMatrix
                data={correlationData}
                showValues={true}
                showSignificance={true}
                colorScale="diverging"
                clusterVariables={uiMode === 'expert'}
                annotateSignificant={true}
                uiMode={uiMode}
                height={400}
                onCellClick={(row, col, value) => {
                  console.log(`Correlation between ${row} and ${col}: ${value}`);
                }}
              />
            </ErrorBoundary>
          </ScientificCard>
        </div>
        
        {/* Form Components */}
        <ScientificCard
          title="Scientific Input Controls"
          subtitle="Validated numeric inputs with constraints"
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <NumericInput
              value={numericValue}
              onChange={setNumericValue}
              label="Significance Level"
              helperText="Statistical significance threshold"
              constraints={{
                min: 0,
                max: 1,
                precision: 4,
                physicalMeaning: 'Probability value for hypothesis testing',
              }}
              unit="α"
              step={0.01}
              showSpinner={true}
              scientificNotation={false}
              uiMode={uiMode}
            />
            
            <NumericInput
              value={2.5e-6}
              onChange={(v) => console.log('Concentration:', v)}
              label="Pollutant Concentration"
              helperText="Enter value in scientific notation"
              constraints={{
                min: 0,
                max: 1000,
                unit: 'μg/m³',
              }}
              scientificNotation={true}
              precision={3}
              uiMode={uiMode}
            />
            
            <NumericInput
              value={273.15}
              onChange={(v) => console.log('Temperature:', v)}
              label="Temperature"
              constraints={{
                min: -273.15,
                max: 1000,
                precision: 2,
              }}
              unit="K"
              engineeringNotation={uiMode === 'expert'}
              uiMode={uiMode}
            />
          </div>
        </ScientificCard>
        
        {/* Progress Indicators */}
        <ScientificCard
          title="Computation Progress"
          subtitle="Long-running operation tracking"
        >
          <div className="space-y-6">
            <div>
              <h4 className="text-sm font-medium mb-2">Linear Progress</h4>
              <ProgressIndicator
                progress={progress}
                variant="linear"
                size="medium"
                showLabel={true}
                showPercentage={true}
                showTimeRemaining={true}
                showSubProgress={uiMode !== 'student'}
                color="primary"
                uiMode={uiMode}
                onCancel={() => console.log('Operation cancelled')}
              />
            </div>
            
            <div className="flex gap-8">
              <div>
                <h4 className="text-sm font-medium mb-2">Circular Progress</h4>
                <ProgressIndicator
                  progress={65}
                  variant="circular"
                  size="large"
                  color="success"
                  uiMode={uiMode}
                />
              </div>
              
              <div>
                <h4 className="text-sm font-medium mb-2">Indeterminate</h4>
                <ProgressIndicator
                  progress={0}
                  variant="circular"
                  size="medium"
                  indeterminate={true}
                  color="primary"
                  uiMode={uiMode}
                />
              </div>
              
              <div>
                <h4 className="text-sm font-medium mb-2">Steps Progress</h4>
                <ProgressIndicator
                  progress={progress.progress}
                  variant="steps"
                  size="medium"
                  color="primary"
                  uiMode={uiMode}
                />
              </div>
            </div>
          </div>
        </ScientificCard>
        
        {/* Theme Controls */}
        <ThemeControls />
      </div>
    </ThemeProvider>
  );
};

// Theme control component
const ThemeControls: React.FC = () => {
  const { theme, toggleTheme, applyColorBlindMode, setFontSize } = useTheme();
  
  return (
    <ScientificCard
      title="Theme & Accessibility Settings"
      subtitle="Customize appearance and accessibility features"
      variant="outlined"
    >
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div>
          <label className="block text-sm font-medium mb-2">Theme Mode</label>
          <button
            onClick={toggleTheme}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Toggle {theme.mode === 'light' ? 'Dark' : 'Light'} Mode
          </button>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-2">Color Blind Mode</label>
          <select
            value={theme.accessibility.colorBlindMode || 'none'}
            onChange={(e) => applyColorBlindMode(e.target.value as any)}
            className="w-full px-3 py-2 border rounded"
          >
            <option value="none">None</option>
            <option value="protanopia">Protanopia</option>
            <option value="deuteranopia">Deuteranopia</option>
            <option value="tritanopia">Tritanopia</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-2">Font Size</label>
          <select
            value={theme.accessibility.fontSize}
            onChange={(e) => setFontSize(e.target.value as any)}
            className="w-full px-3 py-2 border rounded"
          >
            <option value="small">Small</option>
            <option value="medium">Medium</option>
            <option value="large">Large</option>
          </select>
        </div>
      </div>
    </ScientificCard>
  );
};

export default ComponentShowcase;