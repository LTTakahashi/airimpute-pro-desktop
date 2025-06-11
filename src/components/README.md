# AirImpute Pro Component Library

A comprehensive React component library for scientific data analysis applications, built with strict adherence to IEEE HCI guidelines and WCAG 2.1 Level AA accessibility standards.

## Overview

This component library provides a complete set of UI components specifically designed for scientific computing and data analysis workflows. All components support adaptive UI modes (Student/Researcher/Expert) and maintain data integrity throughout the visualization and input process.

## Key Features

- **IEEE HCI Compliance**: All components follow established human-computer interaction guidelines for scientific software
- **WCAG 2.1 Level AA**: Full accessibility support including screen reader compatibility, keyboard navigation, and color contrast
- **Adaptive UI System**: Components adapt their complexity based on user expertise level
- **Scientific Accuracy**: Precise data visualization and validation with proper error handling
- **TypeScript Support**: Complete type safety with comprehensive type definitions
- **Theme System**: Support for light/dark modes and high contrast themes
- **Performance Optimized**: Efficient rendering for large datasets

## Component Categories

### 1. Scientific Visualization Components

#### BenchmarkDashboard
Comprehensive benchmarking interface with real-time progress tracking, GPU acceleration controls, and statistical analysis.

```tsx
import { BenchmarkDashboard } from '@/components/benchmarking';

<BenchmarkDashboard />
```

#### BenchmarkCharts
Suite of specialized charts for benchmark visualization including bar, line, scatter, radar, heatmap, and box plots.

```tsx
import { BarChart, RadarChart, HeatmapChart } from '@/components/benchmarking';

<BarChart
  data={comparisonData}
  title="Method Performance Comparison"
  yAxisLabel="RMSE"
/>

<RadarChart
  data={multiMetricData}
  title="Multi-Metric Analysis"
/>

<HeatmapChart
  data={performanceMatrix}
  title="Performance Heatmap"
  colorScale="viridis"
/>
```

### 2. Advanced Scientific Visualization

#### TimeSeriesChart
Advanced time series visualization with support for confidence intervals, imputed data highlighting, and interactive exploration.

```tsx
import { TimeSeriesChart } from '@/components';

<TimeSeriesChart
  data={timeSeriesData}
  showConfidenceIntervals={true}
  showImputedPoints={true}
  onRangeSelect={(start, end) => console.log('Selected range:', start, end)}
  uiMode="researcher"
/>
```

#### CorrelationMatrix
Interactive correlation matrix with significance testing, clustering, and customizable color schemes.

```tsx
import { CorrelationMatrix } from '@/components';

<CorrelationMatrix
  data={correlationData}
  showSignificance={true}
  significanceThreshold={0.05}
  clusterVariables={true}
  uiMode="expert"
/>
```

### 2. Form Components with Scientific Validation

#### NumericInput
Numeric input with scientific notation support, constraint validation, and precision control.

```tsx
import { NumericInput } from '@/components';

<NumericInput
  value={value}
  onChange={setValue}
  label="Significance Level"
  constraints={{ min: 0, max: 1, precision: 4 }}
  scientificNotation={true}
  unit="α"
  uiMode="researcher"
/>
```

### 3. Layout Components

#### ScientificCard
Flexible card component for grouping related content with status indicators and loading states.

```tsx
import { ScientificCard } from '@/components';

<ScientificCard
  title="Analysis Results"
  subtitle="Principal Component Analysis"
  status="success"
  collapsible={true}
  loading={isLoading}
>
  {/* Card content */}
</ScientificCard>
```

### 4. Benchmarking Components

#### StatisticalTestResults
Display comprehensive statistical test results with visual indicators for significance.

```tsx
import { StatisticalTestResults } from '@/components/benchmarking';

<StatisticalTestResults
  results={benchmarkResults}
  methods={selectedMethods}
  metric="rmse"
/>
```

#### MethodComparison
Interactive method selection and comparison with GPU support indicators.

```tsx
import { MethodComparison } from '@/components/benchmarking';

<MethodComparison
  methods={availableMethods}
  selectedMethods={selected}
  onSelectionChange={setSelected}
/>
```

#### ReproducibilityReport
Generate and display reproducibility certificates with full environment tracking.

```tsx
import { ReproducibilityReport } from '@/components/benchmarking';

<ReproducibilityReport
  results={results}
  datasets={datasets}
  methods={methods}
/>
```

### 5. Feedback Components

#### ProgressIndicator
Comprehensive progress tracking for long-running computations with sub-progress support.

```tsx
import { ProgressIndicator } from '@/components';

<ProgressIndicator
  progress={{
    phase: "Imputation",
    progress: 75,
    estimatedTimeRemaining: 120,
    currentOperation: "Processing missing values"
  }}
  showSubProgress={true}
  onCancel={() => handleCancel()}
/>
```

#### ErrorBoundary
Graceful error handling with scientific context and recovery options.

```tsx
import { ErrorBoundary } from '@/components';

<ErrorBoundary
  onError={(error) => logError(error)}
  showDetails={true}
  allowRetry={true}
>
  {/* Protected components */}
</ErrorBoundary>
```

## Accessibility Features

All components include:

- **ARIA Labels and Descriptions**: Proper semantic markup for screen readers
- **Keyboard Navigation**: Full keyboard support with intuitive shortcuts
- **Focus Management**: Proper focus trapping and restoration
- **Announcements**: Live region updates for dynamic content
- **Color Contrast**: WCAG AA compliant color combinations
- **Reduced Motion**: Respects user preferences for animations

## Theme System

The library includes a comprehensive theme system:

```tsx
import { ThemeProvider, useTheme } from '@/components';

// Wrap your app
<ThemeProvider>
  <App />
</ThemeProvider>

// Use in components
const { theme, toggleTheme, setFontSize } = useTheme();
```

### Available Themes

- **Light Theme**: Default light color scheme
- **Dark Theme**: Eye-friendly dark color scheme
- **High Contrast**: Maximum contrast for accessibility

### Customization Options

- Color blind modes (protanopia, deuteranopia, tritanopia)
- Font size adjustment (small, medium, large)
- Number format preferences (decimal, scientific, engineering)

## UI Modes

Components adapt based on the user's expertise level:

### Student Mode
- Simplified interfaces
- Clear guidance and tooltips
- Limited advanced options
- Focus on learning

### Researcher Mode
- Balanced complexity
- Standard scientific features
- Moderate customization
- Professional workflows

### Expert Mode
- Full feature access
- Advanced customization
- Detailed technical information
- Power user features

## Best Practices

1. **Always specify UI mode**: Pass the appropriate `uiMode` prop based on user preferences
2. **Provide meaningful labels**: Use descriptive `aria-label` attributes for accessibility
3. **Handle errors gracefully**: Wrap components in ErrorBoundary for robust error handling
4. **Validate scientific inputs**: Use constraint validation for data integrity
5. **Consider performance**: Use responsive and exportable props appropriately for large datasets

## TypeScript Usage

All components are fully typed with comprehensive interfaces:

```tsx
import { TimeSeriesDataPoint, ScientificConstraint } from '@/types/components';

const data: TimeSeriesDataPoint[] = [
  {
    timestamp: new Date('2024-01-01'),
    value: 42.5,
    confidence: 0.95,
    isImputed: false
  }
];

const constraints: ScientificConstraint = {
  min: 0,
  max: 100,
  precision: 2,
  unit: 'μg/m³'
};
```

## Examples

### Complete Analysis Dashboard

```tsx
import {
  TimeSeriesChart,
  CorrelationMatrix,
  ScientificCard,
  ProgressIndicator,
  ErrorBoundary,
  ThemeProvider
} from '@/components';

function AnalysisDashboard() {
  return (
    <ThemeProvider>
      <ErrorBoundary>
        <div className="grid grid-cols-2 gap-4">
          <ScientificCard title="Time Series Analysis">
            <TimeSeriesChart
              data={timeSeriesData}
              showConfidenceIntervals={true}
              uiMode={userMode}
            />
          </ScientificCard>
          
          <ScientificCard title="Correlation Analysis">
            <CorrelationMatrix
              data={correlationData}
              showSignificance={true}
              uiMode={userMode}
            />
          </ScientificCard>
        </div>
        
        {isProcessing && (
          <ProgressIndicator
            progress={computationProgress}
            onCancel={handleCancel}
          />
        )}
      </ErrorBoundary>
    </ThemeProvider>
  );
}
```

## Contributing

When contributing new components, ensure they:

1. Follow IEEE HCI guidelines for scientific software
2. Meet WCAG 2.1 Level AA accessibility standards
3. Support all three UI modes (Student/Researcher/Expert)
4. Include comprehensive TypeScript types
5. Have proper error handling and validation
6. Include unit tests with >90% coverage
7. Are documented with clear examples

## License

This component library is part of the AirImpute Pro Desktop application and follows the same MIT license.