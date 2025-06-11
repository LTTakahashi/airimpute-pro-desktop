# AirImpute Pro Desktop - Component Mapping

This document provides a comprehensive mapping of all React components, their relationships, and architectural patterns used in the AirImpute Pro Desktop application.

## Component Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    App (Root)                             │
│                  ├── Router                               │
│                  ├── Redux Provider                       │
│                  └── Theme (Tailwind)                     │
└──────────────────────────────────────────────────────────┘
                           │
    ┌──────────────────────┴──────────────────────┐
    │                                             │
┌───▼────────┐                          ┌────────▼────────┐
│   Pages    │                          │   Components    │
│  (Routes)  │                          │   (Reusable)    │
└────────────┘                          └─────────────────┘
```

## Component Categories

### 1. Scientific Components
Located in `src/components/scientific/`

- **TimeSeriesChart.tsx**: Interactive time series visualization
  - Props: `data`, `missingData`, `imputedData`, `dateRange`, `onZoom`
  - Built on Plotly.js for performance with large datasets
  - Features: Zoom, pan, export, multiple series overlay
  
- **CorrelationMatrix.tsx**: Heatmap for variable correlations
  - Props: `data`, `variables`, `colorScale`, `onClick`
  - Interactive cells with tooltip details
  - Used for feature selection in imputation

### 2. Layout Components
Located in `src/components/layout/`

- **Sidebar.tsx**: Main navigation sidebar
  - Props: `collapsed`, `onCollapse`, `activeSection`
  - Sections: Data, Methods, Analysis, Results, Settings
  
- **Header.tsx**: Top application bar
  - Props: `title`, `user`, `notifications`
  - Contains: Project selector, save status, help button

- **StatusBar.tsx**: Bottom status bar
  - Props: `status`, `progress`, `memory`, `computation`
  - Real-time updates for:
    - Memory usage (with warnings)
    - Computation progress
    - Background tasks
    - Python process status

- **ScientificCard.tsx**: Reusable card for scientific content
  - Props: `title`, `icon`, `children`, `actions`, `collapsible`
  - Consistent styling for all analysis panels
  - Export/fullscreen capabilities

### 3. UI Components (Base Library)
Located in `src/components/ui/`

#### Core Components
- **Card.tsx**: Container component
  - Props: `variant`, `padding`, `shadow`, `children`
  - Used as base for all content sections

- **Badge.tsx**: Status/category indicators
  - Props: `variant`, `size`, `dot`, `children`
  - Variants: `default`, `success`, `warning`, `error`, `info`

- **Progress.tsx**: Progress indicators
  - Props: `value`, `max`, `size`, `showLabel`, `color`
  - Supports both linear and circular variants

- **Tooltip.tsx**: Hover information
  - Built on Radix UI Tooltip
  - Props: `content`, `side`, `align`, `delay`

#### Form Components
- **Select.tsx**: Enhanced dropdown
  - Props: `options`, `value`, `onChange`, `multiple`, `searchable`
  - Special features for method selection

- **Checkbox.tsx**: Checkbox input
  - Props: `checked`, `onChange`, `label`, `description`

- **Tabs.tsx**: Tab navigation
  - Props: `tabs`, `activeTab`, `onChange`, `orientation`
  - Used for method categories, results views

### 4. Feedback Components
Located in `src/components/feedback/`

- **ErrorBoundary.tsx**: Catches React errors
  - Props: `fallback`, `onError`, `resetKeys`
  - Prevents full app crashes from computation errors
  - Provides error reporting to user

- **ProgressIndicator.tsx**: Computation progress
  - Props: `steps`, `currentStep`, `progress`, `estimated`
  - Shows:
    - Current operation
    - Time elapsed/remaining
    - Memory usage
    - Cancel button (when implemented)

### 5. Page Components
Located in `src/pages/`

- **Dashboard**: Main overview and quick actions
- **DataImport**: CSV upload and preview
- **MethodSelection**: Choose imputation methods
- **Analysis**: Run imputation and view progress
- **Results**: View and export results
- **Benchmarking**: Compare methods
- **Settings**: Application preferences

## State Management (Redux Toolkit)

### Store Structure
```
store/
├── slices/
│   ├── dataSlice.ts      # Dataset management
│   ├── imputationSlice.ts # Imputation state
│   ├── uiSlice.ts        # UI preferences
│   └── benchmarkSlice.ts # Benchmarking state
├── hooks.ts              # Typed hooks
└── store.ts              # Store configuration
```

### State Shape
```typescript
{
  data: {
    datasets: Dataset[];
    activeDataset: string | null;
    preview: DataPreview | null;
    loading: boolean;
  },
  imputation: {
    methods: Method[];
    selectedMethods: string[];
    parameters: MethodParameters;
    results: ImputationResult[];
    running: boolean;
    progress: number;
  },
  ui: {
    sidebarCollapsed: boolean;
    theme: 'light' | 'dark';
    notifications: Notification[];
  },
  benchmark: {
    comparisons: Comparison[];
    metrics: BenchmarkMetric[];
    activeComparison: string | null;
  }
}
```

## Component Patterns

### 1. Compound Components
```typescript
// Example: TimeSeriesChart with sub-components
<TimeSeriesChart>
  <TimeSeriesChart.Line data={original} />
  <TimeSeriesChart.Line data={imputed} style="dashed" />
  <TimeSeriesChart.MissingRegions data={missing} />
</TimeSeriesChart>
```

### 2. Render Props for Customization
```typescript
// Example: DataTable with custom cells
<DataTable
  data={data}
  renderCell={({ value, column }) => 
    column.type === 'missing' ? <MissingCell /> : value
  }
/>
```

### 3. Custom Hooks
- `useDataset()`: Dataset operations
- `useImputation()`: Imputation methods
- `useBenchmark()`: Benchmarking operations
- `useMemoryMonitor()`: Memory tracking
- `usePythonBridge()`: Python communication

## Performance Optimizations

### 1. Lazy Loading
- Heavy visualizations loaded on demand
- Python libraries initialized only when needed
- Large datasets paginated

### 2. Memoization Strategy
```typescript
// Example: Expensive correlation calculation
const correlationMatrix = useMemo(
  () => calculateCorrelations(data),
  [data.id] // Only recalculate when data changes
);
```

### 3. Virtual Scrolling
- Data tables use react-window
- Handles datasets with millions of rows
- Only renders visible rows

### 4. Web Workers (Planned)
- Statistical calculations in background
- Non-blocking UI during computation

## Error Handling Patterns

### 1. Python Error Boundary
```typescript
const PythonErrorBoundary: FC = ({ children }) => {
  // Catches Python process crashes
  // Provides recovery options
  // Saves work before crash
};
```

### 2. Validation Layers
- Input validation before Python calls
- Parameter range checking
- Dataset compatibility verification

### 3. Recovery Mechanisms
- Auto-save every 5 minutes
- Crash recovery on restart
- Partial result preservation

## Accessibility Features

### Keyboard Navigation
- Tab order properly managed
- Shortcuts for common actions
- Focus trapping in modals

### Screen Reader Support
- Descriptive labels for all inputs
- Status announcements
- Chart descriptions

### Visual Accessibility
- High contrast mode support
- Configurable font sizes
- Color-blind friendly palettes

## Testing Architecture

### Component Testing
```typescript
// Example test structure
describe('TimeSeriesChart', () => {
  it('renders with data', () => {});
  it('handles missing data regions', () => {});
  it('responds to zoom events', () => {});
  it('exports chart as image', () => {});
});
```

### Test Coverage
- Scientific Components: 60%
- UI Components: 70%
- Layout Components: 80%
- Pages: 40%
- State Management: 85%

## Bundle Analysis

### Component Bundle Sizes
- React + Core: ~150KB
- Plotly.js: ~3MB (largest dependency)
- Three.js: ~600KB (for 3D viz)
- Redux Toolkit: ~50KB
- Radix UI: ~40KB
- Tailwind CSS: ~30KB
- Total: ~4MB (before compression)

### Code Splitting Strategy
```typescript
// Lazy load heavy components
const PlotlyChart = lazy(() => import('./PlotlyChart'));
const ThreeDVisualization = lazy(() => import('./ThreeDVisualization'));
```

## Known Issues & Limitations

### Performance Issues
1. **Large Dataset Rendering**: Slows down above 100K points
2. **State Updates**: Can cause UI freezes during computation
3. **Memory Leaks**: In long-running sessions

### Component Issues
1. **Progress Indicator**: Doesn't update smoothly
2. **Error Boundary**: Sometimes misses Python errors
3. **Select Component**: Slow with many options

## Future Component Plans

### Planned Components
1. **GuidedMethodSelection**: Wizard for method choice
2. **DataQualityDashboard**: Pre-imputation analysis
3. **ResultComparison**: Side-by-side method comparison
4. **ExportWizard**: Advanced export options
5. **GPUMonitor**: GPU usage tracking

### Refactoring Plans
1. Extract visualization library
2. Improve state management patterns
3. Add proper cancellation support
4. Implement streaming for large files

## Component Best Practices

### 1. TypeScript First
```typescript
interface ComponentProps {
  data: TimeSeriesData;
  options?: ChartOptions;
  onError?: (error: Error) => void;
}
```

### 2. Error Handling
```typescript
try {
  await computeImputation(data);
} catch (error) {
  if (error instanceof PythonError) {
    handlePythonError(error);
  } else {
    handleGeneralError(error);
  }
}
```

### 3. Performance Monitoring
```typescript
const startTime = performance.now();
// ... computation ...
const duration = performance.now() - startTime;
trackPerformance('imputation', duration);
```

## Conclusion

The AirImpute Pro Desktop component architecture shows:
- **Scientific Focus**: Specialized components for data analysis
- **Performance Awareness**: Optimizations for large datasets  
- **Error Resilience**: Multiple error boundaries and recovery
- **Research Tool**: Built for experimentation, not production
- **Modular Design**: Easy to add new imputation methods

While less polished than HarpIA Metrics Desktop, the component structure is solid and focused on the scientific computing needs of air quality researchers.