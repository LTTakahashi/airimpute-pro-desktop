/**
 * Visualization component type definitions
 * Following IEEE visualization standards for scientific data
 */

import { TimeSeriesDataPoint, CorrelationMatrix, VisualizationProps } from './index';

// Re-export for convenience
export type { TimeSeriesDataPoint, CorrelationMatrix };

// Time series visualization
export interface TimeSeriesChartProps extends VisualizationProps {
  data: TimeSeriesDataPoint[] | TimeSeriesDataPoint[][];
  xAxisLabel?: string;
  yAxisLabel?: string;
  showConfidenceIntervals?: boolean;
  showImputedPoints?: boolean;
  dateFormat?: string;
  valueFormat?: (value: number) => string;
  zoomable?: boolean;
  pannable?: boolean;
  showLegend?: boolean;
  legendPosition?: 'top' | 'bottom' | 'left' | 'right';
  colorScheme?: string[];
  highlightMissing?: boolean;
  onPointClick?: (point: TimeSeriesDataPoint) => void;
  onRangeSelect?: (start: Date, end: Date) => void;
}

// Correlation matrix visualization
export interface CorrelationMatrixProps extends VisualizationProps {
  data: CorrelationMatrix;
  colorScale?: 'diverging' | 'sequential';
  showValues?: boolean;
  showSignificance?: boolean;
  significanceThreshold?: number;
  clusterVariables?: boolean;
  onCellClick?: (row: string, col: string, value: number) => void;
  annotateSignificant?: boolean;
  maskInsignificant?: boolean;
}

// Distribution visualization
export interface DistributionChartProps extends VisualizationProps {
  data: number[];
  type: 'histogram' | 'density' | 'box' | 'violin' | 'qq';
  bins?: number;
  bandwidth?: number;
  showNormalCurve?: boolean;
  showOutliers?: boolean;
  showStatistics?: boolean;
  referenceLines?: {
    value: number;
    label: string;
    color?: string;
  }[];
}

// Scatter plot with regression
export interface ScatterPlotProps extends VisualizationProps {
  xData: number[];
  yData: number[];
  xLabel: string;
  yLabel: string;
  showRegression?: boolean;
  regressionType?: 'linear' | 'polynomial' | 'exponential' | 'logarithmic';
  showConfidenceBand?: boolean;
  showPredictionBand?: boolean;
  pointSize?: number | number[];
  pointColor?: string | string[];
  groupBy?: string[];
  onPointSelect?: (index: number) => void;
}

// Heatmap visualization
export interface HeatmapProps extends VisualizationProps {
  data: number[][];
  xLabels: string[];
  yLabels: string[];
  colorScale?: string[];
  showValues?: boolean;
  valueFormat?: (value: number) => string;
  missingValueColor?: string;
  onCellHover?: (x: number, y: number, value: number) => void;
}

// 3D surface plot
export interface SurfacePlotProps extends VisualizationProps {
  x: number[];
  y: number[];
  z: number[][];
  colorScale?: string[];
  showContours?: boolean;
  contourLevels?: number;
  cameraPosition?: {
    eye: { x: number; y: number; z: number };
    center: { x: number; y: number; z: number };
    up: { x: number; y: number; z: number };
  };
}

// Statistical chart types
export interface StatisticalChartProps extends VisualizationProps {
  data: any;
  chartType: 'residuals' | 'acf' | 'pacf' | 'qqplot' | 'ppplot';
  confidenceLevel?: number;
  showConfidenceBands?: boolean;
  lags?: number;
}

// Multi-panel visualization
export interface MultiPanelProps extends VisualizationProps {
  panels: {
    id: string;
    type: 'timeseries' | 'scatter' | 'distribution' | 'correlation' | 'heatmap';
    data: any;
    props: any;
    row: number;
    col: number;
    rowSpan?: number;
    colSpan?: number;
  }[];
  rows: number;
  cols: number;
  syncAxes?: boolean;
  sharedLegend?: boolean;
}

// Animation controls for dynamic visualizations
export interface AnimationControls {
  playing: boolean;
  currentFrame: number;
  totalFrames: number;
  fps: number;
  loop: boolean;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onFrameChange: (frame: number) => void;
}

// Export options for visualizations
export interface ExportOptions {
  format: 'png' | 'svg' | 'pdf' | 'csv' | 'json';
  quality?: number; // for raster formats
  includeData?: boolean;
  includeMetadata?: boolean;
  filename?: string;
}