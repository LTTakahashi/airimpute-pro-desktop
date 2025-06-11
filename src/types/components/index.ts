/**
 * Core component type definitions following IEEE HCI guidelines
 * and scientific software interface standards
 */

// UI Mode definitions for adaptive interface
export type UIMode = 'student' | 'researcher' | 'expert';

// Scientific data types
export interface TimeSeriesDataPoint {
  timestamp: Date;
  value: number;
  confidence?: number;
  isImputed?: boolean;
  imputationMethod?: string;
  uncertainty?: {
    lower: number;
    upper: number;
  };
}

export interface CorrelationMatrix {
  variables: string[];
  values: number[][];
  pValues?: number[][];
  significanceLevel?: number;
}

export interface StatisticalResult {
  statistic: number;
  pValue: number;
  confidenceInterval?: [number, number];
  effectSize?: number;
  degreesOfFreedom?: number;
  sampleSize: number;
  method: string;
  assumptions?: {
    name: string;
    satisfied: boolean;
    details?: string;
  }[];
}

// Validation types
export interface ValidationRule<T = any> {
  validate: (value: T) => boolean;
  message: string;
  severity?: 'error' | 'warning' | 'info';
}

export interface ScientificConstraint {
  min?: number;
  max?: number;
  precision?: number;
  unit?: string;
  physicalMeaning?: string;
}

// Progress tracking for long computations
export interface ComputationProgress {
  phase: string;
  progress: number; // 0-100
  estimatedTimeRemaining?: number; // seconds
  currentOperation?: string;
  subProgress?: {
    label: string;
    progress: number;
  }[];
}

// Error handling with scientific context
export interface ScientificError {
  code: string;
  message: string;
  scientificContext?: {
    dataset?: string;
    method?: string;
    parameters?: Record<string, any>;
    suggestion?: string;
  };
  technicalDetails?: string;
  recoverable?: boolean;
}

// Theme system types
export interface ScientificTheme {
  mode: 'light' | 'dark';
  colorScheme: {
    primary: string;
    secondary: string;
    accent: string;
    success: string;
    warning: string;
    error: string;
    info: string;
    // Scientific visualization colors
    heatmapColors: string[];
    divergingColors: string[];
    categoricalColors: string[];
  };
  typography: {
    scientificNotation: boolean;
    significantFigures: number;
    numberFormat: 'decimal' | 'scientific' | 'engineering';
  };
  accessibility: {
    highContrast: boolean;
    colorBlindMode?: 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia';
    fontSize: 'small' | 'medium' | 'large';
  };
}

// Component props interfaces
export interface ScientificComponentProps {
  uiMode?: UIMode;
  'aria-label'?: string;
  'aria-describedby'?: string;
  className?: string;
  testId?: string;
}

// Data visualization component props
export interface VisualizationProps extends ScientificComponentProps {
  data: any;
  width?: number;
  height?: number;
  responsive?: boolean;
  exportable?: boolean;
  interactive?: boolean;
  annotations?: any[];
}

// Form component props with scientific validation
export interface ScientificInputProps<T = any> extends ScientificComponentProps {
  value: T;
  onChange: (value: T) => void;
  label: string;
  helperText?: string;
  error?: string;
  disabled?: boolean;
  required?: boolean;
  constraints?: ScientificConstraint;
  validationRules?: ValidationRule<T>[];
  scientificNotation?: boolean;
  unit?: string;
}

// Layout component props
export interface LayoutProps extends ScientificComponentProps {
  children: React.ReactNode;
  padding?: 'none' | 'small' | 'medium' | 'large';
  gap?: 'none' | 'small' | 'medium' | 'large';
}

// Accessibility types
export interface A11yProps {
  role?: string;
  'aria-live'?: 'polite' | 'assertive' | 'off';
  'aria-busy'?: boolean;
  'aria-atomic'?: boolean;
  'aria-relevant'?: 'additions' | 'removals' | 'text' | 'all';
  tabIndex?: number;
}

// Export all types
export * from './visualization';
export * from './forms';
export * from './layout';
export * from './feedback';