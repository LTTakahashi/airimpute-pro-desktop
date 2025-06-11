/**
 * Form component type definitions with scientific validation
 * Following IEEE standards for data input integrity
 */

import { ScientificInputProps, ScientificConstraint } from './index';

// Numeric input with scientific constraints
export interface NumericInputProps extends ScientificInputProps<number> {
  step?: number;
  precision?: number;
  scientificNotation?: boolean;
  engineeringNotation?: boolean;
  showSpinner?: boolean;
  allowNegative?: boolean;
  allowZero?: boolean;
  significantFigures?: number;
  
  /** @deprecated Use `constraints.min` instead. */
  min?: number;
  /** @deprecated Use `constraints.max` instead. */
  max?: number;
}

// Range input for parameter selection
export interface RangeInputProps extends ScientificInputProps<[number, number]> {
  min: number;
  max: number;
  step?: number;
  showHistogram?: boolean;
  histogramData?: number[];
  marks?: { value: number; label: string }[];
  logarithmic?: boolean;
}

// Matrix input for correlation/covariance matrices
export interface MatrixInputProps extends Omit<ScientificInputProps<number[][]>, 'constraints'> {
  rows: number;
  cols: number;
  symmetric?: boolean;
  diagonal?: 'free' | 'ones' | 'zeros';
  constraints?: ScientificConstraint[][];
  matrixConstraints?: ScientificConstraint;
  cellPrecision?: number;
  showRowLabels?: boolean;
  showColLabels?: boolean;
  rowLabels?: string[];
  colLabels?: string[];
}

// Time series data input
export interface TimeSeriesInputProps extends ScientificInputProps<any[]> {
  columns: {
    key: string;
    label: string;
    type: 'datetime' | 'numeric' | 'categorical';
    required?: boolean;
    constraints?: ScientificConstraint;
  }[];
  dateFormat?: string;
  timeZone?: string;
  allowMissingValues?: boolean;
  missingValueIndicator?: string;
  preview?: boolean;
  previewRows?: number;
}

// File input with validation
export interface FileInputProps extends ScientificInputProps<File | File[]> {
  accept?: string;
  multiple?: boolean;
  maxSize?: number; // bytes
  maxFiles?: number;
  validateContent?: (file: File) => Promise<ValidationResult>;
  preview?: boolean;
  showMetadata?: boolean;
}

// Parameter group input
export interface ParameterGroupProps extends ScientificInputProps<Record<string, any>> {
  parameters: {
    key: string;
    label: string;
    type: 'numeric' | 'select' | 'boolean' | 'range' | 'matrix';
    defaultValue?: any;
    constraints?: ScientificConstraint;
    options?: { value: any; label: string }[];
    dependsOn?: {
      parameter: string;
      condition: (value: any) => boolean;
    };
    tooltip?: string;
    advanced?: boolean;
  }[];
  layout?: 'vertical' | 'horizontal' | 'grid';
  columns?: number;
  showAdvanced?: boolean;
  validation?: (values: Record<string, any>) => ValidationResult;
}

// Method selection with parameter configuration
export interface MethodSelectorProps extends ScientificInputProps<string> {
  methods: {
    id: string;
    name: string;
    category: string;
    description: string;
    parameters: any[];
    requirements?: {
      minSamples?: number;
      maxMissing?: number;
      dataTypes?: string[];
    };
    citations?: string[];
  }[];
  showDescription?: boolean;
  showRequirements?: boolean;
  showCitations?: boolean;
  groupByCategory?: boolean;
  onParametersChange?: (methodId: string, parameters: any) => void;
}

// Formula/equation input
export interface FormulaInputProps extends ScientificInputProps<string> {
  variables?: string[];
  functions?: string[];
  validateSyntax?: boolean;
  validateSemantics?: boolean;
  showPreview?: boolean;
  syntaxHighlighting?: boolean;
  autoComplete?: boolean;
  examples?: { label: string; formula: string }[];
}

// Data column selector
export interface ColumnSelectorProps extends ScientificInputProps<string[]> {
  columns: {
    name: string;
    type: 'numeric' | 'categorical' | 'datetime' | 'text';
    missing: number;
    unique: number;
    stats?: {
      min?: number;
      max?: number;
      mean?: number;
      std?: number;
    };
  }[];
  minSelection?: number;
  maxSelection?: number;
  filterByType?: string[];
  showStats?: boolean;
  showPreview?: boolean;
  allowReordering?: boolean;
}

// Validation result type
export interface ValidationResult {
  valid: boolean;
  errors?: {
    field: string;
    message: string;
    severity: 'error' | 'warning' | 'info';
  }[];
  warnings?: string[];
  suggestions?: string[];
}

// Constraint builder for dynamic validation
export interface ConstraintBuilderProps {
  constraints: ScientificConstraint[];
  onChange: (constraints: ScientificConstraint[]) => void;
  availableConstraints?: string[];
  dataType?: 'numeric' | 'integer' | 'percentage' | 'probability';
  showPhysicalMeaning?: boolean;
}

// Batch input for multiple datasets
export interface BatchInputProps extends ScientificInputProps<any[]> {
  template: any;
  minItems?: number;
  maxItems?: number;
  itemLabel?: (index: number) => string;
  validateItem?: (item: any, index: number) => ValidationResult;
  showComparison?: boolean;
  allowDuplication?: boolean;
}