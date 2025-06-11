/**
 * Scientific validation utilities
 */

import { ScientificConstraint } from '@/types/components';

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

export function validateScientificConstraint(
  value: number,
  constraint: ScientificConstraint
): ValidationResult {
  const errors: string[] = [];
  
  if (constraint.min !== undefined && value < constraint.min) {
    errors.push(`Value must be at least ${constraint.min}${constraint.unit ? ' ' + constraint.unit : ''}`);
  }
  
  if (constraint.max !== undefined && value > constraint.max) {
    errors.push(`Value must be at most ${constraint.max}${constraint.unit ? ' ' + constraint.unit : ''}`);
  }
  
  if (constraint.precision !== undefined) {
    const decimalPlaces = (value.toString().split('.')[1] || '').length;
    if (decimalPlaces > constraint.precision) {
      errors.push(`Value must have at most ${constraint.precision} decimal places`);
    }
  }
  
  return {
    valid: errors.length === 0,
    errors,
  };
}

export function parseScientificNotation(input: string): number | null {
  // Handle scientific notation (e.g., 1.23e-4, 1.23E+4)
  const scientificRegex = /^([+-]?\d*\.?\d+)[eE]([+-]?\d+)$/;
  const match = input.trim().match(scientificRegex);
  
  if (match) {
    const mantissa = parseFloat(match[1]);
    const exponent = parseInt(match[2], 10);
    return mantissa * Math.pow(10, exponent);
  }
  
  // Try regular number parsing
  const parsed = parseFloat(input);
  return isNaN(parsed) ? null : parsed;
}

export function formatScientificNumber(
  value: number,
  options: {
    notation?: 'standard' | 'scientific' | 'engineering';
    precision?: number;
    significantFigures?: number;
  } = {}
): string {
  const { notation = 'standard', precision = 6, significantFigures } = options;
  
  if (significantFigures !== undefined) {
    return value.toPrecision(significantFigures);
  }
  
  switch (notation) {
    case 'scientific':
      return value.toExponential(precision);
      
    case 'engineering': {
      const exp = Math.floor(Math.log10(Math.abs(value)) / 3) * 3;
      const mantissa = value / Math.pow(10, exp);
      return `${mantissa.toFixed(precision)}e${exp >= 0 ? '+' : ''}${exp}`;
    }
    
    default:
      return value.toFixed(precision);
  }
}