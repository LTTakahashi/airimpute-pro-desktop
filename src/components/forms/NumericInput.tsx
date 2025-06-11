/**
 * Numeric Input Component with Scientific Constraints
 * Implements IEEE standards for numerical input validation
 * WCAG 2.1 Level AA compliant
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { NumericInputProps } from '@/types/components/forms';
import { validateScientificConstraint, parseScientificNotation } from '@/lib/validation/scientific';
import { announce, KEYS } from '@/lib/accessibility';
import { cn } from '@/utils/cn';
import { AlertCircle, Info } from 'lucide-react';

export const NumericInput: React.FC<NumericInputProps> = ({
  value,
  onChange,
  label,
  helperText,
  error,
  disabled = false,
  required = false,
  constraints,
  validationRules = [],
  scientificNotation = false,
  engineeringNotation = false,
  unit,
  step = 1,
  precision = 6,
  showSpinner = true,
  allowNegative = true,
  allowZero = true,
  significantFigures,
  min,
  max,
  className,
  'aria-label': ariaLabel,
  'aria-describedby': ariaDescribedBy,
  uiMode = 'researcher',
  testId = 'numeric-input',
}) => {
  // Merge legacy props into constraints
  const finalConstraints = React.useMemo(() => {
    const base = constraints || {};
    if (min !== undefined && base.min === undefined) {
      console.warn('NumericInput: The "min" prop is deprecated. Use "constraints.min" instead.');
      return { ...base, min };
    }
    if (max !== undefined && base.max === undefined) {
      console.warn('NumericInput: The "max" prop is deprecated. Use "constraints.max" instead.');
      return { ...base, max };
    }
    return base;
  }, [constraints, min, max]);
  const [localValue, setLocalValue] = useState<string>(formatValue(value));
  const [isFocused, setIsFocused] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Format value for display
  function formatValue(num: number | undefined): string {
    if (num === undefined || num === null || isNaN(num)) return '';
    
    if (scientificNotation) {
      return num.toExponential(precision);
    }
    
    if (engineeringNotation) {
      const exp = Math.floor(Math.log10(Math.abs(num)) / 3) * 3;
      const mantissa = num / Math.pow(10, exp);
      return `${mantissa.toFixed(precision)}e${exp >= 0 ? '+' : ''}${exp}`;
    }
    
    if (significantFigures !== undefined) {
      return num.toPrecision(significantFigures);
    }
    
    return num.toFixed(precision);
  }
  
  // Parse user input
  const parseValue = useCallback((input: string): number | null => {
    if (!input.trim()) return null;
    
    let parsed: number | null = null;
    
    if (scientificNotation || engineeringNotation) {
      parsed = parseScientificNotation(input);
    } else {
      parsed = parseFloat(input);
    }
    
    if (parsed === null || isNaN(parsed)) return null;
    
    // Apply constraints
    if (!allowNegative && parsed < 0) return null;
    if (!allowZero && parsed === 0) return null;
    
    return parsed;
  }, [scientificNotation, engineeringNotation, allowNegative, allowZero]);
  
  // Validate value
  const validateValue = useCallback((num: number): { valid: boolean; error?: string } => {
    // Check scientific constraints
    if (finalConstraints) {
      const result = validateScientificConstraint(num, finalConstraints);
      if (!result.valid) {
        return { valid: false, error: result.errors[0] };
      }
    }
    
    // Check custom validation rules
    for (const rule of validationRules) {
      if (!rule.validate(num)) {
        return { valid: false, error: rule.message };
      }
    }
    
    return { valid: true };
  }, [finalConstraints, validationRules]);
  
  // Handle input change
  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const input = e.target.value;
    setLocalValue(input);
    
    // Allow intermediate invalid states while typing
    if (!input.trim()) {
      setValidationError(null);
      return;
    }
    
    const parsed = parseValue(input);
    if (parsed === null) {
      setValidationError('Invalid number format');
      return;
    }
    
    const validation = validateValue(parsed);
    if (!validation.valid) {
      setValidationError(validation.error || 'Invalid value');
      return;
    }
    
    setValidationError(null);
    onChange(parsed);
  }, [parseValue, validateValue, onChange]);
  
  // Handle blur - final validation and formatting
  const handleBlur = useCallback(() => {
    setIsFocused(false);
    
    const parsed = parseValue(localValue);
    if (parsed !== null) {
      const validation = validateValue(parsed);
      if (validation.valid) {
        setLocalValue(formatValue(parsed));
        onChange(parsed);
      }
    } else if (localValue.trim() && required) {
      setValidationError('This field is required');
    }
  }, [localValue, parseValue, validateValue, onChange, required]);
  
  // Handle keyboard shortcuts
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (disabled) return;
    
    const currentValue = parseValue(localValue) || 0;
    let newValue: number | null = null;
    
    switch (e.key) {
      case KEYS.ARROW_UP:
        e.preventDefault();
        newValue = currentValue + (e.shiftKey ? step * 10 : step);
        break;
        
      case KEYS.ARROW_DOWN:
        e.preventDefault();
        newValue = currentValue - (e.shiftKey ? step * 10 : step);
        break;
        
      case KEYS.HOME:
        if (e.ctrlKey && finalConstraints?.min !== undefined) {
          e.preventDefault();
          newValue = finalConstraints.min;
        }
        break;
        
      case KEYS.END:
        if (e.ctrlKey && finalConstraints?.max !== undefined) {
          e.preventDefault();
          newValue = finalConstraints.max;
        }
        break;
    }
    
    if (newValue !== null) {
      const validation = validateValue(newValue);
      if (validation.valid) {
        setLocalValue(formatValue(newValue));
        onChange(newValue);
        announce(`Value changed to ${newValue}`);
      }
    }
  }, [localValue, parseValue, validateValue, onChange, step, constraints, disabled]);
  
  // Update local value when prop changes
  useEffect(() => {
    if (!isFocused) {
      setLocalValue(formatValue(value));
    }
  }, [value, isFocused]);
  
  // Spinner controls
  const handleSpinnerClick = useCallback((direction: 'up' | 'down') => {
    const currentValue = parseValue(localValue) || 0;
    const newValue = direction === 'up' ? currentValue + step : currentValue - step;
    
    const validation = validateValue(newValue);
    if (validation.valid) {
      setLocalValue(formatValue(newValue));
      onChange(newValue);
      announce(`Value ${direction === 'up' ? 'increased' : 'decreased'} to ${newValue}`);
    }
  }, [localValue, parseValue, validateValue, onChange, step]);
  
  const showError = error || validationError;
  const showHelper = helperText || (finalConstraints && (finalConstraints.min !== undefined || finalConstraints.max !== undefined) && uiMode !== 'student');
  
  return (
    <div className={cn('numeric-input-container', className)} data-testid={testId}>
      <label 
        htmlFor={`${testId}-input`}
        className={cn(
          'block text-sm font-medium mb-1',
          disabled ? 'text-gray-400' : 'text-gray-700 dark:text-gray-300'
        )}
      >
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </label>
      
      <div className="relative">
        <input
          ref={inputRef}
          id={`${testId}-input`}
          type="text"
          value={localValue}
          onChange={handleChange}
          onBlur={handleBlur}
          onFocus={() => setIsFocused(true)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          required={required}
          aria-label={ariaLabel || label}
          aria-describedby={cn(
            ariaDescribedBy,
            showError && `${testId}-error`,
            showHelper && `${testId}-helper`
          )}
          aria-invalid={!!showError}
          aria-required={required}
          className={cn(
            'w-full px-3 py-2 border rounded-md transition-colors',
            'focus:outline-none focus:ring-2 focus:ring-offset-2',
            showSpinner && 'pr-10',
            unit && 'pr-16',
            showError
              ? 'border-red-500 focus:ring-red-500'
              : 'border-gray-300 dark:border-gray-600 focus:ring-blue-500',
            disabled && 'bg-gray-100 dark:bg-gray-800 cursor-not-allowed',
            'dark:bg-gray-900 dark:text-gray-100'
          )}
          placeholder={
            scientificNotation ? 'e.g., 1.23e-4' :
            engineeringNotation ? 'e.g., 123e3' :
            'Enter number'
          }
        />
        
        {/* Unit display */}
        {unit && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-gray-500 pointer-events-none">
            {unit}
          </span>
        )}
        
        {/* Spinner controls */}
        {showSpinner && !disabled && (
          <div className="absolute right-1 top-1 bottom-1 flex flex-col">
            <button
              type="button"
              onClick={() => handleSpinnerClick('up')}
              aria-label="Increase value"
              className="flex-1 px-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-t"
              tabIndex={-1}
            >
              <svg className="w-3 h-3" viewBox="0 0 12 12">
                <path d="M6 3L10 7H2L6 3Z" fill="currentColor" />
              </svg>
            </button>
            <button
              type="button"
              onClick={() => handleSpinnerClick('down')}
              aria-label="Decrease value"
              className="flex-1 px-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-b"
              tabIndex={-1}
            >
              <svg className="w-3 h-3" viewBox="0 0 12 12">
                <path d="M6 9L2 5H10L6 9Z" fill="currentColor" />
              </svg>
            </button>
          </div>
        )}
      </div>
      
      {/* Error message */}
      {showError && (
        <div
          id={`${testId}-error`}
          role="alert"
          className="mt-1 text-sm text-red-600 dark:text-red-400 flex items-center"
        >
          <AlertCircle className="w-4 h-4 mr-1" />
          {showError}
        </div>
      )}
      
      {/* Helper text */}
      {showHelper && !showError && (
        <div
          id={`${testId}-helper`}
          className="mt-1 text-sm text-gray-500 dark:text-gray-400 flex items-start"
        >
          {uiMode !== 'student' && <Info className="w-4 h-4 mr-1 mt-0.5 flex-shrink-0" />}
          <span>
            {helperText}
            {finalConstraints && uiMode !== 'student' && (
              <span className="block mt-1">
                {finalConstraints.min !== undefined && `Min: ${finalConstraints.min}`}
                {finalConstraints.min !== undefined && finalConstraints.max !== undefined && ', '}
                {finalConstraints.max !== undefined && `Max: ${finalConstraints.max}`}
                {finalConstraints.precision !== undefined && `, Precision: ${finalConstraints.precision} decimals`}
              </span>
            )}
          </span>
        </div>
      )}
    </div>
  );
};

// Export with display name for debugging
NumericInput.displayName = 'NumericInput';