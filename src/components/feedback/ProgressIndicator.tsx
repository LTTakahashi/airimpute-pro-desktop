/**
 * Progress Indicator Component for Long Computations
 * Implements IEEE standards for scientific computing feedback
 * WCAG 2.1 Level AA compliant
 */

import React, { useMemo, useEffect, useCallback } from 'react';
import { ProgressIndicatorProps } from '@/types/components/feedback';
import { ComputationProgress } from '@/types/components';
import { announce, getScientificAriaProps } from '@/lib/accessibility';
import { cn } from '@/utils/cn';
import { X, Pause, Play } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  progress,
  variant = 'linear',
  size = 'medium',
  showLabel = true,
  showPercentage = true,
  showTimeRemaining = true,
  showSubProgress = true,
  color = 'primary',
  indeterminate = false,
  onCancel,
  className,
  'aria-label': ariaLabel,
  'aria-describedby': ariaDescribedBy,
  uiMode = 'researcher',
  testId = 'progress-indicator',
}) => {
  // Normalize progress data
  const progressData = useMemo((): ComputationProgress => {
    if (typeof progress === 'number') {
      return {
        phase: 'Processing',
        progress: Math.min(100, Math.max(0, progress)),
        currentOperation: undefined,
        estimatedTimeRemaining: undefined,
        subProgress: undefined,
      };
    }
    return progress;
  }, [progress]);
  
  // Format time remaining
  const formatTimeRemaining = useCallback((seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
    return `${Math.round(seconds / 86400)}d ${Math.round((seconds % 86400) / 3600)}h`;
  }, []);
  
  // Announce progress updates for screen readers
  useEffect(() => {
    if (progressData.progress % 10 === 0 && progressData.progress > 0) {
      announce(
        `Progress: ${progressData.progress}% complete. ${progressData.phase}`,
        'polite'
      );
    }
  }, [progressData.progress, progressData.phase]);
  
  // Size classes
  const sizeClasses = {
    small: {
      height: 'h-1',
      circleSize: 'w-12 h-12',
      fontSize: 'text-xs',
      strokeWidth: 2,
    },
    medium: {
      height: 'h-2',
      circleSize: 'w-24 h-24',
      fontSize: 'text-sm',
      strokeWidth: 3,
    },
    large: {
      height: 'h-3',
      circleSize: 'w-32 h-32',
      fontSize: 'text-base',
      strokeWidth: 4,
    },
  };
  
  // Color classes
  const colorClasses = {
    primary: 'bg-blue-500 text-blue-500',
    success: 'bg-green-500 text-green-500',
    warning: 'bg-yellow-500 text-yellow-500',
    error: 'bg-red-500 text-red-500',
  };
  
  const progressAriaProps = getScientificAriaProps('progress', {
    label: progressData.phase,
    current: progressData.progress,
    min: 0,
    max: 100,
    text: `${progressData.progress}% complete`,
  });
  
  // Render linear progress
  const renderLinearProgress = () => (
    <div className="w-full">
      {/* Main progress bar */}
      <div className="relative">
        <div 
          className={cn(
            'w-full bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden',
            sizeClasses[size].height
          )}
        >
          {indeterminate ? (
            <motion.div
              className={cn('h-full rounded-full', colorClasses[color])}
              animate={{
                x: ['0%', '100%'],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: 'linear',
              }}
              style={{ width: '30%' }}
            />
          ) : (
            <motion.div
              className={cn('h-full rounded-full', colorClasses[color])}
              initial={{ width: '0%' }}
              animate={{ width: `${progressData.progress}%` }}
              transition={{ duration: 0.3 }}
            />
          )}
        </div>
        
        {/* Cancel button */}
        {onCancel && uiMode !== 'student' && (
          <button
            onClick={onCancel}
            aria-label="Cancel operation"
            className="absolute -right-8 top-1/2 -translate-y-1/2 p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
      
      {/* Sub-progress bars */}
      {showSubProgress && progressData.subProgress && progressData.subProgress.length > 0 && (
        <div className="mt-2 space-y-1">
          {progressData.subProgress.map((sub, index) => (
            <div key={index} className="flex items-center gap-2">
              <span className={cn('text-xs text-gray-600 dark:text-gray-400 w-24 truncate')}>
                {sub.label}
              </span>
              <div className="flex-1 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gray-400 dark:bg-gray-500 rounded-full"
                  initial={{ width: '0%' }}
                  animate={{ width: `${sub.progress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
              <span className="text-xs text-gray-500">{sub.progress}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
  
  // Render circular progress
  const renderCircularProgress = () => {
    const radius = size === 'small' ? 20 : size === 'medium' ? 45 : 60;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (progressData.progress / 100) * circumference;
    
    return (
      <div className="relative inline-flex items-center justify-center">
        <svg className={cn(sizeClasses[size].circleSize, 'transform -rotate-90')}>
          {/* Background circle */}
          <circle
            cx="50%"
            cy="50%"
            r={radius}
            stroke="currentColor"
            strokeWidth={sizeClasses[size].strokeWidth}
            fill="none"
            className="text-gray-200 dark:text-gray-700"
          />
          
          {/* Progress circle */}
          {indeterminate ? (
            <motion.circle
              cx="50%"
              cy="50%"
              r={radius}
              stroke="currentColor"
              strokeWidth={sizeClasses[size].strokeWidth}
              fill="none"
              className={colorClasses[color]}
              strokeDasharray={`${circumference * 0.25} ${circumference * 0.75}`}
              animate={{ rotate: 360 }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: 'linear',
              }}
            />
          ) : (
            <motion.circle
              cx="50%"
              cy="50%"
              r={radius}
              stroke="currentColor"
              strokeWidth={sizeClasses[size].strokeWidth}
              fill="none"
              className={colorClasses[color]}
              strokeDasharray={circumference}
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset }}
              transition={{ duration: 0.3 }}
              strokeLinecap="round"
            />
          )}
        </svg>
        
        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {showPercentage && !indeterminate && (
            <span className={cn(sizeClasses[size].fontSize, 'font-semibold')}>
              {progressData.progress}%
            </span>
          )}
          {onCancel && size !== 'small' && (
            <button
              onClick={onCancel}
              aria-label="Cancel operation"
              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded mt-1"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>
    );
  };
  
  // Render steps progress
  const renderStepsProgress = () => {
    const steps = Math.max(3, Math.ceil(100 / 20)); // Minimum 3 steps
    const activeStep = Math.floor((progressData.progress / 100) * steps);
    
    return (
      <div className="flex items-center gap-2">
        {Array.from({ length: steps }, (_, i) => (
          <div
            key={i}
            className={cn(
              'flex-1 h-2 rounded-full transition-colors duration-300',
              i < activeStep
                ? colorClasses[color]
                : i === activeStep && !indeterminate
                ? 'bg-gray-300 dark:bg-gray-600'
                : 'bg-gray-200 dark:bg-gray-700'
            )}
          >
            {i === activeStep && indeterminate && (
              <motion.div
                className={cn('h-full rounded-full', colorClasses[color])}
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
            )}
          </div>
        ))}
      </div>
    );
  };
  
  return (
    <div
      className={cn('progress-indicator', className)}
      data-testid={testId}
      {...progressAriaProps}
      aria-label={ariaLabel || `Progress: ${progressData.progress}%`}
      aria-describedby={ariaDescribedBy}
      aria-busy={progressData.progress < 100}
    >
      {/* Label and metadata */}
      {(showLabel || showTimeRemaining) && (
        <div className="flex items-center justify-between mb-2">
          {showLabel && (
            <div className="flex flex-col">
              <span className={cn(sizeClasses[size].fontSize, 'font-medium')}>
                {progressData.phase}
              </span>
              {progressData.currentOperation && uiMode !== 'student' && (
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {progressData.currentOperation}
                </span>
              )}
            </div>
          )}
          
          {showTimeRemaining && progressData.estimatedTimeRemaining && (
            <span className={cn(sizeClasses[size].fontSize, 'text-gray-500 dark:text-gray-400')}>
              {formatTimeRemaining(progressData.estimatedTimeRemaining)} remaining
            </span>
          )}
        </div>
      )}
      
      {/* Progress visualization */}
      {variant === 'linear' && renderLinearProgress()}
      {variant === 'circular' && renderCircularProgress()}
      {variant === 'steps' && renderStepsProgress()}
      
      {/* Screen reader announcements */}
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        Progress: {progressData.progress}% complete. 
        {progressData.phase} phase.
        {progressData.estimatedTimeRemaining && 
          ` Estimated time remaining: ${formatTimeRemaining(progressData.estimatedTimeRemaining)}.`
        }
      </div>
    </div>
  );
};

// Export with display name for debugging
ProgressIndicator.displayName = 'ProgressIndicator';