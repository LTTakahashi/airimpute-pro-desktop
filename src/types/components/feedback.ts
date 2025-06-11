/**
 * Feedback component type definitions
 * For progress tracking, status indication, and error handling
 */

import { ScientificComponentProps, ComputationProgress, ScientificError } from './index';

// Progress indicator for long computations
export interface ProgressIndicatorProps extends ScientificComponentProps {
  progress: ComputationProgress | number;
  variant?: 'linear' | 'circular' | 'steps';
  size?: 'small' | 'medium' | 'large';
  showLabel?: boolean;
  showPercentage?: boolean;
  showTimeRemaining?: boolean;
  showSubProgress?: boolean;
  color?: 'primary' | 'success' | 'warning' | 'error';
  indeterminate?: boolean;
  onCancel?: () => void;
}

// Status indicator for system/computation states
export interface StatusIndicatorProps extends ScientificComponentProps {
  status: 'idle' | 'running' | 'success' | 'warning' | 'error' | 'paused';
  label?: string;
  message?: string;
  icon?: boolean;
  pulse?: boolean;
  size?: 'small' | 'medium' | 'large';
  showTimestamp?: boolean;
}

// Alert/notification component
export interface AlertProps extends ScientificComponentProps {
  severity: 'success' | 'info' | 'warning' | 'error';
  title?: string;
  message: string;
  dismissible?: boolean;
  onDismiss?: () => void;
  action?: {
    label: string;
    onClick: () => void;
  };
  icon?: boolean;
  details?: string;
  expandable?: boolean;
  autoHideDuration?: number;
}

// Error boundary component
export interface ErrorBoundaryProps extends ScientificComponentProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error: ScientificError; reset: () => void }>;
  onError?: (error: ScientificError) => void;
  showDetails?: boolean;
  allowRetry?: boolean;
  logErrors?: boolean;
}

// Loading states
export interface LoadingProps extends ScientificComponentProps {
  loading: boolean;
  variant?: 'spinner' | 'skeleton' | 'shimmer' | 'dots';
  size?: 'small' | 'medium' | 'large';
  message?: string;
  overlay?: boolean;
  blur?: boolean;
}

// Empty state component
export interface EmptyStateProps extends ScientificComponentProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  suggestions?: string[];
  variant?: 'default' | 'error' | 'search';
}

// Tooltip with scientific context
export interface TooltipProps extends ScientificComponentProps {
  content: React.ReactNode;
  placement?: 'top' | 'bottom' | 'left' | 'right' | 'auto';
  trigger?: 'hover' | 'click' | 'focus';
  delay?: number;
  interactive?: boolean;
  arrow?: boolean;
  maxWidth?: number;
  scientificFormat?: boolean;
}

// Confirmation dialog
export interface ConfirmationDialogProps extends ScientificComponentProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  severity?: 'info' | 'warning' | 'error';
  showConsequences?: boolean;
  consequences?: string[];
}

// Step indicator for multi-step processes
export interface StepIndicatorProps extends ScientificComponentProps {
  steps: {
    id: string;
    label: string;
    description?: string;
    status: 'pending' | 'active' | 'completed' | 'error' | 'skipped';
    optional?: boolean;
    error?: string;
  }[];
  orientation?: 'horizontal' | 'vertical';
  variant?: 'dots' | 'numbers' | 'icons';
  connector?: boolean;
  clickable?: boolean;
  onStepClick?: (stepId: string) => void;
}

// Badge for status/count indication
export interface BadgeProps extends ScientificComponentProps {
  content: string | number;
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error' | 'info';
  size?: 'small' | 'medium' | 'large';
  max?: number;
  showZero?: boolean;
  dot?: boolean;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
}

// Snackbar/Toast notification
export interface SnackbarProps extends ScientificComponentProps {
  open: boolean;
  onClose: () => void;
  message: string;
  severity?: 'success' | 'info' | 'warning' | 'error';
  duration?: number;
  position?: {
    vertical: 'top' | 'bottom';
    horizontal: 'left' | 'center' | 'right';
  };
  action?: {
    label: string;
    onClick: () => void;
  };
}

// Help/Info component
export interface HelpProps extends ScientificComponentProps {
  title?: string;
  content: React.ReactNode;
  type?: 'info' | 'tip' | 'warning' | 'example';
  collapsible?: boolean;
  defaultExpanded?: boolean;
  showIcon?: boolean;
  citations?: string[];
}