/**
 * AirImpute Pro Component Library
 * Comprehensive React components for scientific data analysis
 * Following IEEE HCI guidelines and WCAG 2.1 Level AA standards
 */

// Scientific Visualization Components
export { TimeSeriesChart } from './scientific/TimeSeriesChart';
export { CorrelationMatrix } from './scientific/CorrelationMatrix';

// Form Components with Scientific Validation
export { NumericInput } from './forms/NumericInput';

// Layout Components
export { ScientificCard } from './layout/ScientificCard';

// Feedback Components
export { ProgressIndicator } from './feedback/ProgressIndicator';
export { ErrorBoundary } from './feedback/ErrorBoundary';

// Provider Components
export { ThemeProvider, useTheme } from './providers/ThemeProvider';

// Re-export existing UI components
export { Button } from './ui/Button';
export { Card } from './ui/Card';
export { Progress } from './ui/Progress';

// NOTE: Type, constant, and utility exports have been removed to prevent circular dependencies.
// Import these directly from their source modules:
// - Types: import from '@/types/components/*'
// - Constants: import from '@/lib/constants/*'
// - Utilities: import from '@/lib/*'