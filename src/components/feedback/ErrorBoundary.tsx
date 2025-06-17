/**
 * Error Boundary Component with Scientific Context
 * Implements graceful error handling for scientific computations
 * WCAG 2.1 Level AA compliant
 */

import type { ErrorInfo } from 'react';
import { Component } from 'react';
import type { ErrorBoundaryProps } from '@/types/components/feedback';
import type { ScientificError } from '@/types/components';
import { AlertTriangle, RefreshCw, ChevronDown, ChevronUp, FileText, Copy } from 'lucide-react';
import { cn } from '@/utils/cn';
import { announce } from '@/lib/accessibility';

interface ErrorBoundaryState {
  hasError: boolean;
  error: ScientificError | null;
  errorInfo: ErrorInfo | null;
  showDetails: boolean;
  errorId: string;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
      errorId: '',
    };
  }
  
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Generate unique error ID for tracking
    const errorId = `error-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Convert standard error to ScientificError
    const scientificError: ScientificError = {
      code: 'RUNTIME_ERROR',
      message: error.message || 'An unexpected error occurred',
      technicalDetails: error.stack,
      recoverable: true,
    };
    
    return {
      hasError: true,
      error: scientificError,
      errorId,
    };
  }
  
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Enhanced error with scientific context
    const scientificError: ScientificError = {
      code: error.name || 'RUNTIME_ERROR',
      message: error.message || 'An unexpected error occurred',
      scientificContext: {
        dataset: this.extractDatasetInfo(error),
        method: this.extractMethodInfo(error),
        parameters: this.extractParametersInfo(error),
        suggestion: this.generateSuggestion(error),
      },
      technicalDetails: error.stack,
      recoverable: this.isRecoverable(error),
    };
    
    this.setState({
      error: scientificError,
      errorInfo,
    });
    
    // Log error if enabled
    if (this.props.logErrors) {
      console.error('Error caught by boundary:', error, errorInfo);
      console.error('Error ID:', this.state.errorId);
    }
    
    // Call error handler if provided
    this.props.onError?.(scientificError);
    
    // Announce error to screen readers
    announce('An error has occurred. Please check the error details.', 'assertive');
  }
  
  // Extract dataset information from error
  private extractDatasetInfo(error: Error): string | undefined {
    // Look for dataset-related information in error message or stack
    const datasetPattern = /dataset[:\s]+([^\s,]+)/i;
    const match = error.message.match(datasetPattern) || error.stack?.match(datasetPattern);
    return match?.[1];
  }
  
  // Extract method information from error
  private extractMethodInfo(error: Error): string | undefined {
    // Look for method-related information
    const methodPattern = /method[:\s]+([^\s,]+)/i;
    const match = error.message.match(methodPattern) || error.stack?.match(methodPattern);
    return match?.[1];
  }
  
  // Extract parameters information from error
  private extractParametersInfo(error: Error): Record<string, any> | undefined {
    // Attempt to extract parameter information
    try {
      const paramsPattern = /parameters[:\s]+({[^}]+})/i;
      const match = error.message.match(paramsPattern);
      if (match?.[1]) {
        return JSON.parse(match[1]);
      }
    } catch {
      // Ignore JSON parsing errors
    }
    return undefined;
  }
  
  // Generate helpful suggestion based on error
  private generateSuggestion(error: Error): string {
    const message = error.message.toLowerCase();
    
    if (message.includes('memory') || message.includes('oom')) {
      return 'Try reducing the dataset size or closing other applications.';
    }
    
    if (message.includes('timeout')) {
      return 'The operation took too long. Try with a smaller dataset or simpler parameters.';
    }
    
    if (message.includes('invalid') || message.includes('validation')) {
      return 'Check your input parameters and ensure they meet the requirements.';
    }
    
    if (message.includes('network') || message.includes('connection')) {
      return 'Check your internet connection and try again.';
    }
    
    if (message.includes('permission') || message.includes('access')) {
      return 'Ensure you have the necessary permissions to perform this operation.';
    }
    
    return 'Try refreshing the page or contact support if the issue persists.';
  }
  
  // Determine if error is recoverable
  private isRecoverable(error: Error): boolean {
    const message = error.message.toLowerCase();
    
    // Non-recoverable errors
    if (message.includes('critical') || message.includes('fatal')) {
      return false;
    }
    
    // Generally recoverable errors
    return true;
  }
  
  // Handle retry
  private handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
    });
    
    announce('Retrying operation', 'polite');
  };
  
  // Toggle details visibility
  private toggleDetails = () => {
    this.setState(prev => ({
      showDetails: !prev.showDetails,
    }));
  };
  
  // Copy error details to clipboard
  private copyErrorDetails = async () => {
    const { error, errorInfo, errorId } = this.state;
    
    const details = `
Error ID: ${errorId}
Time: ${new Date().toISOString()}
Code: ${error?.code}
Message: ${error?.message}

Scientific Context:
${JSON.stringify(error?.scientificContext, null, 2)}

Technical Details:
${error?.technicalDetails}

Component Stack:
${errorInfo?.componentStack}
    `.trim();
    
    try {
      await navigator.clipboard.writeText(details);
      announce('Error details copied to clipboard', 'polite');
    } catch {
      console.error('Failed to copy error details');
    }
  };
  
  // Generate error report
  private generateReport = () => {
    const { error, errorId } = this.state;
    
    const report = {
      id: errorId,
      timestamp: new Date().toISOString(),
      error: {
        code: error?.code,
        message: error?.message,
        context: error?.scientificContext,
      },
      environment: {
        userAgent: navigator.userAgent,
        url: window.location.href,
      },
    };
    
    // Download as JSON file
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `error-report-${errorId}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    announce('Error report downloaded', 'polite');
  };
  
  render() {
    const { hasError, error, errorInfo, showDetails, errorId } = this.state;
    const { children, fallback: Fallback, showDetails: showDetailsProp = true, 
            allowRetry = true, className, uiMode = 'researcher' } = this.props;
    
    if (!hasError) {
      return children;
    }
    
    // Use custom fallback if provided
    if (Fallback) {
      return <Fallback error={error!} reset={this.handleRetry} />;
    }
    
    // Default error UI
    return (
      <div 
        className={cn(
          'error-boundary min-h-[400px] flex items-center justify-center p-8',
          className
        )}
        role="alert"
        aria-live="assertive"
        aria-atomic="true"
      >
        <div className="max-w-2xl w-full">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
            {/* Error header */}
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0">
                <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400" />
              </div>
              
              <div className="flex-1">
                <h2 className="text-lg font-semibold text-red-800 dark:text-red-200">
                  {error?.scientificContext ? 'Scientific Computation Error' : 'Application Error'}
                </h2>
                
                <p className="mt-2 text-sm text-red-700 dark:text-red-300">
                  {error?.message || 'An unexpected error occurred'}
                </p>
                
                {/* Scientific context */}
                {error?.scientificContext && uiMode !== 'student' && (
                  <div className="mt-4 text-sm text-red-600 dark:text-red-400">
                    {error.scientificContext.dataset && (
                      <p>Dataset: {error.scientificContext.dataset}</p>
                    )}
                    {error.scientificContext.method && (
                      <p>Method: {error.scientificContext.method}</p>
                    )}
                    {error.scientificContext.suggestion && (
                      <p className="mt-2 font-medium">
                        Suggestion: {error.scientificContext.suggestion}
                      </p>
                    )}
                  </div>
                )}
                
                {/* Error ID */}
                {uiMode !== 'student' && (
                  <p className="mt-4 text-xs text-red-500 dark:text-red-500">
                    Error ID: {errorId}
                  </p>
                )}
              </div>
            </div>
            
            {/* Actions */}
            <div className="mt-6 flex flex-wrap gap-2">
              {allowRetry && error?.recoverable !== false && (
                <button
                  onClick={this.handleRetry}
                  className="inline-flex items-center px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors"
                  aria-label="Retry operation"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Retry
                </button>
              )}
              
              {showDetailsProp && uiMode !== 'student' && (
                <button
                  onClick={this.toggleDetails}
                  className="inline-flex items-center px-4 py-2 border border-red-300 dark:border-red-700 hover:bg-red-100 dark:hover:bg-red-900/40 rounded-md transition-colors"
                  aria-expanded={showDetails}
                  aria-controls="error-details"
                >
                  {showDetails ? (
                    <>
                      <ChevronUp className="w-4 h-4 mr-2" />
                      Hide Details
                    </>
                  ) : (
                    <>
                      <ChevronDown className="w-4 h-4 mr-2" />
                      Show Details
                    </>
                  )}
                </button>
              )}
              
              {uiMode === 'expert' && (
                <>
                  <button
                    onClick={this.copyErrorDetails}
                    className="inline-flex items-center px-4 py-2 border border-red-300 dark:border-red-700 hover:bg-red-100 dark:hover:bg-red-900/40 rounded-md transition-colors"
                    aria-label="Copy error details"
                  >
                    <Copy className="w-4 h-4 mr-2" />
                    Copy
                  </button>
                  
                  <button
                    onClick={this.generateReport}
                    className="inline-flex items-center px-4 py-2 border border-red-300 dark:border-red-700 hover:bg-red-100 dark:hover:bg-red-900/40 rounded-md transition-colors"
                    aria-label="Download error report"
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    Report
                  </button>
                </>
              )}
            </div>
            
            {/* Error details */}
            {showDetails && showDetailsProp && (
              <div
                id="error-details"
                className="mt-6 p-4 bg-gray-100 dark:bg-gray-800 rounded-md overflow-auto"
              >
                <h3 className="text-sm font-semibold mb-2">Technical Details</h3>
                
                {error?.technicalDetails && (
                  <pre className="text-xs text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                    {error.technicalDetails}
                  </pre>
                )}
                
                {errorInfo?.componentStack && (
                  <>
                    <h3 className="text-sm font-semibold mt-4 mb-2">Component Stack</h3>
                    <pre className="text-xs text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                      {errorInfo.componentStack}
                    </pre>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }
}

// Export for debugging
// ErrorBoundary.displayName = 'ErrorBoundary';