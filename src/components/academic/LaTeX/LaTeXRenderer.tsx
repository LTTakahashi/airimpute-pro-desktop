import React, { useEffect, useRef, useState, useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { cn } from '@/utils/cn';
import { AlertCircle, Copy, Check, Maximize2, X } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Tooltip } from '@/components/ui/Tooltip';

interface LaTeXRendererProps {
  expression: string;
  displayMode?: boolean;
  className?: string;
  numbered?: boolean;
  label?: string;
  onError?: (error: Error) => void;
  macros?: Record<string, string>;
  allowFullscreen?: boolean;
  allowCopy?: boolean;
}

interface RenderCache {
  [key: string]: {
    html: string;
    timestamp: number;
  };
}

// Global cache for rendered expressions
const renderCache: RenderCache = {};
const CACHE_TTL = 3600000; // 1 hour

export const LaTeXRenderer: React.FC<LaTeXRendererProps> = ({
  expression,
  displayMode = false,
  className,
  numbered = false,
  label,
  onError,
  macros = {},
  allowFullscreen = true,
  allowCopy = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [equationNumber, setEquationNumber] = useState<number | null>(null);

  // Global equation counter
  const equationCounter = useRef(0);

  // Default macros for common operations
  const defaultMacros = {
    '\\RR': '\\mathbb{R}',
    '\\NN': '\\mathbb{N}',
    '\\ZZ': '\\mathbb{Z}',
    '\\QQ': '\\mathbb{Q}',
    '\\CC': '\\mathbb{C}',
    '\\PP': '\\mathbb{P}',
    '\\EE': '\\mathbb{E}',
    '\\Var': '\\operatorname{Var}',
    '\\Cov': '\\operatorname{Cov}',
    '\\Corr': '\\operatorname{Corr}',
    '\\MSE': '\\operatorname{MSE}',
    '\\RMSE': '\\operatorname{RMSE}',
    '\\MAE': '\\operatorname{MAE}',
    '\\argmin': '\\operatorname{arg\\,min}',
    '\\argmax': '\\operatorname{arg\\,max}',
    ...macros,
  };

  // Generate cache key
  const cacheKey = useMemo(() => {
    const key = `${expression}_${displayMode}_${JSON.stringify(defaultMacros)}`;
    return btoa(key).replace(/[^a-zA-Z0-9]/g, '');
  }, [expression, displayMode, defaultMacros]);

  useEffect(() => {
    if (!containerRef.current) return;

    // Check cache first
    const cached = renderCache[cacheKey];
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
      containerRef.current.innerHTML = cached.html;
      setError(null);
      if (numbered && !equationNumber) {
        equationCounter.current += 1;
        setEquationNumber(equationCounter.current);
      }
      return;
    }

    try {
      // Clear any previous content
      containerRef.current.innerHTML = '';
      
      // Render with KaTeX
      const html = katex.renderToString(expression, {
        displayMode,
        throwOnError: false,
        errorColor: '#ef4444',
        macros: defaultMacros,
        trust: false,  // Fix XSS vulnerability - set to false
        strict: false,
        output: 'html',
        minRuleThickness: 0.04,
        maxSize: 500,
        maxExpand: 1000,
      });

      // Update cache
      renderCache[cacheKey] = {
        html,
        timestamp: Date.now(),
      };

      containerRef.current.innerHTML = html;
      setError(null);

      // Handle equation numbering
      if (numbered && !equationNumber) {
        equationCounter.current += 1;
        setEquationNumber(equationCounter.current);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      
      if (onError && err instanceof Error) {
        onError(err);
      }

      // Fallback: render as plain text with error styling
      containerRef.current.innerHTML = `
        <span style="color: #ef4444; font-family: monospace;">
          ${expression.replace(/</g, '&lt;').replace(/>/g, '&gt;')}
        </span>
      `;
    }
  }, [expression, displayMode, cacheKey, numbered, equationNumber, onError, defaultMacros]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(expression);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy LaTeX:', err);
    }
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  // Clean up cache periodically
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      Object.keys(renderCache).forEach(key => {
        if (now - renderCache[key].timestamp > CACHE_TTL) {
          delete renderCache[key];
        }
      });
    }, CACHE_TTL);

    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <div
        className={cn(
          'relative group inline-block',
          displayMode && 'block text-center my-4',
          className
        )}
      >
        {/* Equation number */}
        {numbered && equationNumber && displayMode && (
          <span className="absolute right-0 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-sm">
            ({equationNumber})
          </span>
        )}

        {/* Label */}
        {label && (
          <span className="absolute -left-20 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-sm">
            {label}:
          </span>
        )}

        {/* LaTeX content */}
        <div
          ref={containerRef}
          className={cn(
            'katex-container',
            error && 'border border-red-500 rounded p-2'
          )}
        />

        {/* Error indicator */}
        {error && (
          <Tooltip content={error}>
            <AlertCircle className="absolute -right-6 top-1/2 -translate-y-1/2 h-4 w-4 text-red-500" />
          </Tooltip>
        )}

        {/* Action buttons */}
        <div className={cn(
          'absolute top-0 right-0 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity',
          'bg-white dark:bg-gray-800 rounded shadow-sm p-1',
          displayMode ? '-top-8' : '-top-6'
        )}>
          {allowCopy && (
            <Tooltip content={copied ? 'Copied!' : 'Copy LaTeX'}>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCopy}
                className="h-6 w-6 p-0"
              >
                {copied ? (
                  <Check className="h-3 w-3 text-green-500" />
                ) : (
                  <Copy className="h-3 w-3" />
                )}
              </Button>
            </Tooltip>
          )}
          
          {allowFullscreen && displayMode && (
            <Tooltip content="View fullscreen">
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleFullscreen}
                className="h-6 w-6 p-0"
              >
                <Maximize2 className="h-3 w-3" />
              </Button>
            </Tooltip>
          )}
        </div>
      </div>

      {/* Fullscreen modal */}
      {isFullscreen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
          <div className="relative max-w-4xl w-full mx-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleFullscreen}
              className="absolute -top-10 right-0 text-white hover:text-gray-300"
            >
              <X className="h-6 w-6" />
            </Button>
            
            <div className="bg-white dark:bg-gray-900 rounded-lg p-8 shadow-2xl">
              <LaTeXRenderer
                expression={expression}
                displayMode={true}
                macros={macros}
                allowFullscreen={false}
              />
              
              {label && (
                <p className="text-center mt-4 text-gray-600 dark:text-gray-400">
                  {label}
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

// Memoized version for performance
export const MemoizedLaTeXRenderer = React.memo(LaTeXRenderer);

// Batch renderer for multiple equations
export const LaTeXBatchRenderer: React.FC<{
  equations: Array<{
    expression: string;
    label?: string;
    displayMode?: boolean;
  }>;
  numbered?: boolean;
  className?: string;
}> = ({ equations, numbered = true, className }) => {
  return (
    <div className={cn('space-y-4', className)}>
      {equations.map((eq, index) => (
        <LaTeXRenderer
          key={`${eq.expression}_${index}`}
          expression={eq.expression}
          label={eq.label}
          displayMode={eq.displayMode ?? true}
          numbered={numbered}
        />
      ))}
    </div>
  );
};

// Inline LaTeX helper component
export const InlineLaTeX: React.FC<{ children: string }> = ({ children }) => (
  <LaTeXRenderer expression={children} displayMode={false} />
);

// Display LaTeX helper component
export const DisplayLaTeX: React.FC<{ 
  children: string;
  label?: string;
  numbered?: boolean;
}> = ({ children, label, numbered }) => (
  <LaTeXRenderer 
    expression={children} 
    displayMode={true} 
    label={label}
    numbered={numbered}
  />
);