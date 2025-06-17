import React, { useState, useRef, useEffect, useCallback } from 'react';
import { LaTeXRenderer } from './LaTeXRenderer';
import { Button } from '@/components/ui/Button';
import { Tabs } from '@/components/ui/Tabs';
import { cn } from '@/utils/cn';
import { 
  Undo,
  Redo,
  Save,
  HelpCircle
} from 'lucide-react';
import { Tooltip } from '@/components/ui/Tooltip';

interface LaTeXEquationEditorProps {
  initialValue?: string;
  onChange?: (value: string) => void;
  onSave?: (value: string) => void;
  className?: string;
  height?: string;
  showPreview?: boolean;
  showSymbolPalette?: boolean;
  autoSave?: boolean;
  autoSaveDelay?: number;
}

interface SymbolGroup {
  name: string;
  symbols: Array<{
    latex: string;
    display: string;
    tooltip: string;
    insert?: string; // Optional custom insertion string
  }>;
}

const symbolGroups: SymbolGroup[] = [
  {
    name: 'Common',
    symbols: [
      { latex: '\\sum', display: '∑', tooltip: 'Sum', insert: '\\sum_{i=1}^{n}' },
      { latex: '\\prod', display: '∏', tooltip: 'Product', insert: '\\prod_{i=1}^{n}' },
      { latex: '\\int', display: '∫', tooltip: 'Integral', insert: '\\int_{a}^{b}' },
      { latex: '\\frac{}{}', display: '÷', tooltip: 'Fraction', insert: '\\frac{}{|}' },
      { latex: '\\sqrt{}', display: '√', tooltip: 'Square root', insert: '\\sqrt{|}' },
      { latex: '^{}', display: 'x²', tooltip: 'Superscript', insert: '^{|}' },
      { latex: '_{}', display: 'x₁', tooltip: 'Subscript', insert: '_{|}' },
      { latex: '\\infty', display: '∞', tooltip: 'Infinity' },
    ],
  },
  {
    name: 'Greek',
    symbols: [
      { latex: '\\alpha', display: 'α', tooltip: 'Alpha' },
      { latex: '\\beta', display: 'β', tooltip: 'Beta' },
      { latex: '\\gamma', display: 'γ', tooltip: 'Gamma' },
      { latex: '\\delta', display: 'δ', tooltip: 'Delta' },
      { latex: '\\epsilon', display: 'ε', tooltip: 'Epsilon' },
      { latex: '\\theta', display: 'θ', tooltip: 'Theta' },
      { latex: '\\lambda', display: 'λ', tooltip: 'Lambda' },
      { latex: '\\mu', display: 'μ', tooltip: 'Mu' },
      { latex: '\\sigma', display: 'σ', tooltip: 'Sigma' },
      { latex: '\\phi', display: 'φ', tooltip: 'Phi' },
    ],
  },
  {
    name: 'Operators',
    symbols: [
      { latex: '\\times', display: '×', tooltip: 'Times' },
      { latex: '\\div', display: '÷', tooltip: 'Divide' },
      { latex: '\\pm', display: '±', tooltip: 'Plus/minus' },
      { latex: '\\leq', display: '≤', tooltip: 'Less than or equal' },
      { latex: '\\geq', display: '≥', tooltip: 'Greater than or equal' },
      { latex: '\\neq', display: '≠', tooltip: 'Not equal' },
      { latex: '\\approx', display: '≈', tooltip: 'Approximately' },
      { latex: '\\equiv', display: '≡', tooltip: 'Equivalent' },
    ],
  },
  {
    name: 'Statistics',
    symbols: [
      { latex: '\\bar{x}', display: 'x̄', tooltip: 'Mean', insert: '\\bar{|}' },
      { latex: '\\hat{x}', display: 'x̂', tooltip: 'Estimate', insert: '\\hat{|}' },
      { latex: '\\tilde{x}', display: 'x̃', tooltip: 'Median', insert: '\\tilde{|}' },
      { latex: '\\mathbb{E}', display: '𝔼', tooltip: 'Expected value' },
      { latex: '\\mathbb{P}', display: 'ℙ', tooltip: 'Probability' },
      { latex: '\\mathcal{N}', display: '𝒩', tooltip: 'Normal distribution' },
      { latex: '\\sim', display: '∼', tooltip: 'Distributed as' },
      { latex: '\\perp', display: '⊥', tooltip: 'Independent' },
    ],
  },
  {
    name: 'Matrices',
    symbols: [
      { 
        latex: '\\begin{matrix} a & b \\\\ c & d \\end{matrix}', 
        display: 'M', 
        tooltip: 'Matrix',
        insert: '\\begin{matrix}\n  a & b \\\\\n  c & d\n\\end{matrix}'
      },
      { 
        latex: '\\begin{pmatrix} a \\\\ b \\end{pmatrix}', 
        display: '( )', 
        tooltip: 'Parentheses matrix',
        insert: '\\begin{pmatrix}\n  a \\\\\n  b\n\\end{pmatrix}'
      },
      { 
        latex: '\\begin{bmatrix} a \\\\ b \\end{bmatrix}', 
        display: '[ ]', 
        tooltip: 'Bracket matrix',
        insert: '\\begin{bmatrix}\n  a \\\\\n  b\n\\end{bmatrix}'
      },
      { latex: '\\det', display: 'det', tooltip: 'Determinant' },
      { latex: '\\text{rank}', display: 'rank', tooltip: 'Rank' },
    ],
  },
];

const commonTemplates = [
  {
    name: 'Regression',
    template: 'y_i = \\beta_0 + \\beta_1 x_i + \\epsilon_i',
    description: 'Simple linear regression',
  },
  {
    name: 'Mean',
    template: '\\bar{x} = \\frac{1}{n} \\sum_{i=1}^{n} x_i',
    description: 'Sample mean',
  },
  {
    name: 'Variance',
    template: 's^2 = \\frac{1}{n-1} \\sum_{i=1}^{n} (x_i - \\bar{x})^2',
    description: 'Sample variance',
  },
  {
    name: 'Normal',
    template: 'X \\sim \\mathcal{N}(\\mu, \\sigma^2)',
    description: 'Normal distribution',
  },
  {
    name: 'MSE',
    template: '\\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2',
    description: 'Mean squared error',
  },
  {
    name: 'R-squared',
    template: 'R^2 = 1 - \\frac{\\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n} (y_i - \\bar{y})^2}',
    description: 'Coefficient of determination',
  },
];

export const LaTeXEquationEditor: React.FC<LaTeXEquationEditorProps> = ({
  initialValue = '',
  onChange,
  onSave,
  className,
  height = '200px',
  showPreview = true,
  showSymbolPalette = true,
  autoSave = false,
  autoSaveDelay = 2000,
}) => {
  const [value, setValue] = useState(initialValue);
  const [cursorPosition, setCursorPosition] = useState(0);
  const [history, setHistory] = useState<string[]>([initialValue]);
  const [historyIndex, setHistoryIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const autoSaveTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Update cursor position
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.setSelectionRange(cursorPosition, cursorPosition);
      textarea.focus();
    }
  }, [cursorPosition]);

  // Auto-save functionality
  useEffect(() => {
    if (autoSave && onSave) {
      if (autoSaveTimerRef.current) {
        clearTimeout(autoSaveTimerRef.current);
      }
      
      autoSaveTimerRef.current = setTimeout(() => {
        onSave(value);
      }, autoSaveDelay);
    }
    
    return () => {
      if (autoSaveTimerRef.current) {
        clearTimeout(autoSaveTimerRef.current);
      }
    };
  }, [value, autoSave, autoSaveDelay, onSave]);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    setValue(newValue);
    setCursorPosition(e.target.selectionStart);
    
    // Update history
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(newValue);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
    
    if (onChange) {
      onChange(newValue);
    }
  };

  const insertAtCursor = useCallback((text: string) => {
    if (!textareaRef.current) return;
    
    const textarea = textareaRef.current;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const newValue = value.substring(0, start) + text + value.substring(end);
    
    setValue(newValue);
    
    // Find cursor position marker |
    const cursorMarker = text.indexOf('|');
    if (cursorMarker !== -1) {
      // Remove cursor marker and position cursor there
      const cleanText = text.replace('|', '');
      const cleanValue = value.substring(0, start) + cleanText + value.substring(end);
      setValue(cleanValue);
      setCursorPosition(start + cursorMarker);
    } else {
      setCursorPosition(start + text.length);
    }
    
    // Focus back on textarea
    setTimeout(() => textarea.focus(), 0);
  }, [value]);

  const handleSymbolClick = (symbol: typeof symbolGroups[0]['symbols'][0]) => {
    insertAtCursor(symbol.insert || symbol.latex);
  };

  const handleTemplateClick = (template: string) => {
    setValue(template);
    setCursorPosition(template.length);
  };

  const handleUndo = () => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      setValue(history[newIndex]);
    }
  };

  const handleRedo = () => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      setValue(history[newIndex]);
    }
  };

  const handleSave = () => {
    if (onSave) {
      onSave(value);
    }
  };

  const handleLaTeXError = (err: Error) => {
    setError(err.message);
  };

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 p-2">
        <div className="flex items-center gap-1">
          <Tooltip content="Undo">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleUndo}
              disabled={historyIndex === 0}
              className="h-8 w-8 p-0"
            >
              <Undo className="h-4 w-4" />
            </Button>
          </Tooltip>
          
          <Tooltip content="Redo">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRedo}
              disabled={historyIndex === history.length - 1}
              className="h-8 w-8 p-0"
            >
              <Redo className="h-4 w-4" />
            </Button>
          </Tooltip>
          
          <div className="w-px h-6 bg-gray-300 dark:bg-gray-600 mx-1" />
          
          {onSave && (
            <Tooltip content="Save">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleSave}
                className="h-8 px-2"
              >
                <Save className="h-4 w-4 mr-1" />
                Save
              </Button>
            </Tooltip>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          <Tooltip content="LaTeX Reference">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => window.open('https://katex.org/docs/supported.html', '_blank')}
              className="h-8 w-8 p-0"
            >
              <HelpCircle className="h-4 w-4" />
            </Button>
          </Tooltip>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Symbol Palette */}
        {showSymbolPalette && (
          <div className="w-64 border-r border-gray-200 dark:border-gray-700 overflow-y-auto">
            <Tabs defaultValue="symbols" className="h-full">
              <div className="border-b border-gray-200 dark:border-gray-700">
                <div className="flex">
                  <button
                    className="flex-1 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white border-b-2 border-transparent data-[state=active]:border-blue-500"
                    data-state="active"
                  >
                    Symbols
                  </button>
                  <button
                    className="flex-1 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white border-b-2 border-transparent"
                  >
                    Templates
                  </button>
                </div>
              </div>
              
              <div className="p-2">
                {/* Symbols Tab */}
                <div>
                  {symbolGroups.map((group) => (
                    <div key={group.name} className="mb-4">
                      <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
                        {group.name}
                      </h3>
                      <div className="grid grid-cols-4 gap-1">
                        {group.symbols.map((symbol) => (
                          <Tooltip key={symbol.latex} content={symbol.tooltip}>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleSymbolClick(symbol)}
                              className="h-10 w-full text-lg font-mono"
                            >
                              {symbol.display}
                            </Button>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
                
                {/* Templates Tab */}
                <div className="hidden">
                  <div className="space-y-2">
                    {commonTemplates.map((template) => (
                      <button
                        key={template.name}
                        onClick={() => handleTemplateClick(template.template)}
                        className="w-full text-left p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-800"
                      >
                        <div className="text-sm font-medium">{template.name}</div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {template.description}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </Tabs>
          </div>
        )}

        {/* Editor and Preview */}
        <div className="flex-1 flex flex-col">
          <div className={cn('flex-1 flex', showPreview && 'grid grid-cols-2')}>
            {/* Editor */}
            <div className="flex-1 flex flex-col">
              <div className="bg-gray-100 dark:bg-gray-800 px-3 py-1 text-xs font-medium text-gray-600 dark:text-gray-400">
                LaTeX Editor
              </div>
              <textarea
                ref={textareaRef}
                value={value}
                onChange={handleChange}
                className={cn(
                  'flex-1 p-4 font-mono text-sm resize-none',
                  'bg-white dark:bg-gray-900',
                  'text-gray-900 dark:text-gray-100',
                  'border-0 focus:outline-none focus:ring-0',
                  error && 'text-red-600 dark:text-red-400'
                )}
                style={{ minHeight: height }}
                placeholder="Enter LaTeX equation..."
                spellCheck={false}
              />
            </div>

            {/* Preview */}
            {showPreview && (
              <div className="flex-1 flex flex-col border-l border-gray-200 dark:border-gray-700">
                <div className="bg-gray-100 dark:bg-gray-800 px-3 py-1 text-xs font-medium text-gray-600 dark:text-gray-400">
                  Preview
                </div>
                <div className="flex-1 p-4 overflow-auto bg-white dark:bg-gray-900">
                  {value ? (
                    <LaTeXRenderer
                      expression={value}
                      displayMode={true}
                      onError={handleLaTeXError}
                    />
                  ) : (
                    <div className="text-gray-400 dark:text-gray-600 text-center">
                      Preview will appear here
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
          
          {/* Error display */}
          {error && (
            <div className="px-4 py-2 bg-red-50 dark:bg-red-900/20 border-t border-red-200 dark:border-red-800">
              <p className="text-sm text-red-600 dark:text-red-400">
                Error: {error}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};