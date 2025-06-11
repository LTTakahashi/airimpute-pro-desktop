import React, { useState } from 'react';
import { LaTeXRenderer, LaTeXBatchRenderer } from '../LaTeX/LaTeXRenderer';
import { Card } from '@/components/ui/Card';
import { Tabs } from '@/components/ui/Tabs';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { cn } from '@/utils/cn';
import { 
  BookOpen, 
  Code, 
  BarChart, 
  Clock,
  Zap,
  AlertCircle,
  CheckCircle,
  XCircle,
  Copy,
  ExternalLink
} from 'lucide-react';
import { methodDocumentations } from './methodDefinitions';

export interface MethodDocumentation {
  id: string;
  name: string;
  category: 'classical' | 'statistical' | 'machine_learning' | 'deep_learning' | 'hybrid';
  description: string;
  mathematical_formulation: {
    main_equation: string;
    auxiliary_equations?: Array<{
      label: string;
      equation: string;
    }>;
    constraints?: string[];
    objective_function?: string;
  };
  algorithm: {
    steps: string[];
    complexity: {
      time: string;
      space: string;
    };
    convergence?: string;
  };
  parameters: Array<{
    name: string;
    symbol: string;
    description: string;
    default?: string | number;
    range?: string;
  }>;
  assumptions: string[];
  advantages: string[];
  limitations: string[];
  use_cases: string[];
  references: Array<{
    title: string;
    authors: string[];
    year: number;
    journal?: string;
    doi?: string;
    url?: string;
  }>;
  implementation_notes?: string[];
  example_code?: string;
}

interface MethodDocumentationProps {
  methodId: string;
  className?: string;
  showExample?: boolean;
  showReferences?: boolean;
  onCitationGenerate?: (method: MethodDocumentation) => void;
}

export const MethodDocumentationViewer: React.FC<MethodDocumentationProps> = ({
  methodId,
  className,
  showExample = true,
  showReferences = true,
  onCitationGenerate,
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [copiedCode, setCopiedCode] = useState(false);
  
  const method = methodDocumentations[methodId];
  
  if (!method) {
    return (
      <Card className={cn('p-6', className)}>
        <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
          <AlertCircle className="h-5 w-5" />
          <p>Method documentation not found for: {methodId}</p>
        </div>
      </Card>
    );
  }

  const categoryColors = {
    classical: 'blue',
    statistical: 'green',
    machine_learning: 'purple',
    deep_learning: 'orange',
    hybrid: 'pink',
  };

  const handleCopyCode = async () => {
    if (method.example_code) {
      await navigator.clipboard.writeText(method.example_code);
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 2000);
    }
  };

  const handleCitationClick = () => {
    if (onCitationGenerate) {
      onCitationGenerate(method);
    }
  };

  return (
    <Card className={cn('overflow-hidden', className)}>
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                {method.name}
              </h2>
              <Badge variant={categoryColors[method.category] as any}>
                {method.category.replace('_', ' ')}
              </Badge>
            </div>
            <p className="text-gray-600 dark:text-gray-400">
              {method.description}
            </p>
          </div>
          {onCitationGenerate && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleCitationClick}
              className="flex items-center gap-2"
            >
              <BookOpen className="h-4 w-4" />
              Cite
            </Button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
        <div className="border-b border-gray-200 dark:border-gray-700">
          <div className="flex px-6">
            <button
              onClick={() => setActiveTab('overview')}
              className={cn(
                'px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'overview'
                  ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                  : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
              )}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('mathematics')}
              className={cn(
                'px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'mathematics'
                  ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                  : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
              )}
            >
              Mathematics
            </button>
            <button
              onClick={() => setActiveTab('algorithm')}
              className={cn(
                'px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'algorithm'
                  ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                  : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
              )}
            >
              Algorithm
            </button>
            {showExample && method.example_code && (
              <button
                onClick={() => setActiveTab('example')}
                className={cn(
                  'px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                  activeTab === 'example'
                    ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                    : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
                )}
              >
                Example
              </button>
            )}
            {showReferences && (
              <button
                onClick={() => setActiveTab('references')}
                className={cn(
                  'px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                  activeTab === 'references'
                    ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                    : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
                )}
              >
                References
              </button>
            )}
          </div>
        </div>

        <div className="p-6">
          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* Parameters */}
              <div>
                <h3 className="text-lg font-semibold mb-3">Parameters</h3>
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                  <table className="w-full">
                    <thead>
                      <tr className="text-left text-sm text-gray-600 dark:text-gray-400">
                        <th className="pb-2">Parameter</th>
                        <th className="pb-2">Symbol</th>
                        <th className="pb-2">Description</th>
                        <th className="pb-2">Default</th>
                      </tr>
                    </thead>
                    <tbody className="text-sm">
                      {method.parameters.map((param) => (
                        <tr key={param.name} className="border-t border-gray-200 dark:border-gray-700">
                          <td className="py-2 font-mono">{param.name}</td>
                          <td className="py-2">
                            <LaTeXRenderer expression={param.symbol} />
                          </td>
                          <td className="py-2 text-gray-600 dark:text-gray-400">
                            {param.description}
                          </td>
                          <td className="py-2 font-mono">
                            {param.default ?? 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Assumptions */}
              <div>
                <h3 className="text-lg font-semibold mb-3">Assumptions</h3>
                <ul className="space-y-2">
                  {method.assumptions.map((assumption, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <AlertCircle className="h-4 w-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {assumption}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Advantages & Limitations */}
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3 text-green-600 dark:text-green-400">
                    Advantages
                  </h3>
                  <ul className="space-y-2">
                    {method.advantages.map((advantage, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{advantage}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3 text-red-600 dark:text-red-400">
                    Limitations
                  </h3>
                  <ul className="space-y-2">
                    {method.limitations.map((limitation, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <XCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{limitation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Use Cases */}
              <div>
                <h3 className="text-lg font-semibold mb-3">Best Use Cases</h3>
                <div className="flex flex-wrap gap-2">
                  {method.use_cases.map((useCase, idx) => (
                    <Badge key={idx} variant="secondary">
                      {useCase}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Mathematics Tab */}
          {activeTab === 'mathematics' && (
            <div className="space-y-6">
              {/* Main Equation */}
              <div>
                <h3 className="text-lg font-semibold mb-3">Main Formulation</h3>
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
                  <LaTeXRenderer
                    expression={method.mathematical_formulation.main_equation}
                    displayMode={true}
                    numbered={true}
                  />
                </div>
              </div>

              {/* Auxiliary Equations */}
              {method.mathematical_formulation.auxiliary_equations && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Supporting Equations</h3>
                  <LaTeXBatchRenderer
                    equations={method.mathematical_formulation.auxiliary_equations.map(eq => ({
                      expression: eq.equation,
                      label: eq.label,
                      displayMode: true,
                    }))}
                    numbered={true}
                  />
                </div>
              )}

              {/* Objective Function */}
              {method.mathematical_formulation.objective_function && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Objective Function</h3>
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                    <LaTeXRenderer
                      expression={method.mathematical_formulation.objective_function}
                      displayMode={true}
                      numbered={true}
                    />
                  </div>
                </div>
              )}

              {/* Constraints */}
              {method.mathematical_formulation.constraints && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Constraints</h3>
                  <div className="space-y-2">
                    {method.mathematical_formulation.constraints.map((constraint, idx) => (
                      <div key={idx} className="flex items-center gap-3">
                        <span className="text-gray-500">•</span>
                        <LaTeXRenderer expression={constraint} />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Algorithm Tab */}
          {activeTab === 'algorithm' && (
            <div className="space-y-6">
              {/* Algorithm Steps */}
              <div>
                <h3 className="text-lg font-semibold mb-3">Algorithm Steps</h3>
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                  <ol className="space-y-3">
                    {method.algorithm.steps.map((step, idx) => (
                      <li key={idx} className="flex gap-3">
                        <span className="flex-shrink-0 w-8 h-8 bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full flex items-center justify-center text-sm font-semibold">
                          {idx + 1}
                        </span>
                        <span className="pt-1">{step}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              </div>

              {/* Complexity Analysis */}
              <div>
                <h3 className="text-lg font-semibold mb-3">Complexity Analysis</h3>
                <div className="grid grid-cols-2 gap-4">
                  <Card className="p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Clock className="h-5 w-5 text-blue-500" />
                      <h4 className="font-semibold">Time Complexity</h4>
                    </div>
                    <p className="text-2xl font-mono">
                      <LaTeXRenderer expression={method.algorithm.complexity.time} />
                    </p>
                  </Card>
                  
                  <Card className="p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Zap className="h-5 w-5 text-purple-500" />
                      <h4 className="font-semibold">Space Complexity</h4>
                    </div>
                    <p className="text-2xl font-mono">
                      <LaTeXRenderer expression={method.algorithm.complexity.space} />
                    </p>
                  </Card>
                </div>
              </div>

              {/* Convergence */}
              {method.algorithm.convergence && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Convergence Properties</h3>
                  <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                    <p>{method.algorithm.convergence}</p>
                  </div>
                </div>
              )}

              {/* Implementation Notes */}
              {method.implementation_notes && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Implementation Notes</h3>
                  <ul className="space-y-2">
                    {method.implementation_notes.map((note, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <Code className="h-4 w-4 text-gray-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{note}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Example Tab */}
          {activeTab === 'example' && method.example_code && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Example Usage</h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleCopyCode}
                  className="flex items-center gap-2"
                >
                  {copiedCode ? (
                    <>
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="h-4 w-4" />
                      Copy Code
                    </>
                  )}
                </Button>
              </div>
              
              <pre className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                <code className="text-sm">{method.example_code}</code>
              </pre>
            </div>
          )}

          {/* References Tab */}
          {activeTab === 'references' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold mb-3">References</h3>
              <div className="space-y-3">
                {method.references.map((ref, idx) => (
                  <Card key={idx} className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h4 className="font-semibold text-gray-900 dark:text-white">
                          {ref.title}
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          {ref.authors.join(', ')} ({ref.year})
                        </p>
                        {ref.journal && (
                          <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
                            {ref.journal}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-2 ml-4">
                        {ref.doi && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => window.open(`https://doi.org/${ref.doi}`, '_blank')}
                            className="h-8 w-8 p-0"
                          >
                            <ExternalLink className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      </Tabs>
    </Card>
  );
};