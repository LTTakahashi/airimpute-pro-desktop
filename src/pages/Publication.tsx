import { useState } from 'react';
import { 
  FileText, 
  BookOpen, 
  Calculator,
  FileOutput,
  HelpCircle,
  Plus,
  FolderOpen,
  Clock,
  Layout
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Tabs } from '@/components/ui/Tabs';
import { Badge } from '@/components/ui/Badge';
import { Alert } from '@/components/ui/Alert';
import { Progress } from '@/components/ui/Progress';
import { ScientificCard } from '@/components/layout/ScientificCard';
import type {
  Report,
  Citation
} from '@/components/academic';
import { 
  ReportBuilder, 
  CitationGenerator, 
  LaTeXEquationEditor,
  MethodDocumentationViewer
} from '@/components/academic';
import { cn } from '@/utils/cn';

interface RecentReport {
  id: string;
  title: string;
  template: string;
  lastModified: Date;
  completion: number;
  status: 'draft' | 'review' | 'final';
}

export default function Publication() {
  const [activeTab, setActiveTab] = useState<'reports' | 'citations' | 'equations' | 'methods'>('reports');
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [showEquationEditor, setShowEquationEditor] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null);
  
  // Mock data for recent reports
  const [recentReports] = useState<RecentReport[]>([
    {
      id: '1',
      title: 'Air Quality Imputation Methods: A Comparative Study',
      template: 'IEEE Journal',
      lastModified: new Date('2024-01-15'),
      completion: 85,
      status: 'draft'
    },
    {
      id: '2',
      title: 'Novel RAH Algorithm for Environmental Data',
      template: 'Nature',
      lastModified: new Date('2024-01-10'),
      completion: 100,
      status: 'review'
    }
  ]);

  const handleCreateReport = () => {
    setSelectedReport({
      id: `report_${Date.now()}`,
      template: null as any, // Will be set in ReportBuilder
      metadata: {
        title: '',
        authors: [],
        affiliations: [],
        keywords: [],
        date: new Date()
      },
      sections: [],
      citations: [],
      createdAt: new Date(),
      updatedAt: new Date()
    });
    setActiveTab('reports');
  };

  const handleSaveReport = async (report: Report) => {
    // TODO: Implement actual save functionality
    console.log('Saving report:', report);
    // Show success message
  };

  const handleExportReport = async (_report: Report, format: 'pdf' | 'latex' | 'word') => {
    // TODO: Implement actual export functionality
    console.log('Exporting report as:', format);
  };

  const handleCitationGenerate = (method: any) => {
    const newCitation: Citation = {
      id: `cite_${Date.now()}`,
      type: 'article',
      authors: method.references[0]?.authors.map((name: string) => {
        const parts = name.split(', ');
        return {
          family: parts[0],
          given: parts[1] || ''
        };
      }) || [],
      title: method.references[0]?.title || method.name,
      year: method.references[0]?.year || new Date().getFullYear(),
      journal: method.references[0]?.journal,
      doi: method.references[0]?.doi
    };
    setCitations(prev => [...prev, newCitation]);
    setActiveTab('citations');
  };

  if (selectedReport) {
    return (
      <div className="h-full">
        <div className="flex items-center justify-between p-4 border-b">
          <Button
            variant="ghost"
            onClick={() => setSelectedReport(null)}
            className="flex items-center gap-2"
          >
            ← Back to Publications
          </Button>
        </div>
        <div className="h-[calc(100%-65px)]">
          <ReportBuilder
            template={selectedReport.template}
            onSave={handleSaveReport}
            onExport={handleExportReport}
          />
        </div>
      </div>
    );
  }

  if (showEquationEditor) {
    return (
      <div className="h-full">
        <div className="flex items-center justify-between p-4 border-b">
          <div>
            <h1 className="text-2xl font-bold">LaTeX Equation Editor</h1>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Create and edit mathematical equations
            </p>
          </div>
          <Button
            variant="ghost"
            onClick={() => setShowEquationEditor(false)}
            className="flex items-center gap-2"
          >
            ← Back
          </Button>
        </div>
        <div className="h-[calc(100%-100px)] p-6">
          <Card className="h-full">
            <LaTeXEquationEditor
              showPreview={true}
              showSymbolPalette={true}
              height="100%"
              onSave={(latex) => {
                console.log('Saved equation:', latex);
                // Could save to a library or insert into current document
              }}
            />
          </Card>
        </div>
      </div>
    );
  }

  if (selectedMethod) {
    return (
      <div className="h-full overflow-auto">
        <div className="flex items-center justify-between p-4 border-b">
          <Button
            variant="ghost"
            onClick={() => setSelectedMethod(null)}
            className="flex items-center gap-2"
          >
            ← Back
          </Button>
        </div>
        <div className="p-6">
          <MethodDocumentationViewer
            methodId={selectedMethod}
            showExample={true}
            showReferences={true}
            onCitationGenerate={handleCitationGenerate}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-auto bg-gray-50 dark:bg-gray-900">
      <div className="p-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Publication & Documentation
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Create publication-ready reports, manage citations, and document your research
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <Card 
            className="p-4 cursor-pointer hover:shadow-lg transition-shadow"
            onClick={handleCreateReport}
          >
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center">
                <Plus className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <h3 className="font-medium">New Report</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Create from template
                </p>
              </div>
            </div>
          </Card>

          <Card 
            className="p-4 cursor-pointer hover:shadow-lg transition-shadow"
            onClick={() => setShowEquationEditor(true)}
          >
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-green-100 dark:bg-green-900/50 flex items-center justify-center">
                <Calculator className="h-5 w-5 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <h3 className="font-medium">Equation Editor</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  LaTeX equations
                </p>
              </div>
            </div>
          </Card>

          <Card 
            className="p-4 cursor-pointer hover:shadow-lg transition-shadow"
            onClick={() => setActiveTab('citations')}
          >
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-purple-100 dark:bg-purple-900/50 flex items-center justify-center">
                <BookOpen className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <h3 className="font-medium">Citations</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {citations.length} references
                </p>
              </div>
            </div>
          </Card>

          <Card 
            className="p-4 cursor-pointer hover:shadow-lg transition-shadow"
            onClick={() => setActiveTab('methods')}
          >
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-orange-100 dark:bg-orange-900/50 flex items-center justify-center">
                <FileText className="h-5 w-5 text-orange-600 dark:text-orange-400" />
              </div>
              <div>
                <h3 className="font-medium">Method Docs</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  20+ methods
                </p>
              </div>
            </div>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab as any}>
          <div className="flex border-b border-gray-200 dark:border-gray-700 mb-6">
            <button
              onClick={() => setActiveTab('reports')}
              className={cn(
                'px-6 py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'reports'
                  ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                  : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
              )}
            >
              Reports
            </button>
            <button
              onClick={() => setActiveTab('citations')}
              className={cn(
                'px-6 py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'citations'
                  ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                  : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
              )}
            >
              Citations
            </button>
            <button
              onClick={() => setActiveTab('equations')}
              className={cn(
                'px-6 py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'equations'
                  ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                  : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
              )}
            >
              Equations
            </button>
            <button
              onClick={() => setActiveTab('methods')}
              className={cn(
                'px-6 py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'methods'
                  ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                  : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
              )}
            >
              Method Documentation
            </button>
          </div>

          {/* Reports Tab */}
          {activeTab === 'reports' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Recent Reports</h2>
                <Button variant="outline" size="sm" className="flex items-center gap-2">
                  <FolderOpen className="h-4 w-4" />
                  Browse All
                </Button>
              </div>

              {recentReports.length === 0 ? (
                <Card className="p-12 text-center">
                  <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    No reports yet. Create your first academic report.
                  </p>
                  <Button variant="primary" onClick={handleCreateReport}>
                    Create Report
                  </Button>
                </Card>
              ) : (
                <div className="grid gap-4">
                  {recentReports.map(report => (
                    <Card 
                      key={report.id}
                      className="p-6 cursor-pointer hover:shadow-lg transition-shadow"
                      onClick={() => {
                        // Load existing report
                        setSelectedReport({
                          id: report.id,
                          template: null as any,
                          metadata: {
                            title: report.title,
                            authors: [],
                            affiliations: [],
                            keywords: [],
                            date: report.lastModified
                          },
                          sections: [],
                          citations: [],
                          createdAt: report.lastModified,
                          updatedAt: report.lastModified
                        });
                      }}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h3 className="text-lg font-semibold mb-2">{report.title}</h3>
                          <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                            <span className="flex items-center gap-1">
                              <FileText className="h-4 w-4" />
                              {report.template}
                            </span>
                            <span className="flex items-center gap-1">
                              <Clock className="h-4 w-4" />
                              {report.lastModified.toLocaleDateString()}
                            </span>
                          </div>
                          <div className="mt-4">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm font-medium">Completion</span>
                              <span className="text-sm">{report.completion}%</span>
                            </div>
                            <Progress value={report.completion} className="h-2" />
                          </div>
                        </div>
                        <Badge 
                          variant={
                            report.status === 'final' ? 'green' :
                            report.status === 'review' ? 'yellow' : 'secondary'
                          }
                        >
                          {report.status}
                        </Badge>
                      </div>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Citations Tab */}
          {activeTab === 'citations' && (
            <CitationGenerator
              citations={citations}
              onCitationsChange={setCitations}
              allowImport={true}
              allowExport={true}
            />
          )}

          {/* Equations Tab */}
          {activeTab === 'equations' && (
            <div className="space-y-6">
              <Alert variant="info">
                Create and manage LaTeX equations for your publications. Equations can be saved to your library and reused across reports.
              </Alert>
              
              <Card className="p-6">
                <div className="text-center py-8">
                  <Calculator className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    Open the equation editor to create mathematical expressions
                  </p>
                  <Button 
                    variant="primary"
                    onClick={() => setShowEquationEditor(true)}
                    className="flex items-center gap-2 mx-auto"
                  >
                    <Calculator className="h-4 w-4" />
                    Open Equation Editor
                  </Button>
                </div>
              </Card>
            </div>
          )}

          {/* Methods Tab */}
          {activeTab === 'methods' && (
            <div className="space-y-6">
              <Alert variant="info">
                Browse comprehensive documentation for all imputation methods including mathematical formulations, algorithms, and references.
              </Alert>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {[
                  { id: 'mean', name: 'Mean Imputation', category: 'Classical' },
                  { id: 'linear_interpolation', name: 'Linear Interpolation', category: 'Classical' },
                  { id: 'kalman_filter', name: 'Kalman Filter', category: 'Statistical' },
                  { id: 'arima', name: 'ARIMA', category: 'Statistical' },
                  { id: 'random_forest', name: 'Random Forest', category: 'Machine Learning' },
                  { id: 'lstm', name: 'LSTM', category: 'Deep Learning' },
                  { id: 'rah', name: 'RAH (Novel)', category: 'Hybrid' }
                ].map(method => (
                  <Card
                    key={method.id}
                    className="p-4 cursor-pointer hover:shadow-lg transition-shadow"
                    onClick={() => setSelectedMethod(method.id)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="font-medium">{method.name}</h3>
                      <Badge variant="secondary" size="sm">
                        {method.category}
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      View documentation →
                    </p>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </Tabs>

        {/* Help Section */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <ScientificCard
            title="Getting Started"
            subtitle="Learn the basics"
            icon={<HelpCircle className="h-5 w-5" />}
            actions={
              <Button variant="ghost" size="sm">
                View Guide
              </Button>
            }
          >
            <ul className="space-y-2 text-sm">
              <li>• Choose a journal template</li>
              <li>• Import your analysis results</li>
              <li>• Add figures and tables</li>
              <li>• Manage citations</li>
              <li>• Export to PDF or LaTeX</li>
            </ul>
          </ScientificCard>

          <ScientificCard
            title="Templates"
            subtitle="Popular formats"
            icon={<Layout className="h-5 w-5" />}
          >
            <ul className="space-y-2 text-sm">
              <li>• IEEE Transactions</li>
              <li>• Nature</li>
              <li>• Science</li>
              <li>• Elsevier</li>
              <li>• Custom templates</li>
            </ul>
          </ScientificCard>

          <ScientificCard
            title="Export Options"
            subtitle="Multiple formats"
            icon={<FileOutput className="h-5 w-5" />}
          >
            <ul className="space-y-2 text-sm">
              <li>• PDF (print-ready)</li>
              <li>• LaTeX source</li>
              <li>• Word document</li>
              <li>• HTML preview</li>
              <li>• Reproducibility package</li>
            </ul>
          </ScientificCard>
        </div>
      </div>
    </div>
  );
}