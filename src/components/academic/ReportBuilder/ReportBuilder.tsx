import React, { useState, useCallback } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { 
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem
} from '@/components/ui/Select';
import { Tabs } from '@/components/ui/Tabs';
import { Badge } from '@/components/ui/Badge';
import { Alert } from '@/components/ui/Alert';
import { Progress } from '@/components/ui/Progress';
import { cn } from '@/utils/cn';
import { 
  FileText,
  Download,
  Trash2,
  ChevronUp,
  ChevronDown,
  Save,
  Type,
  Image,
  Table,
  List,
  Check,
  Lock
} from 'lucide-react';
import { LaTeXRenderer } from '../LaTeX/LaTeXRenderer';
import type { Citation } from '../Citation/CitationGenerator';
import { CitationGenerator } from '../Citation/CitationGenerator';
import { Tooltip } from '@/components/ui/Tooltip';

export interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  category: 'journal' | 'conference' | 'thesis' | 'technical' | 'custom';
  publisher?: string;
  style: ReportStyle;
  sections: ReportSection[];
  metadata: ReportMetadata;
}

export interface ReportStyle {
  documentClass: string;
  fontSize: number;
  lineSpacing: number;
  margins: {
    top: number;
    bottom: number;
    left: number;
    right: number;
  };
  fonts: {
    main: string;
    headings: string;
    mono: string;
  };
  colors: {
    primary: string;
    secondary: string;
    text: string;
  };
  bibliography: {
    style: string;
    sortBy: 'author' | 'year' | 'appearance';
  };
}

export interface ReportSection {
  id: string;
  type: SectionType;
  title: string;
  level: number;
  required: boolean;
  locked: boolean;
  content: SectionContent[];
  subsections?: ReportSection[];
}

export type SectionType = 
  | 'title'
  | 'abstract'
  | 'keywords'
  | 'introduction'
  | 'literature_review'
  | 'methodology'
  | 'results'
  | 'discussion'
  | 'conclusion'
  | 'acknowledgments'
  | 'references'
  | 'appendix'
  | 'custom';

export interface SectionContent {
  id: string;
  type: ContentType;
  data: any;
}

export type ContentType = 
  | 'text'
  | 'latex'
  | 'figure'
  | 'table'
  | 'equation'
  | 'code'
  | 'citation'
  | 'list';

export interface ReportMetadata {
  title: string;
  subtitle?: string;
  authors: Author[];
  affiliations: string[];
  abstract?: string;
  keywords: string[];
  date: Date;
  version?: string;
  doi?: string;
  funding?: string[];
  correspondingAuthor?: string;
}

export interface Author {
  name: string;
  email?: string;
  orcid?: string;
  affiliation: number[]; // indices of affiliations
}

interface ReportBuilderProps {
  template?: ReportTemplate;
  onSave?: (report: Report) => void;
  onExport?: (report: Report, format: 'pdf' | 'latex' | 'word') => void;
  className?: string;
}

export interface Report {
  id: string;
  template: ReportTemplate;
  metadata: ReportMetadata;
  sections: ReportSection[];
  citations: Citation[];
  createdAt: Date;
  updatedAt: Date;
}

// Predefined templates
const templates: ReportTemplate[] = [
  {
    id: 'ieee_journal',
    name: 'IEEE Journal',
    description: 'IEEE Transactions format for journal papers',
    category: 'journal',
    publisher: 'IEEE',
    style: {
      documentClass: 'IEEEtran',
      fontSize: 10,
      lineSpacing: 1,
      margins: { top: 19, bottom: 19, left: 19, right: 19 },
      fonts: { main: 'Times', headings: 'Times', mono: 'Courier' },
      colors: { primary: '#000000', secondary: '#333333', text: '#000000' },
      bibliography: { style: 'ieee', sortBy: 'appearance' }
    },
    sections: [
      { id: 'title', type: 'title', title: 'Title', level: 0, required: true, locked: true, content: [] },
      { id: 'abstract', type: 'abstract', title: 'Abstract', level: 1, required: true, locked: false, content: [] },
      { id: 'keywords', type: 'keywords', title: 'Index Terms', level: 1, required: true, locked: false, content: [] },
      { id: 'intro', type: 'introduction', title: 'Introduction', level: 1, required: true, locked: false, content: [] },
      { id: 'method', type: 'methodology', title: 'Methodology', level: 1, required: true, locked: false, content: [] },
      { id: 'results', type: 'results', title: 'Results', level: 1, required: true, locked: false, content: [] },
      { id: 'discussion', type: 'discussion', title: 'Discussion', level: 1, required: true, locked: false, content: [] },
      { id: 'conclusion', type: 'conclusion', title: 'Conclusion', level: 1, required: true, locked: false, content: [] },
      { id: 'refs', type: 'references', title: 'References', level: 1, required: true, locked: true, content: [] }
    ],
    metadata: {
      title: '',
      authors: [],
      affiliations: [],
      keywords: [],
      date: new Date()
    }
  },
  {
    id: 'nature',
    name: 'Nature',
    description: 'Nature journal article format',
    category: 'journal',
    publisher: 'Nature Publishing Group',
    style: {
      documentClass: 'nature',
      fontSize: 12,
      lineSpacing: 1.5,
      margins: { top: 25, bottom: 25, left: 25, right: 25 },
      fonts: { main: 'Minion Pro', headings: 'Helvetica Neue', mono: 'Consolas' },
      colors: { primary: '#000000', secondary: '#e3120b', text: '#000000' },
      bibliography: { style: 'nature', sortBy: 'appearance' }
    },
    sections: [
      { id: 'title', type: 'title', title: 'Title', level: 0, required: true, locked: true, content: [] },
      { id: 'abstract', type: 'abstract', title: 'Abstract', level: 1, required: true, locked: false, content: [] },
      { id: 'main', type: 'introduction', title: 'Main', level: 1, required: true, locked: false, content: [] },
      { id: 'methods', type: 'methodology', title: 'Methods', level: 1, required: true, locked: false, content: [] },
      { id: 'refs', type: 'references', title: 'References', level: 1, required: true, locked: true, content: [] }
    ],
    metadata: {
      title: '',
      authors: [],
      affiliations: [],
      keywords: [],
      date: new Date()
    }
  }
];

export const ReportBuilder: React.FC<ReportBuilderProps> = ({
  template: initialTemplate,
  onSave,
  onExport,
  className
}) => {
  const [selectedTemplate, setSelectedTemplate] = useState<ReportTemplate>(
    initialTemplate || templates[0]
  );
  const [report, setReport] = useState<Report>({
    id: `report_${Date.now()}`,
    template: selectedTemplate,
    metadata: { ...selectedTemplate.metadata },
    sections: [...selectedTemplate.sections],
    citations: [],
    createdAt: new Date(),
    updatedAt: new Date()
  });
  const [activeSection, setActiveSection] = useState<string>(report.sections[0]?.id);
  const [activeTab, setActiveTab] = useState<'content' | 'metadata' | 'style' | 'preview'>('content');
  const [isSaving, setIsSaving] = useState(false);
  const [exportProgress, setExportProgress] = useState<number | null>(null);

  // Handle template change
  const handleTemplateChange = useCallback((templateId: string) => {
    const template = templates.find(t => t.id === templateId);
    if (template) {
      setSelectedTemplate(template);
      setReport(prev => ({
        ...prev,
        template,
        sections: [...template.sections],
        updatedAt: new Date()
      }));
    }
  }, []);

  // Handle section content update
  const handleSectionContentUpdate = useCallback((sectionId: string, content: SectionContent[]) => {
    setReport(prev => ({
      ...prev,
      sections: updateSectionContent(prev.sections, sectionId, content),
      updatedAt: new Date()
    }));
  }, []);

  // Helper function to update nested sections
  const updateSectionContent = (
    sections: ReportSection[], 
    sectionId: string, 
    content: SectionContent[]
  ): ReportSection[] => {
    return sections.map(section => {
      if (section.id === sectionId) {
        return { ...section, content };
      }
      if (section.subsections) {
        return {
          ...section,
          subsections: updateSectionContent(section.subsections, sectionId, content)
        };
      }
      return section;
    });
  };

  // Handle metadata update
  const handleMetadataUpdate = useCallback((metadata: Partial<ReportMetadata>) => {
    setReport(prev => ({
      ...prev,
      metadata: { ...prev.metadata, ...metadata },
      updatedAt: new Date()
    }));
  }, []);

  // Handle save
  const handleSave = useCallback(async () => {
    if (!onSave) return;
    
    setIsSaving(true);
    try {
      await onSave(report);
      // Show success message
    } catch (error) {
      // Show error message
    } finally {
      setIsSaving(false);
    }
  }, [report, onSave]);

  // Handle export
  const handleExport = useCallback(async (format: 'pdf' | 'latex' | 'word') => {
    if (!onExport) return;
    
    setExportProgress(0);
    try {
      // Simulate progress
      const interval = setInterval(() => {
        setExportProgress(prev => {
          if (prev === null || prev >= 90) return prev;
          return prev + 10;
        });
      }, 200);
      
      await onExport(report, format);
      
      clearInterval(interval);
      setExportProgress(100);
      
      setTimeout(() => setExportProgress(null), 1000);
    } catch (error) {
      setExportProgress(null);
      // Show error message
    }
  }, [report, onExport]);

  // Calculate completion percentage
  const calculateCompletion = useCallback((): number => {
    const requiredSections = report.sections.filter(s => s.required);
    const completedSections = requiredSections.filter(s => s.content.length > 0);
    return Math.round((completedSections.length / requiredSections.length) * 100);
  }, [report.sections]);

  return (
    <div className={cn('flex h-full', className)}>
      {/* Sidebar - Section Navigation */}
      <div className="w-64 border-r border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <label className="block text-sm font-medium mb-2">Template</label>
          <Select
            value={selectedTemplate.id}
            onValueChange={handleTemplateChange}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a template" />
            </SelectTrigger>
            <SelectContent>
              {templates.map(template => (
                <SelectItem key={template.id} value={template.id}>
                  {template.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        
        <div className="p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium">Sections</h3>
            <Badge variant="secondary" size="sm">
              {calculateCompletion()}% Complete
            </Badge>
          </div>
          
          <div className="space-y-1">
            {report.sections.map((section) => (
              <SectionNavItem
                key={section.id}
                section={section}
                isActive={activeSection === section.id}
                onClick={() => setActiveSection(section.id)}
                level={0}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <div className="px-6 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  {report.metadata.title || 'Untitled Report'}
                </h1>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  Last updated: {report.updatedAt.toLocaleString()}
                </p>
              </div>
              
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSave}
                  disabled={isSaving}
                  className="flex items-center gap-2"
                >
                  <Save className="h-4 w-4" />
                  {isSaving ? 'Saving...' : 'Save'}
                </Button>
                
                <div className="relative">
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={() => handleExport('pdf')}
                    disabled={exportProgress !== null}
                    className="flex items-center gap-2"
                  >
                    <Download className="h-4 w-4" />
                    Export
                  </Button>
                  
                  {exportProgress !== null && (
                    <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-800/80 rounded">
                      <Progress value={exportProgress} className="w-20" />
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
          
          {/* Tabs */}
          <div className="px-6">
            <Tabs value={activeTab} onValueChange={setActiveTab as any}>
              <div className="flex border-b border-gray-200 dark:border-gray-700">
                <button
                  onClick={() => setActiveTab('content')}
                  className={cn(
                    'px-4 py-2 text-sm font-medium border-b-2 transition-colors',
                    activeTab === 'content'
                      ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                      : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
                  )}
                >
                  Content
                </button>
                <button
                  onClick={() => setActiveTab('metadata')}
                  className={cn(
                    'px-4 py-2 text-sm font-medium border-b-2 transition-colors',
                    activeTab === 'metadata'
                      ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                      : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
                  )}
                >
                  Metadata
                </button>
                <button
                  onClick={() => setActiveTab('style')}
                  className={cn(
                    'px-4 py-2 text-sm font-medium border-b-2 transition-colors',
                    activeTab === 'style'
                      ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                      : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
                  )}
                >
                  Style
                </button>
                <button
                  onClick={() => setActiveTab('preview')}
                  className={cn(
                    'px-4 py-2 text-sm font-medium border-b-2 transition-colors',
                    activeTab === 'preview'
                      ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                      : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
                  )}
                >
                  Preview
                </button>
              </div>
            </Tabs>
          </div>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-auto">
          {activeTab === 'content' && (
            <SectionEditor
              section={report.sections.find(s => s.id === activeSection)!}
              onUpdate={(content) => handleSectionContentUpdate(activeSection, content)}
              citations={report.citations}
              onCitationsChange={(citations) => setReport({ ...report, citations })}
            />
          )}
          
          {activeTab === 'metadata' && (
            <MetadataEditor
              metadata={report.metadata}
              onUpdate={handleMetadataUpdate}
            />
          )}
          
          {activeTab === 'style' && (
            <StyleEditor
              style={report.template.style}
              onUpdate={(style) => {
                setReport(prev => ({
                  ...prev,
                  template: { ...prev.template, style },
                  updatedAt: new Date()
                }));
              }}
            />
          )}
          
          {activeTab === 'preview' && (
            <ReportPreview report={report} />
          )}
        </div>
      </div>
    </div>
  );
};

// Section Navigation Item Component
interface SectionNavItemProps {
  section: ReportSection;
  isActive: boolean;
  onClick: () => void;
  level: number;
}

const SectionNavItem: React.FC<SectionNavItemProps> = ({
  section,
  isActive,
  onClick,
  level
}) => {
  const hasContent = section.content.length > 0;
  
  return (
    <>
      <button
        onClick={onClick}
        className={cn(
          'w-full text-left px-3 py-2 rounded text-sm transition-colors',
          'flex items-center justify-between',
          isActive
            ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300'
            : 'hover:bg-gray-100 dark:hover:bg-gray-800',
          level > 0 && 'ml-4'
        )}
        style={{ paddingLeft: `${12 + level * 16}px` }}
      >
        <div className="flex items-center gap-2">
          {section.locked && <Lock className="h-3 w-3" />}
          <span className={cn(!hasContent && section.required && 'text-red-600 dark:text-red-400')}>
            {section.title}
          </span>
          {section.required && <span className="text-red-500">*</span>}
        </div>
        {hasContent && <Check className="h-4 w-4 text-green-500" />}
      </button>
      
      {section.subsections?.map(subsection => (
        <SectionNavItem
          key={subsection.id}
          section={subsection}
          isActive={isActive}
          onClick={onClick}
          level={level + 1}
        />
      ))}
    </>
  );
};

// Section Editor Component
interface SectionEditorProps {
  section: ReportSection;
  onUpdate: (content: SectionContent[]) => void;
  citations: Citation[];
  onCitationsChange: (citations: Citation[]) => void;
}

const SectionEditor: React.FC<SectionEditorProps> = ({
  section,
  onUpdate,
  citations,
  onCitationsChange
}) => {
  const [content, setContent] = useState<SectionContent[]>(section.content);

  const handleAddContent = useCallback((type: ContentType) => {
    const newContent: SectionContent = {
      id: `content_${Date.now()}`,
      type,
      data: getDefaultContentData(type)
    };
    const updated = [...content, newContent];
    setContent(updated);
    onUpdate(updated);
  }, [content, onUpdate]);

  const handleUpdateContent = useCallback((id: string, data: any) => {
    const updated = content.map(c => c.id === id ? { ...c, data } : c);
    setContent(updated);
    onUpdate(updated);
  }, [content, onUpdate]);

  const handleDeleteContent = useCallback((id: string) => {
    const updated = content.filter(c => c.id !== id);
    setContent(updated);
    onUpdate(updated);
  }, [content, onUpdate]);

  const handleReorderContent = useCallback((fromIndex: number, toIndex: number) => {
    const updated = [...content];
    const [removed] = updated.splice(fromIndex, 1);
    updated.splice(toIndex, 0, removed);
    setContent(updated);
    onUpdate(updated);
  }, [content, onUpdate]);

  return (
    <div className="p-6">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <h2 className="text-xl font-semibold mb-2">{section.title}</h2>
          {section.type === 'references' && (
            <CitationGenerator
              citations={citations}
              onCitationsChange={onCitationsChange}
              className="mt-4"
            />
          )}
        </div>

        {section.type !== 'references' && (
          <>
            {/* Content blocks */}
            <div className="space-y-4">
              {content.map((item, index) => (
                <ContentBlock
                  key={item.id}
                  content={item}
                  onUpdate={(data) => handleUpdateContent(item.id, data)}
                  onDelete={() => handleDeleteContent(item.id)}
                  onMoveUp={index > 0 ? () => handleReorderContent(index, index - 1) : undefined}
                  onMoveDown={index < content.length - 1 ? () => handleReorderContent(index, index + 1) : undefined}
                />
              ))}
            </div>

            {/* Add content toolbar */}
            <div className="mt-6 p-4 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Add content:</p>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleAddContent('text')}
                  className="flex items-center gap-2"
                >
                  <Type className="h-4 w-4" />
                  Text
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleAddContent('latex')}
                  className="flex items-center gap-2"
                >
                  <FileText className="h-4 w-4" />
                  LaTeX
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleAddContent('equation')}
                  className="flex items-center gap-2"
                >
                  <FileText className="h-4 w-4" />
                  Equation
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleAddContent('figure')}
                  className="flex items-center gap-2"
                >
                  <Image className="h-4 w-4" />
                  Figure
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleAddContent('table')}
                  className="flex items-center gap-2"
                >
                  <Table className="h-4 w-4" />
                  Table
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleAddContent('list')}
                  className="flex items-center gap-2"
                >
                  <List className="h-4 w-4" />
                  List
                </Button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

// Content Block Component
interface ContentBlockProps {
  content: SectionContent;
  onUpdate: (data: any) => void;
  onDelete: () => void;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
}

const ContentBlock: React.FC<ContentBlockProps> = ({
  content,
  onUpdate,
  onDelete,
  onMoveUp,
  onMoveDown
}) => {
  const renderContent = () => {
    switch (content.type) {
      case 'text':
        return (
          <textarea
            value={content.data.text || ''}
            onChange={(e) => onUpdate({ ...content.data, text: e.target.value })}
            className="w-full min-h-[100px] p-3 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 resize-y"
            placeholder="Enter text..."
          />
        );
      
      case 'latex':
      case 'equation':
        return (
          <div className="space-y-2">
            <textarea
              value={content.data.latex || ''}
              onChange={(e) => onUpdate({ ...content.data, latex: e.target.value })}
              className="w-full min-h-[60px] p-3 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 font-mono text-sm"
              placeholder="Enter LaTeX..."
            />
            {content.data.latex && (
              <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded">
                <LaTeXRenderer
                  expression={content.data.latex}
                  displayMode={content.type === 'equation'}
                />
              </div>
            )}
          </div>
        );
      
      case 'figure':
        return (
          <div className="space-y-2">
            <input
              type="text"
              value={content.data.path || ''}
              onChange={(e) => onUpdate({ ...content.data, path: e.target.value })}
              className="w-full p-2 border rounded bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
              placeholder="Figure path..."
            />
            <input
              type="text"
              value={content.data.caption || ''}
              onChange={(e) => onUpdate({ ...content.data, caption: e.target.value })}
              className="w-full p-2 border rounded bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
              placeholder="Figure caption..."
            />
          </div>
        );
      
      case 'table':
        return (
          <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Table editor would go here
            </p>
          </div>
        );
      
      case 'list':
        return (
          <div className="space-y-2">
            <Select
              value={content.data.type || 'bullet'}
              onValueChange={(value) => onUpdate({ ...content.data, type: value })}
            >
              <SelectTrigger className="w-32">
                <SelectValue placeholder="Select list type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="bullet">Bullet</SelectItem>
                <SelectItem value="numbered">Numbered</SelectItem>
              </SelectContent>
            </Select>
            <textarea
              value={content.data.items?.join('\n') || ''}
              onChange={(e) => onUpdate({ ...content.data, items: e.target.value.split('\n').filter(i => i) })}
              className="w-full min-h-[80px] p-3 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
              placeholder="Enter list items (one per line)..."
            />
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <Card className="p-4">
      <div className="flex items-start justify-between mb-2">
        <Badge variant="secondary" size="sm">
          {content.type}
        </Badge>
        <div className="flex items-center gap-1">
          {onMoveUp && (
            <Tooltip content="Move up">
              <Button variant="ghost" size="sm" onClick={onMoveUp} className="h-8 w-8 p-0">
                <ChevronUp className="h-4 w-4" />
              </Button>
            </Tooltip>
          )}
          {onMoveDown && (
            <Tooltip content="Move down">
              <Button variant="ghost" size="sm" onClick={onMoveDown} className="h-8 w-8 p-0">
                <ChevronDown className="h-4 w-4" />
              </Button>
            </Tooltip>
          )}
          <Tooltip content="Delete">
            <Button
              variant="ghost"
              size="sm"
              onClick={onDelete}
              className="h-8 w-8 p-0 text-red-600 hover:text-red-700"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </Tooltip>
        </div>
      </div>
      {renderContent()}
    </Card>
  );
};

// Metadata Editor Component
interface MetadataEditorProps {
  metadata: ReportMetadata;
  onUpdate: (metadata: Partial<ReportMetadata>) => void;
}

const MetadataEditor: React.FC<MetadataEditorProps> = ({ metadata, onUpdate }) => {
  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div>
        <label className="block text-sm font-medium mb-2">Title</label>
        <input
          type="text"
          value={metadata.title}
          onChange={(e) => onUpdate({ title: e.target.value })}
          className="w-full p-3 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
          placeholder="Report title..."
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Subtitle</label>
        <input
          type="text"
          value={metadata.subtitle || ''}
          onChange={(e) => onUpdate({ subtitle: e.target.value })}
          className="w-full p-3 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
          placeholder="Optional subtitle..."
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Abstract</label>
        <textarea
          value={metadata.abstract || ''}
          onChange={(e) => onUpdate({ abstract: e.target.value })}
          className="w-full min-h-[150px] p-3 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 resize-y"
          placeholder="Report abstract..."
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Keywords</label>
        <input
          type="text"
          value={metadata.keywords.join(', ')}
          onChange={(e) => onUpdate({ 
            keywords: e.target.value.split(',').map(k => k.trim()).filter(k => k) 
          })}
          className="w-full p-3 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
          placeholder="Comma-separated keywords..."
        />
      </div>

      {/* Authors section would be more complex with add/remove functionality */}
      <div>
        <label className="block text-sm font-medium mb-2">Authors</label>
        <Card className="p-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Author management interface would go here
          </p>
        </Card>
      </div>
    </div>
  );
};

// Style Editor Component
interface StyleEditorProps {
  style: ReportStyle;
  onUpdate: (style: ReportStyle) => void;
}

const StyleEditor: React.FC<StyleEditorProps> = ({ style, onUpdate }) => {
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <Alert variant="info" className="mb-6">
        Style settings affect the final PDF output. Some options may be limited by the selected template.
      </Alert>
      
      <div className="space-y-6">
        <Card className="p-4">
          <h3 className="font-medium mb-4">Typography</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Font Size</label>
              <Select
                value={style.fontSize.toString()}
                onValueChange={(value) => onUpdate({ ...style, fontSize: parseInt(value) })}
              >
                <option value="10">10pt</option>
                <option value="11">11pt</option>
                <option value="12">12pt</option>
              </Select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Line Spacing</label>
              <Select
                value={style.lineSpacing.toString()}
                onValueChange={(value) => onUpdate({ ...style, lineSpacing: parseFloat(value) })}
              >
                <option value="1">Single</option>
                <option value="1.5">1.5</option>
                <option value="2">Double</option>
              </Select>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="font-medium mb-4">Page Layout</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Top/Bottom Margin (mm)</label>
              <input
                type="number"
                value={style.margins.top}
                onChange={(e) => onUpdate({
                  ...style,
                  margins: { ...style.margins, top: parseInt(e.target.value), bottom: parseInt(e.target.value) }
                })}
                className="w-full p-2 border rounded bg-white dark:bg-gray-800"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Left/Right Margin (mm)</label>
              <input
                type="number"
                value={style.margins.left}
                onChange={(e) => onUpdate({
                  ...style,
                  margins: { ...style.margins, left: parseInt(e.target.value), right: parseInt(e.target.value) }
                })}
                className="w-full p-2 border rounded bg-white dark:bg-gray-800"
              />
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="font-medium mb-4">Bibliography</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Citation Style</label>
              <Select
                value={style.bibliography.style}
                onValueChange={(value) => onUpdate({
                  ...style,
                  bibliography: { ...style.bibliography, style: value }
                })}
              >
                <option value="ieee">IEEE</option>
                <option value="apa">APA</option>
                <option value="mla">MLA</option>
                <option value="chicago">Chicago</option>
                <option value="nature">Nature</option>
              </Select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Sort By</label>
              <Select
                value={style.bibliography.sortBy}
                onValueChange={(value) => onUpdate({
                  ...style,
                  bibliography: { ...style.bibliography, sortBy: value as any }
                })}
              >
                <option value="appearance">Order of Appearance</option>
                <option value="author">Author Name</option>
                <option value="year">Year</option>
              </Select>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

// Report Preview Component
interface ReportPreviewProps {
  report: Report;
}

const ReportPreview: React.FC<ReportPreviewProps> = ({ report }) => {
  return (
    <div className="p-6">
      <div className="max-w-4xl mx-auto bg-white dark:bg-gray-900 shadow-lg">
        <div className="p-16">
          <div className="text-center mb-12">
            <h1 className="text-3xl font-bold mb-2">{report.metadata.title || 'Untitled Report'}</h1>
            {report.metadata.subtitle && (
              <h2 className="text-xl text-gray-600 dark:text-gray-400">{report.metadata.subtitle}</h2>
            )}
          </div>
          
          {report.metadata.abstract && (
            <div className="mb-8">
              <h3 className="font-bold mb-2">Abstract</h3>
              <p className="text-justify">{report.metadata.abstract}</p>
            </div>
          )}
          
          {report.metadata.keywords.length > 0 && (
            <div className="mb-8">
              <p className="font-medium">
                Keywords: {report.metadata.keywords.join(', ')}
              </p>
            </div>
          )}
          
          <Alert variant="info" className="mt-8">
            Full preview with formatted content would be rendered here
          </Alert>
        </div>
      </div>
    </div>
  );
};

// Helper function to get default content data
function getDefaultContentData(type: ContentType): any {
  switch (type) {
    case 'text':
      return { text: '' };
    case 'latex':
    case 'equation':
      return { latex: '' };
    case 'figure':
      return { path: '', caption: '', label: '' };
    case 'table':
      return { rows: [], columns: [], caption: '', label: '' };
    case 'list':
      return { type: 'bullet', items: [] };
    case 'citation':
      return { ids: [] };
    case 'code':
      return { language: 'python', code: '' };
    default:
      return {};
  }
}