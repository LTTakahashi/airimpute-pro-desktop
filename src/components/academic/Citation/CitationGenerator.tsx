import React, { useState, useCallback } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select';
import { Tabs } from '@/components/ui/Tabs';
import { Badge } from '@/components/ui/Badge';
import { Alert } from '@/components/ui/Alert';
import { cn } from '@/utils/cn';
import { 
  BookOpen, 
  Copy, 
  Download, 
  Upload,
  Plus,
  Trash2,
  Edit2,
  Check,
  AlertCircle,
  ExternalLink,
  FileText,
  Users,
  Calendar,
  Link,
  Hash
} from 'lucide-react';
import { Tooltip } from '@/components/ui/Tooltip';

export interface Citation {
  id: string;
  type: CitationType;
  authors: Author[];
  title: string;
  year: number;
  month?: number;
  day?: number;
  journal?: string;
  volume?: string;
  issue?: string;
  pages?: string;
  doi?: string;
  url?: string;
  isbn?: string;
  publisher?: string;
  edition?: string;
  editors?: Author[];
  booktitle?: string;
  chapter?: string;
  series?: string;
  note?: string;
  abstract?: string;
  keywords?: string[];
  language?: string;
  accessed?: Date;
}

export interface Author {
  given: string;
  family: string;
  suffix?: string;
  affiliation?: string;
  orcid?: string;
}

export type CitationType = 
  | 'article'
  | 'book'
  | 'bookSection'
  | 'conference'
  | 'dataset'
  | 'thesis'
  | 'report'
  | 'webpage'
  | 'software'
  | 'patent';

export type CitationStyle = 
  | 'apa'
  | 'mla'
  | 'chicago'
  | 'ieee'
  | 'nature'
  | 'science'
  | 'vancouver'
  | 'harvard';

export type ExportFormat = 
  | 'bibtex'
  | 'ris'
  | 'endnote'
  | 'csv'
  | 'json'
  | 'word'
  | 'latex';

interface CitationGeneratorProps {
  citations?: Citation[];
  onCitationsChange?: (citations: Citation[]) => void;
  selectedStyle?: CitationStyle;
  onStyleChange?: (style: CitationStyle) => void;
  allowImport?: boolean;
  allowExport?: boolean;
  className?: string;
}

const citationStyles: Record<CitationStyle, string> = {
  apa: 'APA 7th Edition',
  mla: 'MLA 9th Edition',
  chicago: 'Chicago 17th Edition',
  ieee: 'IEEE',
  nature: 'Nature',
  science: 'Science',
  vancouver: 'Vancouver',
  harvard: 'Harvard'
};

const citationTypeIcons = {
  article: BookOpen,
  book: FileText,
  bookSection: FileText,
  conference: Users,
  dataset: Hash,
  thesis: BookOpen,
  report: FileText,
  webpage: Link,
  software: FileText,
  patent: FileText
};

export const CitationGenerator: React.FC<CitationGeneratorProps> = ({
  citations: initialCitations = [],
  onCitationsChange,
  selectedStyle = 'apa',
  onStyleChange,
  allowImport = true,
  allowExport = true,
  className
}) => {
  const [citations, setCitations] = useState<Citation[]>(initialCitations);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [editingCitation, setEditingCitation] = useState<Citation | null>(null);
  const [activeTab, setActiveTab] = useState('list');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [importError, setImportError] = useState<string | null>(null);

  const handleAddCitation = useCallback(() => {
    const newCitation: Citation = {
      id: `cite_${Date.now()}`,
      type: 'article',
      authors: [{ given: '', family: '' }],
      title: '',
      year: new Date().getFullYear()
    };
    setEditingCitation(newCitation);
    setActiveTab('edit');
  }, []);

  const handleSaveCitation = useCallback((citation: Citation) => {
    const updatedCitations = editingCitation && citations.find(c => c.id === editingCitation.id)
      ? citations.map(c => c.id === citation.id ? citation : c)
      : [...citations, citation];
    
    setCitations(updatedCitations);
    if (onCitationsChange) {
      onCitationsChange(updatedCitations);
    }
    setEditingCitation(null);
    setActiveTab('list');
  }, [citations, editingCitation, onCitationsChange]);

  const handleDeleteCitation = useCallback((id: string) => {
    const updatedCitations = citations.filter(c => c.id !== id);
    setCitations(updatedCitations);
    if (onCitationsChange) {
      onCitationsChange(updatedCitations);
    }
    if (selectedCitation?.id === id) {
      setSelectedCitation(null);
    }
  }, [citations, selectedCitation, onCitationsChange]);

  const formatCitation = useCallback((citation: Citation, style: CitationStyle): string => {
    // Format authors
    const formatAuthors = (authors: Author[], style: CitationStyle): string => {
      if (authors.length === 0) return '';
      
      if (style === 'apa' || style === 'harvard') {
        if (authors.length === 1) {
          const author = authors[0];
          if (!author) return '';
          return `${author.family}, ${author.given?.[0] || ''}.`;
        } else if (authors.length === 2) {
          const author1 = authors[0];
          const author2 = authors[1];
          if (!author1 || !author2) return '';
          return `${author1.family}, ${author1.given?.[0] || ''}., & ${author2.family}, ${author2.given?.[0] || ''}.`;
        } else {
          const firstAuthor = authors[0];
          if (!firstAuthor) return '';
          return `${firstAuthor.family}, ${firstAuthor.given?.[0] || ''}., et al.`;
        }
      } else if (style === 'mla') {
        if (authors.length === 1) {
          const author = authors[0];
          if (!author) return '';
          return `${author.family}, ${author.given}.`;
        } else if (authors.length === 2) {
          const author1 = authors[0];
          const author2 = authors[1];
          if (!author1 || !author2) return '';
          return `${author1.family}, ${author1.given}, and ${author2.given} ${author2.family}.`;
        } else {
          const firstAuthor = authors[0];
          if (!firstAuthor) return '';
          return `${firstAuthor.family}, ${firstAuthor.given}, et al.`;
        }
      } else if (style === 'ieee') {
        return authors.map((a, i) => 
          `${a.given[0]}. ${a.family}${i < authors.length - 1 ? ', ' : ''}`
        ).join('');
      }
      return '';
    };

    // Format based on type and style
    const authors = formatAuthors(citation.authors, style);
    const year = `(${citation.year})`;
    const title = citation.type === 'article' ? `"${citation.title}"` : `*${citation.title}*`;
    
    let formatted = '';
    
    if (style === 'apa') {
      if (citation.type === 'article') {
        formatted = `${authors} ${year}. ${citation.title}. *${citation.journal}*`;
        if (citation.volume) formatted += `, *${citation.volume}*`;
        if (citation.issue) formatted += `(${citation.issue})`;
        if (citation.pages) formatted += `, ${citation.pages}`;
        formatted += '.';
      } else if (citation.type === 'book') {
        formatted = `${authors} ${year}. *${citation.title}*. ${citation.publisher}.`;
      }
    } else if (style === 'mla') {
      if (citation.type === 'article') {
        formatted = `${authors} "${citation.title}." *${citation.journal}*`;
        if (citation.volume) formatted += ` ${citation.volume}`;
        if (citation.issue) formatted += `.${citation.issue}`;
        formatted += ` (${citation.year})`;
        if (citation.pages) formatted += `: ${citation.pages}`;
        formatted += '.';
      }
    } else if (style === 'ieee') {
      formatted = `[${citations.indexOf(citation) + 1}] ${authors}, "${citation.title},"`;
      if (citation.journal) formatted += ` *${citation.journal}*,`;
      if (citation.volume) formatted += ` vol. ${citation.volume},`;
      if (citation.issue) formatted += ` no. ${citation.issue},`;
      if (citation.pages) formatted += ` pp. ${citation.pages},`;
      formatted += ` ${citation.year}.`;
    }
    
    if (citation.doi) {
      formatted += ` https://doi.org/${citation.doi}`;
    }
    
    return formatted;
  }, [citations]);

  const generateBibTeX = useCallback((citation: Citation): string => {
    const type = citation.type === 'article' ? '@article' : 
                 citation.type === 'book' ? '@book' :
                 citation.type === 'conference' ? '@inproceedings' : '@misc';
    
    let bibtex = `${type}{${citation.id},\n`;
    
    // Authors
    if (citation.authors.length > 0) {
      bibtex += `  author = {${citation.authors.map(a => `${a.given} ${a.family}`).join(' and ')}},\n`;
    }
    
    // Required fields
    bibtex += `  title = {${citation.title}},\n`;
    bibtex += `  year = {${citation.year}},\n`;
    
    // Optional fields
    if (citation.journal) bibtex += `  journal = {${citation.journal}},\n`;
    if (citation.volume) bibtex += `  volume = {${citation.volume}},\n`;
    if (citation.issue) bibtex += `  number = {${citation.issue}},\n`;
    if (citation.pages) bibtex += `  pages = {${citation.pages}},\n`;
    if (citation.publisher) bibtex += `  publisher = {${citation.publisher}},\n`;
    if (citation.doi) bibtex += `  doi = {${citation.doi}},\n`;
    if (citation.url) bibtex += `  url = {${citation.url}},\n`;
    
    bibtex += '}';
    
    return bibtex;
  }, []);

  const handleCopyBibTeX = useCallback(async (citation: Citation) => {
    const bibtex = generateBibTeX(citation);
    await navigator.clipboard.writeText(bibtex);
    setCopiedId(citation.id);
    setTimeout(() => setCopiedId(null), 2000);
  }, [generateBibTeX]);

  const handleExport = useCallback((format: ExportFormat) => {
    let content = '';
    let filename = `citations_${Date.now()}`;
    let mimeType = 'text/plain';
    
    switch (format) {
      case 'bibtex':
        content = citations.map(c => generateBibTeX(c)).join('\n\n');
        filename += '.bib';
        mimeType = 'application/x-bibtex';
        break;
      case 'json':
        content = JSON.stringify(citations, null, 2);
        filename += '.json';
        mimeType = 'application/json';
        break;
      case 'csv':
        // Simple CSV export
        const headers = ['Type', 'Authors', 'Title', 'Year', 'Journal', 'DOI'];
        const rows = citations.map(c => [
          c.type,
          c.authors.map(a => `${a.given} ${a.family}`).join('; '),
          `"${c.title}"`,
          c.year,
          c.journal || '',
          c.doi || ''
        ]);
        content = [headers, ...rows].map(row => row.join(',')).join('\n');
        filename += '.csv';
        mimeType = 'text/csv';
        break;
    }
    
    // Create and download file
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, [citations, generateBibTeX]);

  const handleImport = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    try {
      const text = await file.text();
      let importedCitations: Citation[] = [];
      
      if (file.name.endsWith('.bib')) {
        // Simple BibTeX parser (would use proper library in production)
        const entries = text.split('@').filter(e => e.trim());
        importedCitations = entries.map((entry, index) => {
          const type = entry.match(/^(\w+)\{/)?.[1] || 'misc';
          const id = entry.match(/\{([^,]+),/)?.[1] || `imported_${index}`;
          
          // Extract fields
          const getField = (name: string): string | undefined => {
            const match = entry.match(new RegExp(`${name}\\s*=\\s*[{"]([^}"]+)[}"]`));
            return match?.[1];
          };
          
          const authorString = getField('author') || '';
          const authors = authorString.split(' and ').map(a => {
            const parts = a.trim().split(' ');
            return {
              given: parts.slice(0, -1).join(' '),
              family: parts[parts.length - 1] || ''
            };
          });
          
          return {
            id,
            type: type === 'article' ? 'article' : 
                  type === 'book' ? 'book' : 
                  type === 'inproceedings' ? 'conference' : 'article',
            authors,
            title: getField('title') || 'Untitled',
            year: parseInt(getField('year') || '2024'),
            journal: getField('journal'),
            volume: getField('volume'),
            issue: getField('number'),
            pages: getField('pages'),
            doi: getField('doi'),
            url: getField('url'),
            publisher: getField('publisher')
          } as Citation;
        });
      } else if (file.name.endsWith('.json')) {
        importedCitations = JSON.parse(text);
      }
      
      if (importedCitations.length > 0) {
        const updatedCitations = [...citations, ...importedCitations];
        setCitations(updatedCitations);
        if (onCitationsChange) {
          onCitationsChange(updatedCitations);
        }
        setImportError(null);
      }
    } catch (error) {
      setImportError('Failed to import citations. Please check the file format.');
    }
    
    // Reset input
    event.target.value = '';
  }, [citations, onCitationsChange]);

  return (
    <Card className={cn('overflow-hidden', className)}>
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold">Citation Manager</h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Manage references and generate citations in multiple formats
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Select
              value={selectedStyle}
              onValueChange={(value) => {
                onStyleChange?.(value as CitationStyle);
              }}
            >
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(citationStyles).map(([value, label]) => (
                  <SelectItem key={value} value={value}>{label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <div className="border-b border-gray-200 dark:border-gray-700">
          <div className="flex px-4">
            <button
              onClick={() => setActiveTab('list')}
              className={cn(
                'px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'list'
                  ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                  : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
              )}
            >
              Citations ({citations.length})
            </button>
            <button
              onClick={() => setActiveTab('edit')}
              className={cn(
                'px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === 'edit'
                  ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400'
                  : 'text-gray-600 dark:text-gray-400 border-transparent hover:text-gray-900 dark:hover:text-white'
              )}
              disabled={!editingCitation}
            >
              {editingCitation ? 'Edit Citation' : 'Edit'}
            </button>
          </div>
        </div>

        {/* Import/Export Toolbar */}
        {(allowImport || allowExport) && activeTab === 'list' && (
          <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2">
              <Button
                variant="default"
                                onClick={handleAddCitation}
                className="flex items-center gap-2"
              >
                <Plus className="h-4 w-4" />
                Add Citation
              </Button>
              
              {allowImport && (
                <>
                  <input
                    type="file"
                    accept=".bib,.json"
                    onChange={handleImport}
                    className="hidden"
                    id="import-citations"
                  />
                  <Tooltip content="Import from BibTeX or JSON">
                    <Button
                      variant="outline"
                                            onClick={() => document.getElementById('import-citations')?.click()}
                      className="flex items-center gap-2"
                    >
                      <Upload className="h-4 w-4" />
                      Import
                    </Button>
                  </Tooltip>
                </>
              )}
            </div>
            
            {allowExport && citations.length > 0 && (
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                                    onClick={() => handleExport('bibtex')}
                  className="flex items-center gap-2"
                >
                  <Download className="h-4 w-4" />
                  BibTeX
                </Button>
                <Button
                  variant="outline"
                                    onClick={() => handleExport('json')}
                  className="flex items-center gap-2"
                >
                  <Download className="h-4 w-4" />
                  JSON
                </Button>
              </div>
            )}
          </div>
        )}

        {/* Content */}
        <div className="p-4">
          {/* Citations List */}
          {activeTab === 'list' && (
            <div className="space-y-4">
              {importError && (
                <Alert variant="destructive">
                  {importError}
                </Alert>
              )}
              
              {citations.length === 0 ? (
                <div className="text-center py-12">
                  <BookOpen className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 dark:text-gray-400">
                    No citations yet. Add your first citation to get started.
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {citations.map((citation) => {
                    const Icon = citationTypeIcons[citation.type];
                    return (
                      <Card
                        key={citation.id}
                        className={cn(
                          'p-4 cursor-pointer transition-colors',
                          selectedCitation?.id === citation.id && 'ring-2 ring-blue-500'
                        )}
                        onClick={() => setSelectedCitation(citation)}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-start gap-3 flex-1">
                            <Icon className="h-5 w-5 text-gray-500 mt-0.5" />
                            <div className="flex-1">
                              <h3 className="font-medium text-gray-900 dark:text-white">
                                {citation.title}
                              </h3>
                              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                                {citation.authors.map(a => `${a.given} ${a.family}`).join(', ')}
                                {citation.year && ` (${citation.year})`}
                              </p>
                              {citation.journal && (
                                <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
                                  {citation.journal}
                                  {citation.volume && `, ${citation.volume}`}
                                  {citation.issue && `(${citation.issue})`}
                                  {citation.pages && `, ${citation.pages}`}
                                </p>
                              )}
                              <div className="flex items-center gap-2 mt-2">
                                <Badge variant="secondary">
                                  {citation.type}
                                </Badge>
                                {citation.doi && (
                                  <Badge variant="default">
                                    DOI
                                  </Badge>
                                )}
                              </div>
                            </div>
                          </div>
                          
                          <div className="flex items-center gap-1">
                            <Tooltip content="Edit citation">
                              <Button
                                variant="ghost"
                                                                onClick={(e) => {
                                  e.stopPropagation();
                                  setEditingCitation(citation);
                                  setActiveTab('edit');
                                }}
                                className="h-8 w-8 p-0"
                              >
                                <Edit2 className="h-4 w-4" />
                              </Button>
                            </Tooltip>
                            
                            <Tooltip content={copiedId === citation.id ? 'Copied!' : 'Copy BibTeX'}>
                              <Button
                                variant="ghost"
                                                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleCopyBibTeX(citation);
                                }}
                                className="h-8 w-8 p-0"
                              >
                                {copiedId === citation.id ? (
                                  <Check className="h-4 w-4 text-green-500" />
                                ) : (
                                  <Copy className="h-4 w-4" />
                                )}
                              </Button>
                            </Tooltip>
                            
                            {citation.doi && (
                              <Tooltip content="Open DOI">
                                <Button
                                  variant="ghost"
                                                                    onClick={(e) => {
                                    e.stopPropagation();
                                    window.open(`https://doi.org/${citation.doi}`, '_blank');
                                  }}
                                  className="h-8 w-8 p-0"
                                >
                                  <ExternalLink className="h-4 w-4" />
                                </Button>
                              </Tooltip>
                            )}
                            
                            <Tooltip content="Delete citation">
                              <Button
                                variant="ghost"
                                                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteCitation(citation.id);
                                }}
                                className="h-8 w-8 p-0 text-red-600 hover:text-red-700"
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </Tooltip>
                          </div>
                        </div>
                        
                        {selectedCitation?.id === citation.id && (
                          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                            <h4 className="text-sm font-medium mb-2">Formatted Citation ({selectedStyle.toUpperCase()})</h4>
                            <p className="text-sm bg-gray-50 dark:bg-gray-800 p-3 rounded font-serif">
                              {formatCitation(citation, selectedStyle)}
                            </p>
                          </div>
                        )}
                      </Card>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Edit Form */}
          {activeTab === 'edit' && editingCitation && (
            <CitationEditForm
              citation={editingCitation}
              onSave={handleSaveCitation}
              onCancel={() => {
                setEditingCitation(null);
                setActiveTab('list');
              }}
            />
          )}
        </div>
      </Tabs>
    </Card>
  );
};

// Citation Edit Form Component
interface CitationEditFormProps {
  citation: Citation;
  onSave: (citation: Citation) => void;
  onCancel: () => void;
}

const CitationEditForm: React.FC<CitationEditFormProps> = ({
  citation,
  onSave,
  onCancel
}) => {
  const [formData, setFormData] = useState<Citation>(citation);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate
    const newErrors: Record<string, string> = {};
    if (!formData.title.trim()) newErrors.title = 'Title is required';
    if (formData.authors.length === 0 || !formData.authors[0].family) {
      newErrors.authors = 'At least one author is required';
    }
    if (!formData.year) newErrors.year = 'Year is required';
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    onSave(formData);
  }, [formData, onSave]);

  const handleAuthorChange = useCallback((index: number, field: keyof Author, value: string) => {
    const newAuthors = [...formData.authors];
    newAuthors[index] = { ...newAuthors[index], [field]: value };
    setFormData({ ...formData, authors: newAuthors });
  }, [formData]);

  const handleAddAuthor = useCallback(() => {
    setFormData({
      ...formData,
      authors: [...formData.authors, { given: '', family: '' }]
    });
  }, [formData]);

  const handleRemoveAuthor = useCallback((index: number) => {
    const newAuthors = formData.authors.filter((_, i) => i !== index);
    setFormData({ ...formData, authors: newAuthors });
  }, [formData]);

  return (
    <form onSubmit={handleSubmit} className="space-y-6 max-w-2xl">
      {/* Citation Type */}
      <div>
        <label className="block text-sm font-medium mb-2">Type</label>
        <Select
          value={formData.type}
          onValueChange={(value) => setFormData({ ...formData, type: value as CitationType })}
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="article">Journal Article</SelectItem>
            <SelectItem value="book">Book</SelectItem>
            <SelectItem value="bookSection">Book Chapter</SelectItem>
            <SelectItem value="conference">Conference Paper</SelectItem>
            <SelectItem value="thesis">Thesis</SelectItem>
            <SelectItem value="report">Report</SelectItem>
            <SelectItem value="webpage">Web Page</SelectItem>
            <SelectItem value="dataset">Dataset</SelectItem>
            <SelectItem value="software">Software</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Title */}
      <div>
        <label className="block text-sm font-medium mb-2">
          Title <span className="text-red-500">*</span>
        </label>
        <input
          type="text"
          value={formData.title}
          onChange={(e) => setFormData({ ...formData, title: e.target.value })}
          className={cn(
            'w-full px-3 py-2 border rounded-md',
            'bg-white dark:bg-gray-800',
            'border-gray-300 dark:border-gray-600',
            'focus:outline-none focus:ring-2 focus:ring-blue-500',
            errors.title && 'border-red-500'
          )}
          placeholder="Enter the title"
        />
        {errors.title && (
          <p className="text-sm text-red-500 mt-1">{errors.title}</p>
        )}
      </div>

      {/* Authors */}
      <div>
        <label className="block text-sm font-medium mb-2">
          Authors <span className="text-red-500">*</span>
        </label>
        <div className="space-y-2">
          {formData.authors.map((author, index) => (
            <div key={index} className="flex items-center gap-2">
              <input
                type="text"
                value={author.given}
                onChange={(e) => handleAuthorChange(index, 'given', e.target.value)}
                className="flex-1 px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
                placeholder="Given name"
              />
              <input
                type="text"
                value={author.family}
                onChange={(e) => handleAuthorChange(index, 'family', e.target.value)}
                className="flex-1 px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
                placeholder="Family name"
              />
              {formData.authors.length > 1 && (
                <Button
                  type="button"
                  variant="ghost"
                                    onClick={() => handleRemoveAuthor(index)}
                  className="h-8 w-8 p-0"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              )}
            </div>
          ))}
        </div>
        <Button
          type="button"
          variant="outline"
                    onClick={handleAddAuthor}
          className="mt-2"
        >
          <Plus className="h-4 w-4 mr-1" />
          Add Author
        </Button>
        {errors.authors && (
          <p className="text-sm text-red-500 mt-1">{errors.authors}</p>
        )}
      </div>

      {/* Year */}
      <div className="grid grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium mb-2">
            Year <span className="text-red-500">*</span>
          </label>
          <input
            type="number"
            value={formData.year}
            onChange={(e) => setFormData({ ...formData, year: parseInt(e.target.value) })}
            className={cn(
              'w-full px-3 py-2 border rounded-md',
              'bg-white dark:bg-gray-800',
              'border-gray-300 dark:border-gray-600',
              errors.year && 'border-red-500'
            )}
            min="1900"
            max={new Date().getFullYear() + 1}
          />
          {errors.year && (
            <p className="text-sm text-red-500 mt-1">{errors.year}</p>
          )}
        </div>
        <div>
          <label className="block text-sm font-medium mb-2">Month</label>
          <input
            type="number"
            value={formData.month || ''}
            onChange={(e) => setFormData({ ...formData, month: parseInt(e.target.value) || undefined })}
            className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
            min="1"
            max="12"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-2">Day</label>
          <input
            type="number"
            value={formData.day || ''}
            onChange={(e) => setFormData({ ...formData, day: parseInt(e.target.value) || undefined })}
            className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
            min="1"
            max="31"
          />
        </div>
      </div>

      {/* Journal/Publisher fields based on type */}
      {(formData.type === 'article' || formData.type === 'conference') && (
        <>
          <div>
            <label className="block text-sm font-medium mb-2">
              {formData.type === 'article' ? 'Journal' : 'Conference Name'}
            </label>
            <input
              type="text"
              value={formData.journal || ''}
              onChange={(e) => setFormData({ ...formData, journal: e.target.value })}
              className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
            />
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Volume</label>
              <input
                type="text"
                value={formData.volume || ''}
                onChange={(e) => setFormData({ ...formData, volume: e.target.value })}
                className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Issue</label>
              <input
                type="text"
                value={formData.issue || ''}
                onChange={(e) => setFormData({ ...formData, issue: e.target.value })}
                className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Pages</label>
              <input
                type="text"
                value={formData.pages || ''}
                onChange={(e) => setFormData({ ...formData, pages: e.target.value })}
                className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
                placeholder="e.g., 123-456"
              />
            </div>
          </div>
        </>
      )}

      {formData.type === 'book' && (
        <>
          <div>
            <label className="block text-sm font-medium mb-2">Publisher</label>
            <input
              type="text"
              value={formData.publisher || ''}
              onChange={(e) => setFormData({ ...formData, publisher: e.target.value })}
              className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Edition</label>
              <input
                type="text"
                value={formData.edition || ''}
                onChange={(e) => setFormData({ ...formData, edition: e.target.value })}
                className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">ISBN</label>
              <input
                type="text"
                value={formData.isbn || ''}
                onChange={(e) => setFormData({ ...formData, isbn: e.target.value })}
                className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
              />
            </div>
          </div>
        </>
      )}

      {/* DOI and URL */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-2">DOI</label>
          <input
            type="text"
            value={formData.doi || ''}
            onChange={(e) => setFormData({ ...formData, doi: e.target.value })}
            className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
            placeholder="10.xxxx/xxxxx"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-2">URL</label>
          <input
            type="url"
            value={formData.url || ''}
            onChange={(e) => setFormData({ ...formData, url: e.target.value })}
            className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
          />
        </div>
      </div>

      {/* Keywords */}
      <div>
        <label className="block text-sm font-medium mb-2">Keywords</label>
        <input
          type="text"
          value={formData.keywords?.join(', ') || ''}
          onChange={(e) => setFormData({ 
            ...formData, 
            keywords: e.target.value.split(',').map(k => k.trim()).filter(k => k) 
          })}
          className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
          placeholder="Comma-separated keywords"
        />
      </div>

      {/* Abstract */}
      <div>
        <label className="block text-sm font-medium mb-2">Abstract</label>
        <textarea
          value={formData.abstract || ''}
          onChange={(e) => setFormData({ ...formData, abstract: e.target.value })}
          className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
          rows={4}
        />
      </div>

      {/* Form Actions */}
      <div className="flex justify-end gap-2 pt-4 border-t">
        <Button type="button" variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit" variant="default">
          Save Citation
        </Button>
      </div>
    </form>
  );
};