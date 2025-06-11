import React, { useState } from 'react';
import { Copy, Check, FileText, Database, Info } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Alert } from '@/components/ui/Alert';
import { Badge } from '@/components/ui/Badge';
import { Tooltip } from '@/components/ui/Tooltip';

interface Citation {
  id: string;
  key: string;
  type: 'article' | 'book' | 'conference' | 'software' | 'dataset';
  title: string;
  authors: string[];
  year: number;
  venue?: string;
  doi?: string;
  url?: string;
  abstract?: string;
  bibtex: string;
  tags?: string[];
}

// Local citation database - no external fetching required
const localCitations: Citation[] = [
  {
    id: '1',
    key: 'takahashi2024rah',
    type: 'article',
    title: 'Robust Adaptive Hybrid Method for Air Quality Data Imputation',
    authors: ['Takahashi, L.', 'Silva, M.', 'Santos, J.'],
    year: 2024,
    venue: 'IEEE Transactions on Environmental Informatics',
    doi: '10.1109/TEI.2024.123456',
    bibtex: `@article{takahashi2024rah,
  title={Robust Adaptive Hybrid Method for Air Quality Data Imputation},
  author={Takahashi, L. and Silva, M. and Santos, J.},
  journal={IEEE Transactions on Environmental Informatics},
  year={2024},
  volume={10},
  number={2},
  pages={123--145},
  doi={10.1109/TEI.2024.123456}
}`,
    abstract: 'We present a novel hybrid approach for imputing missing values in air quality datasets...',
    tags: ['imputation', 'air quality', 'hybrid methods']
  },
  {
    id: '2',
    key: 'breiman2001random',
    type: 'article',
    title: 'Random Forests',
    authors: ['Breiman, L.'],
    year: 2001,
    venue: 'Machine Learning',
    doi: '10.1023/A:1010933404324',
    bibtex: `@article{breiman2001random,
  title={Random Forests},
  author={Breiman, Leo},
  journal={Machine Learning},
  volume={45},
  number={1},
  pages={5--32},
  year={2001},
  publisher={Springer}
}`,
    tags: ['machine learning', 'random forest', 'ensemble methods']
  },
  {
    id: '3',
    key: 'chen2016xgboost',
    type: 'conference',
    title: 'XGBoost: A Scalable Tree Boosting System',
    authors: ['Chen, T.', 'Guestrin, C.'],
    year: 2016,
    venue: 'KDD \'16',
    doi: '10.1145/2939672.2939785',
    bibtex: `@inproceedings{chen2016xgboost,
  title={XGBoost: A Scalable Tree Boosting System},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={785--794},
  year={2016}
}`,
    tags: ['machine learning', 'gradient boosting', 'xgboost']
  },
  {
    id: '4',
    key: 'airimpute2024',
    type: 'software',
    title: 'AirImpute Pro: Advanced Air Quality Data Imputation Software',
    authors: ['Environmental Informatics Lab'],
    year: 2024,
    url: 'https://github.com/envirolab/airimpute-pro',
    bibtex: `@software{airimpute2024,
  title={AirImpute Pro: Advanced Air Quality Data Imputation Software},
  author={{Environmental Informatics Lab}},
  year={2024},
  version={1.0.0},
  url={https://github.com/envirolab/airimpute-pro}
}`,
    tags: ['software', 'air quality', 'imputation']
  }
];

interface OfflineCitationManagerProps {
  onCitationSelect?: (citation: Citation) => void;
  selectedCitations?: string[];
  mode?: 'select' | 'manage';
}

export const OfflineCitationManager: React.FC<OfflineCitationManagerProps> = ({
  onCitationSelect,
  selectedCitations = [],
  mode = 'manage'
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFormat, setSelectedFormat] = useState<'bibtex' | 'apa' | 'mla'>('bibtex');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);

  // Get all unique tags
  const allTags = Array.from(
    new Set(localCitations.flatMap(c => c.tags || []))
  ).sort();

  // Filter citations based on search and tags
  const filteredCitations = localCitations.filter(citation => {
    const matchesSearch = searchQuery === '' || 
      citation.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      citation.authors.some(a => a.toLowerCase().includes(searchQuery.toLowerCase())) ||
      citation.key.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (citation.tags || []).some(t => t.toLowerCase().includes(searchQuery.toLowerCase()));

    const matchesTags = selectedTags.length === 0 ||
      selectedTags.every(tag => citation.tags?.includes(tag));

    return matchesSearch && matchesTags;
  });

  const formatCitation = (citation: Citation, format: 'bibtex' | 'apa' | 'mla'): string => {
    switch (format) {
      case 'bibtex':
        return citation.bibtex;
      
      case 'apa':
        const apaAuthors = citation.authors.length > 2
          ? `${citation.authors[0]} et al.`
          : citation.authors.join(' & ');
        return `${apaAuthors} (${citation.year}). ${citation.title}. ${citation.venue || 'Unpublished'}.${citation.doi ? ` https://doi.org/${citation.doi}` : ''}`;
      
      case 'mla':
        const mlaAuthors = citation.authors.length > 2
          ? `${citation.authors[0]}, et al.`
          : citation.authors.join(', and ');
        return `${mlaAuthors}. "${citation.title}." ${citation.venue || 'Unpublished'}, ${citation.year}.${citation.doi ? ` DOI: ${citation.doi}` : ''}`;
      
      default:
        return citation.bibtex;
    }
  };

  const handleCopy = async (citation: Citation) => {
    const formatted = formatCitation(citation, selectedFormat);
    try {
      await navigator.clipboard.writeText(formatted);
      setCopiedId(citation.id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy citation:', err);
    }
  };

  const toggleTag = (tag: string) => {
    setSelectedTags(prev =>
      prev.includes(tag)
        ? prev.filter(t => t !== tag)
        : [...prev, tag]
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Citation Manager</h2>
        <Badge variant="outline" className="text-green-600 border-green-600">
          <Database className="w-3 h-3 mr-1" />
          Offline Database
        </Badge>
      </div>

      <Alert>
        <Info className="h-4 w-4" />
        <div>
          <p className="font-medium">All citations stored locally</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            No internet required. DOI links are for reference only.
          </p>
        </div>
      </Alert>

      {/* Search and filters */}
      <Card className="p-4 space-y-4">
        <div className="flex gap-4">
          <input
            type="text"
            placeholder="Search citations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <select
            value={selectedFormat}
            onChange={(e) => setSelectedFormat(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="bibtex">BibTeX</option>
            <option value="apa">APA</option>
            <option value="mla">MLA</option>
          </select>
        </div>

        {/* Tag filters */}
        <div className="flex flex-wrap gap-2">
          <span className="text-sm font-medium text-gray-600">Filter by tags:</span>
          {allTags.map(tag => (
            <button
              key={tag}
              onClick={() => toggleTag(tag)}
              className={`px-2 py-1 text-xs rounded-full transition-colors ${
                selectedTags.includes(tag)
                  ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400'
              }`}
            >
              {tag}
            </button>
          ))}
        </div>
      </Card>

      {/* Citations list */}
      <div className="space-y-4">
        {filteredCitations.length === 0 ? (
          <Card className="p-8 text-center">
            <p className="text-gray-500">No citations found matching your criteria.</p>
          </Card>
        ) : (
          filteredCitations.map(citation => (
            <Card
              key={citation.id}
              className={`p-4 space-y-3 transition-all ${
                mode === 'select' && selectedCitations.includes(citation.key)
                  ? 'ring-2 ring-blue-500'
                  : ''
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="font-semibold">{citation.title}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {citation.authors.join(', ')} • {citation.year}
                  </p>
                  {citation.venue && (
                    <p className="text-sm text-gray-500 dark:text-gray-500">
                      {citation.venue}
                    </p>
                  )}
                </div>
                <Badge variant="secondary" className="ml-4">
                  {citation.type}
                </Badge>
              </div>

              {citation.abstract && (
                <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                  {citation.abstract}
                </p>
              )}

              {citation.tags && (
                <div className="flex flex-wrap gap-1">
                  {citation.tags.map(tag => (
                    <Badge key={tag} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              )}

              <div className="flex items-center gap-2">
                {mode === 'select' ? (
                  <Button
                    variant={selectedCitations.includes(citation.key) ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => onCitationSelect?.(citation)}
                  >
                    {selectedCitations.includes(citation.key) ? 'Selected' : 'Select'}
                  </Button>
                ) : (
                  <>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleCopy(citation)}
                    >
                      {copiedId === citation.id ? (
                        <>
                          <Check className="w-4 h-4 mr-1 text-green-500" />
                          Copied!
                        </>
                      ) : (
                        <>
                          <Copy className="w-4 h-4 mr-1" />
                          Copy {selectedFormat.toUpperCase()}
                        </>
                      )}
                    </Button>
                    {citation.doi && (
                      <Tooltip content="DOI stored locally - go online to access">
                        <Badge variant="outline" className="text-xs">
                          DOI: {citation.doi}
                        </Badge>
                      </Tooltip>
                    )}
                  </>
                )}
              </div>
            </Card>
          ))
        )}
      </div>

      {/* Export all citations */}
      {mode === 'manage' && filteredCitations.length > 0 && (
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Export Citations</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Export {filteredCitations.length} citations in {selectedFormat.toUpperCase()} format
              </p>
            </div>
            <Button
              onClick={() => {
                const allCitations = filteredCitations
                  .map(c => formatCitation(c, selectedFormat))
                  .join('\n\n');
                navigator.clipboard.writeText(allCitations);
              }}
            >
              <FileText className="w-4 h-4 mr-2" />
              Export All
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
};

// Citation selector component for use in other parts of the app
export const CitationSelector: React.FC<{
  onSelect: (citations: Citation[]) => void;
  selectedKeys?: string[];
}> = ({ onSelect, selectedKeys = [] }) => {
  const [selected, setSelected] = useState<string[]>(selectedKeys);

  const handleSelect = (citation: Citation) => {
    const newSelected = selected.includes(citation.key)
      ? selected.filter(k => k !== citation.key)
      : [...selected, citation.key];
    
    setSelected(newSelected);
    
    const selectedCitations = localCitations.filter(c => 
      newSelected.includes(c.key)
    );
    onSelect(selectedCitations);
  };

  return (
    <OfflineCitationManager
      mode="select"
      selectedCitations={selected}
      onCitationSelect={handleSelect}
    />
  );
};