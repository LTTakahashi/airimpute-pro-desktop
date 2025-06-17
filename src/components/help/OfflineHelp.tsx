import React, { useState, useEffect } from 'react';
import { HelpCircle, ChevronRight, Search, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { ScientificCard } from '@/components/layout/ScientificCard';
import { Alert } from '@/components/ui/Alert';
import { Badge } from '@/components/ui/Badge';

interface HelpTopic {
  id: string;
  title: string;
  description: string;
  articles: HelpArticle[];
}

interface HelpArticle {
  id: string;
  title: string;
  content: string;
  lastUpdated?: string;
  tags?: string[];
}

interface LocalDoc {
  id: string;
  title: string;
  path: string;
  format: 'pdf' | 'html' | 'md';
  size: string;
  description: string;
}

// Complete offline documentation
const helpTopics: HelpTopic[] = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    description: 'Learn the basics of AirImpute Pro',
    articles: [
      {
        id: 'import-data',
        title: 'Importing Your Data',
        content: `
### Supported File Formats
AirImpute Pro supports multiple data formats:

* **CSV** - Comma-separated values
* **Excel** - .xlsx and .xls files
* **NetCDF** - Network Common Data Form
* **HDF5** - Hierarchical Data Format
* **Parquet** - Columnar storage format

### Data Requirements
Your data should:

* Have a datetime column for time series analysis
* Contain numerical columns for imputation
* Be properly formatted (no merged cells in Excel)

### Import Process
1. Click "Import Data" from the main menu
2. Select your file or drag and drop
3. Configure column mappings if needed
4. Preview the data before confirming
        `,
        tags: ['import', 'data', 'formats'],
        lastUpdated: '2024-01-15'
      },
      {
        id: 'first-imputation',
        title: 'Running Your First Imputation',
        content: `
### Step-by-Step Guide
1. Import your dataset using the Data Import page
2. Navigate to the Imputation page
3. Select an imputation method (we recommend starting with Linear Interpolation)
4. Configure parameters if needed
5. Click "Run Imputation"
6. Review results in the Analysis page

### Choosing the Right Method
For beginners, we recommend:

* **Linear Interpolation** - For small, regular gaps
* **Forward Fill** - For sensor failures
* **RAH Method** - For complex patterns (advanced)
        `,
        tags: ['imputation', 'tutorial', 'methods'],
        lastUpdated: '2024-01-15'
      }
    ]
  },
  {
    id: 'imputation-methods',
    title: 'Imputation Methods',
    description: 'Understanding different imputation algorithms',
    articles: [
      {
        id: 'rah-method',
        title: 'Robust Adaptive Hybrid (RAH)',
        content: `
### Overview
The RAH method is our flagship algorithm that achieved 42.1% improvement over traditional methods.

### Key Features
* Adaptive method selection based on data characteristics
* Spatial-temporal weight optimization
* Uncertainty quantification
* Handles MCAR, MAR, and MNAR patterns

### When to Use
RAH is ideal for:

* Complex missing patterns
* Multi-pollutant datasets
* When accuracy is critical
* Long-term gaps (>24 hours)

### Parameters
* **window_size**: Temporal context (default: 168 hours)
* **spatial_radius**: Distance for spatial correlation (km)
* **confidence_level**: Uncertainty bounds (default: 0.95)
        `,
        tags: ['rah', 'advanced', 'spatial-temporal'],
        lastUpdated: '2024-01-15'
      },
      {
        id: 'ml-methods',
        title: 'Machine Learning Methods',
        content: `
### Random Forest
Ensemble method using multiple decision trees.

* Good for: Non-linear patterns
* Parameters: n_estimators, max_depth
* Pros: Robust to outliers
* Cons: Can be slow for large datasets

### XGBoost
Gradient boosting method optimized for performance.

* Good for: Complex relationships
* Parameters: learning_rate, max_depth, n_estimators
* Pros: High accuracy, fast
* Cons: Requires tuning

### Neural Networks
Deep learning approach for complex patterns.

* Good for: Large datasets with complex patterns
* Parameters: layers, neurons, activation
* Pros: Can capture any pattern
* Cons: Requires lots of data
        `,
        tags: ['machine-learning', 'advanced'],
        lastUpdated: '2024-01-15'
      }
    ]
  },
  {
    id: 'troubleshooting',
    title: 'Troubleshooting',
    description: 'Common issues and solutions',
    articles: [
      {
        id: 'common-errors',
        title: 'Common Errors',
        content: `
### Import Errors
**Problem:** "Failed to parse date column"  
**Solution:** Ensure your date column is in a standard format (YYYY-MM-DD HH:MM:SS)

### Memory Errors
**Problem:** "Out of memory"  
**Solution:** Try reducing chunk size in Settings > Computation

### Imputation Failures
**Problem:** "No valid data points"  
**Solution:** Check that your data contains some non-missing values

### Performance Issues
**Problem:** "Processing taking too long"  
**Solutions:**

* Use a simpler method for initial testing
* Reduce dataset size
* Enable GPU acceleration if available
* Check Settings > Performance
        `,
        tags: ['errors', 'solutions', 'debugging'],
        lastUpdated: '2024-01-15'
      },
      {
        id: 'performance-tips',
        title: 'Performance Optimization',
        content: `
### General Tips
* Process data in chunks for large files
* Use appropriate methods for your data size
* Enable caching in Settings
* Close unnecessary applications

### Method-Specific Tips
* **Linear methods:** Fast for any size
* **ML methods:** Use sampling for testing
* **RAH:** Adjust window_size for speed

### Hardware Recommendations
* RAM: 8GB minimum, 16GB recommended
* CPU: 4+ cores for parallel processing
* GPU: Optional but speeds up deep learning
* Storage: SSD recommended for large files
        `,
        tags: ['performance', 'optimization', 'hardware'],
        lastUpdated: '2024-01-15'
      }
    ]
  },
  {
    id: 'academic-features',
    title: 'Academic Features',
    description: 'Publication-ready outputs and citations',
    articles: [
      {
        id: 'latex-export',
        title: 'LaTeX Export',
        content: `
          <h3>Generating LaTeX Reports</h3>
          <p>Create publication-ready reports with:</p>
          <ul>
            <li>Formatted equations</li>
            <li>High-quality figures</li>
            <li>Statistical tables</li>
            <li>Bibliography management</li>
          </ul>
          
          <h3>Templates Available</h3>
          <ul>
            <li>IEEE Transaction format</li>
            <li>Elsevier article format</li>
            <li>General report format</li>
            <li>Thesis chapter format</li>
          </ul>
          
          <h3>Customization</h3>
          <p>All templates can be customized with your institution's requirements.</p>
        `,
        tags: ['latex', 'export', 'publication'],
        lastUpdated: '2024-01-15'
      },
      {
        id: 'citations',
        title: 'Citations and References',
        content: `
          <h3>Citing AirImpute Pro</h3>
          <p>If you use AirImpute Pro in your research, please cite:</p>
          <pre>
@software{airimpute2024,
  title = {AirImpute Pro: Advanced Air Quality Data Imputation},
  author = {Your Lab Name},
  year = {2024},
  version = {1.0.0},
  url = {https://github.com/yourusername/airimpute-pro}
}
          </pre>
          
          <h3>Method Citations</h3>
          <p>Each method includes proper citations. Access them via the method documentation.</p>
          
          <h3>Reproducibility</h3>
          <p>Export reproducibility certificates for your publications.</p>
        `,
        tags: ['citation', 'bibtex', 'research'],
        lastUpdated: '2024-01-15'
      }
    ]
  }
];

// Local documentation files
const localDocs: LocalDoc[] = [
  {
    id: 'user-manual',
    title: 'Complete User Manual',
    path: '/docs/user-manual.pdf',
    format: 'pdf',
    size: '2.4 MB',
    description: 'Comprehensive guide covering all features'
  },
  {
    id: 'method-comparison',
    title: 'Method Comparison Study',
    path: '/docs/method-comparison.pdf',
    format: 'pdf',
    size: '1.8 MB',
    description: 'Detailed comparison of all imputation methods'
  },
  {
    id: 'api-reference',
    title: 'API Reference',
    path: '/docs/api-reference.html',
    format: 'html',
    size: '890 KB',
    description: 'Technical documentation for developers'
  }
];

export const OfflineHelp: React.FC = () => {
  const [selectedTopic, setSelectedTopic] = useState<string>('getting-started');
  const [selectedArticle, setSelectedArticle] = useState<string>('import-data');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<HelpArticle[]>([]);

  const currentTopic = helpTopics.find(t => t.id === selectedTopic);
  const currentArticle = currentTopic?.articles.find(a => a.id === selectedArticle);

  // Search functionality
  useEffect(() => {
    if (searchQuery.length > 2) {
      const results: HelpArticle[] = [];
      helpTopics.forEach(topic => {
        topic.articles.forEach(article => {
          if (
            article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
            article.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
            article.tags?.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
          ) {
            results.push(article);
          }
        });
      });
      setSearchResults(results);
    } else {
      setSearchResults([]);
    }
  }, [searchQuery]);

  const openLocalDoc = (doc: LocalDoc) => {
    // In a real app, this would use Tauri's API to open the file
    console.log('Opening local document:', doc.path);
  };

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="flex items-center gap-4 mb-6">
        <h1 className="text-3xl font-bold">Help & Documentation</h1>
        <Badge variant="outline" className="text-green-600 border-green-600">
          100% Offline
        </Badge>
      </div>

      <Alert className="mb-6">
        <HelpCircle className="h-4 w-4" />
        <div>
          <p className="font-medium">All documentation is available offline</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            No internet connection required. All help content is stored locally.
          </p>
        </div>
      </Alert>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar */}
        <div className="lg:col-span-1">
          <Card className="p-4 mb-6">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search help..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            {searchResults.length > 0 && (
              <div className="mt-4 space-y-2">
                <p className="text-sm font-medium text-gray-600">Search Results:</p>
                {searchResults.map(article => (
                  <button
                    key={article.id}
                    onClick={() => {
                      const topic = helpTopics.find(t => 
                        t.articles.some(a => a.id === article.id)
                      );
                      if (topic) {
                        setSelectedTopic(topic.id);
                        setSelectedArticle(article.id);
                        setSearchQuery('');
                      }
                    }}
                    className="w-full text-left px-2 py-1 text-sm rounded hover:bg-gray-100 dark:hover:bg-gray-700"
                  >
                    {article.title}
                  </button>
                ))}
              </div>
            )}
          </Card>

          <Card className="p-4">
            <h3 className="font-semibold mb-4">Topics</h3>
            <nav className="space-y-1">
              {helpTopics.map(topic => (
                <div key={topic.id}>
                  <button
                    onClick={() => {
                      setSelectedTopic(topic.id);
                      setSelectedArticle(topic.articles[0]?.id);
                    }}
                    className={`
                      w-full text-left px-3 py-2 rounded-md transition-colors
                      ${selectedTopic === topic.id
                        ? 'bg-blue-50 text-blue-700 font-medium dark:bg-blue-900/20 dark:text-blue-400'
                        : 'hover:bg-gray-50 dark:hover:bg-gray-800'}
                    `}
                  >
                    {topic.title}
                  </button>
                  {selectedTopic === topic.id && (
                    <div className="ml-3 mt-1 space-y-1">
                      {topic.articles.map(article => (
                        <button
                          key={article.id}
                          onClick={() => setSelectedArticle(article.id)}
                          className={`
                            w-full text-left px-3 py-1.5 text-sm rounded-md transition-colors flex items-center
                            ${selectedArticle === article.id
                              ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                              : 'hover:bg-gray-50 dark:hover:bg-gray-800'}
                          `}
                        >
                          <ChevronRight className="w-3 h-3 mr-1" />
                          {article.title}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </nav>
          </Card>
        </div>

        {/* Content */}
        <div className="lg:col-span-3">
          {currentArticle && (
            <Card className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="text-2xl font-bold">{currentArticle.title}</h2>
                  {currentArticle.lastUpdated && (
                    <p className="text-sm text-gray-500 mt-1">
                      Last updated: {currentArticle.lastUpdated}
                    </p>
                  )}
                </div>
                {currentArticle.tags && (
                  <div className="flex gap-2">
                    {currentArticle.tags.map(tag => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <ReactMarkdown>{currentArticle.content}</ReactMarkdown>
              </div>
            </Card>
          )}

          {/* Offline Resources */}
          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-4">Offline Resources</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {localDocs.map(doc => (
                <ScientificCard
                  key={doc.id}
                  title={doc.title}
                  description={doc.description}
                >
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-sm text-gray-500">{doc.size}</span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => openLocalDoc(doc)}
                    >
                      <FileText className="w-4 h-4 mr-1" />
                      Open
                    </Button>
                  </div>
                </ScientificCard>
              ))}
            </div>
          </div>

          {/* Keyboard Shortcuts */}
          <Card className="p-6 mt-6">
            <h3 className="text-lg font-semibold mb-4">Keyboard Shortcuts</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <kbd className="px-2 py-1 bg-gray-100 border border-gray-300 rounded text-sm dark:bg-gray-800 dark:border-gray-600">Ctrl+O</kbd>
                <span className="ml-2 text-sm">Open file</span>
              </div>
              <div>
                <kbd className="px-2 py-1 bg-gray-100 border border-gray-300 rounded text-sm dark:bg-gray-800 dark:border-gray-600">Ctrl+S</kbd>
                <span className="ml-2 text-sm">Save project</span>
              </div>
              <div>
                <kbd className="px-2 py-1 bg-gray-100 border border-gray-300 rounded text-sm dark:bg-gray-800 dark:border-gray-600">Ctrl+Shift+I</kbd>
                <span className="ml-2 text-sm">Quick import</span>
              </div>
              <div>
                <kbd className="px-2 py-1 bg-gray-100 border border-gray-300 rounded text-sm dark:bg-gray-800 dark:border-gray-600">Ctrl+Shift+R</kbd>
                <span className="ml-2 text-sm">Run imputation</span>
              </div>
              <div>
                <kbd className="px-2 py-1 bg-gray-100 border border-gray-300 rounded text-sm dark:bg-gray-800 dark:border-gray-600">Ctrl+E</kbd>
                <span className="ml-2 text-sm">Export data</span>
              </div>
              <div>
                <kbd className="px-2 py-1 bg-gray-100 border border-gray-300 rounded text-sm dark:bg-gray-800 dark:border-gray-600">F1</kbd>
                <span className="ml-2 text-sm">Open help</span>
              </div>
            </div>
          </Card>

          {/* Contact Info (Offline) */}
          <Card className="p-6 mt-6">
            <h3 className="text-lg font-semibold mb-4">Need More Help?</h3>
            <div className="space-y-4">
              <div>
                <p className="font-medium">Email Support</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  support@airimpute.pro - Save this address for when you&apos;re online
                </p>
              </div>
              <div>
                <p className="font-medium">Community Forum</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Visit forum.airimpute.pro when connected to the internet
                </p>
              </div>
              <div>
                <p className="font-medium">Video Tutorials</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Download tutorial videos from the website for offline viewing
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default OfflineHelp;