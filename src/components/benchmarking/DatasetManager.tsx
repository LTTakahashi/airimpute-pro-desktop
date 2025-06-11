import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/Card';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Progress } from '../ui/Progress';
import { Checkbox } from '../ui/Checkbox';
import { 
  Database, 
  Upload, 
  Download, 
  RefreshCw, 
  Plus,
  Trash2,
  FileText,
  HardDrive,
  BarChart3,
  Clock,
  Shuffle
} from 'lucide-react';
import { cn } from '@/utils/cn';

interface Dataset {
  name: string;
  description: string;
  size: number;
  missingRate: number;
  pattern: string;
  metadata: Record<string, any>;
}

interface DatasetManagerProps {
  datasets: Dataset[];
  selectedDatasets: string[];
  onSelectionChange: (datasets: string[]) => void;
  loading?: boolean;
  className?: string;
}

export const DatasetManager: React.FC<DatasetManagerProps> = ({
  datasets,
  selectedDatasets,
  onSelectionChange,
  loading = false,
  className
}) => {
  const [showDetails, setShowDetails] = useState<string | null>(null);
  const [generatingDataset, setGeneratingDataset] = useState(false);

  const toggleDataset = (datasetName: string) => {
    if (selectedDatasets.includes(datasetName)) {
      onSelectionChange(selectedDatasets.filter(d => d !== datasetName));
    } else {
      onSelectionChange([...selectedDatasets, datasetName]);
    }
  };

  const selectAll = () => {
    onSelectionChange(datasets.map(d => d.name));
  };

  const selectNone = () => {
    onSelectionChange([]);
  };

  const formatSize = (size: number) => {
    if (size < 1000) return `${size} values`;
    if (size < 1000000) return `${(size / 1000).toFixed(1)}K values`;
    return `${(size / 1000000).toFixed(1)}M values`;
  };

  const getPatternIcon = (pattern: string) => {
    if (pattern.includes('random')) return <Shuffle className="w-4 h-4" />;
    if (pattern.includes('block')) return <BarChart3 className="w-4 h-4" />;
    if (pattern.includes('temporal')) return <Clock className="w-4 h-4" />;
    return <FileText className="w-4 h-4" />;
  };

  const getDatasetCategory = (name: string) => {
    if (name.includes('synthetic')) return 'synthetic';
    if (name.includes('real')) return 'real-world';
    if (name.includes('benchmark')) return 'standard';
    return 'custom';
  };

  const categoryColors = {
    'synthetic': 'default',
    'real-world': 'secondary',
    'standard': 'success',
    'custom': 'outline'
  } as const;

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="py-8">
          <div className="flex flex-col items-center justify-center space-y-4">
            <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
            <p className="text-sm text-muted-foreground">Loading datasets...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={cn("space-y-4", className)}>
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={selectAll}
            disabled={selectedDatasets.length === datasets.length}
          >
            Select All
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={selectNone}
            disabled={selectedDatasets.length === 0}
          >
            Clear
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setGeneratingDataset(true)}
          >
            <Plus className="w-4 h-4 mr-1" />
            Generate
          </Button>
          <Button variant="outline" size="sm">
            <Upload className="w-4 h-4 mr-1" />
            Import
          </Button>
        </div>
      </div>

      {/* Dataset Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {datasets.map(dataset => {
          const isSelected = selectedDatasets.includes(dataset.name);
          const category = getDatasetCategory(dataset.name);
          
          return (
            <Card
              key={dataset.name}
              className={cn(
                "cursor-pointer transition-all",
                isSelected && "ring-2 ring-primary"
              )}
              onClick={() => toggleDataset(dataset.name)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Checkbox
                        checked={isSelected}
                        onCheckedChange={() => toggleDataset(dataset.name)}
                        onClick={e => e.stopPropagation()}
                      />
                      {dataset.name}
                    </CardTitle>
                    <CardDescription className="text-xs mt-1">
                      {dataset.description}
                    </CardDescription>
                  </div>
                  <Badge variant={categoryColors[category]} className="ml-2">
                    {category}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex items-center gap-1">
                    <Database className="w-3 h-3 text-muted-foreground" />
                    <span>{formatSize(dataset.size)}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <HardDrive className="w-3 h-3 text-muted-foreground" />
                    <span>{(dataset.missingRate * 100).toFixed(1)}% missing</span>
                  </div>
                  <div className="flex items-center gap-1">
                    {getPatternIcon(dataset.pattern)}
                    <span className="truncate">{dataset.pattern}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <FileText className="w-3 h-3 text-muted-foreground" />
                    <span>{Object.keys(dataset.metadata).length} metadata</span>
                  </div>
                </div>
                
                {/* Dataset preview on hover/click */}
                {showDetails === dataset.name && (
                  <div className="mt-3 pt-3 border-t space-y-2">
                    <div className="text-xs space-y-1">
                      <p><strong>Temporal Range:</strong> {dataset.metadata.temporal_range || 'N/A'}</p>
                      <p><strong>Variables:</strong> {dataset.metadata.variables?.join(', ') || 'N/A'}</p>
                      <p><strong>Stations:</strong> {dataset.metadata.stations || 'N/A'}</p>
                    </div>
                    <div className="flex gap-2">
                      <Button size="sm" variant="outline" className="text-xs">
                        <Download className="w-3 h-3 mr-1" />
                        Export
                      </Button>
                      <Button size="sm" variant="outline" className="text-xs">
                        <Trash2 className="w-3 h-3 mr-1" />
                        Remove
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Summary */}
      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>{selectedDatasets.length} of {datasets.length} datasets selected</span>
        <span>
          Total data points: {
            datasets
              .filter(d => selectedDatasets.includes(d.name))
              .reduce((acc, d) => acc + d.size, 0)
              .toLocaleString()
          }
        </span>
      </div>

      {/* Dataset Generation Modal (placeholder) */}
      {generatingDataset && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle>Generate Synthetic Dataset</CardTitle>
              <CardDescription>
                Create a new dataset with custom parameters
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Dataset generation UI would go here...
              </p>
            </CardContent>
            <div className="flex justify-end gap-2 p-4 pt-0">
              <Button variant="outline" onClick={() => setGeneratingDataset(false)}>
                Cancel
              </Button>
              <Button onClick={() => setGeneratingDataset(false)}>
                Generate
              </Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};