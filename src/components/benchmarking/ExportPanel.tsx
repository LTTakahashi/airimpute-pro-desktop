import React, { useState } from 'react';
import { Button } from '../ui/Button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '../ui/DropdownMenu';
import { 
  Download, 
  FileText, 
  FileSpreadsheet, 
  FileJson,
  FileCode,
  Image,
  Package,
  Check
} from 'lucide-react';
import { invoke } from '@tauri-apps/api/tauri';

interface ExportPanelProps {
  results: any[];
  className?: string;
}

export const ExportPanel: React.FC<ExportPanelProps> = ({ results, className }) => {
  const [exporting, setExporting] = useState(false);
  const [exportSuccess, setExportSuccess] = useState<string | null>(null);

  const exportFormats = [
    { 
      format: 'csv', 
      label: 'CSV File', 
      icon: FileSpreadsheet,
      description: 'Comma-separated values for Excel/spreadsheets'
    },
    { 
      format: 'json', 
      label: 'JSON File', 
      icon: FileJson,
      description: 'JavaScript Object Notation for data interchange'
    },
    { 
      format: 'latex', 
      label: 'LaTeX Table', 
      icon: FileCode,
      description: 'Publication-ready LaTeX table code'
    },
    { 
      format: 'html', 
      label: 'HTML Report', 
      icon: FileText,
      description: 'Interactive HTML report with charts'
    },
    { 
      format: 'png', 
      label: 'PNG Charts', 
      icon: Image,
      description: 'High-resolution chart images'
    },
    { 
      format: 'hdf5', 
      label: 'HDF5 Archive', 
      icon: Package,
      description: 'Hierarchical data format for large datasets'
    }
  ];

  const handleExport = async (format: string) => {
    if (!results.length) return;

    setExporting(true);
    setExportSuccess(null);

    try {
      // Get save file path from user
      const { save } = await import('@tauri-apps/api/dialog');
      const filePath = await save({
        defaultPath: `benchmark_results_${new Date().toISOString().split('T')[0]}.${format}`,
        filters: [{
          name: format.toUpperCase(),
          extensions: [format]
        }]
      });

      if (!filePath) {
        setExporting(false);
        return;
      }

      // Call export command
      await invoke('export_benchmark_results', {
        format,
        outputPath: filePath,
        results
      });

      setExportSuccess(format);
      setTimeout(() => setExportSuccess(null), 3000);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setExporting(false);
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button 
          variant="outline" 
          disabled={!results.length || exporting}
          className={className}
        >
          {exportSuccess ? (
            <>
              <Check className="w-4 h-4 mr-2" />
              Exported!
            </>
          ) : (
            <>
              <Download className="w-4 h-4 mr-2" />
              {exporting ? 'Exporting...' : 'Export'}
            </>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64">
        <DropdownMenuLabel>Export Format</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {exportFormats.map(({ format, label, icon: Icon, description }) => (
          <DropdownMenuItem
            key={format}
            onClick={() => handleExport(format)}
            disabled={exporting}
            className="cursor-pointer"
          >
            <div className="flex items-start gap-3 w-full">
              <Icon className="w-4 h-4 mt-0.5 text-muted-foreground" />
              <div className="flex-1">
                <div className="font-medium">{label}</div>
                <div className="text-xs text-muted-foreground">{description}</div>
              </div>
            </div>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};