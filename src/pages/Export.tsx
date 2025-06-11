import React, { useState } from 'react';
import { FileDown, FileText, Table, Archive, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { ScientificCard } from '@/components/layout/ScientificCard';
import { invoke } from '@tauri-apps/api/tauri';
import { save } from '@tauri-apps/api/dialog';
import { useNavigate } from 'react-router-dom';
import { useStore } from '@/store';

interface ExportFormat {
  id: string;
  name: string;
  extension: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}

const exportFormats: ExportFormat[] = [
  {
    id: 'csv',
    name: 'CSV',
    extension: '.csv',
    description: 'Comma-separated values, compatible with Excel and most tools',
    icon: FileText
  },
  {
    id: 'excel',
    name: 'Excel',
    extension: '.xlsx',
    description: 'Microsoft Excel format with multiple sheets',
    icon: Table
  },
  {
    id: 'netcdf',
    name: 'NetCDF',
    extension: '.nc',
    description: 'Network Common Data Form for scientific data',
    icon: Archive
  },
  {
    id: 'hdf5',
    name: 'HDF5',
    extension: '.h5',
    description: 'Hierarchical Data Format for large scientific datasets',
    icon: Archive
  },
  {
    id: 'latex',
    name: 'LaTeX Report',
    extension: '.tex',
    description: 'Publication-ready LaTeX document with results',
    icon: FileText
  }
];

interface ExportOptions {
  includeOriginal: boolean;
  includeImputed: boolean;
  includeMetadata: boolean;
  includeStatistics: boolean;
  includeVisualizations: boolean;
  compressionLevel: number;
}

const Export: React.FC = () => {
  const navigate = useNavigate();
  const { currentDataset, imputationResults } = useStore();
  const [selectedFormat, setSelectedFormat] = useState<string>('csv');
  const [exporting, setExporting] = useState(false);
  const [exportSuccess, setExportSuccess] = useState(false);
  const [options, setOptions] = useState<ExportOptions>({
    includeOriginal: true,
    includeImputed: true,
    includeMetadata: true,
    includeStatistics: true,
    includeVisualizations: false,
    compressionLevel: 5
  });

  const handleExport = async () => {
    if (!currentDataset || !imputationResults) return;

    const format = exportFormats.find(f => f.id === selectedFormat);
    if (!format) return;

    try {
      // Open save dialog
      const filePath = await save({
        defaultPath: `${currentDataset.info.name}_imputed${format.extension}`,
        filters: [{
          name: format.name,
          extensions: [format.extension.slice(1)]
        }]
      });

      if (!filePath) return;

      setExporting(true);
      setExportSuccess(false);

      // Call appropriate export function based on format
      switch (selectedFormat) {
        case 'csv':
          await invoke('export_to_csv', {
            datasetId: currentDataset.id,
            imputationId: imputationResults.id,
            outputPath: filePath,
            options
          });
          break;
        case 'excel':
          await invoke('export_to_excel', {
            datasetId: currentDataset.id,
            imputationId: imputationResults.id,
            outputPath: filePath,
            options
          });
          break;
        case 'netcdf':
          await invoke('export_to_netcdf', {
            datasetId: currentDataset.id,
            imputationId: imputationResults.id,
            outputPath: filePath,
            options
          });
          break;
        case 'hdf5':
          await invoke('export_to_hdf5', {
            datasetId: currentDataset.id,
            imputationId: imputationResults.id,
            outputPath: filePath,
            options
          });
          break;
        case 'latex':
          await invoke('generate_latex_report', {
            datasetId: currentDataset.id,
            imputationId: imputationResults.id,
            outputPath: filePath,
            options
          });
          break;
      }

      setExportSuccess(true);
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      setExporting(false);
    }
  };

  const handleGeneratePackage = async () => {
    try {
      const packagePath = await save({
        defaultPath: `${currentDataset?.info.name}_publication_package.zip`
      });

      if (!packagePath) return;

      await invoke('generate_publication_package', {
        datasetId: currentDataset?.id,
        imputationId: imputationResults?.id,
        outputPath: packagePath
      });

      setExportSuccess(true);
    } catch (err) {
      console.error('Failed to generate publication package:', err);
    }
  };

  if (!currentDataset || !imputationResults) {
    return (
      <div className="container mx-auto p-6">
        <Card className="p-8 text-center">
          <p className="text-gray-600 mb-4">No data to export</p>
          <Button onClick={() => navigate('/data-import')}>
            Import Data
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <h1 className="text-3xl font-bold mb-6">Export Results</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Format Selection */}
        <div className="lg:col-span-2">
          <ScientificCard
            title="Export Format"
            description="Select the format for your exported data"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              {exportFormats.map(format => (
                <button
                  key={format.id}
                  onClick={() => setSelectedFormat(format.id)}
                  className={`
                    p-4 rounded-lg border-2 text-left transition-colors
                    ${selectedFormat === format.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'}
                  `}
                >
                  <div className="flex items-start">
                    <format.icon className="w-6 h-6 mr-3 mt-0.5 text-gray-600" />
                    <div>
                      <p className="font-medium">{format.name}</p>
                      <p className="text-sm text-gray-500">{format.description}</p>
                    </div>
                  </div>
                </button>
              ))}
            </div>

            {/* Export Options */}
            <div className="border-t pt-6">
              <h3 className="font-medium mb-4">Export Options</h3>
              <div className="space-y-3">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={options.includeOriginal}
                    onChange={(e) => setOptions({ ...options, includeOriginal: e.target.checked })}
                    className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                  />
                  <span className="ml-2">Include original data</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={options.includeImputed}
                    onChange={(e) => setOptions({ ...options, includeImputed: e.target.checked })}
                    className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                  />
                  <span className="ml-2">Include imputed data</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={options.includeMetadata}
                    onChange={(e) => setOptions({ ...options, includeMetadata: e.target.checked })}
                    className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                  />
                  <span className="ml-2">Include metadata</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={options.includeStatistics}
                    onChange={(e) => setOptions({ ...options, includeStatistics: e.target.checked })}
                    className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                  />
                  <span className="ml-2">Include statistics report</span>
                </label>
                {selectedFormat === 'latex' && (
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={options.includeVisualizations}
                      onChange={(e) => setOptions({ ...options, includeVisualizations: e.target.checked })}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                    />
                    <span className="ml-2">Include visualizations</span>
                  </label>
                )}
              </div>
            </div>

            {/* Success Message */}
            {exportSuccess && (
              <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center">
                  <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                  <p className="text-green-700">Export completed successfully!</p>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="mt-6 flex justify-end space-x-3">
              <Button
                variant="outline"
                onClick={() => navigate('/analysis')}
              >
                Back to Analysis
              </Button>
              <Button
                onClick={handleExport}
                disabled={exporting}
              >
                <FileDown className="w-4 h-4 mr-2" />
                {exporting ? 'Exporting...' : 'Export'}
              </Button>
            </div>
          </ScientificCard>
        </div>

        {/* Quick Actions */}
        <div className="lg:col-span-1">
          <ScientificCard
            title="Quick Actions"
            description="Additional export options"
          >
            <div className="space-y-3">
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={handleGeneratePackage}
              >
                <Archive className="w-4 h-4 mr-2" />
                Generate Publication Package
              </Button>
              <p className="text-sm text-gray-500 mt-2">
                Creates a complete package with data, results, visualizations, and LaTeX report ready for publication.
              </p>
            </div>
          </ScientificCard>

          <ScientificCard
            title="Dataset Summary"
            description="Data being exported"
            className="mt-6"
          >
            <div className="space-y-2 text-sm">
              <div>
                <span className="text-gray-500">Dataset:</span>
                <span className="ml-2 font-medium">{currentDataset.info.name}</span>
              </div>
              <div>
                <span className="text-gray-500">Original size:</span>
                <span className="ml-2 font-medium">
                  {currentDataset.info.rows.toLocaleString()} × {currentDataset.info.columns}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Imputation method:</span>
                <span className="ml-2 font-medium">{imputationResults.method}</span>
              </div>
              <div>
                <span className="text-gray-500">Values imputed:</span>
                <span className="ml-2 font-medium">
                  {imputationResults.imputedCount?.toLocaleString() || 'N/A'}
                </span>
              </div>
            </div>
          </ScientificCard>
        </div>
      </div>
    </div>
  );
};

export default Export;