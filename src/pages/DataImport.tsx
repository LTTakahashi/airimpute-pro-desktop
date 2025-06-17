import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Progress } from '@/components/ui/Progress';
import { ScientificCard } from '@/components/layout/ScientificCard';
import { invoke } from '@tauri-apps/api/tauri';
import { open } from '@tauri-apps/api/dialog';
import { useNavigate } from 'react-router-dom';
import { useStore } from '@/store';

interface DatasetInfo {
  name: string;
  rows: number;
  columns: number;
  missing_count: number;
  missing_percentage: number;
  file_size: number;
  column_names: string[];
}

interface ImportProgress {
  current: number;
  total: number;
  message: string;
}

const DataImport: React.FC = () => {
  const navigate = useNavigate();
  const { setCurrentDataset } = useStore();
  const [importing, setImporting] = useState(false);
  const [progress, setProgress] = useState<ImportProgress | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    setError(null);
    setImporting(true);
    setProgress({ current: 0, total: 100, message: 'Reading file...' });

    try {
      // In Tauri, we need to use the file path, not the File object
      const filePath = await open({
        multiple: false,
        filters: [{
          name: 'Data Files',
          extensions: ['csv', 'xlsx', 'xls', 'json', 'parquet', 'hdf5', 'nc']
        }]
      });

      if (filePath && typeof filePath === 'string') {
        const result = await invoke<DatasetInfo>('load_dataset', {
          path: filePath,
          options: {
            delimiter: ',',
            parse_dates: true,
            infer_datetime_format: true
          }
        });

        setDatasetInfo(result);
        setCurrentDataset({
          id: Date.now().toString(),
          name: result.name,
          path: filePath,
          info: result
        });
        
        setProgress({ current: 100, total: 100, message: 'Import complete!' });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to import dataset');
    } finally {
      setImporting(false);
    }
  }, [setCurrentDataset]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/json': ['.json'],
      'application/x-parquet': ['.parquet'],
      'application/x-hdf': ['.hdf5', '.h5'],
      'application/x-netcdf': ['.nc']
    },
    maxFiles: 1,
    disabled: importing
  });

  const handleBrowse = async () => {
    const filePath = await open({
      multiple: false,
      filters: [{
        name: 'Data Files',
        extensions: ['csv', 'xlsx', 'xls', 'json', 'parquet', 'hdf5', 'nc']
      }]
    });

    if (filePath && typeof filePath === 'string') {
      onDrop([new File([], filePath)]);
    }
  };

  const handleProceed = () => {
    if (datasetInfo) {
      navigate('/imputation');
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <h1 className="text-3xl font-bold mb-6">Import Dataset</h1>

      <ScientificCard
        title="Upload Data File"
        description="Import your air quality dataset for imputation"
      >
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
            transition-colors duration-200
            ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
            ${importing ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getInputProps()} />
          <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
          <p className="text-lg mb-2">
            {isDragActive
              ? 'Drop the file here...'
              : 'Drag and drop a data file here, or click to browse'}
          </p>
          <p className="text-sm text-gray-500 mb-4">
            Supported formats: CSV, Excel, JSON, Parquet, HDF5, NetCDF
          </p>
          <Button onClick={handleBrowse} disabled={importing}>
            Browse Files
          </Button>
        </div>

        {importing && progress && (
          <div className="mt-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">{progress.message}</span>
              <span className="text-sm text-gray-500">
                {Math.round((progress.current / progress.total) * 100)}%
              </span>
            </div>
            <Progress value={(progress.current / progress.total) * 100} />
          </div>
        )}

        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        )}

        {datasetInfo && !importing && (
          <div className="mt-6">
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg mb-6">
              <div className="flex items-center mb-2">
                <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                <p className="font-medium text-green-700">Dataset imported successfully!</p>
              </div>
            </div>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <FileText className="w-5 h-5 mr-2" />
                Dataset Information
              </h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500">File Name</p>
                  <p className="font-medium">{datasetInfo.name}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">File Size</p>
                  <p className="font-medium">
                    {(datasetInfo.file_size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Rows</p>
                  <p className="font-medium">{datasetInfo.rows.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Columns</p>
                  <p className="font-medium">{datasetInfo.columns}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Missing Values</p>
                  <p className="font-medium">
                    {datasetInfo.missing_count.toLocaleString()} ({datasetInfo.missing_percentage.toFixed(1)}%)
                  </p>
                </div>
              </div>

              <div className="mt-4">
                <p className="text-sm text-gray-500 mb-2">Columns</p>
                <div className="flex flex-wrap gap-2">
                  {datasetInfo.column_names.slice(0, 10).map((col, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-1 bg-gray-100 rounded text-sm"
                    >
                      {col}
                    </span>
                  ))}
                  {datasetInfo.column_names.length > 10 && (
                    <span className="px-2 py-1 text-sm text-gray-500">
                      +{datasetInfo.column_names.length - 10} more
                    </span>
                  )}
                </div>
              </div>

              <div className="mt-6 flex justify-end space-x-3">
                <Button
                  variant="outline"
                  onClick={() => {
                    setDatasetInfo(null);
                    setError(null);
                  }}
                >
                  Import Another
                </Button>
                <Button onClick={handleProceed}>
                  Proceed to Imputation
                </Button>
              </div>
            </Card>
          </div>
        )}
      </ScientificCard>
    </div>
  );
};

export default DataImport;