import { invoke } from '@tauri-apps/api/tauri';
import { save } from '@tauri-apps/api/dialog';

export interface ExportOptions {
  includeOriginal: boolean;
  includeImputed: boolean;
  includeMetadata: boolean;
  includeStatistics: boolean;
  includeVisualizations: boolean;
  compressionLevel: number;
  format?: 'csv' | 'excel' | 'netcdf' | 'hdf5' | 'latex';
}

export interface ExportResult {
  success: boolean;
  path: string;
  size: number;
  error?: string;
}

class ExportService {
  async exportToCSV(
    datasetId: string,
    imputationId: string,
    outputPath?: string,
    options?: Partial<ExportOptions>
  ): Promise<ExportResult> {
    try {
      const path = outputPath || await this.selectSavePath('csv', 'CSV Files');
      if (!path) throw new Error('No path selected');

      return await invoke<ExportResult>('export_to_csv', {
        datasetId,
        imputationId,
        outputPath: path,
        options: options || {}
      });
    } catch (error) {
      console.error('Failed to export to CSV:', error);
      throw error;
    }
  }

  async exportToExcel(
    datasetId: string,
    imputationId: string,
    outputPath?: string,
    options?: Partial<ExportOptions>
  ): Promise<ExportResult> {
    try {
      const path = outputPath || await this.selectSavePath('xlsx', 'Excel Files');
      if (!path) throw new Error('No path selected');

      return await invoke<ExportResult>('export_to_excel', {
        datasetId,
        imputationId,
        outputPath: path,
        options: options || {}
      });
    } catch (error) {
      console.error('Failed to export to Excel:', error);
      throw error;
    }
  }

  async exportToNetCDF(
    datasetId: string,
    imputationId: string,
    outputPath?: string,
    options?: Partial<ExportOptions>
  ): Promise<ExportResult> {
    try {
      const path = outputPath || await this.selectSavePath('nc', 'NetCDF Files');
      if (!path) throw new Error('No path selected');

      return await invoke<ExportResult>('export_to_netcdf', {
        datasetId,
        imputationId,
        outputPath: path,
        options: options || {}
      });
    } catch (error) {
      console.error('Failed to export to NetCDF:', error);
      throw error;
    }
  }

  async exportToHDF5(
    datasetId: string,
    imputationId: string,
    outputPath?: string,
    options?: Partial<ExportOptions>
  ): Promise<ExportResult> {
    try {
      const path = outputPath || await this.selectSavePath('h5', 'HDF5 Files');
      if (!path) throw new Error('No path selected');

      return await invoke<ExportResult>('export_to_hdf5', {
        datasetId,
        imputationId,
        outputPath: path,
        options: options || {}
      });
    } catch (error) {
      console.error('Failed to export to HDF5:', error);
      throw error;
    }
  }

  async generateLatexReport(
    datasetId: string,
    imputationId: string,
    outputPath?: string,
    options?: Partial<ExportOptions>
  ): Promise<ExportResult> {
    try {
      const path = outputPath || await this.selectSavePath('tex', 'LaTeX Files');
      if (!path) throw new Error('No path selected');

      return await invoke<ExportResult>('generate_latex_report', {
        datasetId,
        imputationId,
        outputPath: path,
        options: options || {}
      });
    } catch (error) {
      console.error('Failed to generate LaTeX report:', error);
      throw error;
    }
  }

  async generatePublicationPackage(
    datasetId: string,
    imputationId: string,
    outputPath?: string
  ): Promise<ExportResult> {
    try {
      const path = outputPath || await this.selectSavePath('zip', 'ZIP Archives');
      if (!path) throw new Error('No path selected');

      return await invoke<ExportResult>('generate_publication_package', {
        datasetId,
        imputationId,
        outputPath: path
      });
    } catch (error) {
      console.error('Failed to generate publication package:', error);
      throw error;
    }
  }

  async exportVisualization(
    type: string,
    format: 'png' | 'svg' | 'pdf',
    outputPath?: string,
    options?: any
  ): Promise<ExportResult> {
    try {
      const path = outputPath || await this.selectSavePath(format, `${format.toUpperCase()} Files`);
      if (!path) throw new Error('No path selected');

      return await invoke<ExportResult>('export_visualization', {
        type,
        format,
        outputPath: path,
        options: options || {}
      });
    } catch (error) {
      console.error('Failed to export visualization:', error);
      throw error;
    }
  }

  private async selectSavePath(extension: string, filterName: string): Promise<string | null> {
    try {
      const path = await save({
        filters: [{
          name: filterName,
          extensions: [extension]
        }]
      });
      return path as string | null;
    } catch (error) {
      console.error('Failed to select save path:', error);
      return null;
    }
  }
}

export default new ExportService();