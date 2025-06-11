import { invoke } from '@tauri-apps/api/tauri';
import type { 
  Report, 
  Citation, 
  ExportFormat,
  ReportTemplate 
} from '@/components/academic';

export interface LaTeXRenderRequest {
  expression: string;
  displayMode: boolean;
  macros?: Record<string, string>;
}

export interface LaTeXRenderResponse {
  html: string;
  error?: string;
}

export interface ExportReportRequest {
  report: Report;
  format: ExportFormat;
  options: ExportOptions;
}

export interface ExportOptions {
  includeCode: boolean;
  includeData: boolean;
  highQualityFigures: boolean;
  embedFonts: boolean;
}

export interface CitationFormatRequest {
  citation: Citation;
  style: string;
}

class PublicationService {
  // Report Management
  async saveReport(report: Report): Promise<Report> {
    return invoke('save_report', { report });
  }

  async loadReport(reportId: string): Promise<Report> {
    return invoke('load_report', { reportId });
  }

  async listReports(): Promise<Report[]> {
    return invoke('list_reports');
  }

  async getReportTemplates(): Promise<ReportTemplate[]> {
    return invoke('get_report_templates');
  }

  // LaTeX Rendering
  async renderLatex(expression: string, displayMode: boolean = false): Promise<LaTeXRenderResponse> {
    return invoke('render_latex', { expression, displayMode });
  }

  // Export Functions
  async exportReport(request: ExportReportRequest): Promise<string> {
    return invoke('export_report', { request });
  }

  async exportToPDF(report: Report, options?: Partial<ExportOptions>): Promise<string> {
    const defaultOptions: ExportOptions = {
      includeCode: true,
      includeData: false,
      highQualityFigures: true,
      embedFonts: true,
      ...options
    };

    return this.exportReport({
      report,
      format: 'pdf' as ExportFormat,
      options: defaultOptions
    });
  }

  async exportToLaTeX(report: Report, options?: Partial<ExportOptions>): Promise<string> {
    const defaultOptions: ExportOptions = {
      includeCode: true,
      includeData: true,
      highQualityFigures: true,
      embedFonts: false,
      ...options
    };

    return this.exportReport({
      report,
      format: 'latex' as ExportFormat,
      options: defaultOptions
    });
  }

  async exportToWord(report: Report, options?: Partial<ExportOptions>): Promise<string> {
    const defaultOptions: ExportOptions = {
      includeCode: false,
      includeData: false,
      highQualityFigures: false,
      embedFonts: false,
      ...options
    };

    return this.exportReport({
      report,
      format: 'word' as ExportFormat,
      options: defaultOptions
    });
  }

  // Citation Management
  async importBibtex(content: string): Promise<Citation[]> {
    return invoke('import_bibtex', { content });
  }

  async formatCitation(citation: Citation, style: string): Promise<string> {
    return invoke('format_citation', { 
      request: { citation, style } 
    });
  }

  async generateBibliography(citations: Citation[], style: string): Promise<string> {
    return invoke('generate_bibliography', { citations, style });
  }

  // Utility Functions
  async validateLatex(expression: string): Promise<{ valid: boolean; error?: string }> {
    try {
      const result = await this.renderLatex(expression);
      return { valid: !result.error, error: result.error };
    } catch (error) {
      return { valid: false, error: String(error) };
    }
  }

  async generateCitation(methodId: string): Promise<Citation> {
    // This would be implemented based on method documentation
    // For now, return a placeholder
    return {
      id: `cite_${Date.now()}`,
      type: 'article',
      authors: [],
      title: `Method: ${methodId}`,
      year: new Date().getFullYear()
    } as Citation;
  }

  // Template Management
  async loadTemplate(templateId: string): Promise<ReportTemplate> {
    const templates = await this.getReportTemplates();
    const template = templates.find(t => t.id === templateId);
    if (!template) {
      throw new Error(`Template ${templateId} not found`);
    }
    return template;
  }

  // Auto-save functionality
  private autoSaveTimer: NodeJS.Timeout | null = null;

  enableAutoSave(report: Report, intervalMs: number = 30000): void {
    this.disableAutoSave();
    this.autoSaveTimer = setInterval(async () => {
      try {
        await this.saveReport(report);
        console.log('Report auto-saved');
      } catch (error) {
        console.error('Auto-save failed:', error);
      }
    }, intervalMs);
  }

  disableAutoSave(): void {
    if (this.autoSaveTimer) {
      clearInterval(this.autoSaveTimer);
      this.autoSaveTimer = null;
    }
  }

  // Export presets for common journals
  getExportPreset(journal: string): Partial<ExportOptions> {
    const presets: Record<string, Partial<ExportOptions>> = {
      ieee: {
        includeCode: true,
        includeData: false,
        highQualityFigures: true,
        embedFonts: true
      },
      nature: {
        includeCode: false,
        includeData: true,
        highQualityFigures: true,
        embedFonts: true
      },
      elsevier: {
        includeCode: true,
        includeData: true,
        highQualityFigures: true,
        embedFonts: false
      }
    };

    return presets[journal.toLowerCase()] || {};
  }
}

export const publicationService = new PublicationService();