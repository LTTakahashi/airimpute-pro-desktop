# Phase 1: Publication & Documentation System - Implementation Plan

## Executive Summary

This document outlines the comprehensive architecture and implementation plan for adding academic publication and documentation capabilities to AirImpute Pro Desktop. The system will provide researchers with tools for generating publication-ready reports, managing citations, rendering mathematical content, and creating reproducible documentation packages.

## System Architecture Overview

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React + TypeScript)                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   LaTeX     │  │  Citation    │  │   Report Template  │    │
│  │  Renderer   │  │  Manager     │  │     Editor         │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  Equation   │  │  Bibliography│  │    PDF Preview     │    │
│  │   Editor    │  │    Editor    │  │                    │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                      Backend (Rust/Tauri)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   LaTeX     │  │     CSL      │  │   PDF Generation   │    │
│  │  Processor  │  │  Processor   │  │     Engine         │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  Template   │  │  Citation    │  │    Metadata        │    │
│  │   Engine    │  │  Database    │  │    Manager         │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                   Python Scientific Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  Figure     │  │  Statistical │  │   Report Content   │    │
│  │ Generation  │  │   Analysis   │  │    Generation      │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack Selection

### LaTeX Rendering: KaTeX vs MathJax

After careful analysis, **KaTeX** is recommended for the following reasons:

| Feature | KaTeX | MathJax | Winner |
|---------|-------|---------|---------|
| Rendering Speed | ~10ms | ~100ms | KaTeX ✓ |
| Bundle Size | 280KB | 500KB+ | KaTeX ✓ |
| Server-side Rendering | Yes | Limited | KaTeX ✓ |
| LaTeX Coverage | 85% | 95% | MathJax |
| Accessibility | Good | Excellent | MathJax |
| React Integration | Excellent | Good | KaTeX ✓ |

**Decision**: Use KaTeX as primary renderer with MathJax fallback for complex equations.

### Citation Management

**CSL (Citation Style Language)** ecosystem:
- `citeproc-js`: JavaScript implementation of CSL processor
- `citation-js`: Modern citation parsing and formatting
- Support for 10,000+ citation styles
- BibTeX/BibLaTeX import/export

### PDF Generation

**Hybrid approach**:
- Client-side: `jsPDF` with custom academic templates
- Server-side: `wkhtmltopdf` or `Puppeteer` for complex layouts
- LaTeX compilation: `texlive` integration for perfect typography

## Detailed Component Design

### 1. Frontend Components

#### 1.1 LaTeX Rendering System

```typescript
// src/components/publication/LaTeXRenderer.tsx
interface LaTeXRendererProps {
  content: string;
  displayMode?: boolean;
  throwOnError?: boolean;
  macros?: Record<string, string>;
  trust?: boolean;
}

// src/components/publication/EquationEditor.tsx
interface EquationEditorProps {
  initialValue?: string;
  onChange: (latex: string) => void;
  preview?: boolean;
  toolbar?: boolean;
  symbols?: SymbolCategory[];
}
```

#### 1.2 Citation Management

```typescript
// src/components/publication/CitationManager.tsx
interface Citation {
  id: string;
  type: CitationType;
  authors: Author[];
  title: string;
  year: number;
  doi?: string;
  url?: string;
  journal?: string;
  volume?: string;
  pages?: string;
}

// src/components/publication/BibliographyEditor.tsx
interface BibliographyEditorProps {
  citations: Citation[];
  style: string; // CSL style ID
  locale?: string;
  format?: 'html' | 'text' | 'rtf';
}
```

#### 1.3 Report Template System

```typescript
// src/components/publication/TemplateEditor.tsx
interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  category: 'journal' | 'conference' | 'thesis' | 'custom';
  structure: TemplateSection[];
  metadata: TemplateMetadata;
  styles: TemplateStyles;
}

interface TemplateSection {
  id: string;
  type: 'title' | 'abstract' | 'introduction' | 'methods' | 'results' | 'discussion' | 'conclusion' | 'references' | 'custom';
  title: string;
  required: boolean;
  content?: string;
  subsections?: TemplateSection[];
}
```

### 2. Backend Services

#### 2.1 LaTeX Processing Service

```rust
// src-tauri/src/services/latex.rs
pub struct LaTeXService {
    renderer: KaTeXRenderer,
    fallback: MathJaxRenderer,
    cache: LaTeXCache,
}

impl LaTeXService {
    pub async fn render_equation(&self, latex: &str, options: RenderOptions) -> Result<RenderedEquation> {
        // Try KaTeX first
        match self.renderer.render(latex, &options) {
            Ok(result) => Ok(result),
            Err(_) => {
                // Fallback to MathJax for complex equations
                self.fallback.render(latex, &options).await
            }
        }
    }
    
    pub async fn compile_document(&self, document: LaTeXDocument) -> Result<CompiledDocument> {
        // Full LaTeX compilation for complete documents
        let compiler = LaTeXCompiler::new();
        compiler.compile(document).await
    }
}
```

#### 2.2 Citation Processing Service

```rust
// src-tauri/src/services/citation.rs
pub struct CitationService {
    processor: CSLProcessor,
    styles: StyleRepository,
    database: CitationDatabase,
}

impl CitationService {
    pub async fn format_citation(&self, citation: &Citation, style: &str) -> Result<String> {
        let style_data = self.styles.get(style)?;
        self.processor.format(citation, style_data)
    }
    
    pub async fn generate_bibliography(&self, citations: Vec<Citation>, style: &str) -> Result<Bibliography> {
        let style_data = self.styles.get(style)?;
        self.processor.generate_bibliography(citations, style_data)
    }
    
    pub async fn import_bibtex(&self, content: &str) -> Result<Vec<Citation>> {
        BibtexParser::parse(content)
    }
}
```

#### 2.3 PDF Generation Service

```rust
// src-tauri/src/services/pdf.rs
pub struct PDFService {
    generator: PDFGenerator,
    templates: TemplateEngine,
    latex_service: LaTeXService,
}

impl PDFService {
    pub async fn generate_report(&self, report: Report, template: ReportTemplate) -> Result<PDFDocument> {
        // Render content with template
        let rendered = self.templates.render(report, template)?;
        
        // Process LaTeX content
        let processed = self.latex_service.process_document(rendered).await?;
        
        // Generate PDF
        self.generator.create_pdf(processed).await
    }
}
```

### 3. Python Integration

#### 3.1 Report Content Generation

```python
# scripts/airimpute/reporting.py
class ReportGenerator:
    """Generate publication-ready content from analysis results"""
    
    def __init__(self, style_guide: str = "ieee"):
        self.style_guide = style_guide
        self.figure_generator = FigureGenerator()
        self.table_formatter = TableFormatter()
        self.stats_reporter = StatisticalReporter()
    
    def generate_methods_section(self, imputation_result: ImputationResult) -> MethodsSection:
        """Generate methods section with proper citations"""
        return MethodsSection(
            overview=self._describe_methodology(imputation_result),
            equations=self._extract_equations(imputation_result),
            parameters=self._format_parameters(imputation_result),
            validation=self._describe_validation(imputation_result)
        )
    
    def generate_results_section(self, results: Dict[str, Any]) -> ResultsSection:
        """Generate results section with figures and tables"""
        return ResultsSection(
            summary=self._summarize_results(results),
            figures=self._generate_figures(results),
            tables=self._generate_tables(results),
            statistics=self._report_statistics(results)
        )
```

#### 3.2 Figure Generation with Publication Standards

```python
# scripts/airimpute/publication_figures.py
class PublicationFigureGenerator:
    """Generate publication-quality figures"""
    
    def __init__(self):
        self.setup_publication_style()
    
    def setup_publication_style(self):
        """Configure matplotlib for publication-quality output"""
        plt.style.use(['science', 'ieee'])
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'pdf.fonttype': 42,  # TrueType fonts
            'ps.fonttype': 42,
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}'
        })
    
    def create_figure(self, data: pd.DataFrame, fig_type: str, **kwargs) -> Figure:
        """Create publication-ready figure"""
        fig, ax = plt.subplots(figsize=self._get_figure_size(fig_type))
        
        # Apply specific plotting based on type
        if fig_type == 'time_series':
            self._plot_time_series(ax, data, **kwargs)
        elif fig_type == 'correlation_matrix':
            self._plot_correlation_matrix(ax, data, **kwargs)
        # ... more figure types
        
        self._apply_publication_formatting(fig, ax)
        return PublicationFigure(fig, self._generate_caption(data, fig_type))
```

## Implementation Roadmap

### Phase 1A: Core Infrastructure (Week 1-2)

1. **Frontend Setup**
   - Install and configure KaTeX
   - Set up citation-js and citeproc-js
   - Create base component structure
   - Implement LaTeX renderer component

2. **Backend Setup**
   - Add LaTeX processing crate dependencies
   - Implement CSL processor integration
   - Set up PDF generation pipeline
   - Create citation database schema

3. **Python Integration**
   - Implement report generation module
   - Create publication figure generator
   - Add LaTeX equation extraction

### Phase 1B: Component Development (Week 3-4)

1. **LaTeX Components**
   - Equation editor with live preview
   - Symbol palette and shortcuts
   - Macro management system
   - Error handling and validation

2. **Citation Components**
   - Citation manager interface
   - Bibliography editor
   - Import/export dialogs
   - Style selector

3. **Template Components**
   - Template editor
   - Section management
   - Metadata editor
   - Preview system

### Phase 1C: Integration & Testing (Week 5-6)

1. **System Integration**
   - Connect frontend to backend services
   - Integrate with existing data flow
   - Add to navigation and routing
   - Update state management

2. **Testing & Validation**
   - Unit tests for all components
   - Integration tests for workflows
   - Performance testing
   - Accessibility testing

## File Structure

```
src/
├── components/
│   └── publication/
│       ├── LaTeXRenderer.tsx
│       ├── EquationEditor.tsx
│       ├── CitationManager.tsx
│       ├── BibliographyEditor.tsx
│       ├── TemplateEditor.tsx
│       ├── ReportBuilder.tsx
│       ├── PDFPreview.tsx
│       └── __tests__/
│           ├── LaTeXRenderer.test.tsx
│           └── CitationManager.test.tsx
├── pages/
│   └── Publication.tsx
├── services/
│   ├── publication.service.ts
│   ├── citation.service.ts
│   └── latex.service.ts
├── hooks/
│   ├── useLatex.ts
│   ├── useCitations.ts
│   └── useTemplates.ts
└── types/
    └── publication.ts

src-tauri/
├── src/
│   ├── commands/
│   │   ├── publication.rs
│   │   └── citation.rs
│   ├── services/
│   │   ├── latex.rs
│   │   ├── citation.rs
│   │   ├── pdf.rs
│   │   └── template.rs
│   └── db/
│       └── migrations/
│           └── 002_publication_schema.sql

scripts/
└── airimpute/
    ├── reporting.py
    ├── publication_figures.py
    ├── latex_utils.py
    └── citation_utils.py
```

## Dependencies

### Frontend Dependencies
```json
{
  "dependencies": {
    "katex": "^0.16.9",
    "react-katex": "^3.0.1",
    "citation-js": "^0.6.8",
    "citeproc": "^2.4.63",
    "react-pdf": "^7.5.1",
    "@uiw/react-md-editor": "^3.25.6",
    "react-hook-form": "^7.48.2",
    "react-hotkeys-hook": "^4.4.3"
  }
}
```

### Backend Dependencies
```toml
[dependencies]
katex = "0.16"
tectonic = "0.15"  # LaTeX engine
citeproc = "0.2"
bibtex = "0.1"
lopdf = "0.31"
handlebars = "5.0"
```

### Python Dependencies
```txt
matplotlib>=3.8.0
seaborn>=0.13.0
SciencePlots>=2.1.0
pylatex>=1.4.2
bibtexparser>=1.4.1
scholarly>=1.7.11
```

## Best Practices & Guidelines

### 1. Accessibility
- All LaTeX content must have text alternatives
- Keyboard navigation for all editors
- Screen reader compatibility
- High contrast mode support

### 2. Performance
- Lazy load LaTeX renderer
- Cache rendered equations
- Virtualize long citation lists
- Progressive PDF generation

### 3. User Experience
- Auto-save for all editors
- Undo/redo support
- Keyboard shortcuts
- Contextual help

### 4. Academic Standards
- Support major citation styles (APA, MLA, Chicago, IEEE)
- Configurable formatting rules
- DOI resolution
- ORCID integration

## Security Considerations

1. **LaTeX Security**
   - Sanitize user input
   - Disable dangerous commands
   - Sandbox compilation environment

2. **Citation Data**
   - Validate imported data
   - Secure API keys for citation databases
   - Privacy controls for bibliography

3. **PDF Generation**
   - Validate templates
   - Restrict file system access
   - Monitor resource usage

## Success Metrics

1. **Performance Targets**
   - LaTeX rendering: <50ms for 95% of equations
   - PDF generation: <5s for 20-page report
   - Citation formatting: <100ms per citation

2. **Quality Metrics**
   - Zero rendering errors for standard LaTeX
   - 100% citation style compliance
   - Pixel-perfect PDF output

3. **User Satisfaction**
   - 90% task completion rate
   - <3 clicks to common actions
   - 95% accessibility compliance

## Conclusion

This implementation plan provides a comprehensive roadmap for adding publication and documentation capabilities to AirImpute Pro Desktop. The system will enable researchers to create publication-ready reports directly from their analysis results, manage citations professionally, and generate high-quality documentation that meets academic standards.

The modular architecture ensures maintainability and extensibility, while the careful technology selection balances performance, features, and user experience. With this system in place, AirImpute Pro will truly serve as an end-to-end solution for academic research in air quality data analysis.