"""
Publication-Quality Export System
Generates camera-ready figures, tables, and documents for academic publications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams, font_manager
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import yaml
from datetime import datetime
import warnings
from dataclasses import dataclass
import logging
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)


@dataclass
class PublicationConfig:
    """Configuration for publication-quality exports"""
    # Figure settings
    figure_format: List[str] = None  # ["pdf", "eps", "png", "svg"]
    dpi: int = 300
    figure_width: float = 6.5  # inches (single column)
    figure_height_ratio: float = 0.618  # Golden ratio
    
    # Font settings
    font_family: str = "serif"
    font_size: int = 10
    use_latex: bool = True
    latex_preamble: str = r"\usepackage{amsmath}\usepackage{amssymb}"
    
    # Color settings
    color_palette: str = "colorblind"  # Accessible colors
    use_grayscale: bool = False
    
    # Table settings
    table_format: str = "latex"  # "latex", "html", "markdown"
    decimal_places: int = 3
    use_siunitx: bool = True  # LaTeX siunitx package for numbers
    
    # Journal styles
    journal_style: str = "ieee"  # "ieee", "nature", "science", "elsevier"
    
    # Output settings
    output_dir: Path = Path("publication_figures")
    use_compression: bool = True
    embed_fonts: bool = True
    
    def __post_init__(self):
        if self.figure_format is None:
            self.figure_format = ["pdf", "eps", "png"]


class JournalStyles:
    """Predefined styles for major journals"""
    
    STYLES = {
        "ieee": {
            "figure_width": 3.5,  # Single column
            "figure_width_double": 7.16,  # Double column
            "font_size": 8,
            "font_family": "serif",
            "caption_size": 8,
            "tick_labelsize": 7,
            "legend_fontsize": 7,
            "use_latex": True,
        },
        "nature": {
            "figure_width": 3.42,  # 89mm single column
            "figure_width_double": 7.08,  # 183mm double column
            "font_size": 8,
            "font_family": "sans-serif",
            "caption_size": 8,
            "tick_labelsize": 7,
            "legend_fontsize": 7,
            "use_latex": False,
        },
        "science": {
            "figure_width": 3.27,  # Single column
            "figure_width_double": 6.69,  # Double column
            "font_size": 9,
            "font_family": "sans-serif",
            "caption_size": 8,
            "tick_labelsize": 8,
            "legend_fontsize": 8,
            "use_latex": False,
        },
        "elsevier": {
            "figure_width": 3.54,  # 90mm single column
            "figure_width_double": 7.48,  # 190mm double column
            "font_size": 10,
            "font_family": "serif",
            "caption_size": 9,
            "tick_labelsize": 9,
            "legend_fontsize": 9,
            "use_latex": True,
        },
        "plos": {
            "figure_width": 3.25,
            "figure_width_double": 6.83,
            "font_size": 10,
            "font_family": "sans-serif",
            "caption_size": 10,
            "tick_labelsize": 9,
            "legend_fontsize": 9,
            "use_latex": False,
        }
    }
    
    @classmethod
    def get_style(cls, journal: str) -> Dict[str, Any]:
        """Get style configuration for a specific journal"""
        return cls.STYLES.get(journal, cls.STYLES["ieee"])


class PublicationFigureGenerator:
    """Generate publication-quality figures"""
    
    def __init__(self, config: PublicationConfig):
        self.config = config
        self._setup_matplotlib()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_matplotlib(self):
        """Configure matplotlib for publication quality"""
        # Get journal style
        journal_style = JournalStyles.get_style(self.config.journal_style)
        
        # Update configuration
        rcParams['figure.dpi'] = self.config.dpi
        rcParams['savefig.dpi'] = self.config.dpi
        rcParams['font.size'] = journal_style['font_size']
        rcParams['font.family'] = journal_style['font_family']
        
        # LaTeX settings
        if self.config.use_latex and journal_style.get('use_latex', True):
            rcParams['text.usetex'] = True
            rcParams['text.latex.preamble'] = self.config.latex_preamble
            rcParams['font.serif'] = ['Computer Modern Roman']
            rcParams['font.sans-serif'] = ['Computer Modern Sans serif']
        else:
            rcParams['text.usetex'] = False
        
        # Figure settings
        rcParams['figure.figsize'] = (
            journal_style['figure_width'],
            journal_style['figure_width'] * self.config.figure_height_ratio
        )
        
        # Axes settings
        rcParams['axes.labelsize'] = journal_style['font_size']
        rcParams['axes.titlesize'] = journal_style['font_size']
        rcParams['axes.linewidth'] = 0.5
        
        # Tick settings
        rcParams['xtick.labelsize'] = journal_style['tick_labelsize']
        rcParams['ytick.labelsize'] = journal_style['tick_labelsize']
        rcParams['xtick.major.width'] = 0.5
        rcParams['ytick.major.width'] = 0.5
        rcParams['xtick.major.size'] = 3
        rcParams['ytick.major.size'] = 3
        
        # Legend settings
        rcParams['legend.fontsize'] = journal_style['legend_fontsize']
        rcParams['legend.frameon'] = False
        
        # Line settings
        rcParams['lines.linewidth'] = 1.0
        rcParams['lines.markersize'] = 4
        
        # Grid settings
        rcParams['grid.linewidth'] = 0.5
        rcParams['grid.alpha'] = 0.3
        
        # Save settings
        rcParams['savefig.bbox'] = 'tight'
        rcParams['savefig.pad_inches'] = 0.05
        
        if self.config.embed_fonts:
            rcParams['pdf.fonttype'] = 42  # TrueType fonts
            rcParams['ps.fonttype'] = 42
    
    def _get_colors(self, n: int) -> List[str]:
        """Get publication-appropriate colors"""
        if self.config.use_grayscale:
            # Grayscale for print
            return [str(i/n) for i in range(n)]
        else:
            # Colorblind-safe palette
            if n <= 8:
                return sns.color_palette("colorblind", n)
            else:
                return sns.color_palette("husl", n)
    
    def create_time_series_plot(self,
                               data: pd.DataFrame,
                               title: str = "",
                               ylabel: str = "Value",
                               xlabel: str = "Time",
                               highlight_gaps: bool = True,
                               double_column: bool = False) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create publication-quality time series plot
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with datetime index
        title : str
            Figure title (often omitted in publications)
        ylabel : str
            Y-axis label
        xlabel : str
            X-axis label
        highlight_gaps : bool
            Highlight missing data regions
        double_column : bool
            Use double-column width
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes
        """
        # Get appropriate figure size
        journal_style = JournalStyles.get_style(self.config.journal_style)
        if double_column:
            fig_width = journal_style['figure_width_double']
        else:
            fig_width = journal_style['figure_width']
        
        fig_height = fig_width * self.config.figure_height_ratio
        
        # Create figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Get colors
        n_series = len(data.columns)
        colors = self._get_colors(n_series)
        
        # Plot each series
        for i, col in enumerate(data.columns):
            ax.plot(data.index, data[col], color=colors[i], label=col, linewidth=1.0)
        
        # Highlight gaps if requested
        if highlight_gaps:
            for col in data.columns:
                gaps = data[col].isna()
                if gaps.any():
                    # Find gap regions
                    gap_starts = gaps & ~gaps.shift(1).fillna(False)
                    gap_ends = gaps & ~gaps.shift(-1).fillna(False)
                    
                    for start, end in zip(data.index[gap_starts], data.index[gap_ends]):
                        ax.axvspan(start, end, alpha=0.2, color='gray', zorder=0)
        
        # Styling
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title and not self.config.journal_style in ["nature", "science"]:
            ax.set_title(title)
        
        # Grid
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Legend
        if n_series > 1:
            ax.legend(loc='best', frameon=False)
        
        # Tight layout
        fig.tight_layout()
        
        return fig, ax
    
    def create_comparison_plot(self,
                             results: Dict[str, List[float]],
                             metric_name: str = "RMSE",
                             show_significance: bool = True,
                             significance_data: Optional[Dict] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create method comparison plot with statistical significance
        
        Parameters:
        -----------
        results : dict
            Method name -> list of scores
        metric_name : str
            Name of the metric
        show_significance : bool
            Show significance brackets
        significance_data : dict
            Pairwise comparison results
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes
        """
        # Prepare data
        methods = list(results.keys())
        data = [results[m] for m in methods]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.config.figure_width, 
                                       self.config.figure_width * 0.75))
        
        # Create violin plot
        parts = ax.violinplot(data, positions=range(len(methods)), 
                             widths=0.7, showmeans=True, showextrema=True)
        
        # Customize violin plot
        colors = self._get_colors(len(methods))
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        
        # Customize other elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            if partname in parts:
                vp = parts[partname]
                vp.set_edgecolor('black')
                vp.set_linewidth(0.5)
        
        # Add significance brackets if requested
        if show_significance and significance_data:
            self._add_significance_brackets(ax, methods, significance_data)
        
        # Styling
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45 if len(methods) > 5 else 0, ha='right')
        ax.set_ylabel(metric_name)
        ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        
        return fig, ax
    
    def _add_significance_brackets(self, ax, methods, significance_data):
        """Add significance brackets to comparison plot"""
        y_max = ax.get_ylim()[1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        
        bracket_height = y_range * 0.05
        bracket_gap = y_range * 0.02
        
        current_y = y_max + bracket_gap
        
        # Sort by position to minimize overlaps
        comparisons = []
        for comp in significance_data.get('comparisons', []):
            if comp.get('significant', False):
                idx1 = methods.index(comp['method1'])
                idx2 = methods.index(comp['method2'])
                comparisons.append((min(idx1, idx2), max(idx1, idx2)))
        
        comparisons.sort(key=lambda x: (x[1] - x[0], x[0]))
        
        # Draw brackets
        for i, (idx1, idx2) in enumerate(comparisons):
            y = current_y + i * (bracket_height + bracket_gap)
            
            # Horizontal line
            ax.plot([idx1, idx2], [y, y], 'k-', linewidth=0.5)
            
            # Vertical ticks
            ax.plot([idx1, idx1], [y - bracket_gap/2, y], 'k-', linewidth=0.5)
            ax.plot([idx2, idx2], [y - bracket_gap/2, y], 'k-', linewidth=0.5)
            
            # Significance marker
            ax.text((idx1 + idx2) / 2, y + bracket_gap/2, '*', 
                   ha='center', va='bottom', fontsize=8)
        
        # Adjust y-limit
        ax.set_ylim(ax.get_ylim()[0], current_y + len(comparisons) * (bracket_height + bracket_gap))
    
    def create_heatmap(self,
                      data: pd.DataFrame,
                      title: str = "",
                      cmap: str = "RdBu_r",
                      center: Optional[float] = None,
                      annot: bool = True,
                      fmt: str = ".2f") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create publication-quality heatmap
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data for heatmap
        title : str
            Figure title
        cmap : str
            Colormap name
        center : float
            Center value for diverging colormap
        annot : bool
            Annotate cells with values
        fmt : str
            Format string for annotations
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes
        """
        # Adjust figure size based on data shape
        n_rows, n_cols = data.shape
        aspect_ratio = n_cols / n_rows
        
        fig_width = self.config.figure_width
        fig_height = fig_width / aspect_ratio * 0.8
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create heatmap
        if self.config.use_grayscale:
            cmap = "gray_r"
        
        sns.heatmap(data, ax=ax, cmap=cmap, center=center,
                   annot=annot, fmt=fmt, 
                   annot_kws={'fontsize': 6},
                   cbar_kws={'label': ''},
                   linewidths=0.5, linecolor='white')
        
        # Styling
        if title:
            ax.set_title(title)
        
        # Rotate labels if needed
        if n_cols > 10:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        if n_rows > 10:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        fig.tight_layout()
        
        return fig, ax
    
    def create_multi_panel_figure(self,
                                 panels: List[Callable],
                                 layout: Tuple[int, int],
                                 panel_labels: bool = True,
                                 shared_axes: str = "none") -> plt.Figure:
        """
        Create multi-panel figure for publication
        
        Parameters:
        -----------
        panels : list
            List of functions that create each panel
        layout : tuple
            (n_rows, n_cols)
        panel_labels : bool
            Add panel labels (a, b, c, ...)
        shared_axes : str
            "none", "x", "y", or "both"
        
        Returns:
        --------
        fig : matplotlib figure
        """
        n_rows, n_cols = layout
        
        # Calculate figure size
        journal_style = JournalStyles.get_style(self.config.journal_style)
        fig_width = journal_style['figure_width_double']
        fig_height = fig_width * self.config.figure_height_ratio * n_rows / n_cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height),
                               sharex=(shared_axes in ["x", "both"]),
                               sharey=(shared_axes in ["y", "both"]))
        
        # Flatten axes array for easier iteration
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Create each panel
        for i, (ax, panel_func) in enumerate(zip(axes, panels)):
            panel_func(ax)
            
            # Add panel label
            if panel_labels:
                label = chr(ord('a') + i)
                ax.text(-0.15, 1.05, f'({label})', transform=ax.transAxes,
                       fontsize=10, fontweight='bold', va='top')
        
        # Remove unused axes
        for i in range(len(panels), len(axes)):
            fig.delaxes(axes[i])
        
        # Adjust spacing
        fig.tight_layout()
        
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, 
                   formats: Optional[List[str]] = None):
        """
        Save figure in multiple formats
        
        Parameters:
        -----------
        fig : matplotlib figure
            Figure to save
        filename : str
            Base filename (without extension)
        formats : list
            Output formats (uses config default if None)
        """
        if formats is None:
            formats = self.config.figure_format
        
        for fmt in formats:
            filepath = self.config.output_dir / f"{filename}.{fmt}"
            
            # Format-specific options
            save_kwargs = {
                'dpi': self.config.dpi,
                'bbox_inches': 'tight',
                'pad_inches': 0.05
            }
            
            if fmt == 'pdf':
                save_kwargs['transparent'] = True
                save_kwargs['metadata'] = {
                    'Creator': 'AirImpute Pro',
                    'Producer': 'matplotlib',
                    'CreationDate': datetime.now()
                }
            elif fmt == 'eps':
                save_kwargs['transparent'] = True
            elif fmt == 'png':
                save_kwargs['transparent'] = False
                save_kwargs['facecolor'] = 'white'
            elif fmt == 'svg':
                save_kwargs['transparent'] = True
            
            fig.savefig(filepath, format=fmt, **save_kwargs)
            logger.info(f"Saved figure to {filepath}")
            
            # Compress if requested
            if self.config.use_compression and fmt == 'pdf':
                self._compress_pdf(filepath)
    
    def _compress_pdf(self, filepath: Path):
        """Compress PDF using ghostscript if available"""
        try:
            import subprocess
            
            compressed_path = filepath.with_suffix('.compressed.pdf')
            
            cmd = [
                'gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
                '-dPDFSETTINGS=/prepress', '-dNOPAUSE', '-dQUIET', '-dBATCH',
                f'-sOutputFile={compressed_path}', str(filepath)
            ]
            
            subprocess.run(cmd, check=True)
            
            # Replace original with compressed
            compressed_path.replace(filepath)
            logger.info(f"Compressed {filepath}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("PDF compression failed (ghostscript not available)")


class PublicationTableGenerator:
    """Generate publication-quality tables"""
    
    def __init__(self, config: PublicationConfig):
        self.config = config
    
    def create_results_table(self,
                           results: pd.DataFrame,
                           caption: str = "Performance comparison of imputation methods",
                           label: str = "tab:results",
                           highlight_best: bool = True) -> str:
        """
        Create publication-ready results table
        
        Parameters:
        -----------
        results : pd.DataFrame
            Results data with methods as rows and metrics as columns
        caption : str
            Table caption
        label : str
            LaTeX label
        highlight_best : bool
            Bold the best result in each column
        
        Returns:
        --------
        str : LaTeX table code
        """
        # Copy data to avoid modifying original
        data = results.copy()
        
        # Format numbers
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col] = data[col].round(self.config.decimal_places)
        
        # Highlight best results
        if highlight_best:
            for col in data.select_dtypes(include=[np.number]).columns:
                # Determine if lower or higher is better
                if any(keyword in col.lower() for keyword in ['error', 'rmse', 'mae']):
                    best_idx = data[col].idxmin()
                else:
                    best_idx = data[col].idxmax()
                
                # Bold the best value
                if pd.notna(best_idx):
                    data.loc[best_idx, col] = f"\\textbf{{{data.loc[best_idx, col]}}}"
        
        # Generate LaTeX table
        if self.config.table_format == "latex":
            # Basic LaTeX table
            latex_code = data.to_latex(
                escape=False,
                index=True,
                column_format='l' + 'c' * len(data.columns),
                bold_rows=True
            )
            
            # Enhance with siunitx if requested
            if self.config.use_siunitx:
                latex_code = self._enhance_with_siunitx(latex_code, data)
            
            # Add caption and label
            latex_code = latex_code.replace(
                '\\begin{tabular}',
                f'\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}'
            )
            
            # Wrap in table environment
            latex_code = f"\\begin{{table}}[htbp]\n\\centering\n{latex_code}\\end{{table}}"
            
            return latex_code
        
        elif self.config.table_format == "html":
            return data.to_html(escape=False, index=True)
        
        elif self.config.table_format == "markdown":
            return data.to_markdown(index=True)
        
        else:
            raise ValueError(f"Unknown table format: {self.config.table_format}")
    
    def _enhance_with_siunitx(self, latex_code: str, data: pd.DataFrame) -> str:
        """Enhance LaTeX table with siunitx formatting"""
        # Replace column format for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        n_numeric = len(numeric_cols)
        
        if n_numeric > 0:
            # Replace column format
            old_format = 'c' * len(data.columns)
            new_format = 'l' + 'S[table-format=2.3]' * n_numeric
            
            latex_code = latex_code.replace(
                f'{{{old_format}}}',
                f'{{{new_format}}}'
            )
            
            # Add siunitx setup
            siunitx_setup = r"""
\sisetup{
    table-format = 2.3,
    table-number-alignment = center,
    table-auto-round = true,
    separate-uncertainty = true
}
"""
            latex_code = siunitx_setup + latex_code
        
        return latex_code
    
    def create_statistical_summary_table(self,
                                       statistical_results: Dict[str, Any],
                                       caption: str = "Statistical test results",
                                       label: str = "tab:statistics") -> str:
        """
        Create table summarizing statistical tests
        
        Parameters:
        -----------
        statistical_results : dict
            Results from statistical testing
        caption : str
            Table caption
        label : str
            LaTeX label
        
        Returns:
        --------
        str : LaTeX table code
        """
        # Extract test results
        rows = []
        
        for metric, results in statistical_results.items():
            if 'omnibus' in results:
                omnibus = results['omnibus']
                rows.append({
                    'Metric': metric,
                    'Test': omnibus['test'],
                    'Statistic': f"{omnibus['statistic']:.3f}",
                    'p-value': f"{omnibus['p_value']:.3e}",
                    'Significant': 'Yes' if omnibus['significant'] else 'No'
                })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Generate table
        return self.create_results_table(
            df, caption=caption, label=label, highlight_best=False
        )
    
    def create_dataset_characteristics_table(self,
                                           datasets: Dict[str, pd.DataFrame],
                                           caption: str = "Dataset characteristics",
                                           label: str = "tab:datasets") -> str:
        """
        Create table of dataset characteristics
        
        Parameters:
        -----------
        datasets : dict
            Dataset name -> DataFrame
        caption : str
            Table caption
        label : str
            LaTeX label
        
        Returns:
        --------
        str : LaTeX table code
        """
        rows = []
        
        for name, data in datasets.items():
            n_rows, n_cols = data.shape
            n_missing = data.isnull().sum().sum()
            missing_pct = n_missing / (n_rows * n_cols) * 100
            
            # Time range if datetime index
            if isinstance(data.index, pd.DatetimeIndex):
                time_range = f"{data.index[0]:%Y-%m-%d} to {data.index[-1]:%Y-%m-%d}"
                frequency = pd.infer_freq(data.index) or "Irregular"
            else:
                time_range = "N/A"
                frequency = "N/A"
            
            rows.append({
                'Dataset': name,
                'Samples': n_rows,
                'Features': n_cols,
                'Missing (%)': f"{missing_pct:.1f}",
                'Time Range': time_range,
                'Frequency': frequency
            })
        
        df = pd.DataFrame(rows)
        
        return self.create_results_table(
            df, caption=caption, label=label, highlight_best=False
        )


class PublicationDocumentGenerator:
    """Generate complete publication documents"""
    
    def __init__(self, config: PublicationConfig):
        self.config = config
        self.figure_gen = PublicationFigureGenerator(config)
        self.table_gen = PublicationTableGenerator(config)
    
    def generate_latex_article(self,
                             title: str,
                             authors: List[Dict[str, str]],
                             abstract: str,
                             sections: Dict[str, str],
                             figures: List[Dict[str, Any]],
                             tables: List[Dict[str, Any]],
                             bibliography: Optional[str] = None,
                             document_class: str = "IEEEtran") -> str:
        """
        Generate complete LaTeX article
        
        Parameters:
        -----------
        title : str
            Article title
        authors : list
            List of author dicts with 'name' and 'affiliation'
        abstract : str
            Article abstract
        sections : dict
            Section name -> content
        figures : list
            List of figure specifications
        tables : list
            List of table specifications
        bibliography : str
            BibTeX file path
        document_class : str
            LaTeX document class
        
        Returns:
        --------
        str : Complete LaTeX document
        """
        # Document preamble
        if document_class == "IEEEtran":
            preamble = r"""\documentclass[journal]{IEEEtran}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{hyperref}
"""
        else:
            preamble = f"\\documentclass{{article}}\n"
            preamble += r"""\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}
"""
        
        # Begin document
        document = preamble + "\n\\begin{document}\n\n"
        
        # Title and authors
        document += f"\\title{{{title}}}\n\n"
        
        if document_class == "IEEEtran":
            author_str = "\\author{"
            for i, author in enumerate(authors):
                if i > 0:
                    author_str += ",~"
                author_str += f"\\IEEEauthorblockN{{{author['name']}}}\n"
                author_str += f"\\IEEEauthorblockA{{{author['affiliation']}}}"
            author_str += "}"
            document += author_str + "\n\n"
        else:
            author_str = " \\and ".join([author['name'] for author in authors])
            document += f"\\author{{{author_str}}}\n\n"
        
        document += "\\maketitle\n\n"
        
        # Abstract
        document += f"\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n\n"
        
        # Sections
        for section_name, content in sections.items():
            document += f"\\section{{{section_name}}}\n{content}\n\n"
        
        # Figures
        if figures:
            document += "\\section{Figures}\n\n"
            for fig_spec in figures:
                document += self._format_figure(fig_spec) + "\n\n"
        
        # Tables
        if tables:
            if not figures:
                document += "\\section{Tables}\n\n"
            for table_spec in tables:
                document += table_spec['latex'] + "\n\n"
        
        # Bibliography
        if bibliography:
            document += f"\\bibliographystyle{{IEEEtran}}\n"
            document += f"\\bibliography{{{bibliography}}}\n\n"
        
        # End document
        document += "\\end{document}"
        
        return document
    
    def _format_figure(self, fig_spec: Dict[str, Any]) -> str:
        """Format figure for LaTeX inclusion"""
        latex = "\\begin{figure}[htbp]\n\\centering\n"
        latex += f"\\includegraphics[width=\\linewidth]{{{fig_spec['filename']}}}\n"
        latex += f"\\caption{{{fig_spec['caption']}}}\n"
        latex += f"\\label{{{fig_spec['label']}}}\n"
        latex += "\\end{figure}"
        
        return latex
    
    def generate_supplementary_material(self,
                                      main_results: Dict[str, Any],
                                      additional_analyses: Dict[str, Any],
                                      code_snippets: Dict[str, str]) -> str:
        """
        Generate supplementary material document
        
        Parameters:
        -----------
        main_results : dict
            Main results to supplement
        additional_analyses : dict
            Additional analyses not in main paper
        code_snippets : dict
            Key code implementations
        
        Returns:
        --------
        str : LaTeX supplementary document
        """
        document = r"""\documentclass{article}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}

\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    commentstyle=\color{gray},
    numbers=left,
    numberstyle=\tiny\color{gray},
    frame=single,
    breaklines=true
}

\title{Supplementary Material:\\Air Quality Imputation Methods}
\author{}
\date{}

\begin{document}
\maketitle

\section{Additional Results}
"""
        
        # Add additional analyses
        for analysis_name, content in additional_analyses.items():
            document += f"\\subsection{{{analysis_name}}}\n{content}\n\n"
        
        # Add code snippets
        document += "\\section{Implementation Details}\n\n"
        
        for snippet_name, code in code_snippets.items():
            document += f"\\subsection{{{snippet_name}}}\n"
            document += f"\\begin{{lstlisting}}\n{code}\n\\end{{lstlisting}}\n\n"
        
        document += "\\end{document}"
        
        return document
    
    def export_for_journal_submission(self,
                                    journal: str,
                                    manuscript_data: Dict[str, Any],
                                    output_dir: Path) -> Dict[str, Path]:
        """
        Export complete package for journal submission
        
        Parameters:
        -----------
        journal : str
            Target journal name
        manuscript_data : dict
            All manuscript components
        output_dir : Path
            Output directory
        
        Returns:
        --------
        dict : Paths to generated files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Generate main manuscript
        manuscript = self.generate_latex_article(
            title=manuscript_data['title'],
            authors=manuscript_data['authors'],
            abstract=manuscript_data['abstract'],
            sections=manuscript_data['sections'],
            figures=manuscript_data['figures'],
            tables=manuscript_data['tables'],
            bibliography=manuscript_data.get('bibliography')
        )
        
        manuscript_path = output_dir / "manuscript.tex"
        with open(manuscript_path, 'w') as f:
            f.write(manuscript)
        files['manuscript'] = manuscript_path
        
        # Generate figures
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        
        for fig_data in manuscript_data['figures']:
            if 'figure' in fig_data:
                fig = fig_data['figure']
                filename = fig_data['filename']
                self.figure_gen.save_figure(
                    fig, 
                    str(figure_dir / filename),
                    formats=['pdf', 'eps']
                )
        
        # Generate supplementary material
        if 'supplementary' in manuscript_data:
            supp = self.generate_supplementary_material(
                main_results=manuscript_data['main_results'],
                additional_analyses=manuscript_data['supplementary']['analyses'],
                code_snippets=manuscript_data['supplementary']['code']
            )
            
            supp_path = output_dir / "supplementary.tex"
            with open(supp_path, 'w') as f:
                f.write(supp)
            files['supplementary'] = supp_path
        
        # Journal-specific requirements
        self._add_journal_specific_files(journal, output_dir, files)
        
        # Create ZIP archive
        import zipfile
        
        zip_path = output_dir / f"{journal}_submission.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_type, file_path in files.items():
                if file_path.exists():
                    zf.write(file_path, file_path.name)
            
            # Add figures
            if figure_dir.exists():
                for fig_file in figure_dir.iterdir():
                    zf.write(fig_file, f"figures/{fig_file.name}")
        
        files['submission_package'] = zip_path
        
        logger.info(f"Created submission package: {zip_path}")
        
        return files
    
    def _add_journal_specific_files(self, journal: str, output_dir: Path, 
                                  files: Dict[str, Path]):
        """Add journal-specific required files"""
        if journal.lower() == "ieee":
            # IEEE requires specific formatting
            ieee_cls = output_dir / "IEEEtran.cls"
            # Would download or copy IEEEtran.cls here
            
        elif journal.lower() == "nature":
            # Nature requires specific formatting
            pass
        
        elif journal.lower() == "elsevier":
            # Elsevier CAS template
            pass


# Example usage
if __name__ == "__main__":
    # Configuration
    config = PublicationConfig(
        figure_format=["pdf", "eps"],
        dpi=300,
        journal_style="ieee",
        use_latex=True,
        output_dir=Path("publication_output")
    )
    
    # Create generators
    fig_gen = PublicationFigureGenerator(config)
    table_gen = PublicationTableGenerator(config)
    doc_gen = PublicationDocumentGenerator(config)
    
    # Example: Create time series plot
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'PM2.5': np.sin(np.linspace(0, 4*np.pi, 100)) * 20 + 50 + np.random.normal(0, 5, 100),
        'PM10': np.sin(np.linspace(0, 4*np.pi, 100)) * 30 + 80 + np.random.normal(0, 8, 100)
    }, index=dates)
    
    # Add some missing values
    data.loc[data.index[20:30], 'PM2.5'] = np.nan
    data.loc[data.index[60:70], 'PM10'] = np.nan
    
    fig, ax = fig_gen.create_time_series_plot(
        data,
        ylabel=r"Concentration ($\mu$g/m$^3$)",
        xlabel="Date",
        highlight_gaps=True
    )
    
    fig_gen.save_figure(fig, "time_series_example")
    
    # Example: Create results table
    results = pd.DataFrame({
        'RMSE': [5.23, 4.87, 6.12, 4.92],
        'MAE': [3.45, 3.12, 4.23, 3.21],
        'R²': [0.923, 0.945, 0.876, 0.938]
    }, index=['Linear', 'RF', 'LSTM', 'RAH'])
    
    latex_table = table_gen.create_results_table(
        results,
        caption="Performance comparison of imputation methods on PM2.5 data",
        label="tab:pm25_results",
        highlight_best=True
    )
    
    print("LaTeX Table:")
    print(latex_table)
    
    # Example: Generate complete manuscript structure
    manuscript_data = {
        'title': "Advanced Methods for Air Quality Data Imputation",
        'authors': [
            {'name': 'Jane Doe', 'affiliation': 'University of Example'},
            {'name': 'John Smith', 'affiliation': 'Institute of Technology'}
        ],
        'abstract': "We present a comprehensive evaluation of imputation methods...",
        'sections': {
            'Introduction': "Air quality monitoring is essential...",
            'Methods': "We evaluated four imputation approaches...",
            'Results': "Our experiments demonstrate...",
            'Conclusion': "The RAH method shows superior performance..."
        },
        'figures': [
            {
                'filename': 'time_series_example',
                'caption': 'Time series of PM concentrations with missing data highlighted',
                'label': 'fig:timeseries'
            }
        ],
        'tables': [
            {
                'latex': latex_table
            }
        ],
        'bibliography': 'references'
    }
    
    # Generate manuscript
    manuscript = doc_gen.generate_latex_article(**manuscript_data)
    
    # Save manuscript
    with open(config.output_dir / "manuscript_example.tex", 'w') as f:
        f.write(manuscript)
    
    print(f"\nFiles saved to {config.output_dir}")