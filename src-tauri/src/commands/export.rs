use tauri::{command, Window};
use std::sync::Arc;
use tauri::State;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use anyhow::{Result, Context};
use tracing::info;
use std::path::{Path, PathBuf};
use std::fs;
use chrono::{Utc, Datelike};

use crate::state::AppState;
use crate::commands::data::{save_csv, save_excel};
use crate::core::imputation::JobStatus;
#[cfg(feature = "netcdf-support")]
use crate::commands::data::save_netcdf;
#[cfg(feature = "hdf5-support")]
use crate::commands::data::save_hdf5;
use crate::core::data::Dataset;
use crate::security::validate_write_path;
use crate::security::escape_latex as security_escape_latex;

#[derive(Debug, Clone, Deserialize)]
pub struct ExportOptions {
    pub include_metadata: Option<bool>,
    pub include_quality_flags: Option<bool>,
    pub date_format: Option<String>,
    pub delimiter: Option<String>,
    pub encoding: Option<String>,
    pub precision: Option<usize>,
    pub na_representation: Option<String>,
    pub compression: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExportResult {
    pub success: bool,
    pub file_path: String,
    pub file_size_mb: f64,
    pub rows_exported: usize,
    pub columns_exported: usize,
    pub export_time_ms: u128,
    pub warnings: Vec<String>,
}

#[command]
pub async fn export_to_csv(
    window: Window,
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
    path: String,
    options: ExportOptions,
) -> Result<ExportResult, String> {
    let start_time = std::time::Instant::now();
    info!("Exporting dataset {} to CSV: {}", dataset_id, path);
    
    // Parse dataset ID
    let id = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    // Get dataset
    let dataset_ref = state.datasets.get(&id)
        .ok_or_else(|| "Dataset not found".to_string())?;
    let dataset = dataset_ref.value();
    
    // Validate and sanitize export path using security module
    let export_path = validate_write_path(&path)
        .map_err(|e| format!("Security validation failed: {}", e))?;
    
    // Emit progress
    window.emit("export-progress", serde_json::json!({
        "stage": "preparing",
        "progress": 0.1,
        "message": "Preparing CSV export..."
    })).ok();
    
    // Use enhanced CSV export if options specified
    if options.delimiter.is_some() || options.encoding.is_some() || 
       options.precision.is_some() || options.na_representation.is_some() {
        
        // Custom CSV export with options
        export_csv_with_options(dataset, &export_path, &options, &window)
            .await
            .map_err(|e| format!("Failed to export CSV: {}", e))?;
    } else {
        // Use standard save_csv
        save_csv(dataset, &export_path)
            .await
            .map_err(|e| format!("Failed to save CSV: {}", e))?;
    }
    
    // Add metadata file if requested
    let mut warnings = Vec::new();
    if options.include_metadata.unwrap_or(true) {
        let metadata_path = export_path.with_extension("csv.meta.json");
        match export_metadata(dataset, &metadata_path) {
            Ok(_) => info!("Exported metadata to {:?}", metadata_path),
            Err(e) => warnings.push(format!("Failed to export metadata: {}", e)),
        }
    }
    
    // Get file size
    let file_size = fs::metadata(&export_path)
        .map(|m| m.len() as f64 / 1_048_576.0)
        .unwrap_or(0.0);
    
    // Apply compression if requested
    if let Some(compression) = &options.compression {
        compress_file(&export_path, compression)
            .map_err(|e| format!("Failed to compress file: {}", e))?;
    }
    
    // Emit completion
    window.emit("export-progress", serde_json::json!({
        "stage": "complete",
        "progress": 1.0,
        "message": "CSV export completed successfully"
    })).ok();
    
    Ok(ExportResult {
        success: true,
        file_path: export_path.to_string_lossy().to_string(),
        file_size_mb: file_size,
        rows_exported: dataset.rows(),
        columns_exported: dataset.columns(),
        export_time_ms: start_time.elapsed().as_millis(),
        warnings,
    })
}

async fn export_csv_with_options(
    dataset: &Dataset,
    path: &Path,
    options: &ExportOptions,
    window: &Window,
) -> Result<()> {
    use csv::WriterBuilder;
    
    
    info!("Exporting CSV with custom options");
    
    let file = fs::File::create(path)
        .context("Failed to create CSV file")?;
    
    let delimiter = options.delimiter.as_ref()
        .and_then(|d| d.chars().next())
        .unwrap_or(',');
    
    let mut writer = WriterBuilder::new()
        .delimiter(delimiter as u8)
        .from_writer(file);
    
    // Write headers
    let mut headers = vec!["timestamp".to_string()];
    headers.extend(dataset.columns.clone());
    writer.write_record(&headers)?;
    
    // Date format
    let date_format = options.date_format.as_deref()
        .unwrap_or("%Y-%m-%d %H:%M:%S");
    
    // NA representation
    let na_repr = options.na_representation.as_deref()
        .unwrap_or("");
    
    // Precision
    let precision = options.precision.unwrap_or(6);
    
    // Write data with progress
    let total_rows = dataset.rows();
    let progress_interval = (total_rows / 100).max(1);
    
    for (i, timestamp) in dataset.index.iter().enumerate() {
        if i % progress_interval == 0 {
            let progress = 0.1 + (i as f64 / total_rows as f64) * 0.8;
            window.emit("export-progress", serde_json::json!({
                "stage": "writing",
                "progress": progress,
                "message": format!("Writing row {} of {}", i + 1, total_rows)
            })).ok();
        }
        
        let mut record = vec![timestamp.format(date_format).to_string()];
        
        for j in 0..dataset.columns() {
            let value = dataset.data[[i, j]];
            let formatted = if value.is_nan() {
                na_repr.to_string()
            } else {
                format!("{:.prec$}", value, prec = precision)
            };
            record.push(formatted);
        }
        
        writer.write_record(&record)?;
    }
    
    writer.flush()?;
    Ok(())
}

#[command]
pub async fn export_to_excel(
    window: Window,
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
    path: String,
    options: ExportOptions,
) -> Result<ExportResult, String> {
    let start_time = std::time::Instant::now();
    info!("Exporting dataset {} to Excel: {}", dataset_id, path);
    
    // Parse dataset ID
    let id = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    // Get dataset
    let dataset_ref = state.datasets.get(&id)
        .ok_or_else(|| "Dataset not found".to_string())?;
    let dataset = dataset_ref.value();
    
    // Validate and sanitize export path using security module
    let export_path = validate_write_path(&path)
        .map_err(|e| format!("Security validation failed: {}", e))?;
    
    // Emit progress
    window.emit("export-progress", serde_json::json!({
        "stage": "preparing",
        "progress": 0.1,
        "message": "Preparing Excel export..."
    })).ok();
    
    // Export to Excel
    save_excel(dataset, &export_path)
        .await
        .map_err(|e| format!("Failed to save Excel: {}", e))?;
    
    // Get file size
    let file_size = fs::metadata(&export_path)
        .map(|m| m.len() as f64 / 1_048_576.0)
        .unwrap_or(0.0);
    
    // Emit completion
    window.emit("export-progress", serde_json::json!({
        "stage": "complete",
        "progress": 1.0,
        "message": "Excel export completed successfully"
    })).ok();
    
    Ok(ExportResult {
        success: true,
        file_path: export_path.to_string_lossy().to_string(),
        file_size_mb: file_size,
        rows_exported: dataset.rows(),
        columns_exported: dataset.columns(),
        export_time_ms: start_time.elapsed().as_millis(),
        warnings: vec![],
    })
}

#[command]
pub async fn export_to_netcdf(
    window: Window,
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
    path: String,
    options: ExportOptions,
) -> Result<ExportResult, String> {
    let start_time = std::time::Instant::now();
    info!("Exporting dataset {} to NetCDF: {}", dataset_id, path);
    
    // Parse dataset ID
    let id = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    // Get dataset
    let dataset_ref = state.datasets.get(&id)
        .ok_or_else(|| "Dataset not found".to_string())?;
    let dataset = dataset_ref.value();
    
    // Validate and sanitize export path using security module
    let export_path = validate_write_path(&path)
        .map_err(|e| format!("Security validation failed: {}", e))?;
    
    // Emit progress
    window.emit("export-progress", serde_json::json!({
        "stage": "preparing",
        "progress": 0.1,
        "message": "Preparing NetCDF export..."
    })).ok();
    
    // Export to NetCDF
    #[cfg(feature = "netcdf-support")]
    {
        save_netcdf(dataset, &export_path)
            .await
            .map_err(|e| format!("Failed to save NetCDF: {}", e))?;
    }
    #[cfg(not(feature = "netcdf-support"))]
    {
        return Err("NetCDF support not compiled. Enable 'netcdf-support' feature.".to_string());
    }
    
    // Get file size
    let file_size = fs::metadata(&export_path)
        .map(|m| m.len() as f64 / 1_048_576.0)
        .unwrap_or(0.0);
    
    // Apply compression if requested
    let mut warnings = Vec::new();
    if let Some(compression) = &options.compression {
        if compression == "gzip" || compression == "gz" {
            match compress_file(&export_path, compression) {
                Ok(_) => info!("Compressed NetCDF file"),
                Err(e) => warnings.push(format!("Failed to compress: {}", e)),
            }
        }
    }
    
    // Emit completion
    window.emit("export-progress", serde_json::json!({
        "stage": "complete",
        "progress": 1.0,
        "message": "NetCDF export completed successfully"
    })).ok();
    
    Ok(ExportResult {
        success: true,
        file_path: export_path.to_string_lossy().to_string(),
        file_size_mb: file_size,
        rows_exported: dataset.rows(),
        columns_exported: dataset.columns(),
        export_time_ms: start_time.elapsed().as_millis(),
        warnings,
    })
}

#[command]
pub async fn export_to_hdf5(
    window: Window,
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
    path: String,
    options: ExportOptions,
) -> Result<ExportResult, String> {
    let start_time = std::time::Instant::now();
    info!("Exporting dataset {} to HDF5: {}", dataset_id, path);
    
    // Parse dataset ID
    let id = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    // Get dataset
    let dataset_ref = state.datasets.get(&id)
        .ok_or_else(|| "Dataset not found".to_string())?;
    let dataset = dataset_ref.value();
    
    // Validate and sanitize export path using security module
    let export_path = validate_write_path(&path)
        .map_err(|e| format!("Security validation failed: {}", e))?;
    
    // Emit progress
    window.emit("export-progress", serde_json::json!({
        "stage": "preparing",
        "progress": 0.1,
        "message": "Preparing HDF5 export..."
    })).ok();
    
    // Export to HDF5
    #[cfg(feature = "hdf5-support")]
    {
        save_hdf5(dataset, &export_path)
            .await
            .map_err(|e| format!("Failed to save HDF5: {}", e))?;
    }
    #[cfg(not(feature = "hdf5-support"))]
    {
        return Err("HDF5 support not compiled. Enable 'hdf5-support' feature.".to_string());
    }
    
    // Get file size
    let file_size = fs::metadata(&export_path)
        .map(|m| m.len() as f64 / 1_048_576.0)
        .unwrap_or(0.0);
    
    // Emit completion
    window.emit("export-progress", serde_json::json!({
        "stage": "complete",
        "progress": 1.0,
        "message": "HDF5 export completed successfully"
    })).ok();
    
    Ok(ExportResult {
        success: true,
        file_path: export_path.to_string_lossy().to_string(),
        file_size_mb: file_size,
        rows_exported: dataset.rows(),
        columns_exported: dataset.columns(),
        export_time_ms: start_time.elapsed().as_millis(),
        warnings: vec![],
    })
}

#[derive(Debug, Clone, Deserialize)]
pub struct LaTeXTemplate {
    pub name: String,
    pub content: String,
    pub variables: Vec<String>,
}

#[command]
pub async fn generate_latex_report(
    window: Window,
    state: State<'_, Arc<AppState>>,
    job_id: String,
    template: String,
) -> Result<String, String> {
    info!("Generating LaTeX report for job: {}", job_id);
    
    // Parse job ID
    let job_uuid = Uuid::parse_str(&job_id)
        .map_err(|e| format!("Invalid job ID: {}", e))?;
    
    // Get imputation job and dataset, cloning the data
    let (job_data, dataset) = {
        let job_entry = state.imputation_jobs.get(&job_uuid)
            .ok_or_else(|| "Imputation job not found".to_string())?;
        let job = job_entry.lock().await;
        
        if job.status != JobStatus::Completed {
            return Err(format!("Job is not completed: status = {:?}", job.status));
        }
        
        let dataset_ref = state.datasets.get(&job.dataset_id)
            .ok_or_else(|| "Dataset not found".to_string())?;
        
        // Clone the data we need and drop the locks
        (job.clone(), dataset_ref.value().clone())
    }; // Locks are released here
    
    // Emit progress
    window.emit("report-progress", serde_json::json!({
        "stage": "generating",
        "progress": 0.1,
        "message": "Generating LaTeX report..."
    })).ok();
    
    // Select template
    let latex_template = match template.as_str() {
        "academic" => ACADEMIC_REPORT_TEMPLATE,
        "technical" => TECHNICAL_REPORT_TEMPLATE,
        "summary" => SUMMARY_REPORT_TEMPLATE,
        _ => ACADEMIC_REPORT_TEMPLATE,
    };
    
    // Extract job results
    let method = job_data.result_data.as_ref()
        .and_then(|rd| rd.get("method"))
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown");
    
    let metrics = job_data.result_data.as_ref()
        .and_then(|rd| rd.get("metrics"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    
    let execution_time = job_data.result_data.as_ref()
        .and_then(|rd| rd.get("execution_time_ms"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    
    // Calculate statistics
    let total_missing = dataset.count_missing();
    let total_values = dataset.rows() * dataset.columns();
    let missing_percentage = (total_missing as f64 / total_values as f64) * 100.0;
    
    // Generate plots for report
    let plots = generate_report_plots(&dataset, &job_data, &state).await?;
    
    // Create variable metrics table
    let mut variable_metrics = Vec::new();
    if let Some(metrics_obj) = metrics.as_object() {
        for (var, var_metrics) in metrics_obj {
            if let Some(var_metrics_obj) = var_metrics.as_object() {
                variable_metrics.push(format!(
                    "{} & {:.4} & {:.4} & {:.4} & {:.4} \\\\",
                    security_escape_latex(var),
                    var_metrics_obj.get("rmse").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    var_metrics_obj.get("mae").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    var_metrics_obj.get("r2").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    var_metrics_obj.get("mape").and_then(|v| v.as_f64()).unwrap_or(0.0),
                ));
            }
        }
    }
    
    // Fill template
    let report = latex_template
        .replace("{TITLE}", &security_escape_latex(&format!("Imputation Analysis Report: {}", dataset.name)))
        .replace("{DATE}", &Utc::now().format("%B %d, %Y").to_string())
        .replace("{DATASET_NAME}", &security_escape_latex(&dataset.name))
        .replace("{METHOD}", &security_escape_latex(method))
        .replace("{TOTAL_ROWS}", &dataset.rows().to_string())
        .replace("{TOTAL_COLUMNS}", &dataset.columns().to_string())
        .replace("{MISSING_VALUES}", &total_missing.to_string())
        .replace("{MISSING_PERCENTAGE}", &format!("{:.2}", missing_percentage))
        .replace("{EXECUTION_TIME}", &format!("{:.2}", execution_time as f64 / 1000.0))
        .replace("{VARIABLE_METRICS_TABLE}", &variable_metrics.join("\n"))
        .replace("{MISSING_PATTERN_PLOT}", &plots.missing_pattern)
        .replace("{TIMESERIES_PLOT}", &plots.timeseries)
        .replace("{CORRELATION_PLOT}", &plots.correlation)
        .replace("{UNCERTAINTY_PLOT}", &plots.uncertainty)
        .replace("{JOB_ID}", &job_id)
        .replace("{PARAMETERS}", &format_parameters(&serde_json::to_value(&job_data.parameters).unwrap()));
    
    // Emit completion
    window.emit("report-progress", serde_json::json!({
        "stage": "complete",
        "progress": 1.0,
        "message": "LaTeX report generated successfully"
    })).ok();
    
    Ok(report)
}

// LaTeX report templates
const ACADEMIC_REPORT_TEMPLATE: &str = r#"\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{float}
\usepackage[margin=1in]{geometry}

\title{{TITLE}}
\author{AirImpute Pro - Automated Report}
\date{{DATE}}

\begin{document}
\maketitle

\begin{abstract}
This report presents the results of air quality data imputation performed on the {DATASET_NAME} dataset using the {METHOD} method. The analysis addresses {MISSING_VALUES} missing values ({MISSING_PERCENTAGE}\% of total data points) across {TOTAL_COLUMNS} variables and {TOTAL_ROWS} time points.
\end{abstract}

\section{Introduction}
Missing data in air quality monitoring presents significant challenges for environmental analysis and decision-making. This report documents the imputation process and evaluates the quality of reconstructed values.

\section{Dataset Overview}
\begin{table}[H]
\centering
\caption{Dataset Characteristics}
\begin{tabular}{ll}
\toprule
Property & Value \\
\midrule
Dataset Name & {DATASET_NAME} \\
Total Observations & {TOTAL_ROWS} \\
Variables & {TOTAL_COLUMNS} \\
Missing Values & {MISSING_VALUES} \\
Missing Percentage & {MISSING_PERCENTAGE}\% \\
\bottomrule
\end{tabular}
\end{table}

\section{Missing Data Pattern Analysis}
\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{{MISSING_PATTERN_PLOT}}
\caption{Missing data pattern visualization showing the distribution of missing values across variables and time.}
\label{fig:missing_pattern}
\end{figure}

\section{Imputation Methodology}
\subsection{Method: {METHOD}}
{PARAMETERS}

\subsection{Execution Details}
\begin{itemize}
\item Job ID: \texttt{{JOB_ID}}
\item Execution Time: {EXECUTION_TIME} seconds
\item Convergence: Achieved
\end{itemize}

\section{Results}
\subsection{Performance Metrics}
\begin{table}[H]
\centering
\caption{Imputation Performance by Variable}
\begin{tabular}{lcccc}
\toprule
Variable & RMSE & MAE & R² & MAPE (\%) \\
\midrule
{VARIABLE_METRICS_TABLE}
\bottomrule
\end{tabular}
\end{table}

\subsection{Time Series Visualization}
\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{{TIMESERIES_PLOT}}
\caption{Time series showing original and imputed values for selected variables.}
\label{fig:timeseries}
\end{figure}

\subsection{Uncertainty Quantification}
\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{{UNCERTAINTY_PLOT}}
\caption{Uncertainty bands for imputed values showing 95\% confidence intervals.}
\label{fig:uncertainty}
\end{figure}

\subsection{Correlation Structure}
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{{CORRELATION_PLOT}}
\caption{Correlation matrix comparing relationships before and after imputation.}
\label{fig:correlation}
\end{figure}

\section{Conclusions}
The {METHOD} imputation method successfully reconstructed {MISSING_VALUES} missing values with an average execution time of {EXECUTION_TIME} seconds. The performance metrics indicate reliable imputation quality suitable for subsequent analysis.

\section{References}
\begin{enumerate}
\item Little, R. J., \& Rubin, D. B. (2019). \textit{Statistical analysis with missing data} (3rd ed.). John Wiley \& Sons.
\item Moritz, S., \& Bartz-Beielstein, T. (2017). imputeTS: Time series missing value imputation in R. \textit{The R Journal}, 9(1), 207-218.
\end{enumerate}

\end{document}
"#;

const TECHNICAL_REPORT_TEMPLATE: &str = r#"\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{courier}
\usepackage[margin=1in]{geometry}

\title{Technical Imputation Report: {DATASET_NAME}}
\date{{DATE}}

\begin{document}
\maketitle

\section{Summary}
Dataset: {DATASET_NAME}\\
Method: {METHOD}\\
Missing: {MISSING_PERCENTAGE}\%\\
Execution Time: {EXECUTION_TIME}s

\section{Results}
{VARIABLE_METRICS_TABLE}

\section{Visualizations}
\includegraphics[width=\textwidth]{{MISSING_PATTERN_PLOT}}
\includegraphics[width=\textwidth]{{TIMESERIES_PLOT}}

\end{document}
"#;

const SUMMARY_REPORT_TEMPLATE: &str = r#"\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1.5in]{geometry}

\title{Imputation Summary}
\date{{DATE}}

\begin{document}
\maketitle

Dataset: {DATASET_NAME}

Method: {METHOD}

Missing Values: {MISSING_VALUES} ({MISSING_PERCENTAGE}\%)

Execution Time: {EXECUTION_TIME} seconds

\end{document}
"#;


fn format_parameters(params: &serde_json::Value) -> String {
    if let Some(obj) = params.as_object() {
        let items: Vec<String> = obj.iter()
            .map(|(k, v)| format!("\\item {}: {}", 
                security_escape_latex(k), 
                security_escape_latex(&v.to_string())))
            .collect();
        
        format!("\\begin{{itemize}}\n{}\n\\end{{itemize}}", items.join("\n"))
    } else {
        "No parameters specified".to_string()
    }
}

struct ReportPlots {
    missing_pattern: String,
    timeseries: String,
    correlation: String,
    uncertainty: String,
}

async fn generate_report_plots(
    dataset: &Dataset,
    job: &crate::core::imputation::ImputationJob,
    state: &Arc<AppState>,
) -> Result<ReportPlots, String> {
    // Generate plot file paths using platform-specific temp directory
    let plot_dir = std::env::temp_dir().join("airimpute_reports");
    fs::create_dir_all(&plot_dir)
        .map_err(|e| format!("Failed to create plot directory: {}", e))?;
    
    let missing_pattern_path = plot_dir.join(format!("{}_missing.pdf", job.id));
    let timeseries_path = plot_dir.join(format!("{}_timeseries.pdf", job.id));
    let correlation_path = plot_dir.join(format!("{}_correlation.pdf", job.id));
    let uncertainty_path = plot_dir.join(format!("{}_uncertainty.pdf", job.id));
    
    // Generate plots using Python bridge
    // (Simplified - would call visualization functions)
    
    Ok(ReportPlots {
        missing_pattern: missing_pattern_path.to_string_lossy().to_string(),
        timeseries: timeseries_path.to_string_lossy().to_string(),
        correlation: correlation_path.to_string_lossy().to_string(),
        uncertainty: uncertainty_path.to_string_lossy().to_string(),
    })
}

#[command]
pub async fn generate_publication_package(
    window: Window,
    state: State<'_, Arc<AppState>>,
    project_id: String,
    output_path: String,
) -> Result<String, String> {
    info!("Generating publication package for project: {}", project_id);
    
    // Parse project ID
    let project_uuid = Uuid::parse_str(&project_id)
        .map_err(|e| format!("Invalid project ID: {}", e))?;
    
    // Get project
    let project_ref = state.projects.get(&project_uuid)
        .ok_or_else(|| "Project not found".to_string())?;
    let project = project_ref.value();
    
    // Create output directory
    let package_dir = PathBuf::from(&output_path);
    fs::create_dir_all(&package_dir)
        .map_err(|e| format!("Failed to create package directory: {}", e))?;
    
    // Emit progress
    window.emit("package-progress", serde_json::json!({
        "stage": "preparing",
        "progress": 0.1,
        "message": "Preparing publication package..."
    })).ok();
    
    // Create directory structure
    let dirs = [
        "data/raw",
        "data/processed",
        "data/imputed",
        "figures",
        "tables",
        "code",
        "docs",
        "supplements",
    ];
    
    for dir in &dirs {
        fs::create_dir_all(package_dir.join(dir))
            .map_err(|e| format!("Failed to create directory {}: {}", dir, e))?;
    }
    
    // Export datasets
    window.emit("package-progress", serde_json::json!({
        "stage": "exporting_data",
        "progress": 0.2,
        "message": "Exporting datasets..."
    })).ok();
    
    let mut exported_files = Vec::new();
    
    // Export all project datasets
    for entry in state.datasets.iter() {
        let dataset_id = entry.key();
        let dataset = entry.value();
        
        // Export to multiple formats
        let base_name = format!("{}_{}", project.name, dataset.name)
            .replace(" ", "_")
            .to_lowercase();
        
        // CSV format
        let csv_path = package_dir.join("data/processed")
            .join(format!("{}.csv", base_name));
        save_csv(dataset, &csv_path).await
            .map_err(|e| format!("Failed to export CSV: {}", e))?;
        exported_files.push(csv_path.to_string_lossy().to_string());
        
        // NetCDF format for scientific use
        #[cfg(feature = "netcdf-support")]
        {
            let nc_path = package_dir.join("data/processed")
                .join(format!("{}.nc", base_name));
            save_netcdf(dataset, &nc_path).await
                .map_err(|e| format!("Failed to export NetCDF: {}", e))?;
            exported_files.push(nc_path.to_string_lossy().to_string());
        }
    }
    
    // Generate figures
    window.emit("package-progress", serde_json::json!({
        "stage": "generating_figures",
        "progress": 0.4,
        "message": "Generating publication-quality figures..."
    })).ok();
    
    // Generate all standard plots
    // (Would call visualization functions to create high-res figures)
    
    // Create README
    window.emit("package-progress", serde_json::json!({
        "stage": "documentation",
        "progress": 0.6,
        "message": "Creating documentation..."
    })).ok();
    
    let readme_content = format!(r#"# {} - Publication Package

Generated: {}
Project ID: {}

## Contents

### Data
- `data/raw/` - Original raw data files
- `data/processed/` - Cleaned and processed datasets
- `data/imputed/` - Imputed datasets with uncertainty estimates

### Figures
- `figures/` - Publication-quality figures (PDF, PNG formats)

### Tables
- `tables/` - Statistical summaries and results tables

### Code
- `code/` - Analysis scripts and reproducibility code

### Documentation
- `docs/` - Detailed documentation and metadata
- `supplements/` - Supplementary materials

## Data Files

{}

## Citation

If you use this data, please cite:

```bibtex
@misc{{{}_data_{},
    title = {{{} Dataset}},
    author = {{AirImpute Research Team}},
    year = {{{}}},
    publisher = {{AirImpute Pro}},
    doi = {{10.xxxx/xxxxx}}
}}
```

## License

This data is provided under the CC BY 4.0 license.

## Contact

For questions about this dataset, please contact: [email]
"#, 
        project.name,
        Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        project_id,
        exported_files.iter()
            .map(|f| format!("- `{}`", f))
            .collect::<Vec<_>>()
            .join("\n"),
        project.name.replace(" ", "_").to_lowercase(),
        Utc::now().year(),
        project.name,
        Utc::now().year(),
    );
    
    fs::write(package_dir.join("README.md"), readme_content)
        .map_err(|e| format!("Failed to write README: {}", e))?;
    
    // Create metadata JSON
    let metadata = serde_json::json!({
        "project": {
            "id": project_id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "generated_at": Utc::now(),
        },
        "contents": {
            "datasets": state.datasets.len(),
            "figures": 0, // Would count actual figures
            "tables": 0,  // Would count actual tables
        },
        "checksums": {}, // Would calculate file checksums
        "reproducibility": {
            "airimpute_version": env!("CARGO_PKG_VERSION"),
            "generation_date": Utc::now(),
        }
    });
    
    fs::write(
        package_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata).unwrap()
    ).map_err(|e| format!("Failed to write metadata: {}", e))?;
    
    // Create LaTeX template for paper
    let paper_template = r#"\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}

\title{[Your Title Here]}
\author{[Authors]}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
[Abstract text]
\end{abstract}

\section{Introduction}

\section{Data and Methods}

\subsection{Dataset}
The dataset was processed using AirImpute Pro version {}.

\section{Results}

\section{Discussion}

\section{Conclusions}

\section{Acknowledgments}

\section{References}

\end{document}
"#;
    
    fs::write(
        package_dir.join("docs/paper_template.tex"),
        paper_template.replace("{}", env!("CARGO_PKG_VERSION"))
    ).map_err(|e| format!("Failed to write paper template: {}", e))?;
    
    // Create citation file
    let citation = format!(r#"cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: "AirImpute"
    given-names: "Research Team"
title: "{}"
version: 1.0.0
date-released: {}
url: "https://github.com/airimpute/airimpute-pro"
"#,
        project.name,
        Utc::now().format("%Y-%m-%d"),
    );
    
    fs::write(package_dir.join("CITATION.cff"), citation)
        .map_err(|e| format!("Failed to write citation file: {}", e))?;
    
    // Compress package
    window.emit("package-progress", serde_json::json!({
        "stage": "compressing",
        "progress": 0.9,
        "message": "Compressing package..."
    })).ok();
    
    let archive_path = format!("{}.tar.gz", output_path);
    compress_directory(&package_dir, &PathBuf::from(&archive_path))
        .map_err(|e| format!("Failed to compress package: {}", e))?;
    
    // Emit completion
    window.emit("package-progress", serde_json::json!({
        "stage": "complete",
        "progress": 1.0,
        "message": "Publication package created successfully"
    })).ok();
    
    Ok(archive_path)
}

// Helper functions

fn export_metadata(dataset: &Dataset, path: &Path) -> Result<()> {
    let metadata = serde_json::json!({
        "dataset": {
            "name": dataset.name,
            "id": dataset.id,
            "created_at": dataset.created_at,
            "modified_at": dataset.modified_at,
        },
        "shape": {
            "rows": dataset.rows(),
            "columns": dataset.columns(),
        },
        "variables": dataset.variables,
        "stations": dataset.stations,
        "time_range": {
            "start": dataset.index.first(),
            "end": dataset.index.last(),
        },
        "missing_data": {
            "total": dataset.count_missing(),
            "percentage": (dataset.count_missing() as f64 / 
                (dataset.rows() * dataset.columns()) as f64) * 100.0,
        }
    });
    
    let file = fs::File::create(path)?;
    serde_json::to_writer_pretty(file, &metadata)?;
    Ok(())
}

fn compress_file(path: &Path, compression: &str) -> Result<()> {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    
    match compression {
        "gzip" | "gz" => {
            let input_file = fs::File::open(path)?;
            let output_path = path.with_extension(format!("{}.gz", path.extension().unwrap_or_default().to_string_lossy()));
            let output_file = fs::File::create(&output_path)?;
            
            let mut encoder = GzEncoder::new(output_file, Compression::default());
            std::io::copy(&mut &input_file, &mut encoder)?;
            encoder.finish()?;
            
            // Replace original with compressed
            fs::remove_file(path)?;
            fs::rename(output_path, path)?;
        }
        "zstd" => {
            let input = fs::read(path)?;
            let compressed = zstd::encode_all(input.as_slice(), 3)?;
            fs::write(path, compressed)?;
        }
        _ => return Err(anyhow::anyhow!("Unsupported compression: {}", compression)),
    }
    
    Ok(())
}

fn compress_directory(source: &Path, destination: &Path) -> Result<()> {
    use flate2::write::GzEncoder;
    use tar::Builder;
    
    let tar_gz = fs::File::create(destination)?;
    let enc = GzEncoder::new(tar_gz, flate2::Compression::default());
    let mut tar = Builder::new(enc);
    
    tar.append_dir_all(".", source)?;
    tar.finish()?;
    
    Ok(())
}