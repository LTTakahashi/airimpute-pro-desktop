use tauri::{command, State, Window};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::{Result, Context};
use tracing::{info, warn, error};
use serde_json::json;
use ndarray::{Array2, Axis};
use std::collections::HashMap;

use crate::state::AppState;
use crate::core::data::{Dataset, DataValidation, DataStatistics};
use crate::utils::fs::{get_file_size, sanitize_path};
use crate::security::validate_read_path;

/// Response structure for dataset operations
#[derive(Debug, Clone, Serialize)]
pub struct DatasetResponse {
    pub id: String,
    pub name: String,
    pub path: String,
    pub rows: usize,
    pub columns: usize,
    pub missing_percentage: f64,
    pub statistics: Option<DataStatistics>,
    pub created_at: DateTime<Utc>,
}

/// Request structure for data import
#[derive(Debug, Clone, Deserialize)]
pub struct ImportRequest {
    pub paths: Vec<String>,
    pub format: DataFormat,
    pub options: ImportOptions,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataFormat {
    Csv,
    Excel,
    NetCdf,
    Hdf5,
    Parquet,
    Json,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImportOptions {
    pub delimiter: Option<String>,
    pub encoding: Option<String>,
    pub has_header: bool,
    pub date_column: Option<String>,
    pub parse_dates: bool,
    pub na_values: Option<Vec<String>>,
    pub skip_rows: Option<usize>,
    pub use_cols: Option<Vec<String>>,
}

/// Load a dataset from file system with validation
#[command]
pub async fn load_dataset(
    window: Window,
    state: State<'_, Arc<AppState>>,
    path: String,
    options: ImportOptions,
) -> Result<DatasetResponse, String> {
    info!("Loading dataset from: {}", path);
    
    // Validate and sanitize path using security module
    let sanitized_path = validate_read_path(&path)
        .map_err(|e| format!("Security validation failed: {}", e))?;
    
    // Check file exists and is readable
    if !sanitized_path.exists() {
        return Err(format!("File not found: {}", path));
    }
    
    if !sanitized_path.is_file() {
        return Err(format!("Path is not a file: {}", path));
    }
    
    // Check file size using utility function
    let file_size_mb = get_file_size(&sanitized_path)
        .map_err(|e| format!("Failed to get file size: {}", e))?
        / (1024 * 1024);
    
    if file_size_mb > 500 {
        warn!("Large file detected: {} MB", file_size_mb);
        window.emit("import-warning", json!({
            "message": format!("Large file: {} MB. Processing may take time.", file_size_mb)
        })).ok();
    }
    
    // Emit progress event
    window.emit("import-progress", serde_json::json!({
        "stage": "reading",
        "progress": 0.1,
        "message": "Reading file..."
    })).ok();
    
    // Load dataset using appropriate reader
    let dataset = match detect_format(&sanitized_path) {
        DataFormat::Csv => load_csv(&sanitized_path, &options).await,
        DataFormat::Excel => load_excel(&sanitized_path, &options).await,
        DataFormat::NetCdf => {
            return Err("NetCDF format support is not yet implemented".to_string());
        }
        DataFormat::Hdf5 => {
            return Err("HDF5 format support is not yet implemented".to_string());
        }
        DataFormat::Parquet => {
            return Err("Parquet format is temporarily disabled due to dependency conflicts".to_string());
            // load_parquet(&sanitized_path, &options).await
        }
        DataFormat::Json => load_json(&sanitized_path, &options).await,
    }.map_err(|e| format!("Failed to load dataset: {}", e))?;
    
    // Emit progress
    window.emit("import-progress", serde_json::json!({
        "stage": "validating",
        "progress": 0.5,
        "message": "Validating data..."
    })).ok();
    
    // Validate dataset
    let validation = DataValidation::validate(&dataset)
        .map_err(|e| format!("Data validation failed: {}", e))?;
    
    if !validation.is_valid {
        warn!("Dataset has validation issues: {:?}", validation.issues);
        // Emit warning but don't fail
        window.emit("import-warning", serde_json::json!({
            "issues": validation.issues
        })).ok();
    }
    
    // Calculate statistics
    window.emit("import-progress", serde_json::json!({
        "stage": "analyzing",
        "progress": 0.7,
        "message": "Calculating statistics..."
    })).ok();
    
    let statistics = DataStatistics::calculate(&dataset)
        .map_err(|e| format!("Failed to calculate statistics: {}", e))?;
    
    // Generate unique ID
    let dataset_id = Uuid::new_v4();
    
    // Store in state
    let dataset_clone = dataset.clone();
    state.datasets.insert(dataset_id, Arc::new(dataset_clone));
    
    // Save metadata to database
    // For now, use a placeholder project ID - this should come from active project
    let project_id = Uuid::new_v4(); // TODO: Get from active project
    let column_metadata = serde_json::json!({
        "columns": dataset.columns,
        "variables": dataset.variables,
    });
    
    let db_result = state.db.save_dataset_metadata(
        &project_id,
        &dataset.name,
        &sanitized_path,
        &statistics,
        column_metadata,
    ).await;
    
    if let Err(e) = db_result {
        error!("Failed to save dataset metadata: {}", e);
        // Continue anyway - data is in memory
    }
    
    // Emit completion
    window.emit("import-progress", serde_json::json!({
        "stage": "complete",
        "progress": 1.0,
        "message": "Dataset loaded successfully"
    })).ok();
    
    // Calculate missing percentage
    let total_values = dataset.rows() * dataset.columns();
    let missing_percentage = if total_values > 0 {
        (dataset.count_missing() as f64 / total_values as f64) * 100.0
    } else {
        0.0
    };
    
    Ok(DatasetResponse {
        id: dataset_id.to_string(),
        name: dataset.name.clone(),
        path: sanitized_path.to_string_lossy().to_string(),
        rows: dataset.rows(),
        columns: dataset.columns(),
        missing_percentage,
        statistics: Some(statistics),
        created_at: Utc::now(),
    })
}

/// Save dataset to file system
#[command]
pub async fn save_dataset(
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
    path: String,
    format: DataFormat,
) -> Result<(), String> {
    let id = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    let dataset_ref = state.datasets.get(&id)
        .ok_or_else(|| "Dataset not found".to_string())?;
    let dataset = dataset_ref.value();
    
    let sanitized_path = sanitize_path(&path)
        .map_err(|e| format!("Invalid path: {}", e))?;
    
    // Save based on format
    match format {
        DataFormat::Csv => save_csv(dataset, &sanitized_path).await,
        DataFormat::Excel => save_excel(dataset, &sanitized_path).await,
        DataFormat::NetCdf => {
            return Err("NetCDF format support is not yet implemented".to_string());
        }
        DataFormat::Hdf5 => {
            return Err("HDF5 format support is not yet implemented".to_string());
        }
        DataFormat::Parquet => save_parquet(dataset, &sanitized_path).await,
        DataFormat::Json => save_json(dataset, &sanitized_path).await,
    }.map_err(|e| format!("Failed to save dataset: {}", e))?;
    
    info!("Dataset saved to: {}", sanitized_path.display());
    Ok(())
}

/// Validate dataset structure and content
#[command]
pub async fn validate_dataset(
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
) -> Result<DataValidation, String> {
    let id = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    let dataset_ref = state.datasets.get(&id)
        .ok_or_else(|| "Dataset not found".to_string())?;
    let dataset = dataset_ref.value();
    
    DataValidation::validate(dataset)
        .map_err(|e| format!("Validation failed: {}", e))
}

/// Get comprehensive dataset statistics
#[command]
pub async fn get_dataset_statistics(
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
) -> Result<DataStatistics, String> {
    let id = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    let dataset_ref = state.datasets.get(&id)
        .ok_or_else(|| "Dataset not found".to_string())?;
    let dataset = dataset_ref.value();
    
    DataStatistics::calculate(dataset)
        .map_err(|e| format!("Failed to calculate statistics: {}", e))
}

/// Preview dataset with configurable limits
#[command]
pub async fn preview_dataset(
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
    rows: Option<usize>,
    columns: Option<Vec<String>>,
) -> Result<serde_json::Value, String> {
    let id = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    let dataset_ref = state.datasets.get(&id)
        .ok_or_else(|| "Dataset not found".to_string())?;
    let dataset = dataset_ref.value();
    
    let preview_rows = rows.unwrap_or(100).min(1000); // Max 1000 rows for preview
    
    // Generate preview
    let preview = dataset.preview(preview_rows, columns.as_deref())
        .map_err(|e| format!("Failed to generate preview: {}", e))?;
    
    Ok(preview)
}

/// Import from multiple data sources simultaneously
#[command]
pub async fn import_from_multiple_sources(
    window: Window,
    state: State<'_, Arc<AppState>>,
    requests: Vec<ImportRequest>,
) -> Result<Vec<DatasetResponse>, String> {
    info!("Importing {} datasets", requests.len());
    
    let mut results = Vec::new();
    let total = requests.len();
    
    for (idx, request) in requests.into_iter().enumerate() {
        // Update overall progress
        let progress = (idx as f64) / (total as f64);
        window.emit("batch-import-progress", serde_json::json!({
            "current": idx + 1,
            "total": total,
            "progress": progress,
        })).ok();
        
        // Process each file
        for path in request.paths {
            match load_dataset(
                window.clone(),
                state.clone(),
                path.clone(),
                request.options.clone(),
            ).await {
                Ok(response) => results.push(response),
                Err(e) => {
                    error!("Failed to import {}: {}", path, e);
                    window.emit("import-error", serde_json::json!({
                        "path": path,
                        "error": e.to_string(),
                    })).ok();
                }
            }
        }
    }
    
    info!("Successfully imported {} datasets", results.len());
    Ok(results)
}

// Helper functions for format detection and loading

fn detect_format(path: &PathBuf) -> DataFormat {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("csv") | Some("tsv") => DataFormat::Csv,
        Some("xlsx") | Some("xls") => DataFormat::Excel,
        Some("nc") | Some("netcdf") => DataFormat::NetCdf,
        Some("h5") | Some("hdf5") => DataFormat::Hdf5,
        Some("parquet") => DataFormat::Parquet,
        Some("json") => DataFormat::Json,
        _ => DataFormat::Csv, // Default
    }
}

async fn load_csv(path: &PathBuf, options: &ImportOptions) -> Result<Dataset> {
    use csv::ReaderBuilder;
    use std::fs::File;
    use std::io::BufReader;
    use chrono::NaiveDateTime;
    
    info!("Loading CSV from: {:?}", path);
    
    let file = File::open(path)
        .context("Failed to open CSV file")?;
    let mut reader = ReaderBuilder::new()
        .delimiter(options.delimiter.as_ref().map(|s| s.as_bytes()[0]).unwrap_or(b','))
        .has_headers(options.has_header)
        .from_reader(BufReader::new(file));
    
    // Get headers
    let headers = if options.has_header {
        reader.headers()?
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    } else {
        // Generate column names
        let record = reader.records().next()
            .ok_or_else(|| anyhow::anyhow!("Empty CSV file"))??;
        (0..record.len())
            .map(|i| format!("Column_{}", i + 1))
            .collect()
    };
    
    // Parse data
    let mut data_vec = Vec::new();
    let mut timestamps = Vec::new();
    let date_col_idx = options.date_column.as_ref()
        .and_then(|col| headers.iter().position(|h| h == col));
    
    for (row_idx, result) in reader.records().enumerate() {
        let record = result.context("Failed to read CSV record")?;
        let mut row = Vec::new();
        
        // Parse timestamp if date column specified
        if let Some(date_idx) = date_col_idx {
            if let Some(date_str) = record.get(date_idx) {
                // Try parsing various date formats
                let timestamp = NaiveDateTime::parse_from_str(date_str, "%Y-%m-%d %H:%M:%S")
                    .or_else(|_| NaiveDateTime::parse_from_str(date_str, "%Y-%m-%d"))
                    .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                    .unwrap_or_else(|_| Utc::now());
                timestamps.push(timestamp);
            }
        } else {
            // Use sequential timestamps
            timestamps.push(Utc::now() + chrono::Duration::hours(row_idx as i64));
        }
        
        // Parse numeric values
        for (col_idx, value) in record.iter().enumerate() {
            if Some(col_idx) == date_col_idx {
                continue; // Skip date column in data
            }
            
            let parsed_value = if value.is_empty() || 
                options.na_values.as_ref().map(|na| na.contains(&value.to_string())).unwrap_or(false) {
                f64::NAN
            } else {
                value.parse::<f64>().unwrap_or(f64::NAN)
            };
            
            row.push(parsed_value);
        }
        
        data_vec.push(row);
    }
    
    if data_vec.is_empty() {
        return Err(anyhow::anyhow!("No data found in CSV file"));
    }
    
    // Convert to ndarray
    let n_rows = data_vec.len();
    let n_cols = data_vec[0].len();
    let flat_data: Vec<f64> = data_vec.into_iter().flatten().collect();
    
    let data_array = Array2::from_shape_vec((n_rows, n_cols), flat_data)
        .context("Failed to create data array")?;
    
    // Filter columns if specified
    let (final_data, final_columns) = if let Some(use_cols) = &options.use_cols {
        let col_indices: Vec<usize> = use_cols.iter()
            .filter_map(|col| headers.iter().position(|h| h == col))
            .filter(|&idx| Some(idx) != date_col_idx)
            .collect();
        
        if col_indices.is_empty() {
            (data_array, headers)
        } else {
            let selected_data = data_array.select(ndarray::Axis(1), &col_indices);
            let selected_columns = col_indices.iter()
                .map(|&i| headers[i].clone())
                .collect();
            (selected_data, selected_columns)
        }
    } else {
        // Remove date column from headers if it was parsed
        let mut filtered_headers = headers.clone();
        if let Some(date_idx) = date_col_idx {
            filtered_headers.remove(date_idx);
        }
        (data_array, filtered_headers)
    };
    
    // Create dataset
    let dataset = Dataset::new(
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("dataset")
            .to_string(),
        final_data,
        final_columns,
        timestamps,
    );
    
    info!("Successfully loaded CSV: {} rows, {} columns", dataset.rows(), dataset.columns());
    Ok(dataset)
}

async fn load_excel(path: &PathBuf, options: &ImportOptions) -> Result<Dataset> {
    use calamine::{Reader, Xlsx, open_workbook};
    use chrono::NaiveDate;
    
    info!("Loading Excel from: {:?}", path);
    
    let mut workbook: Xlsx<_> = open_workbook(path)
        .context("Failed to open Excel file")?;
    
    // Get the first sheet or the one specified in options
    let sheet_name = workbook.sheet_names()[0].clone();
    let range = workbook.worksheet_range(&sheet_name)
        .ok_or_else(|| anyhow::anyhow!("Failed to get worksheet"))?
        .context("Failed to read worksheet")?;
    
    let mut data_vec = Vec::new();
    let mut timestamps = Vec::new();
    let mut headers = Vec::new();
    
    // Skip rows if specified
    let skip_rows = options.skip_rows.unwrap_or(0);
    let mut row_iter = range.rows().skip(skip_rows);
    
    // Parse headers
    if options.has_header {
        if let Some(header_row) = row_iter.next() {
            headers = header_row.iter()
                .map(|cell| cell.to_string())
                .collect();
        }
    } else {
        // Generate column names based on first row
        if let Some(first_row) = row_iter.next() {
            headers = (0..first_row.len())
                .map(|i| format!("Column_{}", i + 1))
                .collect();
        }
    }
    
    // Find date column index
    let date_col_idx = options.date_column.as_ref()
        .and_then(|col| headers.iter().position(|h| h == col));
    
    // Parse data rows
    for (row_idx, row) in row_iter.enumerate() {
        let mut data_row = Vec::new();
        let mut timestamp_found = false;
        
        for (col_idx, cell) in row.iter().enumerate() {
            // Handle date column
            if Some(col_idx) == date_col_idx {
                match cell {
                    calamine::DataType::DateTime(excel_date) => {
                        // Excel dates are days since 1900-01-01
                        let base_date = NaiveDate::from_ymd_opt(1900, 1, 1).unwrap();
                        let days = excel_date.floor() as i64;
                        let date = base_date + chrono::Duration::days(days - 2); // Excel leap year bug
                        timestamps.push(DateTime::<Utc>::from_naive_utc_and_offset(
                            date.and_hms_opt(0, 0, 0).unwrap(), 
                            Utc
                        ));
                        timestamp_found = true;
                    }
                    _ => {
                        // Try to parse as string date
                        let date_str = cell.to_string();
                        if let Ok(dt) = DateTime::parse_from_rfc3339(&date_str) {
                            timestamps.push(dt.with_timezone(&Utc));
                            timestamp_found = true;
                        }
                    }
                }
                continue;
            }
            
            // Parse numeric values
            let value = match cell {
                calamine::DataType::Int(i) => *i as f64,
                calamine::DataType::Float(f) => *f,
                calamine::DataType::String(s) | calamine::DataType::DateTimeIso(s) => {
                    if s.is_empty() || options.na_values.as_ref()
                        .map(|na| na.contains(s)).unwrap_or(false) {
                        f64::NAN
                    } else {
                        s.parse::<f64>().unwrap_or(f64::NAN)
                    }
                }
                calamine::DataType::Bool(b) => if *b { 1.0 } else { 0.0 },
                _ => f64::NAN,
            };
            
            data_row.push(value);
        }
        
        // Generate timestamp if not found
        if !timestamp_found {
            timestamps.push(Utc::now() + chrono::Duration::hours(row_idx as i64));
        }
        
        data_vec.push(data_row);
    }
    
    if data_vec.is_empty() {
        return Err(anyhow::anyhow!("No data found in Excel file"));
    }
    
    // Convert to ndarray
    let n_rows = data_vec.len();
    let n_cols = data_vec[0].len();
    let flat_data: Vec<f64> = data_vec.into_iter().flatten().collect();
    
    let data_array = Array2::from_shape_vec((n_rows, n_cols), flat_data)
        .context("Failed to create data array")?;
    
    // Filter columns if specified
    let (final_data, final_columns) = apply_column_filter(
        data_array, headers, &options.use_cols, date_col_idx
    );
    
    // Create dataset
    let dataset = Dataset::new(
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("excel_dataset")
            .to_string(),
        final_data,
        final_columns,
        timestamps,
    );
    
    info!("Successfully loaded Excel: {} rows, {} columns", dataset.rows(), dataset.columns());
    Ok(dataset)
}

#[cfg(feature = "netcdf-support")]
async fn load_netcdf(path: &PathBuf, options: &ImportOptions) -> Result<Dataset> {
    use netcdf;
    
    info!("Loading NetCDF from: {:?}", path);
    
    let file = netcdf::open(path)
        .context("Failed to open NetCDF file")?;
    
    // Get dimensions
    let time_dim = file.dimension("time")
        .ok_or_else(|| anyhow::anyhow!("No time dimension found"))?;
    let n_times = time_dim.len();
    
    // Get all variables
    let mut variables = Vec::new();
    let mut data_arrays = Vec::new();
    
    for var in file.variables() {
        let var_name = var.name();
        
        // Skip coordinate variables
        if var_name == "time" || var_name == "lat" || var_name == "lon" {
            continue;
        }
        
        // Check if we should include this variable
        if let Some(use_cols) = &options.use_cols {
            if !use_cols.contains(&var_name) {
                continue;
            }
        }
        
        variables.push(var_name.clone());
        
        // Read variable data
        let values: Vec<f64> = var.values::<f64>(None, None)?
            .into_iter()
            .collect();
        
        data_arrays.push(values);
    }
    
    if data_arrays.is_empty() {
        return Err(anyhow::anyhow!("No valid variables found in NetCDF file"));
    }
    
    // Read time variable
    let time_var = file.variable("time")
        .ok_or_else(|| anyhow::anyhow!("No time variable found"))?;
    
    let time_values: Vec<f64> = time_var.values::<f64>(None, None)?
        .into_iter()
        .collect();
    
    // Parse time units (e.g., "days since 1970-01-01")
    let time_units = time_var.attribute("units")
        .and_then(|attr| attr.value().ok())
        .and_then(|val| match val {
            netcdf::AttributeValue::Str(s) => Some(s),
            _ => None,
        })
        .unwrap_or_else(|| "days since 1970-01-01".to_string());
    
    // Convert time values to DateTime
    let timestamps = parse_netcdf_time(&time_values, &time_units)?;
    
    // Create data matrix
    let n_rows = n_times;
    let n_cols = variables.len();
    let mut data = Array2::<f64>::zeros((n_rows, n_cols));
    
    for (col_idx, var_data) in data_arrays.iter().enumerate() {
        for (row_idx, &value) in var_data.iter().enumerate() {
            if row_idx < n_rows {
                data[[row_idx, col_idx]] = value;
            }
        }
    }
    
    // Create dataset
    let dataset = Dataset::new(
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("netcdf_dataset")
            .to_string(),
        data,
        variables,
        timestamps,
    );
    
    info!("Successfully loaded NetCDF: {} rows, {} columns", dataset.rows(), dataset.columns());
    Ok(dataset)
}

#[cfg(feature = "hdf5-support")]
async fn load_hdf5(path: &PathBuf, options: &ImportOptions) -> Result<Dataset> {
    use hdf5;
    
    info!("Loading HDF5 from: {:?}", path);
    
    let file = hdf5::File::open(path)
        .context("Failed to open HDF5 file")?;
    
    // Find data group or datasets
    let mut data_arrays = Vec::new();
    let mut variables = Vec::new();
    let mut timestamps = Vec::new();
    
    // Look for standard group names
    let data_group = file.group("data")
        .or_else(|_| file.group("datasets"))
        .or_else(|_| file.group("/"))
        .context("Failed to find data group")?;
    
    // Read time/index dataset
    if let Ok(time_ds) = data_group.dataset("time") {
        let time_data: Vec<f64> = time_ds.read_1d()?
            .to_vec();
        
        // Convert to timestamps (assuming seconds since epoch)
        timestamps = time_data.iter()
            .map(|&t| DateTime::<Utc>::from_timestamp(t as i64, 0).unwrap())
            .collect();
    } else if let Ok(index_ds) = data_group.dataset("index") {
        let index_data: Vec<i64> = index_ds.read_1d()?
            .to_vec();
        
        // Generate timestamps from index
        timestamps = index_data.iter()
            .enumerate()
            .map(|(i, _)| Utc::now() + chrono::Duration::hours(i as i64))
            .collect();
    }
    
    // Read all numeric datasets
    for name in data_group.member_names()? {
        // Skip metadata datasets
        if name == "time" || name == "index" || name.starts_with("_") {
            continue;
        }
        
        // Check if we should include this variable
        if let Some(use_cols) = &options.use_cols {
            if !use_cols.contains(&name) {
                continue;
            }
        }
        
        if let Ok(dataset) = data_group.dataset(&name) {
            // Read data based on dimensionality
            match dataset.ndim() {
                1 => {
                    let data: Vec<f64> = dataset.read_1d()?
                        .to_vec();
                    data_arrays.push(data);
                    variables.push(name);
                }
                2 => {
                    // For 2D datasets, read each column
                    let data: Array2<f64> = dataset.read_2d()?;
                    for (col_idx, col) in data.axis_iter(Axis(1)).enumerate() {
                        data_arrays.push(col.to_vec());
                        variables.push(format!("{}_{}", name, col_idx));
                    }
                }
                _ => {
                    warn!("Skipping dataset {} with {} dimensions", name, dataset.ndim());
                }
            }
        }
    }
    
    if data_arrays.is_empty() {
        return Err(anyhow::anyhow!("No valid datasets found in HDF5 file"));
    }
    
    // Create data matrix
    let n_rows = data_arrays[0].len();
    let n_cols = data_arrays.len();
    let mut data = Array2::<f64>::zeros((n_rows, n_cols));
    
    for (col_idx, var_data) in data_arrays.iter().enumerate() {
        for (row_idx, &value) in var_data.iter().enumerate() {
            data[[row_idx, col_idx]] = value;
        }
    }
    
    // Generate timestamps if not found
    if timestamps.is_empty() {
        timestamps = (0..n_rows)
            .map(|i| Utc::now() + chrono::Duration::hours(i as i64))
            .collect();
    }
    
    // Create dataset
    let dataset = Dataset::new(
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("hdf5_dataset")
            .to_string(),
        data,
        variables,
        timestamps,
    );
    
    info!("Successfully loaded HDF5: {} rows, {} columns", dataset.rows(), dataset.columns());
    Ok(dataset)
}

// Temporarily disabled due to chrono version conflict with arrow/parquet crates
#[allow(dead_code)]
async fn load_parquet(path: &PathBuf, options: &ImportOptions) -> Result<Dataset> {
    Err(anyhow::anyhow!("Parquet support is temporarily disabled due to dependency conflicts"))
    /*
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::{Row, Field};
    use std::fs::File;
    
    info!("Loading Parquet from: {:?}", path);
    
    let file = File::open(path)
        .context("Failed to open Parquet file")?;
    
    let reader = SerializedFileReader::new(file)
        .context("Failed to create Parquet reader")?;
    
    let metadata = reader.metadata();
    let schema = metadata.file_metadata().schema();
    
    // Get column names
    let mut columns = Vec::new();
    let mut date_col_idx = None;
    
    for (idx, field) in schema.get_fields().iter().enumerate() {
        let col_name = field.name();
        
        if let Some(date_col) = &options.date_column {
            if col_name == date_col {
                date_col_idx = Some(idx);
                continue;
            }
        }
        
        // Check if we should include this column
        if let Some(use_cols) = &options.use_cols {
            if !use_cols.contains(&col_name.to_string()) {
                continue;
            }
        }
        
        columns.push(col_name.to_string());
    }
    
    // Read data
    let mut data_vec = Vec::new();
    let mut timestamps = Vec::new();
    let row_iter = reader.get_row_iter(None)?;
    
    // Skip rows if specified
    let skip_rows = options.skip_rows.unwrap_or(0);
    
    for (row_idx, row_result) in row_iter.enumerate().skip(skip_rows) {
        let row = row_result?;
        let mut data_row = Vec::new();
        let mut timestamp_found = false;
        
        for (col_idx, (_, field)) in row.get_column_iter().enumerate() {
            // Handle date column
            if Some(col_idx) == date_col_idx {
                match field {
                    Field::TimestampMillis(ts) => {
                        timestamps.push(Utc.timestamp_millis_opt(*ts).unwrap());
                        timestamp_found = true;
                    }
                    Field::Date(days) => {
                        let base_date = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                        let date = base_date + chrono::Duration::days(*days as i64);
                        timestamps.push(DateTime::<Utc>::from_naive_utc_and_offset(
                            date.and_hms_opt(0, 0, 0).unwrap(),
                            Utc
                        ));
                        timestamp_found = true;
                    }
                    Field::Str(s) => {
                        if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
                            timestamps.push(dt.with_timezone(&Utc));
                            timestamp_found = true;
                        }
                    }
                    _ => {}
                }
                continue;
            }
            
            // Parse numeric values
            let value = match field {
                Field::Null => f64::NAN,
                Field::Bool(b) => if *b { 1.0 } else { 0.0 },
                Field::Byte(b) => *b as f64,
                Field::Short(s) => *s as f64,
                Field::Int(i) => *i as f64,
                Field::Long(l) => *l as f64,
                Field::UByte(u) => *u as f64,
                Field::UShort(u) => *u as f64,
                Field::UInt(u) => *u as f64,
                Field::ULong(u) => *u as f64,
                Field::Float(f) => *f as f64,
                Field::Double(d) => *d,
                Field::Decimal(d) => d.to_string().parse::<f64>().unwrap_or(f64::NAN),
                Field::Str(s) => {
                    if s.is_empty() || options.na_values.as_ref()
                        .map(|na| na.contains(s)).unwrap_or(false) {
                        f64::NAN
                    } else {
                        s.parse::<f64>().unwrap_or(f64::NAN)
                    }
                }
                _ => f64::NAN,
            };
            
            data_row.push(value);
        }
        
        // Generate timestamp if not found
        if !timestamp_found {
            timestamps.push(Utc::now() + chrono::Duration::hours(row_idx as i64));
        }
        
        data_vec.push(data_row);
    }
    
    if data_vec.is_empty() {
        return Err(anyhow::anyhow!("No data found in Parquet file"));
    }
    
    // Convert to ndarray
    let n_rows = data_vec.len();
    let n_cols = columns.len();
    let flat_data: Vec<f64> = data_vec.into_iter().flatten().collect();
    
    let data = Array2::from_shape_vec((n_rows, n_cols), flat_data)
        .context("Failed to create data array")?;
    
    // Create dataset
    let dataset = Dataset::new(
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("parquet_dataset")
            .to_string(),
        data,
        columns,
        timestamps,
    );
    
    info!("Successfully loaded Parquet: {} rows, {} columns", dataset.rows(), dataset.columns());
    Ok(dataset)
    */
}

async fn load_json(path: &PathBuf, options: &ImportOptions) -> Result<Dataset> {
    use std::fs::File;
    use std::io::BufReader;
    
    info!("Loading JSON from: {:?}", path);
    
    let file = File::open(path)
        .context("Failed to open JSON file")?;
    let reader = BufReader::new(file);
    
    // Parse JSON
    let json_value: serde_json::Value = serde_json::from_reader(reader)
        .context("Failed to parse JSON")?;
    
    // Handle different JSON structures
    let (data_vec, columns, timestamps) = match &json_value {
        // Array of objects (most common for time series)
        serde_json::Value::Array(arr) => parse_json_array(arr, options)?,
        
        // Object with data array
        serde_json::Value::Object(obj) => {
            if let Some(data_array) = obj.get("data").and_then(|v| v.as_array()) {
                parse_json_array(data_array, options)?
            } else {
                // Try to parse as column-oriented data
                parse_json_columns(obj, options)?
            }
        }
        
        _ => return Err(anyhow::anyhow!("Unsupported JSON structure")),
    };
    
    if data_vec.is_empty() {
        return Err(anyhow::anyhow!("No data found in JSON file"));
    }
    
    // Convert to ndarray
    let n_rows = data_vec.len();
    let n_cols = columns.len();
    let flat_data: Vec<f64> = data_vec.into_iter().flatten().collect();
    
    let data = Array2::from_shape_vec((n_rows, n_cols), flat_data)
        .context("Failed to create data array")?;
    
    // Create dataset
    let dataset = Dataset::new(
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("json_dataset")
            .to_string(),
        data,
        columns,
        timestamps,
    );
    
    info!("Successfully loaded JSON: {} rows, {} columns", dataset.rows(), dataset.columns());
    Ok(dataset)
}

// Helper functions for data loading

fn apply_column_filter(
    data: Array2<f64>,
    headers: Vec<String>,
    use_cols: &Option<Vec<String>>,
    date_col_idx: Option<usize>,
) -> (Array2<f64>, Vec<String>) {
    if let Some(cols) = use_cols {
        let col_indices: Vec<usize> = cols.iter()
            .filter_map(|col| headers.iter().position(|h| h == col))
            .filter(|&idx| Some(idx) != date_col_idx)
            .collect();
        
        if !col_indices.is_empty() {
            let selected_data = data.select(Axis(1), &col_indices);
            let selected_columns = col_indices.iter()
                .map(|&i| headers[i].clone())
                .collect();
            return (selected_data, selected_columns);
        }
    }
    
    // Remove date column from headers if it was parsed
    let mut filtered_headers = headers.clone();
    if let Some(date_idx) = date_col_idx {
        filtered_headers.remove(date_idx);
    }
    (data, filtered_headers)
}

fn parse_netcdf_time(time_values: &[f64], units: &str) -> Result<Vec<DateTime<Utc>>> {
    // Parse units string (e.g., "days since 1970-01-01")
    let parts: Vec<&str> = units.split_whitespace().collect();
    if parts.len() < 3 || parts[1] != "since" {
        return Err(anyhow::anyhow!("Invalid time units format"));
    }
    
    let time_unit = parts[0];
    let base_date_str = parts[2..].join(" ");
    
    // Parse base date
    let base_date = chrono::NaiveDateTime::parse_from_str(&base_date_str, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| {
            chrono::NaiveDate::parse_from_str(&base_date_str, "%Y-%m-%d")
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
        })
        .context("Failed to parse base date")?;
    
    let base_dt = DateTime::<Utc>::from_naive_utc_and_offset(base_date, Utc);
    
    // Convert time values
    let timestamps = time_values.iter()
        .map(|&value| {
            let duration = match time_unit {
                "seconds" | "second" => chrono::Duration::seconds(value as i64),
                "minutes" | "minute" => chrono::Duration::minutes(value as i64),
                "hours" | "hour" => chrono::Duration::hours(value as i64),
                "days" | "day" => chrono::Duration::days(value as i64),
                _ => return Err(anyhow::anyhow!("Unsupported time unit: {}", time_unit)),
            };
            Ok(base_dt + duration)
        })
        .collect::<Result<Vec<_>>>()?;
    
    Ok(timestamps)
}

fn parse_json_array(
    arr: &[serde_json::Value],
    options: &ImportOptions,
) -> Result<(Vec<Vec<f64>>, Vec<String>, Vec<DateTime<Utc>>)> {
    if arr.is_empty() {
        return Ok((vec![], vec![], vec![]));
    }
    
    // Get column names from first object
    let first_obj = arr[0].as_object()
        .ok_or_else(|| anyhow::anyhow!("Expected array of objects"))?;
    
    let mut columns = Vec::new();
    let mut date_col = None;
    
    for (key, _) in first_obj {
        if let Some(date_col_name) = &options.date_column {
            if key == date_col_name {
                date_col = Some(key.clone());
                continue;
            }
        }
        
        if let Some(use_cols) = &options.use_cols {
            if !use_cols.contains(key) {
                continue;
            }
        }
        
        columns.push(key.clone());
    }
    
    // Parse data
    let mut data_vec = Vec::new();
    let mut timestamps = Vec::new();
    
    for (idx, obj_val) in arr.iter().enumerate() {
        let obj = obj_val.as_object()
            .ok_or_else(|| anyhow::anyhow!("Expected object in array"))?;
        
        let mut row = Vec::new();
        
        // Parse timestamp if date column specified
        if let Some(date_key) = &date_col {
            if let Some(date_val) = obj.get(date_key) {
                match date_val {
                    serde_json::Value::String(s) => {
                        if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
                            timestamps.push(dt.with_timezone(&Utc));
                        } else if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
                            timestamps.push(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
                        } else {
                            timestamps.push(Utc::now() + chrono::Duration::hours(idx as i64));
                        }
                    }
                    serde_json::Value::Number(n) => {
                        let ts = n.as_i64().unwrap_or(0);
                        timestamps.push(DateTime::<Utc>::from_timestamp(ts, 0).unwrap());
                    }
                    _ => timestamps.push(Utc::now() + chrono::Duration::hours(idx as i64)),
                }
            } else {
                timestamps.push(Utc::now() + chrono::Duration::hours(idx as i64));
            }
        } else {
            timestamps.push(Utc::now() + chrono::Duration::hours(idx as i64));
        }
        
        // Parse numeric values
        for col in &columns {
            let value = obj.get(col)
                .and_then(|v| match v {
                    serde_json::Value::Number(n) => n.as_f64(),
                    serde_json::Value::String(s) => {
                        if s.is_empty() || options.na_values.as_ref()
                            .map(|na| na.contains(s)).unwrap_or(false) {
                            Some(f64::NAN)
                        } else {
                            s.parse::<f64>().ok()
                        }
                    }
                    serde_json::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
                    serde_json::Value::Null => Some(f64::NAN),
                    _ => None,
                })
                .unwrap_or(f64::NAN);
            
            row.push(value);
        }
        
        data_vec.push(row);
    }
    
    Ok((data_vec, columns, timestamps))
}

fn parse_json_columns(
    obj: &serde_json::Map<String, serde_json::Value>,
    options: &ImportOptions,
) -> Result<(Vec<Vec<f64>>, Vec<String>, Vec<DateTime<Utc>>)> {
    let mut columns = Vec::new();
    let mut column_data = HashMap::new();
    let mut timestamps = Vec::new();
    let mut n_rows = 0;
    
    // Parse each column
    for (key, value) in obj {
        if let Some(date_col) = &options.date_column {
            if key == date_col {
                // Parse timestamps
                if let Some(arr) = value.as_array() {
                    timestamps = arr.iter()
                        .enumerate()
                        .map(|(idx, v)| {
                            match v {
                                serde_json::Value::String(s) => {
                                    DateTime::parse_from_rfc3339(s)
                                        .map(|dt| dt.with_timezone(&Utc))
                                        .unwrap_or_else(|_| Utc::now() + chrono::Duration::hours(idx as i64))
                                }
                                serde_json::Value::Number(n) => {
                                    let ts = n.as_i64().unwrap_or(0);
                                    DateTime::<Utc>::from_timestamp(ts, 0).unwrap()
                                }
                                _ => Utc::now() + chrono::Duration::hours(idx as i64),
                            }
                        })
                        .collect();
                }
                continue;
            }
        }
        
        if let Some(use_cols) = &options.use_cols {
            if !use_cols.contains(key) {
                continue;
            }
        }
        
        if let Some(arr) = value.as_array() {
            n_rows = arr.len();
            let data: Vec<f64> = arr.iter()
                .map(|v| match v {
                    serde_json::Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
                    serde_json::Value::String(s) => s.parse::<f64>().unwrap_or(f64::NAN),
                    serde_json::Value::Bool(b) => if *b { 1.0 } else { 0.0 },
                    _ => f64::NAN,
                })
                .collect();
            
            columns.push(key.clone());
            column_data.insert(key.clone(), data);
        }
    }
    
    // Generate timestamps if not found
    if timestamps.is_empty() {
        timestamps = (0..n_rows)
            .map(|i| Utc::now() + chrono::Duration::hours(i as i64))
            .collect();
    }
    
    // Convert to row-oriented format
    let mut data_vec = Vec::new();
    for i in 0..n_rows {
        let mut row = Vec::new();
        for col in &columns {
            if let Some(col_data) = column_data.get(col) {
                row.push(col_data.get(i).copied().unwrap_or(f64::NAN));
            } else {
                row.push(f64::NAN);
            }
        }
        data_vec.push(row);
    }
    
    Ok((data_vec, columns, timestamps))
}

pub async fn save_csv(dataset: &Dataset, path: &PathBuf) -> Result<()> {
    use csv::Writer;
    use std::fs::File;
    
    info!("Saving dataset to CSV: {:?}", path);
    
    let file = File::create(path)
        .context("Failed to create CSV file")?;
    let mut writer = Writer::from_writer(file);
    
    // Write headers
    let mut headers = vec!["timestamp".to_string()];
    headers.extend(dataset.columns.clone());
    writer.write_record(&headers)?;
    
    // Write data
    for (i, timestamp) in dataset.index.iter().enumerate() {
        let mut record = vec![timestamp.to_rfc3339()];
        
        for j in 0..dataset.columns() {
            let value = dataset.data[[i, j]];
            record.push(if value.is_nan() {
                String::new()
            } else {
                value.to_string()
            });
        }
        
        writer.write_record(&record)?;
    }
    
    writer.flush()?;
    info!("Successfully saved {} rows to CSV", dataset.rows());
    Ok(())
}

pub async fn save_excel(dataset: &Dataset, path: &PathBuf) -> Result<()> {
    use xlsxwriter::Workbook;
    use xlsxwriter::prelude::{FormatColor, FormatBorder};
    use tracing::debug;
    
    info!("Saving dataset to Excel: {:?}", path);
    
    // Create workbook
    let workbook = Workbook::new(path.to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid path for Excel file"))?
    ).context("Failed to create Excel workbook")?;
    
    // Create worksheet
    let mut sheet = workbook.add_worksheet(Some(&dataset.name))
        .context("Failed to add worksheet")?;
    
    // Create formats
    let mut header_format = xlsxwriter::Format::new();
    header_format
        .set_bold()
        .set_bg_color(FormatColor::Custom(0xE0E0E0))
        .set_border(FormatBorder::Thin);
    
    let mut date_format = xlsxwriter::Format::new();
    date_format
        .set_num_format("yyyy-mm-dd hh:mm:ss")
        .set_border(FormatBorder::Thin);
    
    let mut number_format = xlsxwriter::Format::new();
    number_format
        .set_num_format("0.00000")
        .set_border(FormatBorder::Thin);
    
    let mut na_format = xlsxwriter::Format::new();
    na_format
        .set_font_color(FormatColor::Red)
        .set_italic()
        .set_border(FormatBorder::Thin);
    
    // Write headers
    sheet.write_string(0, 0, "Timestamp", Some(&header_format))
        .context("Failed to write timestamp header")?;
    
    for (col_idx, col_name) in dataset.columns.iter().enumerate() {
        sheet.write_string(0, (col_idx + 1) as u16, col_name, Some(&header_format))
            .context("Failed to write column header")?;
    }
    
    // Write data with progress tracking
    let total_rows = dataset.rows();
    let progress_interval = (total_rows / 100).max(1);
    
    for (row_idx, timestamp) in dataset.index.iter().enumerate() {
        // Progress tracking
        if row_idx % progress_interval == 0 {
            debug!("Excel save progress: {:.1}%", 
                (row_idx as f64 / total_rows as f64) * 100.0);
        }
        
        // Write timestamp
        let excel_row = (row_idx + 1) as u32;
        let excel_date = excel_datetime(timestamp);
        sheet.write_number(excel_row, 0, excel_date, Some(&date_format))
            .context("Failed to write timestamp")?;
        
        // Write data values
        for col_idx in 0..dataset.columns() {
            let value = dataset.data[[row_idx, col_idx]];
            let excel_col = (col_idx + 1) as u16;
            
            if value.is_nan() {
                sheet.write_string(excel_row, excel_col, "NA", Some(&na_format))
                    .context("Failed to write NA value")?;
            } else {
                sheet.write_number(excel_row, excel_col, value, Some(&number_format))
                    .context("Failed to write numeric value")?;
            }
        }
    }
    
    // Auto-fit columns
    for col_idx in 0..=dataset.columns() {
        sheet.set_column(col_idx as u16, col_idx as u16, 15.0, None)
            .context("Failed to set column width")?;
    }
    
    // Add metadata sheet
    let mut meta_sheet = workbook.add_worksheet(Some("Metadata"))
        .context("Failed to add metadata worksheet")?;
    
    meta_sheet.write_string(0, 0, "Property", Some(&header_format))?;
    meta_sheet.write_string(0, 1, "Value", Some(&header_format))?;
    
    let metadata = vec![
        ("Dataset Name", dataset.name.clone()),
        ("Total Rows", dataset.rows().to_string()),
        ("Total Columns", dataset.columns().to_string()),
        ("Date Range", format!("{} to {}", 
            dataset.index.first().map(|d| d.to_rfc3339()).unwrap_or_default(),
            dataset.index.last().map(|d| d.to_rfc3339()).unwrap_or_default()
        )),
        ("Missing Values", dataset.count_missing().to_string()),
        ("Missing Percentage", format!("{:.2}%", 
            (dataset.count_missing() as f64 / (dataset.rows() * dataset.columns()) as f64) * 100.0
        )),
        ("Created At", dataset.created_at.to_rfc3339()),
        ("Modified At", dataset.modified_at.to_rfc3339()),
    ];
    
    for (row_idx, (key, value)) in metadata.iter().enumerate() {
        let row = (row_idx + 1) as u32;
        meta_sheet.write_string(row, 0, key, None)?;
        meta_sheet.write_string(row, 1, value, None)?;
    }
    
    // Write variable information
    if !dataset.variables.is_empty() {
        meta_sheet.write_string(metadata.len() as u32 + 3, 0, "Variables", Some(&header_format))?;
        meta_sheet.write_string(metadata.len() as u32 + 4, 0, "Name", Some(&header_format))?;
        meta_sheet.write_string(metadata.len() as u32 + 4, 1, "Unit", Some(&header_format))?;
        meta_sheet.write_string(metadata.len() as u32 + 4, 2, "Description", Some(&header_format))?;
        meta_sheet.write_string(metadata.len() as u32 + 4, 3, "Min Valid", Some(&header_format))?;
        meta_sheet.write_string(metadata.len() as u32 + 4, 4, "Max Valid", Some(&header_format))?;
        
        for (idx, var) in dataset.variables.iter().enumerate() {
            let row = (metadata.len() + 5 + idx) as u32;
            meta_sheet.write_string(row, 0, &var.name, None)?;
            meta_sheet.write_string(row, 1, &var.unit, None)?;
            meta_sheet.write_string(row, 2, &var.description, None)?;
            if let Some(min) = var.min_valid {
                meta_sheet.write_number(row, 3, min, None)?;
            }
            if let Some(max) = var.max_valid {
                meta_sheet.write_number(row, 4, max, None)?;
            }
        }
    }
    
    // Close workbook
    workbook.close()
        .context("Failed to close Excel workbook")?;
    
    info!("Successfully saved {} rows to Excel", dataset.rows());
    Ok(())
}

/// Convert DateTime to Excel serial date
fn excel_datetime(dt: &DateTime<Utc>) -> f64 {
    // Excel uses days since 1900-01-01
    let base = chrono::NaiveDate::from_ymd_opt(1900, 1, 1).unwrap()
        .and_hms_opt(0, 0, 0).unwrap();
    let naive = dt.naive_utc();
    let duration = naive.signed_duration_since(base);
    // Add 2 for Excel's leap year bug (1900 is not a leap year)
    duration.num_seconds() as f64 / 86400.0 + 2.0
}

#[cfg(feature = "netcdf-support")]
async fn save_netcdf(dataset: &Dataset, path: &PathBuf) -> Result<()> {
    use netcdf::{self, Dimension};
    
    info!("Saving dataset to NetCDF: {:?}", path);
    
    // Create NetCDF file
    let mut file = netcdf::create(path)
        .context("Failed to create NetCDF file")?;
    
    // Add dimensions
    let time_dim = file.add_dimension("time", dataset.rows())
        .context("Failed to add time dimension")?;
    
    let station_dim = if let Some(stations) = &dataset.stations {
        Some(file.add_dimension("station", stations.len())
            .context("Failed to add station dimension")?)
    } else {
        None
    };
    
    // Add global attributes
    file.add_attribute("title", &dataset.name)
        .context("Failed to add title attribute")?;
    
    file.add_attribute("Conventions", "CF-1.8")
        .context("Failed to add conventions attribute")?;
    
    file.add_attribute("created_at", dataset.created_at.to_rfc3339().as_str())
        .context("Failed to add created_at attribute")?;
    
    file.add_attribute("missing_value", f64::NAN)
        .context("Failed to add missing_value attribute")?;
    
    // Add time variable
    let mut time_var = file.add_variable::<f64>("time", &[&time_dim])
        .context("Failed to add time variable")?;
    
    time_var.add_attribute("units", "seconds since 1970-01-01 00:00:00")
        .context("Failed to add time units")?;
    
    time_var.add_attribute("calendar", "proleptic_gregorian")
        .context("Failed to add calendar attribute")?;
    
    time_var.add_attribute("standard_name", "time")
        .context("Failed to add standard_name attribute")?;
    
    // Write time values
    let time_values: Vec<f64> = dataset.index.iter()
        .map(|dt| dt.timestamp() as f64)
        .collect();
    
    time_var.put_values(&time_values, None, None)
        .context("Failed to write time values")?;
    
    // Add station variables if spatial data
    if let Some(stations) = &dataset.stations {
        if let Some(station_dim) = &station_dim {
            // Latitude
            let mut lat_var = file.add_variable::<f64>("lat", &[station_dim])
                .context("Failed to add latitude variable")?;
            
            lat_var.add_attribute("units", "degrees_north")
                .context("Failed to add lat units")?;
            
            lat_var.add_attribute("standard_name", "latitude")
                .context("Failed to add lat standard_name")?;
            
            let lat_values: Vec<f64> = stations.iter()
                .map(|s| s.latitude)
                .collect();
            
            lat_var.put_values(&lat_values, None, None)
                .context("Failed to write latitude values")?;
            
            // Longitude
            let mut lon_var = file.add_variable::<f64>("lon", &[station_dim])
                .context("Failed to add longitude variable")?;
            
            lon_var.add_attribute("units", "degrees_east")
                .context("Failed to add lon units")?;
            
            lon_var.add_attribute("standard_name", "longitude")
                .context("Failed to add lon standard_name")?;
            
            let lon_values: Vec<f64> = stations.iter()
                .map(|s| s.longitude)
                .collect();
            
            lon_var.put_values(&lon_values, None, None)
                .context("Failed to write longitude values")?;
            
            // Station names
            let mut name_var = file.add_string_variable("station_name", &[station_dim])
                .context("Failed to add station_name variable")?;
            
            name_var.add_attribute("long_name", "station name")
                .context("Failed to add station_name attribute")?;
            
            for (i, station) in stations.iter().enumerate() {
                name_var.put_string(&station.name, Some(&[i]), None)
                    .context("Failed to write station name")?;
            }
        }
    }
    
    // Add data variables
    for (col_idx, (col_name, variable)) in dataset.columns.iter()
        .zip(dataset.variables.iter())
        .enumerate() {
        
        let dims = if station_dim.is_some() {
            vec![&time_dim, station_dim.as_ref().unwrap()]
        } else {
            vec![&time_dim]
        };
        
        let mut var = file.add_variable::<f64>(col_name, &dims)
            .context(format!("Failed to add variable {}", col_name))?;
        
        // Add variable attributes
        var.add_attribute("units", variable.unit.as_str())
            .context("Failed to add units attribute")?;
        
        if !variable.description.is_empty() {
            var.add_attribute("long_name", variable.description.as_str())
                .context("Failed to add long_name attribute")?;
        }
        
        if let Some(min) = variable.min_valid {
            var.add_attribute("valid_min", min)
                .context("Failed to add valid_min attribute")?;
        }
        
        if let Some(max) = variable.max_valid {
            var.add_attribute("valid_max", max)
                .context("Failed to add valid_max attribute")?;
        }
        
        var.add_attribute("_FillValue", f64::NAN)
            .context("Failed to add _FillValue attribute")?;
        
        // Extract column data
        let col_data: Vec<f64> = dataset.data.column(col_idx)
            .iter()
            .copied()
            .collect();
        
        // Write data
        var.put_values(&col_data, None, None)
            .context(format!("Failed to write data for {}", col_name))?;
    }
    
    // Add quality flags if available
    if let Some(quality_flags) = &dataset.quality_flags {
        let mut qc_var = file.add_variable::<u8>("quality_flag", &[&time_dim])
            .context("Failed to add quality_flag variable")?;
        
        qc_var.add_attribute("long_name", "quality control flags")
            .context("Failed to add qc long_name")?;
        
        qc_var.add_attribute("flag_values", vec![0u8, 1, 2, 3, 4, 5, 6, 7])
            .context("Failed to add flag_values")?;
        
        qc_var.add_attribute("flag_meanings", 
            "valid missing below_detection above_detection suspect invalid interpolated imputed")
            .context("Failed to add flag_meanings")?;
        
        // Write quality flags (simplified - would need proper conversion)
        let qc_values: Vec<u8> = vec![0u8; dataset.rows()];
        qc_var.put_values(&qc_values, None, None)
            .context("Failed to write quality flags")?;
    }
    
    info!("Successfully saved {} rows to NetCDF", dataset.rows());
    Ok(())
}

#[cfg(feature = "hdf5-support")]
async fn save_hdf5(dataset: &Dataset, path: &PathBuf) -> Result<()> {
    use hdf5;
    use ndarray::s;
    
    info!("Saving dataset to HDF5: {:?}", path);
    
    // Create HDF5 file
    let file = hdf5::File::create(path)
        .context("Failed to create HDF5 file")?;
    
    // Create root group
    let root = file.group("/")
        .context("Failed to get root group")?;
    
    // Add file attributes
    root.new_attr::<&str>()
        .create("title")
        .context("Failed to create title attribute")?
        .write_scalar(&dataset.name.as_str())
        .context("Failed to write title")?;
    
    root.new_attr::<&str>()
        .create("created_at")
        .context("Failed to create created_at attribute")?
        .write_scalar(&dataset.created_at.to_rfc3339().as_str())
        .context("Failed to write created_at")?;
    
    root.new_attr::<&str>()
        .create("format_version")
        .context("Failed to create format_version attribute")?
        .write_scalar(&"1.0")
        .context("Failed to write format_version")?;
    
    // Create data group
    let data_group = file.create_group("data")
        .context("Failed to create data group")?;
    
    // Save time index
    let time_values: Vec<i64> = dataset.index.iter()
        .map(|dt| dt.timestamp())
        .collect();
    
    let time_ds = data_group.new_dataset::<i64>()
        .shape([dataset.rows()])
        .create("time")
        .context("Failed to create time dataset")?;
    
    time_ds.write(&time_values)
        .context("Failed to write time values")?;
    
    time_ds.new_attr::<&str>()
        .create("units")
        .context("Failed to create time units attribute")?
        .write_scalar(&"seconds since 1970-01-01 00:00:00")
        .context("Failed to write time units")?;
    
    // Save data matrix as individual columns for better accessibility
    for (col_idx, col_name) in dataset.columns.iter().enumerate() {
        let col_data: Vec<f64> = dataset.data.column(col_idx)
            .iter()
            .copied()
            .collect();
        
        let col_ds = data_group.new_dataset::<f64>()
            .shape([dataset.rows()])
            .create(col_name)
            .context(format!("Failed to create dataset for {}", col_name))?;
        
        col_ds.write(&col_data)
            .context(format!("Failed to write data for {}", col_name))?;
        
        // Add variable metadata if available
        if let Some(var) = dataset.variables.iter().find(|v| &v.name == col_name) {
            col_ds.new_attr::<&str>()
                .create("units")
                .context("Failed to create units attribute")?
                .write_scalar(&var.unit.as_str())
                .context("Failed to write units")?;
            
            if !var.description.is_empty() {
                col_ds.new_attr::<&str>()
                    .create("description")
                    .context("Failed to create description attribute")?
                    .write_scalar(&var.description.as_str())
                    .context("Failed to write description")?;
            }
            
            if let Some(min) = var.min_valid {
                col_ds.new_attr::<f64>()
                    .create("valid_min")
                    .context("Failed to create valid_min attribute")?
                    .write_scalar(&min)
                    .context("Failed to write valid_min")?;
            }
            
            if let Some(max) = var.max_valid {
                col_ds.new_attr::<f64>()
                    .create("valid_max")
                    .context("Failed to create valid_max attribute")?
                    .write_scalar(&max)
                    .context("Failed to write valid_max")?;
            }
        }
    }
    
    // Save full data matrix as well for compatibility
    let full_data_ds = data_group.new_dataset::<f64>()
        .shape([dataset.rows(), dataset.columns()])
        .create("matrix")
        .context("Failed to create matrix dataset")?;
    
    full_data_ds.write(&dataset.data)
        .context("Failed to write data matrix")?;
    
    // Create metadata group
    let meta_group = file.create_group("metadata")
        .context("Failed to create metadata group")?;
    
    // Save column names
    let col_names_ds = meta_group.new_dataset::<hdf5::types::VarLenUnicode>()
        .shape([dataset.columns.len()])
        .create("column_names")
        .context("Failed to create column_names dataset")?;
    
    let col_names_vlen: Vec<hdf5::types::VarLenUnicode> = dataset.columns.iter()
        .map(|s| hdf5::types::VarLenUnicode::from_str(s).unwrap())
        .collect();
    
    col_names_ds.write(&col_names_vlen)
        .context("Failed to write column names")?;
    
    // Save statistics
    let stats_group = meta_group.create_group("statistics")
        .context("Failed to create statistics group")?;
    
    stats_group.new_attr::<usize>()
        .create("n_rows")
        .context("Failed to create n_rows attribute")?
        .write_scalar(&dataset.rows())
        .context("Failed to write n_rows")?;
    
    stats_group.new_attr::<usize>()
        .create("n_columns")
        .context("Failed to create n_columns attribute")?
        .write_scalar(&dataset.columns())
        .context("Failed to write n_columns")?;
    
    stats_group.new_attr::<usize>()
        .create("missing_values")
        .context("Failed to create missing_values attribute")?
        .write_scalar(&dataset.count_missing())
        .context("Failed to write missing_values")?;
    
    let missing_pct = (dataset.count_missing() as f64 / 
        (dataset.rows() * dataset.columns()) as f64) * 100.0;
    
    stats_group.new_attr::<f64>()
        .create("missing_percentage")
        .context("Failed to create missing_percentage attribute")?
        .write_scalar(&missing_pct)
        .context("Failed to write missing_percentage")?;
    
    // Save spatial information if available
    if let Some(stations) = &dataset.stations {
        let spatial_group = file.create_group("spatial")
            .context("Failed to create spatial group")?;
        
        // Station coordinates
        let lat_values: Vec<f64> = stations.iter().map(|s| s.latitude).collect();
        let lon_values: Vec<f64> = stations.iter().map(|s| s.longitude).collect();
        
        let lat_ds = spatial_group.new_dataset::<f64>()
            .shape([stations.len()])
            .create("latitude")
            .context("Failed to create latitude dataset")?;
        
        lat_ds.write(&lat_values)
            .context("Failed to write latitude values")?;
        
        let lon_ds = spatial_group.new_dataset::<f64>()
            .shape([stations.len()])
            .create("longitude")
            .context("Failed to create longitude dataset")?;
        
        lon_ds.write(&lon_values)
            .context("Failed to write longitude values")?;
        
        // Station names
        let names_ds = spatial_group.new_dataset::<hdf5::types::VarLenUnicode>()
            .shape([stations.len()])
            .create("station_names")
            .context("Failed to create station_names dataset")?;
        
        let station_names_vlen: Vec<hdf5::types::VarLenUnicode> = stations.iter()
            .map(|s| hdf5::types::VarLenUnicode::from_str(&s.name).unwrap())
            .collect();
        
        names_ds.write(&station_names_vlen)
            .context("Failed to write station names")?;
        
        // Station IDs
        let ids_ds = spatial_group.new_dataset::<hdf5::types::VarLenUnicode>()
            .shape([stations.len()])
            .create("station_ids")
            .context("Failed to create station_ids dataset")?;
        
        let station_ids_vlen: Vec<hdf5::types::VarLenUnicode> = stations.iter()
            .map(|s| hdf5::types::VarLenUnicode::from_str(&s.id).unwrap())
            .collect();
        
        ids_ds.write(&station_ids_vlen)
            .context("Failed to write station IDs")?;
        
        // Altitude if available
        let alt_values: Vec<f64> = stations.iter()
            .map(|s| s.altitude.unwrap_or(f64::NAN))
            .collect();
        
        let alt_ds = spatial_group.new_dataset::<f64>()
            .shape([stations.len()])
            .create("altitude")
            .context("Failed to create altitude dataset")?;
        
        alt_ds.write(&alt_values)
            .context("Failed to write altitude values")?;
    }
    
    info!("Successfully saved {} rows to HDF5", dataset.rows());
    Ok(())
}

// Temporarily disabled due to chrono version conflict with arrow/parquet crates
#[allow(dead_code)]
async fn save_parquet(dataset: &Dataset, path: &PathBuf) -> Result<()> {
    Err(anyhow::anyhow!("Parquet support is temporarily disabled due to dependency conflicts"))
    /*
    use parquet::{
        file::{
            properties::WriterProperties,
            writer::{FileWriter, SerializedFileWriter},
        },
        schema::types::Type,
        data::RecordBatch,
        record::RecordWriter,
    };
    use parquet::basic::{Compression, Encoding};
    use arrow::{
        array::{Float64Array, TimestampSecondArray, StringArray},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch as ArrowRecordBatch,
    };
    use std::sync::Arc;
    use std::fs::File;
    
    info!("Saving dataset to Parquet: {:?}", path);
    
    // Create Arrow schema
    let mut fields = vec![
        Field::new("timestamp", DataType::Timestamp(TimeUnit::Second, None), false),
    ];
    
    for col_name in &dataset.columns {
        fields.push(Field::new(col_name, DataType::Float64, true));
    }
    
    let schema = Arc::new(Schema::new(fields));
    
    // Convert data to Arrow arrays
    let timestamp_array = TimestampSecondArray::from_iter_values(
        dataset.index.iter().map(|dt| dt.timestamp())
    );
    
    let mut arrays: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new(timestamp_array),
    ];
    
    // Add data columns
    for col_idx in 0..dataset.columns() {
        let col_data: Vec<Option<f64>> = dataset.data.column(col_idx)
            .iter()
            .map(|&v| if v.is_nan() { None } else { Some(v) })
            .collect();
        
        let array = Float64Array::from(col_data);
        arrays.push(Arc::new(array));
    }
    
    // Create record batch
    let batch = ArrowRecordBatch::try_new(schema.clone(), arrays)
        .context("Failed to create Arrow record batch")?;
    
    // Configure writer properties
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_encoding(Encoding::PLAIN)
        .set_statistics_enabled(true)
        .build();
    
    // Write to file using Arrow
    let file = File::create(path)
        .context("Failed to create Parquet file")?;
    
    let mut writer = arrow::parquet::arrow::ArrowWriter::try_new(
        file,
        schema.clone(),
        Some(props),
    ).context("Failed to create Arrow writer")?;
    
    writer.write(&batch)
        .context("Failed to write record batch")?;
    
    // Add metadata
    let mut metadata = HashMap::new();
    metadata.insert("dataset_name".to_string(), dataset.name.clone());
    metadata.insert("created_at".to_string(), dataset.created_at.to_rfc3339());
    metadata.insert("rows".to_string(), dataset.rows().to_string());
    metadata.insert("columns".to_string(), dataset.columns().to_string());
    metadata.insert("missing_values".to_string(), dataset.count_missing().to_string());
    
    // Write variable metadata as JSON
    if !dataset.variables.is_empty() {
        let var_metadata = dataset.variables.iter()
            .map(|v| {
                json!({
                    "name": v.name,
                    "unit": v.unit,
                    "description": v.description,
                    "min_valid": v.min_valid,
                    "max_valid": v.max_valid,
                })
            })
            .collect::<Vec<_>>();
        
        metadata.insert(
            "variables".to_string(),
            serde_json::to_string(&var_metadata).unwrap_or_default()
        );
    }
    
    // Add station metadata if available
    if let Some(stations) = &dataset.stations {
        let station_metadata = stations.iter()
            .map(|s| {
                json!({
                    "id": s.id,
                    "name": s.name,
                    "latitude": s.latitude,
                    "longitude": s.longitude,
                    "altitude": s.altitude,
                })
            })
            .collect::<Vec<_>>();
        
        metadata.insert(
            "stations".to_string(),
            serde_json::to_string(&station_metadata).unwrap_or_default()
        );
    }
    
    writer.close()
        .context("Failed to close Parquet writer")?;
    
    info!("Successfully saved {} rows to Parquet", dataset.rows());
    Ok(())
    */
}

pub async fn save_json(dataset: &Dataset, path: &PathBuf) -> Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use serde_json::{json, to_writer_pretty};
    
    info!("Saving dataset to JSON: {:?}", path);
    
    let file = File::create(path)
        .context("Failed to create JSON file")?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file); // 1MB buffer
    
    // Create JSON structure
    let mut json_data = json!({
        "metadata": {
            "name": dataset.name,
            "created_at": dataset.created_at.to_rfc3339(),
            "modified_at": dataset.modified_at.to_rfc3339(),
            "source_path": dataset.source_path,
            "shape": [dataset.rows(), dataset.columns()],
            "missing_values": dataset.count_missing(),
            "missing_percentage": (dataset.count_missing() as f64 / 
                (dataset.rows() * dataset.columns()) as f64) * 100.0,
        },
        "columns": dataset.columns,
        "variables": dataset.variables.iter().map(|v| {
            json!({
                "name": v.name,
                "unit": v.unit,
                "description": v.description,
                "min_valid": v.min_valid,
                "max_valid": v.max_valid,
                "precision": v.precision,
                "measurement_type": v.measurement_type,
            })
        }).collect::<Vec<_>>(),
    });
    
    // Add spatial information if available
    if let Some(stations) = &dataset.stations {
        json_data["stations"] = json!(stations.iter().map(|s| {
            json!({
                "id": s.id,
                "name": s.name,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "altitude": s.altitude,
                "metadata": s.metadata,
            })
        }).collect::<Vec<_>>());
    }
    
    // Choose format based on data size
    let total_values = dataset.rows() * dataset.columns();
    
    if total_values < 100_000 {
        // Small dataset: use array of objects format (more readable)
        let mut data_array = Vec::with_capacity(dataset.rows());
        
        for (row_idx, timestamp) in dataset.index.iter().enumerate() {
            let mut row_obj = json!({
                "timestamp": timestamp.to_rfc3339(),
            });
            
            for (col_idx, col_name) in dataset.columns.iter().enumerate() {
                let value = dataset.data[[row_idx, col_idx]];
                row_obj[col_name] = if value.is_nan() {
                    json!(null)
                } else {
                    json!(value)
                };
            }
            
            data_array.push(row_obj);
        }
        
        json_data["data"] = json!(data_array);
        
        // Write pretty-printed JSON
        to_writer_pretty(&mut writer, &json_data)
            .context("Failed to write JSON data")?;
    } else {
        // Large dataset: use column-oriented format (more efficient)
        json_data["format"] = json!("columnar");
        
        // Write metadata first
        writeln!(&mut writer, "{{")?;
        write!(&mut writer, "  \"metadata\": ")?;
        to_writer_pretty(&mut writer, &json_data["metadata"])
            .context("Failed to write metadata")?;
        writeln!(&mut writer, ",")?;
        
        write!(&mut writer, "  \"columns\": ")?;
        to_writer_pretty(&mut writer, &json_data["columns"])
            .context("Failed to write columns")?;
        writeln!(&mut writer, ",")?;
        
        write!(&mut writer, "  \"variables\": ")?;
        to_writer_pretty(&mut writer, &json_data["variables"])
            .context("Failed to write variables")?;
        writeln!(&mut writer, ",")?;
        
        if json_data.get("stations").is_some() {
            write!(&mut writer, "  \"stations\": ")?;
            to_writer_pretty(&mut writer, &json_data["stations"])
                .context("Failed to write stations")?;
            writeln!(&mut writer, ",")?;
        }
        
        writeln!(&mut writer, "  \"format\": \"columnar\",")?;
        
        // Write timestamps
        writeln!(&mut writer, "  \"timestamps\": [")?;
        for (idx, timestamp) in dataset.index.iter().enumerate() {
            writeln!(&mut writer, "    \"{}\"{}", 
                timestamp.to_rfc3339(),
                if idx < dataset.index.len() - 1 { "," } else { "" }
            )?;
        }
        writeln!(&mut writer, "  ],")?;
        
        // Write data columns
        writeln!(&mut writer, "  \"data\": {{")?;
        
        for (col_idx, col_name) in dataset.columns.iter().enumerate() {
            writeln!(&mut writer, "    \"{}\": [", col_name)?;
            
            let col_data = dataset.data.column(col_idx);
            for (row_idx, &value) in col_data.iter().enumerate() {
                if value.is_nan() {
                    write!(&mut writer, "      null")?;
                } else {
                    write!(&mut writer, "      {:.6}", value)?;
                }
                
                if row_idx < col_data.len() - 1 {
                    writeln!(&mut writer, ",")?;
                } else {
                    writeln!(&mut writer)?;
                }
            }
            
            if col_idx < dataset.columns.len() - 1 {
                writeln!(&mut writer, "    ],")?;
            } else {
                writeln!(&mut writer, "    ]")?;
            }
        }
        
        writeln!(&mut writer, "  }}")?;
        writeln!(&mut writer, "}}")?;
    }
    
    writer.flush()
        .context("Failed to flush JSON writer")?;
    
    info!("Successfully saved {} rows to JSON", dataset.rows());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_detection() {
        assert!(matches!(
            detect_format(&PathBuf::from("data.csv")),
            DataFormat::Csv
        ));
        assert!(matches!(
            detect_format(&PathBuf::from("data.xlsx")),
            DataFormat::Excel
        ));
        assert!(matches!(
            detect_format(&PathBuf::from("data.nc")),
            DataFormat::NetCdf
        ));
    }
}