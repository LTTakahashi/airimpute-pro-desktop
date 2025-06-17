use crate::error::CommandError;
use pyo3::types::PyString;
use crate::state::AppState;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tauri::State;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDataset {
    pub name: String,
    pub description: String,
    pub size: usize,
    pub missing_rate: f64,
    pub pattern: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMethod {
    pub name: String,
    pub category: String,
    pub has_gpu_support: bool,
    pub parameters: HashMap<String, serde_json::Value>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub method_name: String,
    pub dataset_name: String,
    pub metrics: HashMap<String, f64>,
    pub runtime: f64,
    pub memory_usage: f64,
    pub parameters: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
    pub hardware_info: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkProgress {
    pub progress: f64,
    pub task: String,
    pub current_method: Option<String>,
    pub current_dataset: Option<String>,
}

#[tauri::command]
pub async fn get_benchmark_datasets(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<BenchmarkDataset>, String> {
    let python = &state.python_runtime;
    
    Python::with_gil(|py| {
        let benchmarking = py.import("airimpute.benchmarking")?;
        let manager_class = benchmarking.getattr("BenchmarkDatasetManager")?;
        let manager = manager_class.call0()?;
        
        // Get standard datasets
        let datasets_py = manager.call_method0("get_standard_datasets")?;
        let dataset_names: Vec<String> = datasets_py.extract()?;
        
        let mut datasets = Vec::new();
        
        for name in dataset_names {
            let dataset_py = manager.call_method1("load_dataset", (name.clone(),))?;
            
            // Extract dataset properties
            let data = dataset_py.getattr("data")?;
            let shape: (usize, usize) = data.call_method0("shape")?.extract()?;
            
            // Extract metadata as JSON string and parse
            let metadata_py = dataset_py.getattr("metadata")?;
            let json_module = py.import("json")?;
            let metadata_json_str: String = json_module
                .call_method1("dumps", (metadata_py,))?
                .extract()?;
            let metadata: HashMap<String, serde_json::Value> = 
                serde_json::from_str(&metadata_json_str).unwrap_or_default();
            
            let missing_patterns = dataset_py.getattr("missing_patterns")?;
            let pattern_keys: Vec<String> = missing_patterns
                .call_method0("keys")?
                .extract()?;
            
            // Calculate average missing rate
            let mut total_missing_rate = 0.0;
            for pattern in &pattern_keys {
                let mask = missing_patterns.get_item(pattern)?;
                let missing_count: f64 = mask.call_method0("sum")?.extract()?;
                let total_count = shape.0 * shape.1;
                total_missing_rate += missing_count / total_count as f64;
            }
            let avg_missing_rate = total_missing_rate / pattern_keys.len() as f64;
            
            datasets.push(BenchmarkDataset {
                name: name.clone(),
                description: dataset_py.getattr("description")?.extract()?,
                size: shape.0 * shape.1,
                missing_rate: avg_missing_rate,
                pattern: pattern_keys.join(", "),
                metadata,
            });
        }
        
        Ok(datasets)
    }).map_err(|e: PyErr| format!("Python error: {}", e))
}

#[tauri::command]
pub async fn run_benchmark(
    datasets: Vec<String>,
    methods: Vec<String>,
    use_gpu: bool,
    cv_splits: Option<usize>,
    save_predictions: Option<bool>,
    state: State<'_, Arc<AppState>>,
    window: tauri::Window,
) -> Result<Vec<BenchmarkResult>, String> {
    let python = &state.python_runtime;
    let cv_splits = cv_splits.unwrap_or(5);
    let save_predictions = save_predictions.unwrap_or(false);
    
    Python::with_gil(|py| {
        // Import necessary modules
        let benchmarking = py.import("airimpute.benchmarking")?;
        let runner_class = benchmarking.getattr("BenchmarkRunner")?;
        
        // Create benchmark runner
        let runner = runner_class.call1((use_gpu,))?;
        
        // Load datasets
        let manager_class = benchmarking.getattr("BenchmarkDatasetManager")?;
        let manager = manager_class.call0()?;
        
        let mut benchmark_datasets = Vec::new();
        for dataset_name in &datasets {
            let dataset = manager.call_method1("load_dataset", (dataset_name,))?;
            benchmark_datasets.push(dataset);
        }
        
        // Create progress callback
        let progress_callback = |current: usize, total: usize, method: &str, dataset: &str| {
            let progress = (current as f64 / total as f64) * 100.0;
            let _ = window.emit("benchmark-progress", BenchmarkProgress {
                progress,
                task: format!("Running {} on {}", method, dataset),
                current_method: Some(method.to_string()),
                current_dataset: Some(dataset.to_string()),
            });
        };
        
        // Run benchmarks
        let mut all_results = Vec::new();
        let total_runs = methods.len() * datasets.len();
        let mut current_run = 0;
        
        for (dataset_idx, dataset_py) in benchmark_datasets.iter().enumerate() {
            let dataset_name = &datasets[dataset_idx];
            
            for method_name in &methods {
                current_run += 1;
                progress_callback(current_run, total_runs, method_name, dataset_name);
                
                // Run single benchmark
                let args = pyo3::types::PyTuple::new(py, [
                    dataset_py,
                    method_name.into_py(py).as_ref(py),
                    cv_splits.into_py(py).as_ref(py),
                    save_predictions.into_py(py).as_ref(py),
                ]);
                let result_py = runner.call_method1("run_single_benchmark", args)?;
                
                // Extract result
                let metrics: HashMap<String, f64> = result_py
                    .getattr("metrics")?
                    .extract()?;
                let runtime: f64 = result_py.getattr("runtime")?.extract()?;
                let memory_usage: f64 = result_py.getattr("memory_usage")?.extract()?;
                
                // Extract parameters as JSON
                let params_py = result_py.getattr("parameters")?;
                let params_json_str: String = py.import("json")?
                    .call_method1("dumps", (params_py,))?
                    .extract()?;
                let parameters: HashMap<String, serde_json::Value> = 
                    serde_json::from_str(&params_json_str).unwrap_or_default();
                let hardware_info: HashMap<String, String> = result_py
                    .getattr("hardware_info")?
                    .extract()?;
                
                all_results.push(BenchmarkResult {
                    method_name: method_name.clone(),
                    dataset_name: dataset_name.clone(),
                    metrics,
                    runtime,
                    memory_usage,
                    parameters,
                    timestamp: Utc::now(),
                    hardware_info,
                });
            }
        }
        
        // Save results to database
        save_benchmark_results(&all_results, &state)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Database error: {}", e)))?;
        
        Ok(all_results)
    }).map_err(|e: PyErr| format!("Python error: {}", e))
}

#[tauri::command]
pub async fn get_benchmark_results(
    datasets: Option<Vec<String>>,
    methods: Option<Vec<String>>,
    limit: Option<usize>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<BenchmarkResult>, String> {
    let db_path = state.app_data_dir.join("benchmarks.db");
    let conn = Connection::open(db_path).map_err(|e| format!("Database error: {}", e))?;
    
    // Build query
    let mut query = String::from(
        "SELECT method_name, dataset_name, metrics, runtime, memory_usage, 
         parameters, timestamp, hardware_info 
         FROM benchmark_results WHERE 1=1"
    );
    
    let mut params: Vec<String> = Vec::new();
    
    if let Some(ref ds) = datasets {
        if !ds.is_empty() {
            let placeholders = ds.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            query.push_str(&format!(" AND dataset_name IN ({})", placeholders));
            params.extend(ds.clone());
        }
    }
    
    if let Some(ref ms) = methods {
        if !ms.is_empty() {
            let placeholders = ms.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            query.push_str(&format!(" AND method_name IN ({})", placeholders));
            params.extend(ms.clone());
        }
    }
    
    query.push_str(" ORDER BY timestamp DESC");
    
    if let Some(limit) = limit {
        query.push_str(&format!(" LIMIT {}", limit));
    }
    
    let mut stmt = conn.prepare(&query).map_err(|e| format!("Query error: {}", e))?;
    let results = stmt.query_map(
        params.iter().map(|s| s as &dyn rusqlite::ToSql).collect::<Vec<_>>().as_slice(),
        |row| {
            let metrics_json: String = row.get(2)?;
            let parameters_json: String = row.get(5)?;
            let hardware_info_json: String = row.get(7)?;
            let timestamp_str: String = row.get(6)?;
            
            Ok(BenchmarkResult {
                method_name: row.get(0)?,
                dataset_name: row.get(1)?,
                metrics: serde_json::from_str(&metrics_json).unwrap_or_default(),
                runtime: row.get(3)?,
                memory_usage: row.get(4)?,
                parameters: serde_json::from_str(&parameters_json).unwrap_or_default(),
                timestamp: DateTime::parse_from_rfc3339(&timestamp_str)
                    .unwrap_or_default()
                    .with_timezone(&Utc),
                hardware_info: serde_json::from_str(&hardware_info_json).unwrap_or_default(),
            })
        }
    ).map_err(|e| format!("Query execution error: {}", e))?;
    
    let mut all_results = Vec::new();
    for result in results {
        all_results.push(result.map_err(|e| format!("Result error: {}", e))?);
    }
    
    Ok(all_results)
}

#[tauri::command]
pub async fn export_benchmark_results(
    format: String,
    output_path: PathBuf,
    results: Vec<BenchmarkResult>,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let python = &state.python_runtime;
    
    Python::with_gil(|py| {
        let benchmarking = py.import("airimpute.benchmarking")?;
        let exporter_class = benchmarking.getattr("BenchmarkExporter")?;
        let exporter = exporter_class.call0()?;
        
        // Convert results to JSON string
        let results_json_str = serde_json::to_string(&results).map_err(|e| 
            pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to serialize results: {}", e))
        )?.to_string();
        let results_py = PyString::new(py, &results_json_str);
        
        // Export based on format (Python methods need to accept JSON string)
        match format.as_str() {
            "csv" => exporter.call_method1("export_csv_from_json", (results_py, output_path.to_str().unwrap()))?,
            "json" => exporter.call_method1("export_json_from_json", (results_py, output_path.to_str().unwrap()))?,
            "latex" => exporter.call_method1("export_latex_from_json", (results_py, output_path.to_str().unwrap()))?,
            "html" => exporter.call_method1("export_html_from_json", (results_py, output_path.to_str().unwrap()))?,
            _ => return Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported export format: {}", format)
            )),
        };
        
        Ok(())
    }).map_err(|e: PyErr| format!("Python error: {}", e))
}

#[tauri::command]
pub async fn generate_reproducibility_certificate(
    benchmark_ids: Vec<String>,
    output_path: PathBuf,
    state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    // Get benchmark results outside of Python::with_gil
    let results = get_benchmark_results(
        None,
        None,
        None,
        state.clone(),
    ).await?;
    
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", ("scripts",))?;
        
        let cert_generator = py.import("generate_reproducibility_cert")?;
        let generate_fn = cert_generator.getattr("generate_certificate")?;
        
        // Filter by IDs if provided
        let filtered_results: Vec<_> = if benchmark_ids.is_empty() {
            results
        } else {
            results.into_iter()
                .filter(|r| {
                    let id = format!("{}_{}", r.method_name, r.dataset_name);
                    benchmark_ids.contains(&id)
                })
                .collect()
        };
        
        // Convert to JSON and generate certificate
        let results_json_str = serde_json::to_string(&filtered_results).map_err(|e| 
            pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to serialize results: {}", e))
        )?.to_string();
        let results_py = PyString::new(py, &results_json_str);
        let cert_path: String = generate_fn
            .call1((results_py, output_path.to_str().unwrap()))?
            .extract()?;
        
        Ok(cert_path)
    }).map_err(|e: PyErr| format!("Python error: {}", e))
}

// Helper function to save benchmark results to database
fn save_benchmark_results(
    results: &[BenchmarkResult],
    state: &Arc<AppState>,
) -> Result<(), CommandError> {
    let db_path = state.app_data_dir.join("benchmarks.db");
    let conn = Connection::open(&db_path)?;
    
    // Create table if not exists
    conn.execute(
        "CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            method_name TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            metrics TEXT NOT NULL,
            runtime REAL NOT NULL,
            memory_usage REAL NOT NULL,
            parameters TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            hardware_info TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    
    // Insert results
    for result in results {
        conn.execute(
            "INSERT INTO benchmark_results 
             (method_name, dataset_name, metrics, runtime, memory_usage, 
              parameters, timestamp, hardware_info)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                result.method_name,
                result.dataset_name,
                serde_json::to_string(&result.metrics).map_err(|e| CommandError::SerializationError { reason: e.to_string() })?,
                result.runtime,
                result.memory_usage,
                serde_json::to_string(&result.parameters).map_err(|e| CommandError::SerializationError { reason: e.to_string() })?,
                result.timestamp.to_rfc3339(),
                serde_json::to_string(&result.hardware_info).map_err(|e| CommandError::SerializationError { reason: e.to_string() })?
            ],
        )?;
    }
    
    Ok(())
}