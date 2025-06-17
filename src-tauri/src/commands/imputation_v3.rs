use crate::core::data::Dataset;
use crate::core::imputation_result::ImputationResult;
use crate::python::arrow_bridge::{
    PythonWorkerPool, SafePythonAction, PythonTask, ndarray_to_arrow, arrow_to_ndarray, 
    serialize_record_batch, deserialize_record_batch
};
use crate::state::AppState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tauri::{AppHandle, State};
use tracing::{error, info};
use uuid::Uuid;

/// Imputation request with security-validated parameters
#[derive(Debug, Serialize, Deserialize)]
pub struct SecureImputationRequest {
    pub dataset_id: Uuid,
    pub method: ImputationMethodV3,
    pub parameters: HashMap<String, serde_json::Value>,
    pub validation_config: ValidationConfig,
}

/// Validation configuration for imputation
#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub validate_input: bool,
    pub validate_output: bool,
    pub max_iterations: Option<u32>,
    pub convergence_threshold: Option<f64>,
}

/// Secure imputation method enumeration
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ImputationMethodV3 {
    // Statistical Methods
    Mean,
    Median,
    Mode,
    ForwardFill,
    BackwardFill,
    LinearInterpolation,
    Spline { order: u32 },
    
    // Machine Learning Methods
    KNN { k: u32 },
    RandomForest { n_estimators: u32, max_depth: Option<u32> },
    MICE { max_iter: u32 },
    XGBoost { n_estimators: u32, learning_rate: f32 },
    
    // Deep Learning Methods
    Autoencoder { hidden_dims: Vec<u32>, epochs: u32, use_gpu: bool },
    GAIN { hidden_dims: Vec<u32>, epochs: u32, use_gpu: bool },
    Transformer { model_name: String, context_length: u32 },
    
    // Spatial Methods
    Kriging { variogram_model: String },
    GNN { hidden_dims: Vec<u32>, num_layers: u32 },
    
    // Ensemble Methods
    Ensemble { methods: Vec<String>, weights: Option<Vec<f64>> },
}

impl ImputationMethodV3 {
    /// Convert to secure Python action
    fn to_python_action(&self) -> SafePythonAction {
        match self {
            ImputationMethodV3::Mean => SafePythonAction::ImputeMean,
            ImputationMethodV3::Median => SafePythonAction::ImputeMedian,
            ImputationMethodV3::Mode => SafePythonAction::ImputeMode,
            ImputationMethodV3::ForwardFill => SafePythonAction::ImputeForwardFill,
            ImputationMethodV3::BackwardFill => SafePythonAction::ImputeBackwardFill,
            ImputationMethodV3::LinearInterpolation => SafePythonAction::ImputeLinearInterpolation,
            ImputationMethodV3::Spline { order } => SafePythonAction::ImputeSpline { order: *order },
            
            ImputationMethodV3::KNN { k } => SafePythonAction::ImputeKNN { k: *k },
            ImputationMethodV3::RandomForest { n_estimators, max_depth } => {
                SafePythonAction::ImputeRandomForest {
                    n_estimators: *n_estimators,
                    max_depth: *max_depth,
                }
            }
            ImputationMethodV3::MICE { max_iter } => SafePythonAction::ImputeMICE { max_iter: *max_iter },
            ImputationMethodV3::XGBoost { n_estimators, learning_rate } => {
                SafePythonAction::ImputeXGBoost {
                    n_estimators: *n_estimators,
                    learning_rate: *learning_rate,
                }
            }
            
            ImputationMethodV3::Autoencoder { hidden_dims, epochs, use_gpu } => {
                SafePythonAction::ImputeAutoencoder {
                    hidden_dims: hidden_dims.clone(),
                    epochs: *epochs,
                }
            }
            ImputationMethodV3::GAIN { hidden_dims, epochs, use_gpu } => {
                SafePythonAction::ImputeGAIN {
                    hidden_dims: hidden_dims.clone(),
                    epochs: *epochs,
                }
            }
            ImputationMethodV3::Transformer { model_name, context_length } => {
                SafePythonAction::ImputeTransformer {
                    model_name: model_name.clone(),
                    context_length: *context_length,
                }
            }
            
            ImputationMethodV3::Kriging { variogram_model } => {
                SafePythonAction::ImputeKriging {
                    variogram_model: variogram_model.clone(),
                }
            }
            ImputationMethodV3::GNN { hidden_dims, num_layers } => {
                SafePythonAction::ImputeGNN {
                    hidden_dims: hidden_dims.clone(),
                    num_layers: *num_layers,
                }
            }
            
            ImputationMethodV3::Ensemble { methods, weights } => {
                SafePythonAction::ImputeEnsemble {
                    methods: methods.clone(),
                    weights: weights.clone(),
                }
            }
        }
    }
}

/// Imputation job with enhanced tracking
#[derive(Debug, Clone, Serialize)]
pub struct ImputationJobV3 {
    pub id: Uuid,
    pub dataset_id: Uuid,
    pub method: ImputationMethodV3,
    pub status: JobStatus,
    pub progress: f32,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub result_id: Option<Uuid>,
    pub error: Option<String>,
    pub metrics: HashMap<String, f64>,
    pub original_data: Arc<ndarray::Array2<f64>>, // Use Arc to avoid cloning
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Initialize Python worker pool on application startup
#[tauri::command]
pub async fn initialize_worker_pool(
    state: State<'_, AppState>,
    num_workers: Option<usize>,
) -> Result<(), String> {
    let num_workers = num_workers.unwrap_or_else(|| {
        // Default to number of CPU cores, max 8 for desktop app
        std::cmp::min(num_cpus::get(), 8)
    });
    
    info!("Initializing Python worker pool with {} workers", num_workers);
    
    match PythonWorkerPool::new(num_workers).await {
        Ok(pool) => {
            // Store the pool in app state
            let mut worker_pool = state.worker_pool.write();
            *worker_pool = Some(pool);
            info!("Worker pool initialized successfully");
            Ok(())
        }
        Err(e) => {
            error!("Failed to initialize worker pool: {}", e);
            Err(format!("Failed to initialize worker pool: {}", e))
        }
    }
}

/// Validate imputation method for security
fn validate_imputation_method(method: &ImputationMethodV3) -> Result<(), String> {
    match method {
        ImputationMethodV3::Transformer { model_name, .. } => {
            const ALLOWED_MODELS: &[&str] = &["imputeformer-base", "imputeformer-large", "imputeformer-small"];
            if !ALLOWED_MODELS.contains(&model_name.as_str()) {
                return Err(format!("Disallowed transformer model: {}. Allowed models: {:?}", 
                    model_name, ALLOWED_MODELS));
            }
        }
        ImputationMethodV3::Ensemble { methods, .. } => {
            const ALLOWED_METHODS: &[&str] = &[
                "mean", "median", "mode", "knn", "random_forest", 
                "mice", "xgboost", "linear_interpolation"
            ];
            for method_name in methods {
                if !ALLOWED_METHODS.contains(&method_name.as_str()) {
                    return Err(format!("Disallowed ensemble method: {}. Allowed methods: {:?}", 
                        method_name, ALLOWED_METHODS));
                }
            }
        }
        _ => {} // Other methods are safe as they don't accept user-provided strings
    }
    Ok(())
}

/// Run imputation with the new secure architecture
#[tauri::command]
pub async fn run_imputation_v3(
    app: AppHandle,
    state: State<'_, AppState>,
    request: SecureImputationRequest,
) -> Result<Uuid, String> {
    info!("Starting imputation v3 with method: {:?}", request.method);
    
    // Validate method for security
    validate_imputation_method(&request.method)?;
    
    // Validate dataset exists
    let dataset = state.datasets
        .get(&request.dataset_id)
        .ok_or_else(|| format!("Dataset {} not found", request.dataset_id))?;
    
    // Validate worker pool is initialized
    let worker_pool_guard = state.worker_pool.read();
    let worker_pool = worker_pool_guard
        .as_ref()
        .ok_or_else(|| "Worker pool not initialized. Call initialize_worker_pool first.".to_string())?;
    
    // Create job
    let job_id = Uuid::new_v4();
    let job = ImputationJobV3 {
        id: job_id,
        dataset_id: request.dataset_id,
        method: request.method.clone(),
        status: JobStatus::Pending,
        progress: 0.0,
        started_at: chrono::Utc::now(),
        completed_at: None,
        result_id: None,
        error: None,
        metrics: HashMap::new(),
        original_data: Arc::new(dataset.data.clone()), // Use Arc to share data
    };
    
    // Store job
    {
        let mut jobs = state.imputation_jobs_v3.write();
        jobs.insert(job_id, job.clone());
    }
    
    // Clone necessary data for async task
    let dataset_clone = dataset.value().as_ref().clone();
    let method_clone = request.method.clone();
    let state_clone = state.inner().clone();
    let app_handle = app.clone();
    
    // Spawn imputation task
    tokio::task::spawn_blocking(move || {
        let result = execute_imputation_v3(
            &app_handle,
            &state_clone,
            job_id,
            dataset_clone,
            method_clone,
            request.parameters,
            request.validation_config,
        );
        
        // Update job status
        let mut jobs = state_clone.imputation_jobs_v3.write();
        if let Some(job) = jobs.get_mut(&job_id) {
            match result {
                Ok(result_id) => {
                    job.status = JobStatus::Completed;
                    job.result_id = Some(result_id);
                    job.completed_at = Some(chrono::Utc::now());
                    job.progress = 1.0;
                    info!("Imputation job {} completed successfully", job_id);
                }
                Err(e) => {
                    job.status = JobStatus::Failed;
                    job.error = Some(e.clone());
                    job.completed_at = Some(chrono::Utc::now());
                    error!("Imputation job {} failed: {}", job_id, e);
                }
            }
        }
    });
    
    Ok(job_id)
}

/// Execute imputation using Arrow-based worker pool
fn execute_imputation_v3(
    app: &AppHandle,
    state: &AppState,
    job_id: Uuid,
    dataset: Dataset,
    method: ImputationMethodV3,
    parameters: HashMap<String, serde_json::Value>,
    validation_config: ValidationConfig,
) -> Result<Uuid, String> {
    // Update job status to running
    {
        let mut jobs = state.imputation_jobs_v3.write();
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Running;
        }
    }
    
    // TODO: Re-enable progress tracking when we refactor to async
    // let progress_service = ProgressService::new(app.clone());
    // let progress_tracker = progress_service
    //     .create_tracker("imputation".to_string(), 100).await;
    
    // Convert data to Arrow format
    let column_names: Vec<String> = dataset.columns.clone();
    
    let arrow_batch = ndarray_to_arrow(&dataset.data, &column_names)
        .map_err(|e| format!("Failed to convert data to Arrow format: {}", e))?;
    
    // Serialize Arrow data for IPC
    let serialized_data = serialize_record_batch(&arrow_batch)
        .map_err(|e| format!("Failed to serialize Arrow data: {}", e))?;
    
    // Prepare task
    let python_action = method.to_python_action();
    let mut task_metadata = parameters.clone();
    task_metadata.insert("job_id".to_string(), serde_json::json!(job_id.to_string()));
    task_metadata.insert("use_gpu".to_string(), serde_json::json!(true)); // Enable GPU if available
    
    let task = PythonTask {
        id: job_id,
        action: python_action,
        data: serialized_data,
        metadata: task_metadata.into_iter()
            .map(|(k, v)| (k, v.to_string()))
            .collect(),
    };
    
    // Execute task on worker pool
    // Progress: 10% - Executing imputation algorithm
    
    let worker_pool_guard = state.worker_pool.read();
    let worker_pool = worker_pool_guard.as_ref().unwrap();
    
    let response = futures::executor::block_on(worker_pool.execute(task))
        .map_err(|e| format!("Failed to execute imputation: {}", e))?;
    
    // Handle response
    match response.status {
        crate::python::arrow_bridge::TaskStatus::Success => {
            if let Some(result) = response.result {
                // Progress: 90% - Processing results
                
                // Deserialize Arrow result
                let result_batch = deserialize_record_batch(&result.data)
                    .map_err(|e| format!("Failed to deserialize result: {}", e))?;
                
                // Convert back to ndarray
                let imputed_data = arrow_to_ndarray(&result_batch)
                    .map_err(|e| format!("Failed to convert result to ndarray: {}", e))?;
                
                // Convert imputed data to Vec<Vec<f64>>
                let imputed_vec: Vec<Vec<f64>> = (0..imputed_data.nrows())
                    .map(|row| imputed_data.row(row).to_vec())
                    .collect();
                
                // Convert metrics to proper format
                let mut column_metrics = HashMap::new();
                for (i, col_name) in dataset.columns.iter().enumerate() {
                    if let Some(rmse) = result.metrics.get(&format!("col_{}_rmse", i)) {
                        column_metrics.insert(col_name.clone(), crate::core::imputation_result::MetricValues {
                            rmse: *rmse,
                            mae: result.metrics.get(&format!("col_{}_mae", i)).copied().unwrap_or(0.0),
                            mape: result.metrics.get(&format!("col_{}_mape", i)).copied(),
                            r2: result.metrics.get(&format!("col_{}_r2", i)).copied(),
                            correlation: None,
                            bias: None,
                            coverage_rate: None,
                        });
                    }
                }
                
                // Create imputation result
                let imputation_result = ImputationResult {
                    imputed_data: imputed_vec,
                    columns: dataset.columns.clone(),
                    index: dataset.index.clone(),
                    method: format!("{:?}", method),
                    parameters: parameters.clone(),
                    metrics: column_metrics,
                    uncertainty: None,
                    validation_results: None,
                    method_outputs: None,
                    execution_time_seconds: 0.0,  // TODO: Track actual time
                    memory_usage_mb: None,
                    convergence_info: None,
                };
                
                // Store result with new ID
                let result_id = Uuid::new_v4();
                let mut results = state.imputation_results.write();
                results.insert(result_id, imputation_result);
                
                // Update job metrics
                {
                    let mut jobs = state.imputation_jobs_v3.write();
                    if let Some(job) = jobs.get_mut(&job_id) {
                        job.metrics = result.metrics;
                    }
                }
                
                // Progress complete: Imputation completed successfully
                Ok(result_id)
            } else {
                Err("No result data returned from imputation".to_string())
            }
        }
        crate::python::arrow_bridge::TaskStatus::Failed => {
            let error_msg = response.error
                .map(|e| format!("{}: {}", e.error_type, e.message))
                .unwrap_or_else(|| "Unknown error".to_string());
            // Progress error: {error_msg}
            Err(error_msg)
        }
        crate::python::arrow_bridge::TaskStatus::Cancelled => {
            // Progress error: Imputation was cancelled
            Err("Imputation was cancelled".to_string())
        }
        _ => {
            Err("Unexpected task status".to_string())
        }
    }
}

/// Get status of an imputation job
#[tauri::command]
pub async fn get_imputation_status_v3(
    state: State<'_, AppState>,
    job_id: Uuid,
) -> Result<serde_json::Value, String> {
    let jobs = state.imputation_jobs_v3.read();
    let job = jobs
        .get(&job_id)
        .ok_or_else(|| format!("Job {} not found", job_id))?;
    
    serde_json::to_value(job)
        .map_err(|e| format!("Failed to serialize job status: {}", e))
}

/// Cancel an imputation job
#[tauri::command]
pub async fn cancel_imputation_v3(
    state: State<'_, AppState>,
    job_id: Uuid,
) -> Result<(), String> {
    // Update job status
    {
        let mut jobs = state.imputation_jobs_v3.write();
        if let Some(job) = jobs.get_mut(&job_id) {
            if matches!(job.status, JobStatus::Pending | JobStatus::Running) {
                job.status = JobStatus::Cancelled;
                job.completed_at = Some(chrono::Utc::now());
            } else {
                return Err(format!("Job {} is not cancellable in status {:?}", job_id, job.status));
            }
        } else {
            return Err(format!("Job {} not found", job_id));
        }
    }
    
    // Send cancellation request to worker pool
    let worker_pool_guard = state.worker_pool.read();
    if let Some(pool) = worker_pool_guard.as_ref() {
        let cancel_task = PythonTask {
            id: Uuid::new_v4(),
            action: SafePythonAction::CancelJob { job_id },
            data: Vec::new(),  // No data needed for cancel
            metadata: HashMap::new(),
        };
        
        let _ = pool.execute(cancel_task);
    }
    
    Ok(())
}

/// Get available imputation methods with metadata
#[tauri::command]
pub async fn get_imputation_methods_v3() -> Result<Vec<MethodInfo>, String> {
    Ok(vec![
        // Statistical Methods
        MethodInfo {
            id: "mean".to_string(),
            name: "Mean Imputation".to_string(),
            category: "statistical".to_string(),
            description: "Replace missing values with column mean".to_string(),
            complexity: "O(n)".to_string(),
            parameters: vec![],
            gpu_accelerated: false,
            streaming_capable: true,
        },
        MethodInfo {
            id: "knn".to_string(),
            name: "K-Nearest Neighbors".to_string(),
            category: "machine_learning".to_string(),
            description: "Impute using weighted average of k nearest neighbors".to_string(),
            complexity: "O(n²)".to_string(),
            parameters: vec![
                ParameterInfo {
                    name: "k".to_string(),
                    type_: "integer".to_string(),
                    default: serde_json::json!(5),
                    min: Some(1.0),
                    max: Some(20.0),
                    description: "Number of neighbors".to_string(),
                }
            ],
            gpu_accelerated: false,
            streaming_capable: false,
        },
        MethodInfo {
            id: "autoencoder".to_string(),
            name: "Denoising Autoencoder".to_string(),
            category: "deep_learning".to_string(),
            description: "Neural network-based imputation using autoencoder architecture".to_string(),
            complexity: "O(n·m·h)".to_string(),
            parameters: vec![
                ParameterInfo {
                    name: "hidden_dims".to_string(),
                    type_: "array".to_string(),
                    default: serde_json::json!([64, 32, 64]),
                    min: None,
                    max: None,
                    description: "Hidden layer dimensions".to_string(),
                },
                ParameterInfo {
                    name: "epochs".to_string(),
                    type_: "integer".to_string(),
                    default: serde_json::json!(100),
                    min: Some(10.0),
                    max: Some(1000.0),
                    description: "Training epochs".to_string(),
                },
            ],
            gpu_accelerated: true,
            streaming_capable: false,
        },
        MethodInfo {
            id: "transformer".to_string(),
            name: "Transformer Imputation".to_string(),
            category: "deep_learning".to_string(),
            description: "State-of-the-art transformer model for time series imputation".to_string(),
            complexity: "O(n²·d)".to_string(),
            parameters: vec![
                ParameterInfo {
                    name: "model_name".to_string(),
                    type_: "string".to_string(),
                    default: serde_json::json!("imputeformer-base"),
                    min: None,
                    max: None,
                    description: "Pre-trained model name".to_string(),
                },
                ParameterInfo {
                    name: "context_length".to_string(),
                    type_: "integer".to_string(),
                    default: serde_json::json!(96),
                    min: Some(24.0),
                    max: Some(512.0),
                    description: "Context window size".to_string(),
                },
            ],
            gpu_accelerated: true,
            streaming_capable: true,
        },
        MethodInfo {
            id: "gnn".to_string(),
            name: "Graph Neural Network".to_string(),
            category: "spatial".to_string(),
            description: "Spatial imputation using graph neural networks".to_string(),
            complexity: "O(n·e·h)".to_string(),
            parameters: vec![
                ParameterInfo {
                    name: "hidden_dims".to_string(),
                    type_: "array".to_string(),
                    default: serde_json::json!([64, 128, 64]),
                    min: None,
                    max: None,
                    description: "Hidden layer dimensions".to_string(),
                },
                ParameterInfo {
                    name: "num_layers".to_string(),
                    type_: "integer".to_string(),
                    default: serde_json::json!(3),
                    min: Some(1.0),
                    max: Some(10.0),
                    description: "Number of GNN layers".to_string(),
                },
            ],
            gpu_accelerated: true,
            streaming_capable: false,
        },
    ])
}

#[derive(Debug, Serialize)]
pub struct MethodInfo {
    pub id: String,
    pub name: String,
    pub category: String,
    pub description: String,
    pub complexity: String,
    pub parameters: Vec<ParameterInfo>,
    pub gpu_accelerated: bool,
    pub streaming_capable: bool,
}

#[derive(Debug, Serialize)]
pub struct ParameterInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub default: serde_json::Value,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub description: String,
}

/// Health check for worker pool
#[tauri::command]
pub async fn check_worker_health(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<serde_json::Value>, String> {
    // Check if worker pool exists and get num_workers
    let num_workers = {
        let worker_pool_guard = state.worker_pool.read();
        let pool = worker_pool_guard
            .as_ref()
            .ok_or_else(|| "Worker pool not initialized".to_string())?;
        pool.num_workers
    };
    
    // For now, return mock health status
    let mut health_reports = Vec::new();
    for i in 0..num_workers {
        health_reports.push(serde_json::json!({
            "worker_id": i,
            "status": "healthy",
            "details": {
                "uptime": 3600,
                "memory_mb": 256,
                "tasks_completed": 42
            }
        }));
    }
    
    Ok(health_reports)
}