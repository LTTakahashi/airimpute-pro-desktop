// Simplified imputation commands using the new Python bridge

use tauri::{command, State, Window};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tracing::{info, error, warn};
use serde_json::json;
use tokio::sync::Mutex;
use std::collections::HashMap;

use crate::state::AppState;
use crate::python::{SafePythonBridge, PythonOperation};
use crate::core::imputation::{ImputationJob, JobStatus};
use crate::error::simple_error::{AppError, Result as AppResult};
use crate::core::progress_tracker::{ProgressManager, ProgressTracker};
use crate::validation::data_validator::{DataValidator, ValidationConfig};

/// Available imputation methods with descriptions
#[derive(Debug, Clone, Serialize)]
pub struct MethodInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub complexity: String,
    pub suitable_for: Vec<String>,
    pub parameters: serde_json::Value,
    pub requires_gpu: bool,
}

/// Imputation job response
#[derive(Debug, Clone, Serialize)]
pub struct ImputationJobResponse {
    pub job_id: String,
    pub dataset_id: String,
    pub method: String,
    pub status: String,
    pub progress: f64,
    pub eta_seconds: Option<u64>,
    pub created_at: DateTime<Utc>,
    pub message: Option<String>,
}

/// Get available imputation methods
#[command]
pub async fn get_imputation_methods(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<MethodInfo>, String> {
    info!("Getting available imputation methods");
    
    // Return built-in methods directly
    // TODO: In the future, we can query Python for additional methods
    Ok(get_builtin_methods())
}

/// Get built-in methods as fallback
fn get_builtin_methods() -> Vec<MethodInfo> {
    vec![
        MethodInfo {
            id: "mean".to_string(),
            name: "Mean Imputation".to_string(),
            description: "Replace missing values with column mean".to_string(),
            category: "Simple".to_string(),
            complexity: "O(n)".to_string(),
            suitable_for: vec!["Small gaps".to_string(), "Normal distribution".to_string()],
            parameters: json!({}),
            requires_gpu: false,
        },
        MethodInfo {
            id: "linear".to_string(),
            name: "Linear Interpolation".to_string(),
            description: "Connect points with straight lines".to_string(),
            category: "Interpolation".to_string(),
            complexity: "O(n)".to_string(),
            suitable_for: vec!["Smooth data".to_string(), "Regular sampling".to_string()],
            parameters: json!({
                "limit": { "type": "int", "default": null, "description": "Maximum gap size" }
            }),
            requires_gpu: false,
        },
        MethodInfo {
            id: "forward_fill".to_string(),
            name: "Forward Fill".to_string(),
            description: "Propagate last valid observation forward".to_string(),
            category: "Simple".to_string(),
            complexity: "O(n)".to_string(),
            suitable_for: vec!["Time series".to_string(), "Small gaps".to_string()],
            parameters: json!({
                "limit": { "type": "int", "default": null, "description": "Maximum gap size" }
            }),
            requires_gpu: false,
        },
    ]
}

/// Validate data before imputation
#[command]
pub async fn validate_imputation_data(
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
) -> Result<serde_json::Value, String> {
    info!("Validating dataset {} for imputation", dataset_id);
    
    let dataset_uuid = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    let dataset = state.datasets.get(&dataset_uuid)
        .ok_or_else(|| "Dataset not found".to_string())?;
    
    // Use Rust validator first
    let validator = DataValidator::new(ValidationConfig::default());
    let validation_result = validator.validate(&dataset);
    
    // If basic validation passes, do Python validation for more details
    if validation_result.is_valid {
        let data_json = json!({
            "data": dataset.to_json_split(),
            "columns": dataset.get_variable_names(),
        });
        
        let operation = PythonOperation {
            module: "airimpute.desktop_integration".to_string(),
            function: "validate_data_json".to_string(),
            args: vec![data_json.to_string()],
            kwargs: HashMap::new(),
            timeout_ms: Some(10000),
        };
        
        match state.python_bridge.execute_operation(&operation, None).await {
            Ok(result) => Ok(serde_json::from_str(&result).unwrap_or_else(|_| {
                serde_json::to_value(&validation_result).unwrap()
            })),
            Err(e) => {
                warn!("Python validation failed, using Rust result: {}", e);
                Ok(serde_json::to_value(&validation_result).unwrap())
            }
        }
    } else {
        Ok(serde_json::to_value(&validation_result).unwrap())
    }
}

/// Run imputation on a dataset
#[command]
pub async fn run_imputation_v2(
    window: Window,
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
    method: String,
    parameters: serde_json::Value,
) -> Result<ImputationJobResponse, String> {
    info!("Starting imputation job for dataset {} with method {}", dataset_id, method);
    
    let dataset_uuid = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    // Check if dataset exists
    let dataset = state.datasets.get(&dataset_uuid)
        .ok_or_else(|| "Dataset not found".to_string())?;
    
    // Quick validation
    let validator = DataValidator::new(ValidationConfig::default());
    let validation = validator.validate(&dataset.value());
    
    if !validation.is_valid {
        let critical_errors: Vec<String> = validation.errors.iter()
            .filter(|e| matches!(e.severity, crate::validation::data_validator::ErrorSeverity::Critical))
            .map(|e| e.message.clone())
            .collect();
        
        if !critical_errors.is_empty() {
            return Err(format!("Data validation failed: {}", critical_errors.join(", ")));
        }
    }
    
    // Create job
    let job_id = Uuid::new_v4();
    let job = ImputationJob {
        id: job_id.to_string(),
        dataset_id: dataset_uuid,
        dataset_name: dataset.value().name.clone(),
        method: method.clone(),
        parameters: if let serde_json::Value::Object(map) = parameters.clone() {
            map.into_iter().collect()
        } else {
            HashMap::new()
        },
        original_data: dataset.value().data.clone(),
        result: None,
        result_data: None,
        status: JobStatus::Pending,
        progress: 0.0,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    
    // Store job
    state.imputation_jobs.insert(job_id, Arc::new(Mutex::new(job.clone())));
    
    // Spawn imputation task
    let state_clone = state.inner().clone();
    let window_clone = window.clone();
    let dataset_clone = dataset.clone();
    
    tokio::spawn(async move {
        execute_imputation_v2(
            state_clone,
            window_clone,
            job_id,
            dataset_clone,
            method,
            parameters,
        ).await;
    });
    
    Ok(ImputationJobResponse {
        job_id: job_id.to_string(),
        dataset_id: dataset_id.clone(),
        method: job.method,
        status: "pending".to_string(),
        progress: 0.0,
        eta_seconds: None,
        created_at: job.created_at,
        message: Some("Imputation job started".to_string()),
    })
}

/// Execute imputation in background
async fn execute_imputation_v2(
    state: Arc<AppState>,
    window: Window,
    job_id: Uuid,
    dataset: Arc<crate::core::data::Dataset>,
    method: String,
    parameters: serde_json::Value,
) {
    // Update job status
    if let Some(job_arc) = state.imputation_jobs.get(&job_id) {
        let mut job = job_arc.lock().await;
        job.status = JobStatus::Running;
        job.started_at = Some(Utc::now());
    }
    
    // Emit start event
    window.emit("imputation:started", json!({
        "job_id": job_id.to_string(),
        "method": method,
    })).ok();
    
    // Create progress tracker
    let tracker = state.progress_manager.create_tracker(
        format!("Imputation: {}", method),
        100,
        true,
    );
    
    // Monitor progress in background
    let window_clone = window.clone();
    let job_id_str = job_id.to_string();
    let progress_task = tokio::spawn(async move {
        loop {
            if let Some(update) = state.progress_manager.next_update().await {
                if update.id == tracker.get_id() {
                    window_clone.emit("imputation:progress", json!({
                        "job_id": job_id_str,
                        "progress": update.percentage,
                        "message": update.message,
                        "elapsed_seconds": update.elapsed_seconds,
                        "eta_seconds": update.estimated_remaining_seconds,
                    })).ok();
                }
            }
            
            if tracker.is_cancelled() {
                break;
            }
            
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    });
    
    // Prepare data for Python
    let data_json = json!({
        "data": dataset.to_json_split(),
        "method": method,
        "parameters": parameters,
        "columns": dataset.get_variable_names(),
    });
    
    // Create imputation request
    let request_json = json!({
        "data": dataset.to_json_split(),
        "method": method,
        "parameters": parameters,
        "columns": dataset.get_variable_names(),
    });
    
    let operation = PythonOperation {
        module: "airimpute.desktop_integration".to_string(),
        function: "get_integration().impute".to_string(),
        args: vec![
            json!(request_json).to_string(),
        ],
        kwargs: HashMap::new(),
        timeout_ms: Some(300000), // 5 minutes
    };
    
    // Execute imputation
    match state.python_bridge.execute_operation(&operation, Some(&tracker)).await {
        Ok(result_json) => {
            match serde_json::from_str::<serde_json::Value>(&result_json) {
                Ok(result) => {
                    if result["success"].as_bool().unwrap_or(false) {
                        // Update job with results
                        if let Some(job_arc) = state.imputation_jobs.get(&job_id) {
                            let mut job = job_arc.lock().await;
                            job.status = JobStatus::Completed;
                            job.completed_at = Some(Utc::now());
                            job.progress = 1.0;
                            job.result = Some(Arc::new(result.clone()));
                            
                            info!("Imputation completed successfully for job {}", job_id);
                        }
                        
                        tracker.complete("Imputation completed successfully".to_string());
                        
                        // Emit completion
                        window.emit("imputation:completed", json!({
                            "job_id": job_id.to_string(),
                            "metadata": result["metadata"],
                            "warnings": result["warnings"],
                        })).ok();
                    } else {
                        handle_imputation_error(
                            &state,
                            &window,
                            job_id,
                            result["error"].as_str().unwrap_or("Unknown error"),
                        ).await;
                    }
                }
                Err(e) => {
                    handle_imputation_error(
                        &state,
                        &window,
                        job_id,
                        &format!("Invalid result format: {}", e),
                    ).await;
                }
            }
        }
        Err(e) => {
            handle_imputation_error(
                &state,
                &window,
                job_id,
                &e.user_message(),
            ).await;
            
            // Emit error with user-friendly message
            window.emit("imputation:error", json!({
                "job_id": job_id.to_string(),
                "error": e.user_message(),
                "code": e.code(),
                "suggestion": e.suggested_action(),
            })).ok();
        }
    }
    
    // Cancel progress monitoring
    progress_task.abort();
}

async fn handle_imputation_error(
    state: &Arc<AppState>,
    window: &Window,
    job_id: Uuid,
    error_msg: &str,
) {
    error!("Imputation failed: {}", error_msg);
    
    // Update job with error
    if let Some(job_arc) = state.imputation_jobs.get(&job_id) {
        let mut job = job_arc.lock().await;
        job.status = JobStatus::Failed;
        job.completed_at = Some(Utc::now());
        job.error = Some(error_msg.to_string());
    }
    
    // Emit error
    window.emit("imputation:failed", json!({
        "job_id": job_id.to_string(),
        "error": error_msg,
    })).ok();
}

/// Get imputation job status
#[command]
pub async fn get_imputation_status(
    state: State<'_, Arc<AppState>>,
    job_id: String,
) -> Result<serde_json::Value, String> {
    let job_uuid = Uuid::parse_str(&job_id)
        .map_err(|e| format!("Invalid job ID: {}", e))?;
    
    let job_arc = state.imputation_jobs.get(&job_uuid)
        .ok_or_else(|| "Job not found".to_string())?;
    
    let job = job_arc.lock().await;
    
    Ok(json!({
        "job_id": job.id.to_string(),
        "dataset_id": job.dataset_id.to_string(),
        "method": job.method,
        "status": format!("{:?}", job.status),
        "progress": job.progress,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error": job.error,
        "has_result": job.result.is_some(),
    }))
}

/// Get imputation result
#[command]
pub async fn get_imputation_result(
    state: State<'_, Arc<AppState>>,
    job_id: String,
) -> Result<serde_json::Value, String> {
    let job_uuid = Uuid::parse_str(&job_id)
        .map_err(|e| format!("Invalid job ID: {}", e))?;
    
    let job_arc = state.imputation_jobs.get(&job_uuid)
        .ok_or_else(|| "Job not found".to_string())?;
    
    let job = job_arc.lock().await;
    
    if job.status != JobStatus::Completed {
        return Err("Job not completed".to_string());
    }
    
    match &job.result {
        Some(result) => Ok((**result).clone()),
        None => Err("No result available".to_string()),
    }
}

/// Cancel imputation job
#[command]
pub async fn cancel_imputation(
    state: State<'_, Arc<AppState>>,
    job_id: String,
) -> Result<(), String> {
    let job_uuid = Uuid::parse_str(&job_id)
        .map_err(|e| format!("Invalid job ID: {}", e))?;
    
    if let Some(job_arc) = state.imputation_jobs.get(&job_uuid) {
        let mut job = job_arc.lock().await;
        if matches!(job.status, JobStatus::Pending | JobStatus::Running) {
            job.status = JobStatus::Failed;
            job.error = Some("Cancelled by user".to_string());
            job.completed_at = Some(Utc::now());
            
            // TODO: Actually cancel the Python operation
            info!("Cancelled imputation job {}", job_id);
            Ok(())
        } else {
            Err("Job is not running".to_string())
        }
    } else {
        Err("Job not found".to_string())
    }
}

/// Estimate processing time
#[command]
pub async fn estimate_imputation_time(
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
    method: String,
) -> Result<serde_json::Value, String> {
    let dataset_uuid = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    let dataset_ref = state.datasets.get(&dataset_uuid)
        .ok_or_else(|| "Dataset not found".to_string())?;
    let dataset = dataset_ref.value();
    
    let n_missing = dataset.count_missing();
    let dataset_size = dataset.rows() * dataset.columns();
    
    // Simple estimation based on method and size
    let base_time_ms = match method.as_str() {
        "mean" | "median" => 10.0 + (n_missing as f64 * 0.01),
        "forward_fill" => 20.0 + (n_missing as f64 * 0.02),
        "linear" => 50.0 + (n_missing as f64 * 0.05),
        "spline" => 100.0 + (n_missing as f64 * 0.1),
        "random_forest" => 500.0 + (dataset_size as f64 * 0.1),
        "lstm" => 1000.0 + (dataset_size as f64 * 0.5),
        _ => 100.0 + (n_missing as f64 * 0.1),
    };
    
    Ok(json!({
        "estimated_ms": base_time_ms as u64,
        "estimated_readable": format_duration(base_time_ms as u64),
        "confidence": 0.7,
        "factors": {
            "missing_values": n_missing,
            "dataset_size": dataset_size,
            "method_complexity": get_method_complexity(&method),
        }
    }))
}

fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{} ms", ms)
    } else if ms < 60000 {
        format!("{:.1} seconds", ms as f64 / 1000.0)
    } else {
        format!("{:.1} minutes", ms as f64 / 60000.0)
    }
}

fn get_method_complexity(method: &str) -> &'static str {
    match method {
        "mean" | "median" | "forward_fill" => "Low",
        "linear" | "spline" => "Medium",
        "random_forest" | "lstm" => "High",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(500), "500 ms");
        assert_eq!(format_duration(1500), "1.5 seconds");
        assert_eq!(format_duration(90000), "1.5 minutes");
    }
    
    #[test]
    fn test_builtin_methods() {
        let methods = get_builtin_methods();
        assert!(methods.len() >= 3);
        assert!(methods.iter().any(|m| m.id == "mean"));
        assert!(methods.iter().any(|m| m.id == "linear"));
    }
}