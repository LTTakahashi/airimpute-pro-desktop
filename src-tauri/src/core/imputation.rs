use serde::{Deserialize, Serialize};
use ndarray::Array2;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Represents an imputation job with its state and results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImputationJob {
    /// Unique job ID
    pub id: String,
    
    /// Dataset ID this job is for
    pub dataset_id: uuid::Uuid,
    
    /// Dataset name for display
    pub dataset_name: String,
    
    /// The imputation method being used
    pub method: String,
    
    /// Parameters for the imputation method
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Original data before imputation
    pub original_data: Array2<f64>,
    
    /// Result of the imputation (optional until completed)
    pub result: Option<std::sync::Arc<ImputationResult>>,
    
    /// Legacy result_data field for compatibility
    pub result_data: Option<serde_json::Value>,
    
    /// Job status
    pub status: JobStatus,
    
    /// Progress percentage (0.0 to 1.0)
    pub progress: f64,
    
    /// Job creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Job start timestamp
    pub started_at: Option<DateTime<Utc>>,
    
    /// Job completion timestamp  
    pub completed_at: Option<DateTime<Utc>>,
    
    /// Error message if failed
    pub error: Option<String>,
}

/// Result of an imputation operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImputationResult {
    /// Imputed data
    pub imputed_data: Array2<f64>,
    
    /// Confidence intervals if available
    pub confidence_intervals: Option<ConfidenceIntervals>,
    
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    
    /// Method-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Confidence intervals for imputed values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    pub lower_bound: Array2<f64>,
    pub upper_bound: Array2<f64>,
    pub confidence_level: f64,
}

/// Quality metrics for imputation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub rmse: Option<f64>,
    pub mae: Option<f64>,
    pub r_squared: Option<f64>,
    pub coverage_rate: Option<f64>,
}

/// Job status enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Cancelled,
}