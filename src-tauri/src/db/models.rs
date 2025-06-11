use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Recent project information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentProject {
    pub path: String,
    pub name: String,
    pub last_opened: String,
    pub exists: bool,
}

/// Project model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub version: i32,
    pub metadata: Option<serde_json::Value>,
}

/// Dataset model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Dataset {
    pub id: String,
    pub project_id: String,
    pub name: String,
    pub file_path: String,
    pub file_hash: String,
    pub rows: i32,
    pub columns: i32,
    pub missing_count: i32,
    pub missing_percentage: f64,
    pub statistics: serde_json::Value,
    pub column_metadata: serde_json::Value,
    pub temporal_info: Option<serde_json::Value>,
    pub spatial_info: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Imputation job status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl From<String> for JobStatus {
    fn from(s: String) -> Self {
        match s.as_str() {
            "pending" => JobStatus::Pending,
            "running" => JobStatus::Running,
            "completed" => JobStatus::Completed,
            "failed" => JobStatus::Failed,
            "cancelled" => JobStatus::Cancelled,
            _ => JobStatus::Pending,
        }
    }
}

impl ToString for JobStatus {
    fn to_string(&self) -> String {
        match self {
            JobStatus::Pending => "pending".to_string(),
            JobStatus::Running => "running".to_string(),
            JobStatus::Completed => "completed".to_string(),
            JobStatus::Failed => "failed".to_string(),
            JobStatus::Cancelled => "cancelled".to_string(),
        }
    }
}

/// Imputation job model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ImputationJob {
    pub id: String,
    pub project_id: String,
    pub dataset_id: String,
    pub method: String,
    pub parameters: serde_json::Value,
    #[sqlx(rename = "status")]
    pub status_str: String,
    pub priority: i32,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub duration_ms: Option<i32>,
    pub error_message: Option<String>,
    pub error_details: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl ImputationJob {
    pub fn status(&self) -> JobStatus {
        JobStatus::from(self.status_str.clone())
    }
}

/// Imputation result model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ImputationResult {
    pub id: String,
    pub job_id: String,
    pub dataset_id: String,
    pub imputed_data_path: String,
    pub imputed_data_hash: String,
    pub quality_metrics: serde_json::Value,
    pub uncertainty_metrics: Option<serde_json::Value>,
    pub validation_results: Option<serde_json::Value>,
    pub method_specific_outputs: Option<serde_json::Value>,
    pub data_preview: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

/// Method performance model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MethodPerformance {
    pub id: i64,
    pub method: String,
    pub dataset_id: String,
    pub job_id: String,
    pub rmse: Option<f64>,
    pub mae: Option<f64>,
    pub r2_score: Option<f64>,
    pub execution_time_ms: i32,
    pub memory_usage_mb: Option<f64>,
    pub cpu_usage_percent: Option<f64>,
    pub gpu_usage_percent: Option<f64>,
    pub parameters: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

/// User preference model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct UserPreference {
    pub key: String,
    pub value: serde_json::Value,
    pub category: String,
    pub description: Option<String>,
    pub updated_at: DateTime<Utc>,
}

/// Audit log model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct AuditLog {
    pub id: i64,
    pub table_name: String,
    pub operation: String,
    pub record_id: String,
    pub old_values: Option<serde_json::Value>,
    pub new_values: Option<serde_json::Value>,
    pub user_id: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Cache entry model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct CacheEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub expires_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub access_count: i32,
    pub last_accessed: DateTime<Utc>,
}

/// Active job view model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ActiveJobView {
    pub id: String,
    pub project_id: String,
    pub dataset_id: String,
    pub method: String,
    pub parameters: serde_json::Value,
    pub status: String,
    pub priority: i32,
    pub dataset_name: String,
    pub project_name: String,
    pub created_at: DateTime<Utc>,
}

/// Method performance summary view
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MethodPerformanceSummary {
    pub method: String,
    pub run_count: i64,
    pub avg_rmse: Option<f64>,
    pub min_rmse: Option<f64>,
    pub max_rmse: Option<f64>,
    pub avg_mae: Option<f64>,
    pub avg_r2: Option<f64>,
    pub avg_time_ms: Option<f64>,
    pub avg_memory_mb: Option<f64>,
}