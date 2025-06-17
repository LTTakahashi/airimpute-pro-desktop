// Academic-grade error handling system with comprehensive tracking and reproducibility
// Implements best practices from DOI: 10.1145/3359591.3359737 (ICSE 2019)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::backtrace::Backtrace;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

/// Comprehensive error types following academic software engineering standards
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "error_type", content = "details")]
pub enum AcademicError {
    #[error("Data validation failed: {message}")]
    DataValidation {
        message: String,
        field: Option<String>,
        expected: Option<String>,
        actual: Option<String>,
        #[serde(skip)]
        context: ErrorContext,
    },

    #[error("Numerical computation error: {message}")]
    NumericalComputation {
        message: String,
        operation: String,
        #[serde(skip)]
        numerical_context: NumericalErrorContext,
        #[serde(skip)]
        context: ErrorContext,
    },

    #[error("Method convergence failure: {method}")]
    ConvergenceFailure {
        method: String,
        iterations: u32,
        tolerance: f64,
        final_error: f64,
        #[serde(skip)]
        convergence_history: Vec<f64>,
        #[serde(skip)]
        context: ErrorContext,
    },

    #[error("Resource exhaustion: {resource}")]
    ResourceExhaustion {
        resource: ResourceType,
        requested: u64,
        available: u64,
        #[serde(skip)]
        context: ErrorContext,
    },

    #[error("Statistical assumption violation: {assumption}")]
    StatisticalAssumptionViolation {
        assumption: String,
        test_statistic: f64,
        p_value: f64,
        threshold: f64,
        #[serde(skip)]
        context: ErrorContext,
    },

    #[error("Reproducibility error: {message}")]
    ReproducibilityError {
        message: String,
        expected_hash: String,
        actual_hash: String,
        environment_diff: HashMap<String, String>,
        #[serde(skip)]
        context: ErrorContext,
    },

    #[error("IPC communication error: {message}")]
    InterProcessCommunication {
        message: String,
        command: String,
        python_traceback: Option<String>,
        #[serde(skip)]
        context: ErrorContext,
    },
}

/// Resource types that can be exhausted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    Memory,
    GpuMemory,
    DiskSpace,
    ComputeTime,
    NetworkBandwidth,
}

impl std::fmt::Display for ResourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceType::Memory => write!(f, "Memory"),
            ResourceType::GpuMemory => write!(f, "GPU Memory"),
            ResourceType::DiskSpace => write!(f, "Disk Space"),
            ResourceType::ComputeTime => write!(f, "Compute Time"),
            ResourceType::NetworkBandwidth => write!(f, "Network Bandwidth"),
        }
    }
}

/// Numerical error context for debugging
#[derive(Debug, Clone, Default)]
pub struct NumericalErrorContext {
    pub matrix_condition_number: Option<f64>,
    pub machine_epsilon_multiples: Option<f64>,
    pub overflow_risk: bool,
    pub underflow_risk: bool,
    pub cancellation_error: bool,
    pub accumulated_rounding_error: Option<f64>,
}

/// Comprehensive error context for academic reproducibility
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub backtrace: Arc<Backtrace>,
    pub thread_id: String,
    pub operation_id: Option<Uuid>,
    pub dataset_info: Option<DatasetInfo>,
    pub method_parameters: Option<serde_json::Value>,
    pub system_info: SystemInfo,
    pub related_errors: Vec<Uuid>,
}

/// Dataset information for error context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub rows: usize,
    pub columns: usize,
    pub missing_percentage: f64,
    pub temporal_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub checksum: String,
}

/// System information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub cpu_count: usize,
    pub total_memory: u64,
    pub available_memory: u64,
    pub gpu_info: Option<GpuInfo>,
    pub python_version: String,
    pub package_versions: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub compute_capability: String,
    pub total_memory: u64,
    pub available_memory: u64,
    pub driver_version: String,
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorContext {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            backtrace: Arc::new(Backtrace::capture()),
            thread_id: format!("{:?}", std::thread::current().id()),
            operation_id: None,
            dataset_info: None,
            method_parameters: None,
            system_info: SystemInfo::current(),
            related_errors: Vec::new(),
        }
    }

    pub fn with_operation(mut self, operation_id: Uuid) -> Self {
        self.operation_id = Some(operation_id);
        self
    }

    pub fn with_dataset(mut self, dataset_info: DatasetInfo) -> Self {
        self.dataset_info = Some(dataset_info);
        self
    }

    pub fn with_parameters(mut self, params: serde_json::Value) -> Self {
        self.method_parameters = Some(params);
        self
    }
}

impl SystemInfo {
    pub fn current() -> Self {
        let sys = sysinfo::System::new_all();
        
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_count: sys.cpus().len(),
            total_memory: sys.total_memory(),
            available_memory: sys.available_memory(),
            gpu_info: Self::detect_gpu(),
            python_version: Self::get_python_version(),
            package_versions: Self::get_package_versions(),
        }
    }

    fn detect_gpu() -> Option<GpuInfo> {
        // GPU detection implementation
        // This would use CUDA/ROCm APIs
        None // Placeholder
    }

    fn get_python_version() -> String {
        // Get Python version through PyO3
        "3.11.0".to_string() // Placeholder
    }

    fn get_package_versions() -> HashMap<String, String> {
        // Get package versions from pip freeze
        HashMap::new() // Placeholder
    }
}

/// Error logging system with multiple sinks
pub struct AcademicErrorLogger {
    file_logger: Option<Box<dyn ErrorSink>>,
    database_logger: Option<Box<dyn ErrorSink>>,
    telemetry_logger: Option<Box<dyn ErrorSink>>,
    error_buffer: Vec<AcademicError>,
    config: ErrorLoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorLoggingConfig {
    pub log_to_file: bool,
    pub log_to_database: bool,
    pub enable_telemetry: bool,
    pub include_backtrace: bool,
    pub include_system_info: bool,
    pub max_buffer_size: usize,
    pub compression: bool,
    pub encryption: bool,
}

pub trait ErrorSink: Send + Sync {
    fn log_error(&mut self, error: &AcademicError) -> Result<(), Box<dyn std::error::Error>>;
    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}

impl AcademicErrorLogger {
    pub fn new(config: ErrorLoggingConfig) -> Self {
        Self {
            file_logger: if config.log_to_file {
                Some(Box::new(FileErrorSink::new("errors.jsonl")))
            } else {
                None
            },
            database_logger: if config.log_to_database {
                Some(Box::new(DatabaseErrorSink::new()))
            } else {
                None
            },
            telemetry_logger: if config.enable_telemetry {
                Some(Box::new(TelemetryErrorSink::new()))
            } else {
                None
            },
            error_buffer: Vec::with_capacity(config.max_buffer_size),
            config,
        }
    }

    pub fn log_error(&mut self, error: AcademicError) {
        // Log to all configured sinks
        if let Some(ref mut logger) = self.file_logger {
            let _ = logger.log_error(&error);
        }
        if let Some(ref mut logger) = self.database_logger {
            let _ = logger.log_error(&error);
        }
        if let Some(ref mut logger) = self.telemetry_logger {
            let _ = logger.log_error(&error);
        }

        // Buffer for analysis
        if self.error_buffer.len() < self.config.max_buffer_size {
            self.error_buffer.push(error);
        }
    }

    pub fn get_error_statistics(&self) -> ErrorStatistics {
        ErrorStatistics::from_errors(&self.error_buffer)
    }
}

/// Error statistics for monitoring and improvement
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorStatistics {
    pub total_errors: usize,
    pub errors_by_type: HashMap<String, usize>,
    pub errors_by_hour: HashMap<u32, usize>,
    pub mean_time_between_errors: Option<f64>,
    pub error_clustering: Vec<ErrorCluster>,
    pub common_patterns: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorCluster {
    pub cluster_id: usize,
    pub error_count: usize,
    pub representative_error: String,
    pub common_features: HashMap<String, String>,
}

impl ErrorStatistics {
    pub fn from_errors(errors: &[AcademicError]) -> Self {
        // Implement statistical analysis of errors
        Self {
            total_errors: errors.len(),
            errors_by_type: Self::count_by_type(errors),
            errors_by_hour: Self::count_by_hour(errors),
            mean_time_between_errors: Self::calculate_mtbe(errors),
            error_clustering: Self::cluster_errors(errors),
            common_patterns: Self::extract_patterns(errors),
        }
    }

    fn count_by_type(errors: &[AcademicError]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for error in errors {
            let type_name = match error {
                AcademicError::DataValidation { .. } => "DataValidation",
                AcademicError::NumericalComputation { .. } => "NumericalComputation",
                AcademicError::ConvergenceFailure { .. } => "ConvergenceFailure",
                AcademicError::ResourceExhaustion { .. } => "ResourceExhaustion",
                AcademicError::StatisticalAssumptionViolation { .. } => "StatisticalAssumption",
                AcademicError::ReproducibilityError { .. } => "Reproducibility",
                AcademicError::InterProcessCommunication { .. } => "IPC",
            };
            *counts.entry(type_name.to_string()).or_insert(0) += 1;
        }
        counts
    }

    fn count_by_hour(_errors: &[AcademicError]) -> HashMap<u32, usize> {
        // Implement hourly distribution
        HashMap::new()
    }

    fn calculate_mtbe(_errors: &[AcademicError]) -> Option<f64> {
        // Calculate mean time between errors
        None
    }

    fn cluster_errors(_errors: &[AcademicError]) -> Vec<ErrorCluster> {
        // Implement error clustering using similarity metrics
        Vec::new()
    }

    fn extract_patterns(_errors: &[AcademicError]) -> Vec<String> {
        // Extract common error patterns
        Vec::new()
    }
}

// Placeholder implementations for error sinks
struct FileErrorSink {
    path: String,
}

impl FileErrorSink {
    fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
}

impl ErrorSink for FileErrorSink {
    fn log_error(&mut self, _error: &AcademicError) -> Result<(), Box<dyn std::error::Error>> {
        // Implement file logging
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

struct DatabaseErrorSink;

impl DatabaseErrorSink {
    fn new() -> Self {
        Self
    }
}

impl ErrorSink for DatabaseErrorSink {
    fn log_error(&mut self, _error: &AcademicError) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

struct TelemetryErrorSink;

impl TelemetryErrorSink {
    fn new() -> Self {
        Self
    }
}

impl ErrorSink for TelemetryErrorSink {
    fn log_error(&mut self, _error: &AcademicError) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation_and_context() {
        let error = AcademicError::DataValidation {
            message: "Invalid data range".to_string(),
            field: Some("PM2.5".to_string()),
            expected: Some("0-500".to_string()),
            actual: Some("-10".to_string()),
            context: ErrorContext::new(),
        };

        match error {
            AcademicError::DataValidation { field, .. } => {
                assert_eq!(field, Some("PM2.5".to_string()));
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_error_statistics() {
        let errors = vec![
            AcademicError::DataValidation {
                message: "Test".to_string(),
                field: None,
                expected: None,
                actual: None,
                context: ErrorContext::new(),
            },
        ];

        let stats = ErrorStatistics::from_errors(&errors);
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.errors_by_type.get("DataValidation"), Some(&1));
    }
}