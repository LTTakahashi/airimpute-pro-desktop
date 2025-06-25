// Stub types for when Python support is disabled
use serde::{Deserialize, Serialize};
use ndarray::Array2;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Request structure for imputation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImputationRequest {
    pub data: Array2<f64>,
    pub method: ImputationMethod,
    pub parameters: ImputationParameters,
    pub metadata: DatasetMetadata,
}

/// Available imputation methods with academic rigor
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImputationMethod {
    /// Robust Adaptive Hybrid - our flagship method
    RAH {
        spatial_weight: f64,
        temporal_weight: f64,
        adaptive_threshold: f64,
    },
    /// Spline interpolation for smooth time series
    SplineInterpolation {
        order: u32,
        smoothing: f64,
    },
    /// Enhanced KNN with adaptive neighbors
    AdaptiveKNN {
        max_neighbors: u32,
        distance_metric: String,
    },
    /// Random Forest with optimized hyperparameters
    RandomForest {
        n_estimators: u32,
        max_depth: Option<u32>,
    },
    /// Deep learning approaches
    DeepLearning {
        architecture: String,
        hidden_layers: Vec<u32>,
        epochs: u32,
    },
    /// Classical methods for comparison
    Mean,
    Median,
    ForwardFill,
    BackwardFill,
    LinearInterpolation,
}

/// Imputation parameters with validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImputationParameters {
    pub physical_bounds: HashMap<String, (f64, f64)>,
    pub confidence_level: f64,
    pub preserve_statistics: bool,
    pub validation_strategy: ValidationStrategy,
    pub random_seed: Option<u64>,
    pub optimization: OptimizationSettings,
    pub quality_threshold: f64,
    pub max_iterations: u32,
    pub convergence_tolerance: f64,
}

/// Validation strategies for imputation quality
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStrategy {
    CrossValidation { folds: u32 },
    TimeSeriesSplit { n_splits: u32 },
    LeaveOneOut,
    Bootstrap { n_samples: u32 },
    None,
}

/// Optimization settings for performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSettings {
    pub use_gpu: bool,
    pub parallel_chunks: Option<usize>,
    pub cache_intermediate: bool,
    pub early_stopping: bool,
    pub chunk_size: usize,
    pub parallel_threads: Option<usize>,
    pub memory_limit_mb: Option<usize>,
}

/// Dataset metadata for context-aware imputation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub station_names: Vec<String>,
    pub variable_names: Vec<String>,
    pub timestamps: Vec<String>,
    pub coordinates: Option<Vec<(f64, f64)>>,
    pub units: HashMap<String, String>,
    pub missing_pattern: MissingPattern,
    pub name: String,
    pub variables: Vec<VariableInfo>,
    pub temporal_info: Option<TemporalInfo>,
    pub spatial_info: Option<SpatialInfo>,
}

/// Variable information for typed handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableInfo {
    pub name: String,
    pub unit: Option<String>,
    pub data_type: DataType,
    pub valid_range: Option<(f64, f64)>,
    pub missing_percentage: f64,
}

/// Data types for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    Temperature,
    Humidity,
    Pressure,
    PM25,
    PM10,
    NO2,
    O3,
    Custom(String),
}

/// Temporal information for time-aware methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInfo {
    pub start_time: DateTime<Utc>,
    pub frequency: String, // e.g., "1H", "30T", "D"
    pub timezone: String,
}

/// Spatial information for location-aware methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialInfo {
    pub locations: Vec<Location>,
    pub coordinate_system: String,
}

/// Geographic location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub id: String,
    pub name: String,
    pub latitude: f64,
    pub longitude: f64,
    pub elevation: Option<f64>,
}

/// Missing data patterns for method selection
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingPattern {
    Random,
    Systematic,
    Temporal,
    Spatial,
    Mixed,
}

/// Comprehensive imputation result with diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImputationResult {
    pub imputed_data: Array2<f64>,
    pub metadata: ImputationMetadata,
    pub quality_metrics: QualityMetrics,
    pub diagnostics: ImputationDiagnostics,
}

/// Metadata about the imputation process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImputationMetadata {
    pub method_used: String,
    pub parameters_used: HashMap<String, serde_json::Value>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub version: String,
}

/// Quality metrics for imputation assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub rmse: Option<f64>,
    pub mae: Option<f64>,
    pub r_squared: Option<f64>,
    pub coverage: f64,
    pub confidence_intervals: Option<Array2<f64>>,
}

/// Diagnostic information for troubleshooting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImputationDiagnostics {
    pub convergence_achieved: bool,
    pub iterations_used: u32,
    pub warnings: Vec<String>,
    pub performance_stats: PerformanceStats,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_time_ms: u64,
    pub memory_peak_mb: f64,
    pub gpu_utilization: Option<f64>,
}

/// Stub implementation of PythonBridge
pub struct PythonBridge;

impl PythonBridge {
    pub fn new(_scripts_path: &std::path::Path) -> anyhow::Result<Self> {
        Ok(Self)
    }
    
    pub fn run_imputation(&self, _request: ImputationRequest) -> anyhow::Result<ImputationResult> {
        Err(anyhow::anyhow!("Python support is disabled in this build"))
    }
}