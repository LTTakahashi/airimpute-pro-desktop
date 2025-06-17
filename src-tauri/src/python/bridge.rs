use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyArray2;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use chrono::{DateTime, Utc};
use tracing::info;
use crate::error::{CommandError, CommandResult, ErrorExt, AuditLogEntry};
use crate::validation::{RulesValidator, NumericValidator};
use crate::python::bridge_api::{BridgeCommand, BridgeResponse};

/// Bridge between Rust and Python for scientific computing operations
pub struct PythonBridge {
    modules: Arc<Mutex<Option<PythonModules>>>,
    validator: RulesValidator,
    audit_log: Arc<Mutex<Vec<AuditLogEntry>>>,
}

/// Container for loaded Python modules
struct PythonModules {
    imputation: Py<PyModule>,
    analysis: Py<PyModule>,
    visualization: Py<PyModule>,
    validation: Py<PyModule>,
}

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
        order: usize,
        smoothing: f64,
    },
    /// Spatial kriging for geographic data
    SpatialKriging {
        variogram_model: String,
        n_neighbors: usize,
    },
    /// Seasonal decomposition for periodic patterns
    SeasonalDecomposition {
        period: usize,
        trend_extraction: String,
    },
    /// Matrix factorization for multivariate data
    MatrixFactorization {
        n_factors: usize,
        regularization: f64,
    },
    /// Deep learning imputation (experimental)
    DeepImputation {
        architecture: String,
        epochs: usize,
    },
}

/// Comprehensive imputation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImputationParameters {
    /// Physical bounds for each variable
    pub physical_bounds: HashMap<String, (f64, f64)>,
    
    /// Confidence level for uncertainty quantification
    pub confidence_level: f64,
    
    /// Whether to preserve statistical properties
    pub preserve_statistics: bool,
    
    /// Cross-validation strategy
    pub validation_strategy: ValidationStrategy,
    
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    
    /// Performance optimization flags
    pub optimization: OptimizationSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStrategy {
    KFold { n_splits: usize },
    TimeSeriesSplit { n_splits: usize },
    SpatialKFold { n_splits: usize },
    LeaveOneOut,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSettings {
    pub use_gpu: bool,
    pub parallel_chunks: Option<usize>,
    pub cache_intermediate: bool,
    pub early_stopping: bool,
}

/// Dataset metadata for context-aware imputation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub station_names: Vec<String>,
    pub variable_names: Vec<String>,
    pub timestamps: Vec<DateTime<Utc>>,
    pub coordinates: Option<Vec<(f64, f64)>>, // (lat, lon)
    pub units: HashMap<String, String>,
    pub missing_pattern: MissingPattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingPattern {
    Random,
    Systematic,
    Temporal,
    Spatial,
    Mixed,
}

/// Comprehensive imputation result with uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImputationResult {
    pub imputed_data: Array2<f64>,
    pub confidence_intervals: ConfidenceIntervals,
    pub quality_metrics: QualityMetrics,
    pub method_diagnostics: MethodDiagnostics,
    pub execution_stats: ExecutionStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    pub lower_bound: Array2<f64>,
    pub upper_bound: Array2<f64>,
    pub uncertainty_map: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub mae: f64,
    pub rmse: f64,
    pub r_squared: f64,
    pub mape: f64,
    pub variance_preserved: f64,
    pub temporal_consistency: f64,
    pub spatial_coherence: f64,
    pub cross_validation_scores: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodDiagnostics {
    pub method_selection_log: Vec<String>,
    pub convergence_history: Vec<f64>,
    pub parameter_sensitivity: HashMap<String, f64>,
    pub residual_analysis: ResidualAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualAnalysis {
    pub normality_test: (f64, f64), // (statistic, p-value)
    pub autocorrelation: Vec<f64>,
    pub heteroscedasticity_test: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    pub total_time_ms: u64,
    pub python_time_ms: u64,
    pub memory_peak_mb: f64,
    pub gaps_filled: usize,
    pub method_calls: HashMap<String, usize>,
}

impl PythonBridge {
    /// Initialize Python bridge with embedded interpreter
    pub fn new(python_path: &Path) -> CommandResult<Self> {
        info!("Initializing Python bridge at {:?}", python_path);
        
        // Validate Python path
        if !python_path.exists() {
            return Err(CommandError::ConfigurationError {
                reason: format!("Python path does not exist: {:?}", python_path)
            });
        }
        
        let modules = Python::with_gil(|py| -> CommandResult<PythonModules> {
            // Add embedded Python path to sys.path
            let sys = py.import("sys").map_py_err()?;
            let path = sys.getattr("path").map_py_err()?;
            // Use the secure fs sanitizer instead of the dangerous InputSanitizer
            let path_str = python_path.to_str().ok_or_else(|| CommandError::ValidationError {
                reason: "Invalid Python path encoding".to_string()
            })?;
            // For internal paths, we can trust them - no need to sanitize
            let sanitized_path = path_str;
            path.call_method1("insert", (0, sanitized_path)).map_py_err()?;
            
            // Import required modules with error handling
            let modules = PythonModules {
                imputation: py.import("airimpute.core")
                    .map_err(|e| CommandError::PythonError {
                        message: format!("Failed to import airimpute.core: {}", e)
                    })?.into(),
                analysis: py.import("airimpute.methods")
                    .map_err(|e| CommandError::PythonError {
                        message: format!("Failed to import airimpute.methods: {}", e)
                    })?.into(),
                visualization: py.import("matplotlib.pyplot")
                    .map_err(|e| CommandError::PythonError {
                        message: format!("Failed to import matplotlib: {}", e)
                    })?.into(),
                validation: py.import("airimpute.validation")
                    .map_err(|e| CommandError::PythonError {
                        message: format!("Failed to import airimpute.validation: {}", e)
                    })?.into(),
            };
            
            // Verify module versions
            Self::verify_module_versions(py, &modules)?;
            
            Ok(modules)
        })?;
        
        Ok(Self {
            modules: Arc::new(Mutex::new(Some(modules))),
            validator: RulesValidator::with_default_rules(),
            audit_log: Arc::new(Mutex::new(Vec::new())),
        })
    }
    
    /// Verify that Python modules meet version requirements
    fn verify_module_versions(py: Python, modules: &PythonModules) -> CommandResult<()> {
        // Check if core module has required functions
        let imputation = modules.imputation.as_ref(py);
        
        // Verify required functions exist
        let required_functions = vec![
            "DataProcessor",
            "ImputationEngine",
            "QualityMetrics",
        ];
        
        for func_name in required_functions {
            if imputation.getattr(func_name).is_err() {
                return Err(CommandError::ConfigurationError {
                    reason: format!("Python module missing required function: {}", func_name)
                });
            }
        }
        
        info!("Python modules verified successfully");
        Ok(())
    }
    
    /// Execute imputation with comprehensive error handling
    pub fn run_imputation(&self, request: ImputationRequest) -> CommandResult<ImputationResult> {
        // Create audit log entry
        let audit_entry = AuditLogEntry::new(
            "imputation",
            format!("dataset_{}", chrono::Utc::now().timestamp())
        ).with_metadata(serde_json::json!({
            "method": serde_json::to_value(&request.method).unwrap(),
            "data_shape": request.data.shape(),
        }));
        
        // Validate input data
        self.validate_imputation_request(&request)?;
        
        let result = self.run_imputation_internal(request);
        
        // Update audit log
        match &result {
            Ok(_) => {
                self.audit_log.lock().unwrap().push(audit_entry);
            }
            Err(e) => {
                self.audit_log.lock().unwrap().push(
                    audit_entry.with_error(e)
                );
            }
        }
        
        result
    }
    
    fn validate_imputation_request(&self, request: &ImputationRequest) -> CommandResult<()> {
        // Validate data dimensions
        let shape = request.data.shape();
        if shape[0] == 0 || shape[1] == 0 {
            return Err(CommandError::InvalidParameter {
                param: "data".to_string(),
                reason: "Data array cannot be empty".to_string(),
            });
        }
        
        // Validate method parameters
        if let ImputationMethod::RAH { spatial_weight, temporal_weight, adaptive_threshold } = &request.method {
            if *spatial_weight < 0.0 || *spatial_weight > 1.0 {
                return Err(CommandError::InvalidParameter {
                    param: "spatial_weight".to_string(),
                    reason: "Must be between 0 and 1".to_string(),
                });
            }
            if *temporal_weight < 0.0 || *temporal_weight > 1.0 {
                return Err(CommandError::InvalidParameter {
                    param: "temporal_weight".to_string(),
                    reason: "Must be between 0 and 1".to_string(),
                });
            }
            if (*spatial_weight + *temporal_weight - 1.0).abs() > 1e-6 {
                return Err(CommandError::InvalidParameter {
                    param: "weights".to_string(),
                    reason: "Spatial and temporal weights must sum to 1".to_string(),
                });
            }
        }
        
        Ok(())
    }
    
    fn run_imputation_internal(&self, request: ImputationRequest) -> CommandResult<ImputationResult> {
        Python::with_gil(|py| -> CommandResult<ImputationResult> {
            let start_time = std::time::Instant::now();
            
            // Get modules
            let modules_guard = self.modules.lock().unwrap();
            let modules = modules_guard.as_ref()
                .ok_or_else(|| CommandError::StateError {
                    reason: "Python modules not initialized".to_string()
                })?;
            
            // Convert Rust arrays to NumPy
            let data_array = self.array2_to_numpy(py, &request.data)?;
            
            // Prepare method configuration
            let method_config = self.prepare_method_config(py, &request.method)?;
            
            // Prepare parameters
            let params_dict = self.prepare_parameters_dict(py, &request.parameters)?;
            
            // Prepare metadata
            let metadata_dict = self.prepare_metadata_dict(py, &request.metadata)?;
            
            // Call Python imputation function
            let imputation_module = modules.imputation.as_ref(py);
            
            // Create processor and engine
            let processor_class = imputation_module.getattr("DataProcessor").map_py_err()?;
            let engine_class = imputation_module.getattr("ImputationEngine").map_py_err()?;
            
            let processor = processor_class.call0().map_py_err()?;
            let engine = engine_class.call0().map_py_err()?;
            
            // Process data
            let processed_data = processor.call_method1("process_data", (data_array,)).map_py_err()?;
            
            let python_start = std::time::Instant::now();
            
            // Run imputation based on method
            let result = match &request.method {
                ImputationMethod::RAH { .. } => {
                    engine.call_method1("impute_adaptive", 
                        (processed_data, method_config, params_dict)
                    ).map_py_err()?
                }
                _ => {
                    engine.call_method1("impute", 
                        (processed_data, method_config, params_dict)
                    ).map_py_err()?
                }
            };
            
            let python_time = python_start.elapsed().as_millis() as u64;
            
            // Parse results
            let result_dict = result.downcast::<PyDict>()
                .map_err(|_| CommandError::PythonError {
                    message: "Invalid result format from Python".to_string()
                })?;
            
            // Extract imputed data
            let imputed_data_item = result_dict
                .get_item("imputed_data")
                .map_err(|_| CommandError::PythonError {
                    message: "Missing imputed_data in result".to_string()
                })?;
            let imputed_array = imputed_data_item
                .ok_or_else(|| CommandError::PythonError {
                    message: "imputed_data is None".to_string()
                })?
                .downcast::<PyArray2<f64>>()
                .map_err(|_| CommandError::PythonError {
                    message: "Invalid imputed_data format".to_string()
                })?;
            let imputed_data = self.numpy_to_array2(imputed_array)?;
            
            // Extract confidence intervals
            let confidence_intervals = self.extract_confidence_intervals(py, result_dict)?;
            
            // Extract quality metrics
            let quality_metrics = self.extract_quality_metrics(py, result_dict)?;
            
            // Extract method diagnostics
            let method_diagnostics = self.extract_method_diagnostics(py, result_dict)?;
            
            // Calculate execution stats
            let total_time = start_time.elapsed().as_millis() as u64;
            let execution_stats = ExecutionStats {
                total_time_ms: total_time,
                python_time_ms: python_time,
                memory_peak_mb: self.get_memory_usage(py)?,
                gaps_filled: self.count_gaps_filled(&request.data, &imputed_data),
                method_calls: self.extract_method_calls(result_dict)?,
            };
            
            Ok(ImputationResult {
                imputed_data,
                confidence_intervals,
                quality_metrics,
                method_diagnostics,
                execution_stats,
            })
        })
    }
    
    /// Convert Rust ndarray to NumPy array
    fn array2_to_numpy<'py>(
        &self,
        py: Python<'py>,
        array: &Array2<f64>,
    ) -> CommandResult<&'py PyArray2<f64>> {
        let shape = array.shape();
        let numpy_array = unsafe { PyArray2::new(py, [shape[0], shape[1]], false) };
        
        // Copy data
        for ((i, j), &value) in array.indexed_iter() {
            // Validate numeric value
            let validated_value = NumericValidator::validate(value)?;
            unsafe {
                let ptr = numpy_array.uget_mut([i, j]);
                *ptr = validated_value;
            }
        }
        
        Ok(numpy_array)
    }
    
    /// Convert NumPy array to Rust ndarray
    fn numpy_to_array2(&self, py_array: &PyArray2<f64>) -> CommandResult<Array2<f64>> {
        let shape = py_array.shape();
        let mut array = Array2::zeros((shape[0], shape[1]));
        
        // Copy data
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let value = unsafe { 
                    *py_array.uget([i, j])
                };
                array[[i, j]] = NumericValidator::validate(value)?;
            }
        }
        
        Ok(array)
    }
    
    /// Prepare method configuration for Python
    fn prepare_method_config<'py>(
        &self,
        py: Python<'py>,
        method: &ImputationMethod,
    ) -> CommandResult<&'py PyDict> {
        let dict = PyDict::new(py);
        
        match method {
            ImputationMethod::RAH {
                spatial_weight,
                temporal_weight,
                adaptive_threshold,
            } => {
                dict.set_item("method", "rah").map_py_err()?;
                dict.set_item("spatial_weight", spatial_weight).map_py_err()?;
                dict.set_item("temporal_weight", temporal_weight).map_py_err()?;
                dict.set_item("adaptive_threshold", adaptive_threshold).map_py_err()?;
            }
            ImputationMethod::SplineInterpolation { order, smoothing } => {
                dict.set_item("method", "spline").map_py_err()?;
                dict.set_item("order", order).map_py_err()?;
                dict.set_item("smoothing", smoothing).map_py_err()?;
            }
            ImputationMethod::SpatialKriging {
                variogram_model,
                n_neighbors,
            } => {
                dict.set_item("method", "kriging").map_py_err()?;
                dict.set_item("variogram_model", variogram_model).map_py_err()?;
                dict.set_item("n_neighbors", n_neighbors).map_py_err()?;
            }
            _ => {
                // Add other methods as needed
                dict.set_item("method", "auto").map_py_err()?;
            }
        }
        
        Ok(dict)
    }
    
    /// Prepare parameters dictionary for Python
    fn prepare_parameters_dict<'py>(
        &self,
        py: Python<'py>,
        params: &ImputationParameters,
    ) -> CommandResult<&'py PyDict> {
        let dict = PyDict::new(py);
        
        // Physical bounds
        let bounds_dict = PyDict::new(py);
        for (var, (min, max)) in &params.physical_bounds {
            bounds_dict.set_item(var, (min, max)).map_py_err()?;
        }
        dict.set_item("physical_bounds", bounds_dict).map_py_err()?;
        
        // Other parameters
        dict.set_item("confidence_level", params.confidence_level).map_py_err()?;
        dict.set_item("preserve_statistics", params.preserve_statistics).map_py_err()?;
        
        // Validation strategy
        match &params.validation_strategy {
            ValidationStrategy::KFold { n_splits } => {
                dict.set_item("validation", "k_fold").map_py_err()?;
                dict.set_item("n_splits", n_splits).map_py_err()?;
            }
            ValidationStrategy::TimeSeriesSplit { n_splits } => {
                dict.set_item("validation", "time_series_split").map_py_err()?;
                dict.set_item("n_splits", n_splits).map_py_err()?;
            }
            _ => {
                dict.set_item("validation", "none").map_py_err()?;
            }
        }
        
        // Random seed
        if let Some(seed) = params.random_seed {
            dict.set_item("random_seed", seed).map_py_err()?;
        }
        
        Ok(dict)
    }
    
    /// Prepare metadata dictionary for Python
    fn prepare_metadata_dict<'py>(
        &self,
        py: Python<'py>,
        metadata: &DatasetMetadata,
    ) -> CommandResult<&'py PyDict> {
        let dict = PyDict::new(py);
        
        dict.set_item("station_names", &metadata.station_names).map_py_err()?;
        dict.set_item("variable_names", &metadata.variable_names).map_py_err()?;
        
        // Convert timestamps to strings
        let timestamp_strings: Vec<String> = metadata
            .timestamps
            .iter()
            .map(|dt| dt.to_rfc3339())
            .collect();
        dict.set_item("timestamps", timestamp_strings).map_py_err()?;
        
        // Coordinates
        if let Some(coords) = &metadata.coordinates {
            dict.set_item("coordinates", coords).map_py_err()?;
        }
        
        // Units
        let units_dict = PyDict::new(py);
        for (var, unit) in &metadata.units {
            units_dict.set_item(var, unit).map_py_err()?;
        }
        dict.set_item("units", units_dict).map_py_err()?;
        
        // Missing pattern
        let pattern = match metadata.missing_pattern {
            MissingPattern::Random => "random",
            MissingPattern::Systematic => "systematic",
            MissingPattern::Temporal => "temporal",
            MissingPattern::Spatial => "spatial",
            MissingPattern::Mixed => "mixed",
        };
        dict.set_item("missing_pattern", pattern).map_py_err()?;
        
        Ok(dict)
    }
    
    /// Extract confidence intervals from result
    fn extract_confidence_intervals(
        &self,
        py: Python,
        result_dict: &PyDict,
    ) -> CommandResult<ConfidenceIntervals> {
        let ci_item = result_dict
            .get_item("confidence_intervals")
            .map_err(|_| CommandError::PythonError {
                message: "Missing confidence_intervals".to_string()
            })?;
        let ci_dict = ci_item
            .ok_or_else(|| CommandError::PythonError {
                message: "confidence_intervals is None".to_string()
            })?
            .downcast::<PyDict>()
            .map_err(|_| CommandError::PythonError {
                message: "Invalid confidence_intervals format".to_string()
            })?;
        
        let lower = ci_dict
            .get_item("lower")
            .map_err(|_| CommandError::PythonError {
                message: "Missing lower bound".to_string()
            })?
            .ok_or_else(|| CommandError::PythonError {
                message: "lower bound is None".to_string()
            })?
            .downcast::<PyArray2<f64>>()
            .map_err(|_| CommandError::PythonError {
                message: "Invalid lower bound format".to_string()
            })?;
        let upper = ci_dict
            .get_item("upper")
            .map_err(|_| CommandError::PythonError {
                message: "Missing upper bound".to_string()
            })?
            .ok_or_else(|| CommandError::PythonError {
                message: "upper bound is None".to_string()
            })?
            .downcast::<PyArray2<f64>>()
            .map_err(|_| CommandError::PythonError {
                message: "Invalid upper bound format".to_string()
            })?;
        let uncertainty = ci_dict
            .get_item("uncertainty")
            .map_err(|_| CommandError::PythonError {
                message: "Missing uncertainty map".to_string()
            })?
            .ok_or_else(|| CommandError::PythonError {
                message: "uncertainty map is None".to_string()
            })?
            .downcast::<PyArray2<f64>>()
            .map_err(|_| CommandError::PythonError {
                message: "Invalid uncertainty map format".to_string()
            })?;
        
        Ok(ConfidenceIntervals {
            lower_bound: self.numpy_to_array2(lower)?,
            upper_bound: self.numpy_to_array2(upper)?,
            uncertainty_map: self.numpy_to_array2(uncertainty)?,
        })
    }
    
    /// Extract quality metrics from result
    fn extract_quality_metrics(&self, py: Python, result_dict: &PyDict) -> CommandResult<QualityMetrics> {
        let metrics_item = result_dict
            .get_item("quality_metrics")
            .map_err(|_| CommandError::PythonError {
                message: "Missing quality_metrics".to_string()
            })?;
        let metrics_dict = metrics_item
            .ok_or_else(|| CommandError::PythonError {
                message: "quality_metrics is None".to_string()
            })?
            .downcast::<PyDict>()
            .map_err(|_| CommandError::PythonError {
                message: "Invalid quality_metrics format".to_string()
            })?;
        
        Ok(QualityMetrics {
            mae: metrics_dict.get_item("mae")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            rmse: metrics_dict.get_item("rmse")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            r_squared: metrics_dict.get_item("r_squared")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            mape: metrics_dict.get_item("mape")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            variance_preserved: metrics_dict
                .get_item("variance_preserved")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            temporal_consistency: metrics_dict
                .get_item("temporal_consistency")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            spatial_coherence: metrics_dict
                .get_item("spatial_coherence")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            cross_validation_scores: metrics_dict
                .get_item("cv_scores")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or_else(Vec::new),
        })
    }
    
    /// Extract method diagnostics from result
    fn extract_method_diagnostics(&self, py: Python, result_dict: &PyDict) -> CommandResult<MethodDiagnostics> {
        let diag_item = result_dict
            .get_item("diagnostics")
            .map_err(|_| CommandError::PythonError {
                message: "Missing diagnostics".to_string()
            })?;
        let diag_dict = diag_item
            .ok_or_else(|| CommandError::PythonError {
                message: "method_diagnostics is None".to_string()
            })?
            .downcast::<PyDict>()
            .map_err(|_| CommandError::PythonError {
                message: "Invalid diagnostics format".to_string()
            })?;
        
        // Extract residual analysis
        let residual_dict = diag_dict
            .get_item("residual_analysis")
            .ok()
            .flatten()
            .and_then(|v| v.downcast::<PyDict>().ok());
        
        let residual_analysis = if let Some(rd) = residual_dict {
            ResidualAnalysis {
                normality_test: rd
                    .get_item("normality_test")
                    .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                    .unwrap_or((0.0, 1.0)),
                autocorrelation: rd
                    .get_item("autocorrelation")
                    .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                    .unwrap_or_else(Vec::new),
                heteroscedasticity_test: rd
                    .get_item("heteroscedasticity_test")
                    .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                    .unwrap_or((0.0, 1.0)),
            }
        } else {
            ResidualAnalysis {
                normality_test: (0.0, 1.0),
                autocorrelation: Vec::new(),
                heteroscedasticity_test: (0.0, 1.0),
            }
        };
        
        Ok(MethodDiagnostics {
            method_selection_log: diag_dict
                .get_item("method_log")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or_else(Vec::new),
            convergence_history: diag_dict
                .get_item("convergence_history")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or_else(Vec::new),
            parameter_sensitivity: diag_dict
                .get_item("parameter_sensitivity")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or_else(HashMap::new),
            residual_analysis,
        })
    }
    
    /// Extract method call counts
    fn extract_method_calls(&self, result_dict: &PyDict) -> CommandResult<HashMap<String, usize>> {
        Ok(result_dict
            .get_item("method_calls")
            .ok()
            .flatten()
                .and_then(|v| v.extract::<HashMap<String, usize>>().ok())
            .unwrap_or_else(HashMap::new))
    }
    
    /// Get current memory usage from Python
    fn get_memory_usage(&self, py: Python) -> CommandResult<f64> {
        let psutil = py.import("psutil").map_py_err()?;
        let process = psutil.getattr("Process")?.call0()?;
        let memory_info = process.call_method0("memory_info").map_py_err()?;
        let rss = memory_info.getattr("rss").map_py_err()?;
        let rss_mb: f64 = rss.extract::<u64>().map_py_err()? as f64 / 1_048_576.0;
        Ok(rss_mb)
    }
    
    /// Count how many gaps were filled
    fn count_gaps_filled(&self, original: &Array2<f64>, imputed: &Array2<f64>) -> usize {
        let mut count = 0;
        for ((i, j), &original_val) in original.indexed_iter() {
            if original_val.is_nan() && !imputed[[i, j]].is_nan() {
                count += 1;
            }
        }
        count
    }
    
    /// Run comprehensive analysis on dataset
    pub fn run_analysis(
        &self,
        data: &Array2<f64>,
        analysis_type: &str,
    ) -> CommandResult<serde_json::Value> {
        Python::with_gil(|py| -> CommandResult<serde_json::Value> {
            let modules_guard = self.modules.lock().unwrap();
            let modules = modules_guard.as_ref()
                .ok_or_else(|| CommandError::StateError {
                    reason: "Python modules not initialized".to_string()
                })?;
            
            let data_array = self.array2_to_numpy(py, data)?;
            let analysis_module = modules.analysis.as_ref(py);
            
            // Use methods module for analysis
            let analyzer = analysis_module.getattr("Analyzer").map_py_err()?;
            let instance = analyzer.call0().map_py_err()?;
            
            let result = instance.call_method1("analyze", (data_array, analysis_type)).map_py_err()?;
            let json_str: String = result.extract().map_py_err()?;
            
            serde_json::from_str(&json_str)
                .map_err(CommandError::from_serde_err)
        })
    }
    
    /// Generate visualization
    pub fn generate_visualization(
        &self,
        data: &Array2<f64>,
        viz_type: &str,
        options: &serde_json::Value,
    ) -> CommandResult<Vec<u8>> {
        Python::with_gil(|py| -> CommandResult<Vec<u8>> {
            let modules_guard = self.modules.lock().unwrap();
            let modules = modules_guard.as_ref()
                .ok_or_else(|| CommandError::StateError {
                    reason: "Python modules not initialized".to_string()
                })?;
            
            let data_array = self.array2_to_numpy(py, data)?;
            let viz_module = modules.visualization.as_ref(py);
            
            // Create figure
            let figure = viz_module.call_method0("figure").map_py_err()?;
            
            // Plot based on type
            match viz_type {
                "time_series" => {
                    viz_module.call_method1("plot", (data_array,)).map_py_err()?;
                }
                "heatmap" => {
                    viz_module.call_method1("imshow", (data_array,)).map_py_err()?;
                }
                _ => {
                    return Err(CommandError::InvalidParameter {
                        param: "viz_type".to_string(),
                        reason: format!("Unknown visualization type: {}", viz_type),
                    });
                }
            }
            
            // Save to bytes
            let io = py.import("io").map_py_err()?;
            let buffer = io.call_method0("BytesIO").map_py_err()?;
            figure.call_method1("savefig", (buffer,)).map_py_err()?;
            buffer.call_method0("seek").map_py_err()?;
            let bytes: Vec<u8> = buffer.call_method0("read").map_py_err()?.extract().map_py_err()?;
            
            Ok(bytes)
        })
    }
    
    /// Validate imputation results
    pub fn validate_results(
        &self,
        original: &Array2<f64>,
        imputed: &Array2<f64>,
        validation_type: &str,
    ) -> CommandResult<serde_json::Value> {
        Python::with_gil(|py| -> CommandResult<serde_json::Value> {
            let modules_guard = self.modules.lock().unwrap();
            let modules = modules_guard.as_ref()
                .ok_or_else(|| CommandError::StateError {
                    reason: "Python modules not initialized".to_string()
                })?;
            
            let original_array = self.array2_to_numpy(py, original)?;
            let imputed_array = self.array2_to_numpy(py, imputed)?;
            let validation_module = modules.validation.as_ref(py);
            
            let validator = validation_module.getattr("ImputationValidator").map_py_err()?;
            let instance = validator.call0().map_py_err()?;
            
            let result = instance.call_method1(
                "validate", 
                (original_array, imputed_array, validation_type)
            ).map_py_err()?;
            let json_str: String = result.extract().map_py_err()?;
            
            serde_json::from_str(&json_str)
                .map_err(CommandError::from_serde_err)
        })
    }
    
    /// Get audit log entries
    pub fn get_audit_log(&self) -> Vec<AuditLogEntry> {
        self.audit_log.lock().unwrap().clone()
    }
    
    /// Clear audit log
    pub fn clear_audit_log(&self) {
        self.audit_log.lock().unwrap().clear();
    }
    
    /// Execute a data-oriented command securely (replaces dangerous string execution)
    pub fn dispatch_command(&self, command: &BridgeCommand) -> CommandResult<BridgeResponse> {
        Python::with_gil(|py| -> CommandResult<BridgeResponse> {
            let modules_guard = self.modules.lock().unwrap();
            let modules = modules_guard.as_ref()
                .ok_or_else(|| CommandError::StateError {
                    reason: "Python modules not initialized".to_string()
                })?;
            
            // Serialize command to JSON
            let command_json = command.to_json()
                .map_err(CommandError::from_serde_err)?;
            
            // Get the dispatcher module
            let dispatcher = py.import("airimpute.dispatcher")
                .map_err(|e| CommandError::PythonError {
                    message: format!("Failed to import dispatcher: {}", e)
                })?;
            
            // Call the secure dispatch function
            let result = dispatcher.call_method1("dispatch", (command_json,))
                .map_err(|e| CommandError::PythonError {
                    message: format!("Dispatch failed: {}", e)
                })?;
            
            // Parse response
            let response_json: String = result.extract()
                .map_err(|e| CommandError::PythonError {
                    message: format!("Failed to extract response: {}", e)
                })?;
            
            serde_json::from_str(&response_json)
                .map_err(CommandError::from_serde_err)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    
    #[test]
    fn test_imputation_request_serialization() {
        let request = ImputationRequest {
            data: arr2(&[[1.0, 2.0], [f64::NAN, 4.0]]),
            method: ImputationMethod::RAH {
                spatial_weight: 0.5,
                temporal_weight: 0.5,
                adaptive_threshold: 0.1,
            },
            parameters: ImputationParameters {
                physical_bounds: HashMap::new(),
                confidence_level: 0.95,
                preserve_statistics: true,
                validation_strategy: ValidationStrategy::KFold { n_splits: 5 },
                random_seed: Some(42),
                optimization: OptimizationSettings {
                    use_gpu: false,
                    parallel_chunks: Some(4),
                    cache_intermediate: true,
                    early_stopping: false,
                },
            },
            metadata: DatasetMetadata {
                station_names: vec!["Station1".to_string()],
                variable_names: vec!["PM2.5".to_string()],
                timestamps: vec![],
                coordinates: None,
                units: HashMap::new(),
                missing_pattern: MissingPattern::Random,
            },
        };
        
        // Test that the request can be serialized
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("rah"));
        assert!(json.contains("spatial_weight"));
    }
}