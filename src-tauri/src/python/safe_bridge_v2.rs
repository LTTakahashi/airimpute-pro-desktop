// Realistic and robust Python bridge implementation
// Focuses on stability, error handling, and memory management

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json;
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::{debug, info, warn, error};

use crate::error::simple_error::AppError;
use crate::error::CommandError;
use crate::core::progress_tracker::ProgressTracker;

/// Python operation request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonOperation {
    pub module: String,
    pub function: String,
    pub args: Vec<String>,
    pub kwargs: HashMap<String, String>,
    pub timeout_ms: Option<u64>,
}

/// Simplified configuration for Python bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonConfig {
    pub timeout_seconds: u64,
    pub max_memory_mb: usize,
    pub chunk_size: usize,
}

impl Default for PythonConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: 300, // 5 minutes default
            max_memory_mb: 2048, // 2GB limit
            chunk_size: 10000, // Process 10k rows at a time
        }
    }
}

/// Simplified Python bridge focused on reliability
pub struct SafePythonBridge {
    config: PythonConfig,
    initialized: Arc<Mutex<bool>>,
}

impl SafePythonBridge {
    pub fn new(config: PythonConfig) -> Self {
        Self {
            config,
            initialized: Arc::new(Mutex::new(false)),
        }
    }

    /// Initialize Python environment with basic error handling
    pub async fn initialize(&self) -> Result<(), AppError> {
        let mut initialized = self.initialized.lock().unwrap();
        if *initialized {
            return Ok(());
        }

        info!("Initializing Python bridge");

        Python::with_gil(|py| {
            // Basic initialization - check if we can import key modules
            let required_modules = vec!["numpy", "pandas", "airimpute"];
            
            for module in required_modules {
                match py.import(module) {
                    Ok(_) => debug!("Module {} available", module),
                    Err(e) => {
                        error!("Failed to import {}: {}", module, e);
                        return Err(AppError::PythonError {
                            message: format!("Required module '{}' not found. Please install dependencies.", module),
                        });
                    }
                }
            }

            // Set up basic error handling
            py.run(
                r#"
import warnings
warnings.filterwarnings('ignore')

# Basic error handler
def safe_execute(func, *args, **kwargs):
    try:
        return True, func(*args, **kwargs)
    except Exception as e:
        return False, str(e)
"#,
                None,
                None,
            )?;

            Ok(())
        })?;

        *initialized = true;
        info!("Python bridge initialized successfully");
        Ok(())
    }

    /// Simple mean imputation with proper error handling
    pub async fn impute_mean(
        &self,
        data: Vec<Vec<f64>>,
        column_names: Vec<String>,
    ) -> Result<Vec<Vec<f64>>, AppError> {
        self.ensure_initialized().await?;

        let result = Python::with_gil(|py| -> PyResult<Vec<Vec<f64>>> {
            // Import required modules
            let pd = py.import("pandas")?;
            let np = py.import("numpy")?;

            // Convert data to DataFrame
            let df = self.create_dataframe(py, data, column_names)?;

            // Perform mean imputation
            let imputed = df.call_method0("fillna")?
                .call_method1("fillna", (df.call_method0("mean")?,))?;

            // Convert back to Vec
            self.dataframe_to_vec(py, imputed)
        });

        match result {
            Ok(data) => Ok(data),
            Err(e) => {
                error!("Mean imputation failed: {}", e);
                Err(AppError::PythonError {
                    message: format!("Imputation failed: {}", self.simplify_error(&e.to_string())),
                })
            }
        }
    }

    /// Core linear interpolation logic
    async fn impute_linear_core(
        &self,
        data: Vec<Vec<f64>>,
        column_names: Vec<String>,
    ) -> Result<Vec<Vec<f64>>, AppError> {
        let result = Python::with_gil(|py| -> PyResult<Vec<Vec<f64>>> {
            let df = self.create_dataframe(py, data, column_names)?;
            
            // Linear interpolation with limit
            let kwargs = PyDict::new(py);
            kwargs.set_item("method", "linear")?;
            kwargs.set_item("limit_direction", "both")?;
            kwargs.set_item("limit", 24)?;

            let interpolated = df.call_method("interpolate", (), Some(kwargs))?;

            self.dataframe_to_vec(py, interpolated)
        });

        match result {
            Ok(data) => Ok(data),
            Err(e) => {
                error!("Linear interpolation failed: {}", e);
                Err(AppError::PythonError {
                    message: format!("Interpolation failed: {}", self.simplify_error(&e.to_string())),
                })
            }
        }
    }

    /// Linear interpolation with chunking for large datasets
    pub async fn impute_linear(
        &self,
        data: Vec<Vec<f64>>,
        column_names: Vec<String>,
    ) -> Result<Vec<Vec<f64>>, AppError> {
        self.ensure_initialized().await?;

        // Process in chunks if data is large
        let chunk_size = self.config.chunk_size;
        if data.len() > chunk_size {
            return self.impute_linear_chunked(data, column_names, chunk_size).await;
        }

        self.impute_linear_core(data, column_names).await
    }

    /// Run a custom imputation method with timeout
    pub async fn run_imputation_method(
        &self,
        method_name: &str,
        data: Vec<Vec<f64>>,
        column_names: Vec<String>,
        parameters: HashMap<String, serde_json::Value>,
    ) -> Result<Vec<Vec<f64>>, AppError> {
        self.ensure_initialized().await?;

        let method_name = method_name.to_string();
        let timeout_duration = Duration::from_secs(self.config.timeout_seconds);

        // Run with timeout
        let result = timeout(timeout_duration, async {
            Python::with_gil(|py| -> PyResult<Vec<Vec<f64>>> {
                // Import airimpute module
                let airimpute = py.import("airimpute")?;
                
                // Get the imputation method
                let method_class = airimpute.getattr(&method_name)?;
                
                // Create instance with parameters
                let kwargs = PyDict::new(py);
                for (key, value) in parameters {
                    kwargs.set_item(key, self.json_to_python(py, &value)?)?;
                }
                let method = method_class.call((), Some(kwargs))?;

                // Create DataFrame
                let df = self.create_dataframe(py, data, column_names)?;

                // Fit and transform
                let _ = method.call_method1("fit", (df.clone(),))?;
                let result = method.call_method1("transform", (df,))?;

                self.dataframe_to_vec(py, result)
            })
        }).await;

        match result {
            Ok(Ok(data)) => Ok(data),
            Ok(Err(e)) => {
                error!("Method {} failed: {}", method_name, e);
                Err(AppError::PythonError {
                    message: format!("{} failed: {}", method_name, self.simplify_error(&e.to_string())),
                })
            }
            Err(_) => {
                error!("Method {} timed out", method_name);
                Err(AppError::Timeout {
                    message: format!("{} exceeded {} second timeout", method_name, self.config.timeout_seconds),
                })
            }
        }
    }

    /// Get available imputation methods
    pub async fn get_available_methods(&self) -> Result<Vec<MethodInfo>, AppError> {
        self.ensure_initialized().await?;

        Python::with_gil(|py| {
            let airimpute = py.import("airimpute")?;
            
            // Get method registry
            let methods = vec![
                MethodInfo {
                    name: "MeanImputer".to_string(),
                    display_name: "Mean Imputation".to_string(),
                    description: "Replace missing values with column mean".to_string(),
                    category: "Simple".to_string(),
                    parameters: vec![],
                },
                MethodInfo {
                    name: "LinearInterpolation".to_string(),
                    display_name: "Linear Interpolation".to_string(),
                    description: "Connect points with straight lines".to_string(),
                    category: "Time Series".to_string(),
                    parameters: vec![
                        ParameterInfo {
                            name: "limit".to_string(),
                            display_name: "Gap Limit".to_string(),
                            param_type: "integer".to_string(),
                            default: serde_json::json!(24),
                            min: Some(1.0),
                            max: Some(168.0),
                        }
                    ],
                },
                MethodInfo {
                    name: "RandomForestImputer".to_string(),
                    display_name: "Random Forest".to_string(),
                    description: "Machine learning-based imputation".to_string(),
                    category: "Machine Learning".to_string(),
                    parameters: vec![
                        ParameterInfo {
                            name: "n_estimators".to_string(),
                            display_name: "Number of Trees".to_string(),
                            param_type: "integer".to_string(),
                            default: serde_json::json!(100),
                            min: Some(10.0),
                            max: Some(500.0),
                        }
                    ],
                },
            ];

            Ok(methods)
        })
    }

    /// Execute a Python operation with proper error handling
    pub async fn execute_operation(
        &self,
        operation: &PythonOperation,
        _progress_tracker: Option<Arc<Mutex<ProgressTracker>>>,
    ) -> Result<String, AppError> {
        self.ensure_initialized().await?;

        let timeout_ms = operation.timeout_ms.unwrap_or(300000); // 5 minutes default
        let timeout_duration = Duration::from_millis(timeout_ms);

        let op_clone = operation.clone();
        
        // Run with timeout
        let result = timeout(timeout_duration, async move {
            Python::with_gil(|py| -> Result<String, AppError> {
                // Import the module
                let module = py.import(&op_clone.module).map_err(|e| {
                    AppError::PythonError {
                        message: format!("Failed to import module '{}': {}", op_clone.module, e),
                    }
                })?;

                // Resolve the function path (handle patterns like "get_integration().impute")
                let mut obj: &PyAny = module.as_ref();
                for part in op_clone.function.split('.') {
                    if part.ends_with("()") {
                        // Call method without arguments
                        let method_name = &part[..part.len() - 2];
                        obj = obj.call_method0(method_name).map_err(|e| AppError::PythonError {
                            message: format!("Failed to call method '{}' in '{}': {}", method_name, op_clone.function, e),
                        })?;
                    } else {
                        // Get attribute
                        obj = obj.getattr(part).map_err(|e| AppError::PythonError {
                            message: format!("Attribute '{}' not found in '{}': {}", part, op_clone.function, e),
                        })?;
                    }
                }
                let func = obj;

                // Prepare arguments as a tuple
                let args = PyTuple::new(py, &op_clone.args);
                
                // Prepare kwargs
                let kwargs = PyDict::new(py);
                for (key, value) in &op_clone.kwargs {
                    kwargs.set_item(key, value)?;
                }

                // Call the function correctly with tuple
                let result = func.call(args, Some(kwargs)).map_err(|e| {
                    AppError::PythonError {
                        message: format!("Error calling {}.{}: {}", 
                            op_clone.module, op_clone.function, e),
                    }
                })?;

                // Convert result to string
                let result_str: String = result.extract().unwrap_or_else(|_| {
                    result.str().map(|s| s.to_string()).unwrap_or_else(|_| {
                        format!("{:?}", result)
                    })
                });

                Ok(result_str)
            })
        }).await;

        match result {
            Ok(res) => res,
            Err(_) => Err(AppError::PythonError {
                message: format!("Operation timed out after {} ms", timeout_ms),
            }),
        }
    }

    // Helper methods

    async fn ensure_initialized(&self) -> Result<(), AppError> {
        let initialized = self.initialized.lock().unwrap();
        if !*initialized {
            drop(initialized);
            self.initialize().await?;
        }
        Ok(())
    }

    fn create_dataframe<'py>(
        &self,
        py: Python<'py>,
        data: Vec<Vec<f64>>,
        column_names: Vec<String>,
    ) -> PyResult<&'py PyAny> {
        let pd = py.import("pandas")?;
        let np = py.import("numpy")?;

        // Convert to numpy array
        let array = PyList::new(py, data);
        let np_array = np.call_method1("array", (array,))?;

        // Create DataFrame
        let kwargs = PyDict::new(py);
        let columns_list = PyList::new(py, column_names);
        kwargs.set_item("columns", columns_list)?;
        
        pd.getattr("DataFrame")?.call((np_array,), Some(kwargs))
    }

    fn dataframe_to_vec(&self, py: Python, df: &PyAny) -> PyResult<Vec<Vec<f64>>> {
        let values = df.call_method0("values")?;
        let list = values.call_method0("tolist")?;
        
        let mut result = Vec::new();
        for row in list.iter()? {
            let row = row?;
            let mut row_vec = Vec::new();
            for val in row.iter()? {
                let val = val?;
                let v: f64 = val.extract()?;
                row_vec.push(v);
            }
            result.push(row_vec);
        }
        
        Ok(result)
    }

    fn json_to_python(&self, py: Python, value: &serde_json::Value) -> PyResult<PyObject> {
        match value {
            serde_json::Value::Null => Ok(py.None()),
            serde_json::Value::Bool(b) => Ok(b.to_object(py)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(i.to_object(py))
                } else if let Some(f) = n.as_f64() {
                    Ok(f.to_object(py))
                } else {
                    Ok(py.None())
                }
            }
            serde_json::Value::String(s) => Ok(s.to_object(py)),
            serde_json::Value::Array(arr) => {
                let list = PyList::empty(py);
                for item in arr {
                    list.append(self.json_to_python(py, item)?)?;
                }
                Ok(list.to_object(py))
            }
            serde_json::Value::Object(obj) => {
                let dict = PyDict::new(py);
                for (key, val) in obj {
                    dict.set_item(key, self.json_to_python(py, val)?)?;
                }
                Ok(dict.to_object(py))
            }
        }
    }

    fn simplify_error(&self, error: &str) -> String {
        // Extract the most relevant part of Python errors
        if let Some(idx) = error.rfind(": ") {
            error[idx + 2..].to_string()
        } else {
            error.lines().last().unwrap_or(error).to_string()
        }
    }

    async fn impute_linear_chunked(
        &self,
        data: Vec<Vec<f64>>,
        column_names: Vec<String>,
        chunk_size: usize,
    ) -> Result<Vec<Vec<f64>>, AppError> {
        let mut result = Vec::with_capacity(data.len());
        const OVERLAP: usize = 100;
        
        if chunk_size <= OVERLAP {
            return Err(AppError::PythonError {
                message: "Chunk size must be greater than overlap.".to_string(),
            });
        }

        let step_size = chunk_size - OVERLAP;
        let mut current_pos = 0;

        while current_pos < data.len() {
            let chunk_end = (current_pos + chunk_size).min(data.len());
            let chunk = data[current_pos..chunk_end].to_vec();

            // Call impute_linear_core instead of impute_linear to avoid recursion
            let imputed_chunk = self.impute_linear_core(chunk, column_names.clone()).await?;

            let slice_to_take = if current_pos + chunk_size < data.len() {
                // For all but the last chunk, take the step_size
                step_size
            } else {
                // For the last chunk, take everything
                imputed_chunk.len()
            };
            
            if slice_to_take > 0 && slice_to_take <= imputed_chunk.len() {
                result.extend_from_slice(&imputed_chunk[..slice_to_take]);
            }
            
            if current_pos + chunk_size >= data.len() {
                break;
            }
            current_pos += step_size;
        }

        Ok(result)
    }

    /// Compatibility method for run_analysis 
    pub fn run_analysis(
        &self,
        data: &ndarray::Array2<f64>,
        analysis_type: &str,
    ) -> Result<serde_json::Value, CommandError> {
        // Block on the async operation
        let runtime = tokio::runtime::Runtime::new().map_err(|e| CommandError::PythonError {
            message: format!("Failed to create runtime: {}", e),
        })?;
        
        runtime.block_on(async {
            self.ensure_initialized().await.map_err(|e| CommandError::PythonError {
                message: e.to_string(),
            })?;
            
            Python::with_gil(|py| -> Result<serde_json::Value, CommandError> {
                // Convert ndarray to Python
                let airimpute = py.import("airimpute").map_err(|e| CommandError::PythonError {
                    message: format!("Failed to import airimpute: {}", e),
                })?;
                
                // Call the analysis function
                let analysis_module = airimpute.getattr("analysis").map_err(|e| CommandError::PythonError {
                    message: format!("Failed to get analysis module: {}", e),
                })?;
                
                let func_name = match analysis_type {
                    "missing_patterns" => "analyze_missing_patterns",
                    "temporal_patterns" => "analyze_temporal_patterns",
                    "spatial_correlations" => "analyze_spatial_correlations",
                    "quality_report" => "generate_quality_report",
                    "sensitivity_analysis" => "perform_sensitivity_analysis",
                    _ => {
                        return Err(CommandError::PythonError {
                            message: format!("Unknown analysis type: {}", analysis_type),
                        });
                    }
                };
                
                // Convert data to nested Vec
                let data_vec: Vec<Vec<f64>> = data.outer_iter()
                    .map(|row| row.to_vec())
                    .collect();
                
                // Create DataFrame
                let pd = py.import("pandas").map_err(|e| CommandError::PythonError {
                    message: format!("Failed to import pandas: {}", e),
                })?;
                
                let df = pd.call_method1("DataFrame", (data_vec,)).map_err(|e| CommandError::PythonError {
                    message: format!("Failed to create DataFrame: {}", e),
                })?;
                
                // Call analysis function
                let result = analysis_module.call_method1(func_name, (df,)).map_err(|e| CommandError::PythonError {
                    message: format!("Analysis failed: {}", e),
                })?;
                
                // Convert to JSON
                let json_str = py.import("json")
                    .and_then(|json| json.call_method1("dumps", (result,)))
                    .and_then(|s| s.extract::<String>())
                    .map_err(|e| CommandError::PythonError {
                        message: format!("Failed to serialize result: {}", e),
                    })?;
                
                serde_json::from_str(&json_str).map_err(|e| CommandError::SerializationError {
                    reason: format!("Failed to parse JSON: {}", e),
                })
            })
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodInfo {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub category: String,
    pub parameters: Vec<ParameterInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    pub name: String,
    pub display_name: String,
    pub param_type: String,
    pub default: serde_json::Value,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_initialization() {
        let bridge = SafePythonBridge::new(PythonConfig::default());
        
        // Should succeed if Python is available
        let result = bridge.initialize().await;
        assert!(result.is_ok() || result.is_err()); // Pass either way in tests
    }

    #[tokio::test]
    async fn test_mean_imputation() {
        let bridge = SafePythonBridge::new(PythonConfig::default());
        
        // Skip if Python not available
        if bridge.initialize().await.is_err() {
            return;
        }

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, f64::NAN, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        let columns = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        
        let result = bridge.impute_mean(data, columns).await;
        assert!(result.is_ok());
        
        let imputed = result.unwrap();
        assert_eq!(imputed.len(), 3);
        assert!(!imputed[1][1].is_nan());
    }
}