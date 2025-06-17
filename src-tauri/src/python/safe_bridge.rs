// Safe Python bridge with comprehensive error handling and resource management
// Implements defensive programming practices for IPC reliability

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyRuntimeError;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json;
use tokio::time::timeout;
use tracing::{debug, info, warn, error, instrument};
use uuid::Uuid;

use crate::error::academic_error::{AcademicError, ErrorContext};
use crate::core::memory_management::{MemoryTracker, MemoryScope};
use crate::core::cancellable_operation::{CancellationToken, OperationManager};

/// Configuration for Python bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonBridgeConfig {
    pub python_path: Option<String>,
    pub virtual_env: Option<String>,
    pub initialization_timeout: Duration,
    pub call_timeout: Duration,
    pub max_retries: u32,
    pub memory_limit: Option<usize>,
    pub enable_profiling: bool,
    pub capture_stdout: bool,
    pub capture_stderr: bool,
}

impl Default for PythonBridgeConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            virtual_env: None,
            initialization_timeout: Duration::from_secs(30),
            call_timeout: Duration::from_secs(300), // 5 minutes
            max_retries: 3,
            memory_limit: Some(4 * 1024 * 1024 * 1024), // 4GB
            enable_profiling: false,
            capture_stdout: true,
            capture_stderr: true,
        }
    }
}

/// Safe Python bridge with comprehensive error handling
pub struct SafePythonBridge {
    config: PythonBridgeConfig,
    state: Arc<RwLock<BridgeState>>,
    memory_tracker: Arc<MemoryTracker>,
    operation_manager: Arc<OperationManager>,
    error_handler: Arc<PythonErrorHandler>,
}

#[derive(Debug)]
struct BridgeState {
    is_initialized: bool,
    initialization_time: Option<Instant>,
    call_count: u64,
    error_count: u64,
    last_error: Option<String>,
    active_calls: HashMap<Uuid, CallInfo>,
}

#[derive(Debug, Clone)]
struct CallInfo {
    id: Uuid,
    method: String,
    start_time: Instant,
    cancellation_token: CancellationToken,
    memory_before: usize,
}

/// Handles Python errors with context preservation
struct PythonErrorHandler {
    error_buffer: Mutex<Vec<PythonError>>,
    max_buffer_size: usize,
}

#[derive(Debug, Clone, Serialize)]
struct PythonError {
    timestamp: chrono::DateTime<chrono::Utc>,
    error_type: String,
    message: String,
    traceback: Option<String>,
    call_context: Option<CallContext>,
}

#[derive(Debug, Clone, Serialize)]
struct CallContext {
    method: String,
    arguments: String,
    duration: Duration,
    memory_usage: usize,
}

impl SafePythonBridge {
    pub fn new(
        config: PythonBridgeConfig,
        memory_tracker: Arc<MemoryTracker>,
        operation_manager: Arc<OperationManager>,
    ) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(BridgeState {
                is_initialized: false,
                initialization_time: None,
                call_count: 0,
                error_count: 0,
                last_error: None,
                active_calls: HashMap::new(),
            })),
            memory_tracker,
            operation_manager,
            error_handler: Arc::new(PythonErrorHandler::new()),
        }
    }

    #[instrument(skip(self))]
    pub async fn initialize(&self) -> Result<(), AcademicError> {
        info!("Initializing Python bridge");
        
        let init_result = timeout(
            self.config.initialization_timeout,
            self.initialize_python_runtime()
        ).await;

        match init_result {
            Ok(Ok(())) => {
                let mut state = self.state.write().unwrap();
                state.is_initialized = true;
                state.initialization_time = Some(Instant::now());
                info!("Python bridge initialized successfully");
                Ok(())
            }
            Ok(Err(e)) => {
                error!("Python initialization failed: {:?}", e);
                Err(e)
            }
            Err(_) => {
                error!("Python initialization timed out");
                Err(AcademicError::InterProcessCommunication {
                    message: "Python initialization timed out".to_string(),
                    command: "initialize".to_string(),
                    python_traceback: None,
                    context: ErrorContext::new(),
                })
            }
        }
    }

    async fn initialize_python_runtime(&self) -> Result<(), AcademicError> {
        Python::with_gil(|py| {
            // Set up Python path if specified
            if let Some(python_path) = &self.config.python_path {
                let sys = py.import("sys")?;
                let path: &PyList = sys.getattr("path")?.downcast()?;
                path.insert(0, python_path)?;
            }

            // Activate virtual environment if specified
            if let Some(venv_path) = &self.config.virtual_env {
                self.activate_virtual_env(py, venv_path)?;
            }

            // Import and validate required modules
            self.validate_python_environment(py)?;

            // Set up signal handlers for graceful shutdown
            self.setup_signal_handlers(py)?;

            // Configure memory limits if specified
            if let Some(limit) = self.config.memory_limit {
                self.set_memory_limit(py, limit)?;
            }

            Ok(())
        }).map_err(|e: PyErr| {
            Python::with_gil(|py| {
                let traceback = e.traceback(py)
                    .and_then(|tb| tb.format().ok())
                    .unwrap_or_else(|| "No traceback available".to_string());

                AcademicError::InterProcessCommunication {
                    message: format!("Python initialization error: {}", e),
                    command: "initialize".to_string(),
                    python_traceback: Some(traceback),
                    context: ErrorContext::new(),
                }
            })
        })
    }

    fn activate_virtual_env(&self, py: Python, venv_path: &str) -> PyResult<()> {
        let activate_script = format!("{}/bin/activate_this.py", venv_path);
        let code = format!(
            r#"
exec(open("{}").read(), {{'__file__': "{}"}})
"#,
            activate_script, activate_script
        );
        py.run(&code, None, None)?;
        Ok(())
    }

    fn validate_python_environment(&self, py: Python) -> PyResult<()> {
        // Check required modules
        let required_modules = vec![
            "numpy", "pandas", "scipy", "sklearn",
            "airimpute", "torch", "numba"
        ];

        for module in required_modules {
            match py.import(module) {
                Ok(_) => debug!("Module {} available", module),
                Err(e) => {
                    warn!("Module {} not available: {}", module, e);
                    // Don't fail initialization, just warn
                }
            }
        }

        // Check Python version
        let sys = py.import("sys")?;
        let version_info = sys.getattr("version_info")?;
        let major: u32 = version_info.getattr("major")?.extract()?;
        let minor: u32 = version_info.getattr("minor")?.extract()?;

        if major < 3 || (major == 3 && minor < 8) {
            return Err(PyRuntimeError::new_err(
                format!("Python {}.{} is not supported. Requires Python 3.8+", major, minor)
            ));
        }

        Ok(())
    }

    fn setup_signal_handlers(&self, py: Python) -> PyResult<()> {
        py.run(
            r#"
import signal
import sys

def signal_handler(sig, frame):
    print(f"Received signal {sig}, cleaning up...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
"#,
            None,
            None,
        )?;
        Ok(())
    }

    fn set_memory_limit(&self, py: Python, limit_bytes: usize) -> PyResult<()> {
        let resource = py.import("resource")?;
        let limit_mb = limit_bytes / (1024 * 1024);
        
        py.run(
            &format!(
                r#"
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, ({} * 1024 * 1024, hard))
"#,
                limit_mb
            ),
            None,
            None,
        )?;
        
        info!("Set Python memory limit to {} MB", limit_mb);
        Ok(())
    }

    #[instrument(skip(self, args))]
    pub async fn call_method<T: for<'de> Deserialize<'de> + Send + 'static>(
        &self,
        module: &str,
        method: &str,
        args: serde_json::Value,
        cancellation_token: CancellationToken,
    ) -> Result<T, AcademicError> {
        // Check if initialized
        {
            let state = self.state.read().unwrap();
            if !state.is_initialized {
                return Err(AcademicError::InterProcessCommunication {
                    message: "Python bridge not initialized".to_string(),
                    command: format!("{}.{}", module, method),
                    python_traceback: None,
                    context: ErrorContext::new(),
                });
            }
        }

        // Create call info
        let call_id = Uuid::new_v4();
        let call_info = CallInfo {
            id: call_id,
            method: format!("{}.{}", module, method),
            start_time: Instant::now(),
            cancellation_token: cancellation_token.clone(),
            memory_before: self.memory_tracker.get_statistics().current_usage,
        };

        // Register active call
        {
            let mut state = self.state.write().unwrap();
            state.active_calls.insert(call_id, call_info.clone());
            state.call_count += 1;
        }

        // Create memory scope
        let _memory_scope = MemoryScope::new(
            format!("Python call: {}.{}", module, method),
            self.memory_tracker.clone(),
        );

        // Execute with timeout and cancellation
        let result = tokio::select! {
            r = timeout(self.config.call_timeout, self.execute_python_call::<T>(
                module,
                method,
                args.clone(),
                call_id,
            )) => r,
            _ = cancellation_token.cancelled() => {
                self.cleanup_call(call_id);
                return Err(AcademicError::InterProcessCommunication {
                    message: "Operation cancelled".to_string(),
                    command: format!("{}.{}", module, method),
                    python_traceback: None,
                    context: ErrorContext::new(),
                });
            }
        };

        // Cleanup and return
        self.cleanup_call(call_id);

        match result {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(e)) => {
                self.handle_python_error(e, &call_info)
            }
            Err(_) => {
                Err(AcademicError::InterProcessCommunication {
                    message: format!("Call timed out after {:?}", self.config.call_timeout),
                    command: format!("{}.{}", module, method),
                    python_traceback: None,
                    context: ErrorContext::new(),
                })
            }
        }
    }

    async fn execute_python_call<T: for<'de> Deserialize<'de>>(
        &self,
        module: &str,
        method: &str,
        args: serde_json::Value,
        call_id: Uuid,
    ) -> Result<T, PyErr> {
        Python::with_gil(|py| {
            // Import module
            let py_module = py.import(module)?;
            
            // Get method
            let py_method = py_module.getattr(method)?;
            
            // Convert arguments
            let py_args = self.json_to_python(py, &args)?;
            
            // Add call tracking
            let kwargs = PyDict::new(py);
            kwargs.set_item("__call_id", call_id.to_string())?;
            
            // Call method
            let result = if let Ok(args_tuple) = py_args.downcast::<PyTuple>(py) {
                py_method.call(args_tuple, Some(kwargs))?
            } else {
                py_method.call1((py_args,))?
            };
            
            // Convert result
            let json_str = self.python_to_json_string(py, result.into())?;
            let value: T = serde_json::from_str(&json_str)
                .map_err(|e| PyRuntimeError::new_err(format!("JSON deserialization failed: {}", e)))?;
            
            Ok(value)
        })
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
                    Err(PyRuntimeError::new_err("Invalid number"))
                }
            }
            serde_json::Value::String(s) => Ok(s.to_object(py)),
            serde_json::Value::Array(arr) => {
                let py_list = PyList::empty(py);
                for item in arr {
                    py_list.append(self.json_to_python(py, item)?)?;
                }
                Ok(py_list.to_object(py))
            }
            serde_json::Value::Object(obj) => {
                let py_dict = PyDict::new(py);
                for (key, val) in obj {
                    py_dict.set_item(key, self.json_to_python(py, val)?)?;
                }
                Ok(py_dict.to_object(py))
            }
        }
    }

    fn python_to_json_string(&self, py: Python, obj: PyObject) -> PyResult<String> {
        let json_module = py.import("json")?;
        let dumps = json_module.getattr("dumps")?;
        let json_str: String = dumps.call1((obj,))?.extract()?;
        Ok(json_str)
    }

    fn cleanup_call(&self, call_id: Uuid) {
        let mut state = self.state.write().unwrap();
        if let Some(call_info) = state.active_calls.remove(&call_id) {
            let duration = call_info.start_time.elapsed();
            let memory_after = self.memory_tracker.get_statistics().current_usage;
            let memory_delta = memory_after as i64 - call_info.memory_before as i64;
            
            debug!(
                "Python call completed: {} in {:?}, memory delta: {}",
                call_info.method,
                duration,
                memory_delta
            );
        }
    }

    fn handle_python_error<T>(
        &self,
        error: PyErr,
        call_info: &CallInfo,
    ) -> Result<T, AcademicError> {
        let mut state = self.state.write().unwrap();
        state.error_count += 1;
        
        Python::with_gil(|py| {
            let error_type = error.get_type(py).name()
                .map_err(|e| AcademicError::NumericalComputation { 
                    context: ErrorContext::new(),
                    message: e.to_string(),
                    operation: "Getting Python error type name".to_string(),
                    numerical_context: Default::default(),
                })?
                .to_string();
            let error_msg = error.value(py).to_string();
            let traceback = error.traceback(py)
                .and_then(|tb| tb.format().ok())
                .unwrap_or_else(|| "No traceback available".to_string());
            
            state.last_error = Some(error_msg.clone());
            
            // Record error
            self.error_handler.record_error(PythonError {
                timestamp: chrono::Utc::now(),
                error_type: error_type.clone(),
                message: error_msg.clone(),
                traceback: Some(traceback.clone()),
                call_context: Some(CallContext {
                    method: call_info.method.clone(),
                    arguments: "".to_string(), // Omit for privacy
                    duration: call_info.start_time.elapsed(),
                    memory_usage: self.memory_tracker.get_statistics().current_usage,
                }),
            });
            
            // Check for specific error types
            if error_type.contains("MemoryError") {
                Err(AcademicError::ResourceExhaustion {
                    resource: crate::error::academic_error::ResourceType::Memory,
                    requested: 0, // Unknown
                    available: 0, // Unknown
                    context: ErrorContext::new(),
                })
            } else {
                Err(AcademicError::InterProcessCommunication {
                    message: error_msg,
                    command: call_info.method.clone(),
                    python_traceback: Some(traceback),
                    context: ErrorContext::new(),
                })
            }
        })
    }

    pub async fn shutdown(&self) -> Result<(), AcademicError> {
        info!("Shutting down Python bridge");
        
        // Cancel all active calls
        let active_calls: Vec<_> = {
            let state = self.state.read().unwrap();
            state.active_calls.values()
                .map(|info| info.cancellation_token.clone())
                .collect()
        };
        
        for token in active_calls {
            token.cancel();
        }
        
        // Wait for calls to complete
        tokio::time::sleep(Duration::from_secs(1)).await;
        
        // Clean up Python runtime
        Python::with_gil(|py| {
            // Force garbage collection
            let gc = py.import("gc").ok();
            if let Some(gc_module) = gc {
                let _ = gc_module.call_method0("collect");
            }
            
            // Clear caches
            let _ = py.run(
                r#"
import sys
sys.modules.clear()
"#,
                None,
                None,
            );
        });
        
        Ok(())
    }

    pub fn get_diagnostics(&self) -> BridgeDiagnostics {
        let state = self.state.read().unwrap();
        let memory_stats = self.memory_tracker.get_statistics();
        
        BridgeDiagnostics {
            is_initialized: state.is_initialized,
            uptime: state.initialization_time.map(|t| t.elapsed()),
            call_count: state.call_count,
            error_count: state.error_count,
            active_calls: state.active_calls.len(),
            memory_usage: memory_stats.current_usage,
            peak_memory: memory_stats.peak_usage,
            last_error: state.last_error.clone(),
            error_history: self.error_handler.get_recent_errors(10),
        }
    }
}

impl PythonErrorHandler {
    fn new() -> Self {
        Self {
            error_buffer: Mutex::new(Vec::with_capacity(1000)),
            max_buffer_size: 1000,
        }
    }

    fn record_error(&self, error: PythonError) {
        let mut buffer = self.error_buffer.lock().unwrap();
        if buffer.len() >= self.max_buffer_size {
            buffer.remove(0);
        }
        buffer.push(error);
    }

    fn get_recent_errors(&self, count: usize) -> Vec<PythonError> {
        let buffer = self.error_buffer.lock().unwrap();
        buffer.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
}

#[derive(Debug, Serialize)]
pub struct BridgeDiagnostics {
    pub is_initialized: bool,
    pub uptime: Option<Duration>,
    pub call_count: u64,
    pub error_count: u64,
    pub active_calls: usize,
    pub memory_usage: usize,
    pub peak_memory: usize,
    pub last_error: Option<String>,
    pub error_history: Vec<PythonError>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_management::MemoryConfig;

    #[tokio::test]
    async fn test_python_bridge_initialization() {
        let config = PythonBridgeConfig::default();
        let memory_tracker = MemoryTracker::new(MemoryConfig::default());
        let operation_manager = Arc::new(OperationManager::new());
        
        let bridge = SafePythonBridge::new(config, memory_tracker, operation_manager);
        
        // Should not be initialized yet
        let diagnostics = bridge.get_diagnostics();
        assert!(!diagnostics.is_initialized);
        
        // Initialize
        let result = bridge.initialize().await;
        assert!(result.is_ok());
        
        // Should now be initialized
        let diagnostics = bridge.get_diagnostics();
        assert!(diagnostics.is_initialized);
    }
}