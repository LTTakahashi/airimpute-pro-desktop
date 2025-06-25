// Conditional compilation based on python-support feature
#[cfg(feature = "python-support")]
pub mod bridge;
#[cfg(feature = "python-support")]
pub mod bridge_api;

// Stub for when Python is disabled
#[cfg(not(feature = "python-support"))]
pub mod bridge_stub;

#[cfg(not(feature = "python-support"))]
pub use bridge_stub as bridge;
#[cfg(feature = "python-support")]
pub mod embedded_runtime;
#[cfg(feature = "python-support")]
pub mod safe_bridge;
#[cfg(feature = "python-support")]
pub mod arrow_bridge;

// Conditional compilation for safe_bridge_v2
#[cfg(feature = "python-support")]
pub mod safe_bridge_v2;

#[cfg(not(feature = "python-support"))]
pub mod safe_bridge_v2_stub;

#[cfg(not(feature = "python-support"))]
pub use safe_bridge_v2_stub as safe_bridge_v2;

// Conditional compilation for arrow_bridge  
#[cfg(feature = "python-support")]
pub use arrow_bridge::{PythonWorkerPool, SafePythonAction, PythonTask, PythonResponse, ndarray_to_arrow, arrow_to_ndarray, 
    serialize_record_batch, deserialize_record_batch, TaskStatus};

#[cfg(not(feature = "python-support"))]
pub mod arrow_bridge_stub;

#[cfg(not(feature = "python-support"))]
pub use arrow_bridge_stub::{PythonWorkerPool, SafePythonAction, PythonTask, ndarray_to_arrow, arrow_to_ndarray, 
    serialize_record_batch, deserialize_record_batch, TaskStatus};

// Stub module for when Python is disabled
#[cfg(not(feature = "python-support"))]
pub mod stub;

#[cfg(test)]
mod test_bridge;

// Re-export commonly used types
pub use safe_bridge_v2::PythonOperation;

// When Python support is disabled, use stub implementations
#[cfg(not(feature = "python-support"))]
pub use stub::{PythonRuntime, RuntimeStatus, HealthStatus};

// When Python support is enabled, use real implementations
#[cfg(feature = "python-support")]
pub use python_impl::{PythonRuntime, RuntimeStatus, HealthStatus};

#[cfg(feature = "python-support")]
mod python_impl {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use anyhow::{Result, Context};
    use pyo3::prelude::*;
    use serde::Serialize;
    use tokio::sync::Mutex;
    use tracing::{info, debug, warn, error};
    
    use crate::python::bridge::PythonBridge;

    /// Python runtime manager for the embedded Python interpreter
    pub struct PythonRuntime {
    /// Path to Python modules
    python_path: PathBuf,
    
    /// Python-Rust bridge
    pub bridge: Arc<PythonBridge>,
    
    /// Runtime status
    status: Arc<Mutex<RuntimeStatus>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeStatus {
    Uninitialized,
    Initializing,
    Ready,
    Busy,
    Error(String),
    ShuttingDown,
}

impl PythonRuntime {
    /// Create a new Python runtime
    pub fn new(_python_path: &Path) -> Result<Self> {
        info!("Initializing Python runtime");
        
        // Use the scripts directory relative to the executable
        let exe_path = std::env::current_exe()?;
        let exe_dir = exe_path.parent().context("Failed to get exe directory")?;
        
        // In development, scripts are in ../scripts relative to target/debug
        // In production, they'll be bundled differently
        let scripts_path = if cfg!(debug_assertions) {
            exe_dir.parent()
                .and_then(|p| p.parent())
                .map(|p| p.join("scripts"))
                .unwrap_or_else(|| exe_dir.join("scripts"))
        } else {
            exe_dir.join("scripts")
        };
        
        info!("Using Python modules from: {:?}", scripts_path);
        
        // Initialize Python interpreter
        pyo3::prepare_freethreaded_python();
        
        // Add scripts path to Python sys.path
        Python::with_gil(|py| {
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            let scripts_path_str = scripts_path.to_str()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Python scripts path contains non-UTF-8 characters"))?;
            path.call_method1("insert", (0, scripts_path_str))?;
            
            // Also add the src/python directory for our fixes
            let python_src_path = exe_dir.join("src").join("python");
            let python_src_path_str = python_src_path.to_str()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Python source path contains non-UTF-8 characters"))?;
            path.call_method1("insert", (0, python_src_path_str))?;
            
            // CRITICAL: Import GI version fix BEFORE any other imports
            // This prevents libsoup3 from being loaded
            match py.import("gi_version_fix") {
                Ok(_) => info!("GI version fix applied successfully"),
                Err(e) => info!("GI version fix not needed or failed: {}", e),
            }
            
            Ok::<(), PyErr>(())
        }).context("Failed to initialize Python paths")?;
        
        // Create bridge
        let bridge = PythonBridge::new(&scripts_path)?;
        
        Ok(Self {
            python_path: scripts_path,
            bridge: Arc::new(bridge),
            status: Arc::new(Mutex::new(RuntimeStatus::Ready)),
        })
    }
    
    /// Check if Python runtime is healthy
    pub fn check_health(&self) -> Result<HealthStatus> {
        Python::with_gil(|py| {
            // Check Python version
            let sys = py.import("sys")?;
            let version_info = sys.getattr("version_info")?;
            let major: u8 = version_info.getattr("major")?.extract()?;
            let minor: u8 = version_info.getattr("minor")?.extract()?;
            let patch: u8 = version_info.getattr("micro")?.extract()?;
            
            let version = format!("{}.{}.{}", major, minor, patch);
            
            // Check required packages
            let mut missing_packages = Vec::new();
            let required_packages = vec![
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
            ];
            
            for package in &required_packages {
                match py.import(*package) {
                    Ok(_) => debug!("Package {} is available", package),
                    Err(_) => {
                        warn!("Package {} is missing", package);
                        missing_packages.push(package.to_string());
                    }
                }
            }
            
            // Check custom modules
            let airimpute_available = py.import("airimpute").is_ok();
            
            Ok(HealthStatus {
                healthy: missing_packages.is_empty() && airimpute_available,
                python_version: version,
                missing_packages,
                airimpute_available,
                memory_usage_mb: 0.0, // Simplified for now
            })
        })
    }
    
    /// Execute Python code (for development/debugging ONLY)
    #[cfg(debug_assertions)]
    pub async fn execute(&self, code: &str) -> Result<String> {
        let status = self.status.lock().await.clone();
        if status != RuntimeStatus::Ready {
            return Err(anyhow::anyhow!("Python runtime is not ready: {:?}", status));
        }
        
        *self.status.lock().await = RuntimeStatus::Busy;
        
        let result = Python::with_gil(|py| {
            match py.run(code, None, None) {
                Ok(_) => Ok("Code executed successfully".to_string()),
                Err(e) => {
                    let error_msg = format!("Python error: {}", e);
                    error!("{}", error_msg);
                    Err(anyhow::anyhow!(error_msg))
                }
            }
        });
        
        *self.status.lock().await = RuntimeStatus::Ready;
        result
    }
    
    /// Get runtime status
    pub async fn get_status(&self) -> RuntimeStatus {
        self.status.lock().await.clone()
    }
    
    /// Shutdown Python runtime
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Python runtime");
        *self.status.lock().await = RuntimeStatus::ShuttingDown;
        Ok(())
    }
}

/// Health status of the Python runtime
#[derive(Debug, Clone, Serialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub python_version: String,
    pub missing_packages: Vec<String>,
    pub airimpute_available: bool,
    pub memory_usage_mb: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_runtime_status_transitions() {
        let status = RuntimeStatus::Uninitialized;
        assert_eq!(status, RuntimeStatus::Uninitialized);
        
        let status = RuntimeStatus::Ready;
        assert_eq!(status, RuntimeStatus::Ready);
    }
}
}
