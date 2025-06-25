// Stub implementations for when Python support is disabled
use std::path::Path;
use anyhow::Result;
use serde::Serialize;

#[derive(Clone)]
pub struct PythonRuntime;

impl PythonRuntime {
    pub fn new(_python_path: &Path) -> Result<Self> {
        Ok(Self)
    }
    
    pub fn check_health(&self) -> Result<HealthStatus> {
        Ok(HealthStatus {
            healthy: false,
            python_version: "Python support disabled".to_string(),
            missing_packages: vec!["all".to_string()],
            airimpute_available: false,
            memory_usage_mb: 0.0,
        })
    }
    
    #[cfg(debug_assertions)]
    pub async fn execute(&self, _code: &str) -> Result<String> {
        Err(anyhow::anyhow!("Python support is disabled in this build"))
    }
    
    pub async fn get_status(&self) -> RuntimeStatus {
        RuntimeStatus::Error("Python support disabled".to_string())
    }
    
    pub async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
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

#[derive(Debug, Clone, Serialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub python_version: String,
    pub missing_packages: Vec<String>,
    pub airimpute_available: bool,
    pub memory_usage_mb: f64,
}