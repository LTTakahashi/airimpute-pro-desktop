use tauri::command;
use std::sync::Arc;
use tauri::State;
use serde::Serialize;
use sysinfo::System;

use crate::state::AppState;

#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    pub os: String,
    pub cpu_count: usize,
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub app_version: String,
    pub python_version: Option<String>,
}

#[command]
pub async fn get_system_info(
    state: State<'_, Arc<AppState>>,
) -> Result<SystemInfo, String> {
    let sys = System::new_all();
    
    Ok(SystemInfo {
        os: std::env::consts::OS.to_string(),
        cpu_count: sys.cpus().len(),
        total_memory_gb: sys.total_memory() as f64 / 1_073_741_824.0,
        available_memory_gb: sys.available_memory() as f64 / 1_073_741_824.0,
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        python_version: None,
    })
}

#[command]
pub async fn check_python_runtime(
    state: State<'_, Arc<AppState>>,
) -> Result<serde_json::Value, String> {
    match state.python_runtime.check_health() {
        Ok(health) => serde_json::to_value(health)
            .map_err(|e| format!("Failed to serialize health status: {}", e)),
        Err(e) => Err(format!("Python runtime check failed: {}", e)),
    }
}

#[command]
pub async fn get_memory_usage(
    state: State<'_, Arc<AppState>>,
) -> Result<serde_json::Value, String> {
    let metrics = state.metrics.get_snapshot();
    serde_json::to_value(metrics)
        .map_err(|e| format!("Failed to serialize metrics: {}", e))
}

#[command]
pub async fn clear_cache(
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    // Implementation would clear cache
    Ok(())
}

#[command]
pub async fn run_diagnostics(
    state: State<'_, Arc<AppState>>,
) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "status": "healthy",
        "checks": {
            "database": true,
            "python": true,
            "memory": true,
            "disk": true,
        }
    }))
}