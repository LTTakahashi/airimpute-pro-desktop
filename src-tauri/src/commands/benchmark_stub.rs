// Stub implementation of benchmark commands when Python support is disabled
use crate::error::CommandError;
use crate::state::AppState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tauri::{State, command};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDataset {
    pub name: String,
    pub size: usize,
    pub missing_percentage: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub dataset: String,
    pub method: String,
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
    pub runtime_seconds: f64,
    pub memory_mb: f64,
    pub parameters: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub datasets: Vec<BenchmarkDataset>,
    pub methods: Vec<String>,
    pub results: Vec<BenchmarkResult>,
    pub summary: BenchmarkSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub best_method: String,
    pub average_improvement: f64,
    pub total_runtime: f64,
}

#[command]
pub async fn get_benchmark_datasets(
    _state: State<'_, Arc<AppState>>,
) -> Result<Vec<BenchmarkDataset>, String> {
    Err("Python support is disabled in this build".to_string())
}

#[command]
pub async fn run_benchmark(
    _dataset_name: String,
    _methods: Vec<String>,
    _save_results: bool,
    _state: State<'_, Arc<AppState>>,
) -> Result<BenchmarkComparison, String> {
    Err("Python support is disabled in this build".to_string())
}

#[command]
pub async fn get_benchmark_history(
    _limit: Option<usize>,
    _state: State<'_, Arc<AppState>>,
) -> Result<Vec<BenchmarkResult>, String> {
    Err("Python support is disabled in this build".to_string())
}

#[command]
pub async fn export_benchmark_report(
    _results: BenchmarkComparison,
    _format: String,
    _state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    Err("Python support is disabled in this build".to_string())
}

#[command]
pub async fn get_benchmark_results(
    _state: State<'_, Arc<AppState>>,
) -> Result<Vec<BenchmarkResult>, String> {
    Err("Python support is disabled in this build".to_string())
}

#[command]
pub async fn export_benchmark_results(
    _results: Vec<BenchmarkResult>,
    _format: String,
    _state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    Err("Python support is disabled in this build".to_string())
}

#[command]
pub async fn generate_reproducibility_certificate(
    _benchmark_id: String,
    _state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    Err("Python support is disabled in this build".to_string())
}