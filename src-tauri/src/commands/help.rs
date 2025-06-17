use tauri::{command, State};
use std::sync::Arc;
use serde::Serialize;

use crate::state::AppState;
use crate::services::offline_resources::{HelpContent, Tutorial, MethodDocumentation};

#[derive(Debug, Serialize)]
pub struct HelpSearchResult {
    pub results: Vec<HelpContent>,
    pub query: String,
    pub total_found: usize,
}

#[derive(Debug, Serialize)]
pub struct TutorialListResponse {
    pub tutorials: Vec<TutorialInfo>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct TutorialInfo {
    pub id: String,
    pub title: String,
    pub description: String,
    pub difficulty: String,
    pub estimated_time: String,
}

/// Search offline help content
#[command]
pub fn search_help(
    state: State<'_, Arc<AppState>>,
    query: String,
) -> Result<HelpSearchResult, String> {
    let results = state.offline_resources
        .search_help(&query)
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    
    let total_found = results.len();
    
    Ok(HelpSearchResult {
        results,
        query,
        total_found,
    })
}

/// Get method documentation
#[command]
pub fn get_method_documentation(
    state: State<'_, Arc<AppState>>,
    method_id: String,
) -> Result<Option<MethodDocumentation>, String> {
    Ok(state.offline_resources
        .get_method_doc(&method_id)
        .cloned())
}

/// List available tutorials
#[command]
pub fn list_tutorials(
    state: State<'_, Arc<AppState>>,
) -> Result<TutorialListResponse, String> {
    let tutorials: Vec<TutorialInfo> = state.offline_resources
        .list_tutorials()
        .into_iter()
        .map(|(id, tutorial)| TutorialInfo {
            id: id.clone(),
            title: tutorial.title.clone(),
            description: tutorial.description.clone(),
            difficulty: tutorial.difficulty.clone(),
            estimated_time: tutorial.estimated_time.clone(),
        })
        .collect();
    
    let total = tutorials.len();
    
    Ok(TutorialListResponse {
        tutorials,
        total,
    })
}

/// Get specific tutorial
#[command]
pub fn get_tutorial(
    state: State<'_, Arc<AppState>>,
    tutorial_id: String,
) -> Result<Option<Tutorial>, String> {
    Ok(state.offline_resources
        .get_tutorial(&tutorial_id)
        .cloned())
}

/// Get offline status information
#[command]
pub fn get_offline_status() -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "mode": "offline",
        "authentication_required": false,
        "network_required": false,
        "telemetry_enabled": false,
        "update_check_enabled": false,
        "cloud_features": false,
        "data_privacy": "All data remains on your device",
        "features": {
            "local_processing": true,
            "gpu_acceleration": true,
            "embedded_documentation": true,
            "offline_help": true,
            "local_caching": true,
            "auto_save": true,
            "crash_recovery": true
        }
    }))
}

/// Get quick start guide
#[command]
pub fn get_quick_start_guide() -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "title": "Quick Start Guide",
        "steps": [
            {
                "step": 1,
                "title": "Import Your Data",
                "description": "Click 'Import Data' or drag and drop your CSV/Excel file",
                "tip": "Supported formats: CSV, Excel, NetCDF, HDF5"
            },
            {
                "step": 2,
                "title": "Analyze Missing Patterns",
                "description": "View automatic analysis of missing data patterns",
                "tip": "Understanding patterns helps choose the best imputation method"
            },
            {
                "step": 3,
                "title": "Select Imputation Method",
                "description": "Choose from statistical, ML, or deep learning methods",
                "tip": "Use the recommendation system for guidance"
            },
            {
                "step": 4,
                "title": "Configure Parameters",
                "description": "Adjust method-specific parameters or use defaults",
                "tip": "Hover over parameters for detailed explanations"
            },
            {
                "step": 5,
                "title": "Run Imputation",
                "description": "Click 'Run' and monitor progress",
                "tip": "Large datasets can use GPU acceleration if available"
            },
            {
                "step": 6,
                "title": "Review & Export",
                "description": "Validate results and export in your preferred format",
                "tip": "Generate reproducibility certificates for publications"
            }
        ],
        "keyboard_shortcuts": {
            "Ctrl+O": "Open/Import file",
            "Ctrl+S": "Save project",
            "Ctrl+E": "Export results",
            "F1": "Context help",
            "F5": "Run imputation",
            "Ctrl+Z": "Undo",
            "Ctrl+Shift+Z": "Redo"
        }
    }))
}

/// Get sample datasets info
#[command]
pub fn get_sample_datasets() -> Result<Vec<serde_json::Value>, String> {
    Ok(vec![
        serde_json::json!({
            "name": "Air Quality Station Network",
            "description": "Multi-station air quality measurements with spatial correlations",
            "file": "samples/air_quality_network.csv",
            "size": "2.5 MB",
            "variables": ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"],
            "missing_rate": 0.15,
            "pattern": "Random (MCAR)",
            "recommended_methods": ["KNN", "Random Forest", "GNN"]
        }),
        serde_json::json!({
            "name": "Long-term Monitoring Time Series",
            "description": "5-year hourly measurements from single station",
            "file": "samples/longterm_timeseries.csv",
            "size": "8.7 MB",
            "variables": ["PM2.5", "Temperature", "Humidity", "Wind Speed"],
            "missing_rate": 0.22,
            "pattern": "Temporal blocks",
            "recommended_methods": ["ARIMA", "Transformer", "LSTM"]
        }),
        serde_json::json!({
            "name": "Urban Pollution Grid",
            "description": "Gridded urban air quality data with high spatial resolution",
            "file": "samples/urban_grid.nc",
            "size": "15.3 MB",
            "variables": ["PM2.5", "NO2", "Traffic Density"],
            "missing_rate": 0.30,
            "pattern": "Spatial clusters",
            "recommended_methods": ["Kriging", "GNN", "Spatial RF"]
        })
    ])
}