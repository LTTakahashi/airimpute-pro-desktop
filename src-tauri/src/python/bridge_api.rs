use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Data-oriented API for Python bridge
/// This replaces dangerous string-based code execution with structured commands

/// An enum that explicitly lists all allowed operations
/// Anything not in this list is impossible to call
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum DataFrameOp {
    // Data loading operations
    LoadCsv,
    LoadExcel,
    LoadParquet,
    
    // Data transformation operations
    SelectColumns,
    FilterByValue,
    DropNa,
    FillNa,
    
    // Mathematical operations
    MultiplyConstant,
    AddColumns,
    SubtractColumns,
    ApplyFunction,
    
    // Imputation operations
    ImputeMean,
    ImputeMedian,
    ImputeInterpolate,
    ImputeAdvanced,
    
    // Statistical operations
    CalculateStats,
    CalculateCorrelation,
    CalculateDistribution,
    
    // Validation operations
    ValidateData,
    CheckMissing,
    CheckOutliers,
}

/// The main command structure sent over the bridge
#[derive(Serialize, Deserialize, Debug)]
pub struct BridgeCommand {
    pub operation: DataFrameOp,
    /// Parameters for the operation - using JSON for flexibility
    pub params: HashMap<String, serde_json::Value>,
}

/// Response structure from Python operations
#[derive(Serialize, Deserialize, Debug)]
pub struct BridgeResponse {
    pub status: ResponseStatus,
    pub data: Option<serde_json::Value>,
    pub message: Option<String>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Success,
    Error,
    Warning,
}

/// Helper to create commands safely
impl BridgeCommand {
    pub fn new(operation: DataFrameOp) -> Self {
        Self {
            operation,
            params: HashMap::new(),
        }
    }
    
    pub fn with_param<T: Serialize>(mut self, key: &str, value: T) -> Self {
        self.params.insert(key.to_string(), serde_json::to_value(value).unwrap());
        self
    }
    
    /// Serialize to JSON for transmission
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Safe parameter builders for common operations
pub mod builders {
    use super::*;
    
    
    pub fn multiply_constant(column: &str, value: f64, target_column: &str) -> BridgeCommand {
        BridgeCommand::new(DataFrameOp::MultiplyConstant)
            .with_param("source_column", column)
            .with_param("value", value)
            .with_param("target_column", target_column)
    }
    
    pub fn select_columns(columns: Vec<String>) -> BridgeCommand {
        BridgeCommand::new(DataFrameOp::SelectColumns)
            .with_param("columns", columns)
    }
    
    pub fn filter_by_value(column: &str, operator: &str, value: serde_json::Value) -> BridgeCommand {
        BridgeCommand::new(DataFrameOp::FilterByValue)
            .with_param("column", column)
            .with_param("operator", operator)
            .with_param("value", value)
    }
    
    pub fn calculate_stats(columns: Option<Vec<String>>) -> BridgeCommand {
        let mut cmd = BridgeCommand::new(DataFrameOp::CalculateStats);
        if let Some(cols) = columns {
            cmd = cmd.with_param("columns", cols);
        }
        cmd
    }
    
    pub fn impute_advanced(method: &str, params: HashMap<String, serde_json::Value>) -> BridgeCommand {
        BridgeCommand::new(DataFrameOp::ImputeAdvanced)
            .with_param("method", method)
            .with_param("params", params)
    }
}