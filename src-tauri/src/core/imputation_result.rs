use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Canonical ImputationResult struct that bridges between Python and Rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImputationResult {
    pub imputed_data: Vec<Vec<f64>>,
    pub columns: Vec<String>,
    pub index: Vec<DateTime<Utc>>,
    pub method: String,
    pub parameters: HashMap<String, serde_json::Value>,
    
    // Metrics for each column
    pub metrics: HashMap<String, MetricValues>,
    
    // Uncertainty quantification
    pub uncertainty: Option<Vec<Vec<f64>>>,
    
    // Validation results
    pub validation_results: Option<ValidationResults>,
    
    // Method-specific outputs
    pub method_outputs: Option<HashMap<String, serde_json::Value>>,
    
    // Execution metadata
    pub execution_time_seconds: f64,
    pub memory_usage_mb: Option<f64>,
    pub convergence_info: Option<ConvergenceInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValues {
    pub rmse: f64,
    pub mae: f64,
    pub mape: Option<f64>,
    pub r2: Option<f64>,
    pub correlation: Option<f64>,
    pub bias: Option<f64>,
    pub coverage_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub cross_validation_score: Option<f64>,
    pub temporal_consistency: Option<f64>,
    pub spatial_consistency: Option<f64>,
    pub confidence_intervals_valid: Option<bool>,
    pub statistical_tests: Option<HashMap<String, TestResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub passed: bool,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    pub converged: bool,
    pub iterations: usize,
    pub final_error: f64,
    pub convergence_history: Option<Vec<f64>>,
}

impl ImputationResult {
    /// Create a new ImputationResult from basic components
    pub fn new(
        imputed_data: Vec<Vec<f64>>,
        columns: Vec<String>,
        index: Vec<DateTime<Utc>>,
        method: String,
        parameters: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            imputed_data,
            columns,
            index,
            method,
            parameters,
            metrics: HashMap::new(),
            uncertainty: None,
            validation_results: None,
            method_outputs: None,
            execution_time_seconds: 0.0,
            memory_usage_mb: None,
            convergence_info: None,
        }
    }
    
    /// Add metrics for a specific column
    pub fn add_column_metrics(&mut self, column: String, metrics: MetricValues) {
        self.metrics.insert(column, metrics);
    }
    
    /// Set uncertainty estimates
    pub fn set_uncertainty(&mut self, uncertainty: Vec<Vec<f64>>) {
        self.uncertainty = Some(uncertainty);
    }
    
    /// Set validation results
    pub fn set_validation_results(&mut self, results: ValidationResults) {
        self.validation_results = Some(results);
    }
    
    /// Convert to the format expected by the database
    pub fn to_db_format(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
    
    /// Create from Python dictionary result
    pub fn from_python_dict(dict: HashMap<String, serde_json::Value>) -> Result<Self, String> {
        // Extract required fields
        let imputed_data = dict.get("imputed_data")
            .and_then(|v| serde_json::from_value::<Vec<Vec<f64>>>(v.clone()).ok())
            .ok_or("Missing or invalid imputed_data")?;
            
        let columns = dict.get("columns")
            .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok())
            .ok_or("Missing or invalid columns")?;
            
        let index = dict.get("index")
            .and_then(|v| serde_json::from_value::<Vec<DateTime<Utc>>>(v.clone()).ok())
            .ok_or("Missing or invalid index")?;
            
        let method = dict.get("method")
            .and_then(|v| v.as_str())
            .ok_or("Missing or invalid method")?
            .to_string();
            
        let parameters = dict.get("parameters")
            .and_then(|v| serde_json::from_value::<HashMap<String, serde_json::Value>>(v.clone()).ok())
            .unwrap_or_default();
            
        let mut result = Self::new(imputed_data, columns, index, method, parameters);
        
        // Extract optional fields
        if let Some(metrics) = dict.get("metrics") {
            if let Ok(metrics_map) = serde_json::from_value::<HashMap<String, MetricValues>>(metrics.clone()) {
                result.metrics = metrics_map;
            }
        }
        
        if let Some(uncertainty) = dict.get("uncertainty") {
            if let Ok(uncertainty_data) = serde_json::from_value::<Vec<Vec<f64>>>(uncertainty.clone()) {
                result.uncertainty = Some(uncertainty_data);
            }
        }
        
        if let Some(validation) = dict.get("validation_results") {
            if let Ok(validation_results) = serde_json::from_value::<ValidationResults>(validation.clone()) {
                result.validation_results = Some(validation_results);
            }
        }
        
        if let Some(outputs) = dict.get("method_outputs") {
            if let Ok(method_outputs) = serde_json::from_value::<HashMap<String, serde_json::Value>>(outputs.clone()) {
                result.method_outputs = Some(method_outputs);
            }
        }
        
        if let Some(time) = dict.get("execution_time_seconds") {
            if let Some(exec_time) = time.as_f64() {
                result.execution_time_seconds = exec_time;
            }
        }
        
        if let Some(memory) = dict.get("memory_usage_mb") {
            if let Some(mem_usage) = memory.as_f64() {
                result.memory_usage_mb = Some(mem_usage);
            }
        }
        
        if let Some(convergence) = dict.get("convergence_info") {
            if let Ok(conv_info) = serde_json::from_value::<ConvergenceInfo>(convergence.clone()) {
                result.convergence_info = Some(conv_info);
            }
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_imputation_result_creation() {
        let result = ImputationResult::new(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec!["col1".to_string(), "col2".to_string()],
            vec![Utc::now(), Utc::now()],
            "linear".to_string(),
            HashMap::new(),
        );
        
        assert_eq!(result.method, "linear");
        assert_eq!(result.columns.len(), 2);
        assert_eq!(result.imputed_data.len(), 2);
    }
    
    #[test]
    fn test_add_metrics() {
        let mut result = ImputationResult::new(
            vec![vec![1.0, 2.0]],
            vec!["col1".to_string()],
            vec![Utc::now()],
            "linear".to_string(),
            HashMap::new(),
        );
        
        let metrics = MetricValues {
            rmse: 0.5,
            mae: 0.3,
            mape: Some(0.1),
            r2: Some(0.95),
            correlation: Some(0.98),
            bias: Some(0.02),
            coverage_rate: Some(0.95),
        };
        
        result.add_column_metrics("col1".to_string(), metrics);
        assert_eq!(result.metrics.len(), 1);
        assert_eq!(result.metrics.get("col1").unwrap().rmse, 0.5);
    }
}