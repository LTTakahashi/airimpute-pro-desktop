use ndarray::Array2;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use anyhow::Result;

/// Core dataset structure for scientific computing
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Unique identifier
    pub id: uuid::Uuid,
    
    /// Human-readable name
    pub name: String,
    
    /// Raw data as 2D array (time x variables)
    pub data: Array2<f64>,
    
    /// Column names (variable names)
    pub columns: Vec<String>,
    
    /// Row index (timestamps)
    pub index: Vec<DateTime<Utc>>,
    
    /// Station metadata if spatial data
    pub stations: Option<Vec<Station>>,
    
    /// Variable metadata
    pub variables: Vec<Variable>,
    
    /// Data quality flags
    pub quality_flags: Option<Array2<QualityFlag>>,
    
    /// Original file path
    pub source_path: Option<String>,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
}

/// Station information for spatial data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Station {
    pub id: String,
    pub name: String,
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
    pub metadata: HashMap<String, String>,
}

/// Variable metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub unit: String,
    pub description: String,
    pub min_valid: Option<f64>,
    pub max_valid: Option<f64>,
    pub precision: Option<usize>,
    pub measurement_type: MeasurementType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementType {
    Continuous,
    Discrete,
    Categorical,
    Binary,
}

/// Data quality flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum QualityFlag {
    Valid = 0,
    Missing = 1,
    BelowDetection = 2,
    AboveDetection = 3,
    Suspect = 4,
    Invalid = 5,
    Interpolated = 6,
    Imputed = 7,
}

/// Data validation results
#[derive(Debug, Clone, Serialize)]
pub struct DataValidation {
    pub is_valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub warnings: Vec<ValidationWarning>,
    pub summary: ValidationSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub issue_type: IssueType,
    pub location: Option<(usize, usize)>, // (row, col)
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueType {
    MissingValue,
    OutOfBounds,
    NegativeValue,
    ExtremeValue,
    InconsistentType,
    DuplicateTimestamp,
    NonMonotonic,
    GapTooLarge,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationWarning {
    pub warning_type: String,
    pub message: String,
    pub affected_columns: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationSummary {
    pub total_values: usize,
    pub missing_values: usize,
    pub invalid_values: usize,
    pub suspicious_values: usize,
    pub completeness_percentage: f64,
    pub quality_score: f64,
}

/// Comprehensive data statistics
#[derive(Debug, Clone, Serialize)]
pub struct DataStatistics {
    pub basic_stats: BasicStatistics,
    pub missing_stats: MissingStatistics,
    pub temporal_stats: TemporalStatistics,
    pub spatial_stats: Option<SpatialStatistics>,
    pub distribution_stats: DistributionStatistics,
    pub correlation_matrix: Array2<f64>,
    // Convenience fields for database storage
    pub total_rows: usize,
    pub total_columns: usize,
    pub total_missing: usize,
    pub missing_percentage: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BasicStatistics {
    pub count: HashMap<String, usize>,
    pub mean: HashMap<String, f64>,
    pub std: HashMap<String, f64>,
    pub min: HashMap<String, f64>,
    pub max: HashMap<String, f64>,
    pub quartiles: HashMap<String, [f64; 3]>, // Q1, Q2, Q3
}

#[derive(Debug, Clone, Serialize)]
pub struct MissingStatistics {
    pub total_missing: usize,
    pub missing_percentage: f64,
    pub missing_by_column: HashMap<String, usize>,
    pub missing_by_row: Vec<usize>,
    pub gap_lengths: Vec<usize>,
    pub longest_gap: usize,
    pub pattern_type: MissingPatternType,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingPatternType {
    CompletelyRandom,
    Systematic,
    Temporal,
    Spatial,
    Mixed,
}

#[derive(Debug, Clone, Serialize)]
pub struct TemporalStatistics {
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
    pub sampling_frequency: String,
    pub regular_sampling: bool,
    pub trend_components: HashMap<String, TrendComponent>,
    pub seasonality: HashMap<String, SeasonalityInfo>,
    pub autocorrelation: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrendComponent {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SeasonalityInfo {
    pub period: f64,
    pub amplitude: f64,
    pub phase: f64,
    pub strength: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpatialStatistics {
    pub spatial_autocorrelation: f64,
    pub morans_i: f64,
    pub gearys_c: f64,
    pub hotspots: Vec<(f64, f64)>, // (lat, lon)
    pub spatial_variance: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DistributionStatistics {
    pub skewness: HashMap<String, f64>,
    pub kurtosis: HashMap<String, f64>,
    pub normality_test: HashMap<String, NormalityTest>,
    pub outliers: HashMap<String, Vec<usize>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NormalityTest {
    pub statistic: f64,
    pub p_value: f64,
    pub is_normal: bool,
}

impl Dataset {
    /// Create a new dataset
    pub fn new(
        name: String,
        data: Array2<f64>,
        columns: Vec<String>,
        index: Vec<DateTime<Utc>>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4(),
            name,
            data,
            columns: columns.clone(),
            index,
            stations: None,
            variables: columns
                .into_iter()
                .map(|name| Variable {
                    name: name.clone(),
                    unit: "unknown".to_string(),
                    description: String::new(),
                    min_valid: None,
                    max_valid: None,
                    precision: None,
                    measurement_type: MeasurementType::Continuous,
                })
                .collect(),
            quality_flags: None,
            source_path: None,
            created_at: now,
            modified_at: now,
        }
    }
    
    /// Get number of rows
    pub fn rows(&self) -> usize {
        self.data.nrows()
    }
    
    /// Get number of columns
    pub fn columns(&self) -> usize {
        self.data.ncols()
    }
    
    /// Count missing values
    pub fn count_missing(&self) -> usize {
        self.data.iter().filter(|&&x| x.is_nan()).count()
    }
    
    /// Get physical bounds for all variables
    pub fn get_physical_bounds(&self) -> HashMap<String, (f64, f64)> {
        self.variables
            .iter()
            .filter_map(|var| {
                if let (Some(min), Some(max)) = (var.min_valid, var.max_valid) {
                    Some((var.name.clone(), (min, max)))
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Get station names
    pub fn get_station_names(&self) -> Vec<String> {
        self.stations
            .as_ref()
            .map(|stations| stations.iter().map(|s| s.name.clone()).collect())
            .unwrap_or_default()
    }
    
    /// Get variable names
    pub fn get_variable_names(&self) -> Vec<String> {
        self.columns.clone()
    }
    
    /// Get timestamps
    pub fn get_timestamps(&self) -> Vec<DateTime<Utc>> {
        self.index.clone()
    }
    
    /// Get coordinates
    pub fn get_coordinates(&self) -> Option<Vec<(f64, f64)>> {
        self.stations.as_ref().map(|stations| {
            stations
                .iter()
                .map(|s| (s.latitude, s.longitude))
                .collect()
        })
    }
    
    /// Get units mapping
    pub fn get_units(&self) -> HashMap<String, String> {
        self.variables
            .iter()
            .map(|var| (var.name.clone(), var.unit.clone()))
            .collect()
    }
    
    /// Convert to ndarray Array2
    pub fn to_array2(&self) -> Array2<f64> {
        self.data.clone()
    }
    
    /// Generate preview
    pub fn preview(
        &self,
        max_rows: usize,
        columns: Option<&[String]>,
    ) -> Result<serde_json::Value> {
        let rows = self.rows().min(max_rows);
        
        let col_indices: Vec<usize> = if let Some(cols) = columns {
            cols.iter()
                .filter_map(|col| self.columns.iter().position(|c| c == col))
                .collect()
        } else {
            (0..self.columns()).collect()
        };
        
        let mut preview_data = Vec::new();
        
        for i in 0..rows {
            let mut row = serde_json::Map::new();
            row.insert("_index".to_string(), json!(self.index[i].to_rfc3339()));
            
            for &j in &col_indices {
                row.insert(
                    self.columns[j].clone(),
                    json!(self.data[[i, j]]),
                );
            }
            
            preview_data.push(serde_json::Value::Object(row));
        }
        
        Ok(json!({
            "data": preview_data,
            "shape": [self.rows(), self.columns()],
            "columns": self.columns,
            "preview_rows": rows,
            "total_rows": self.rows(),
        }))
    }
    
    /// Convert dataset to JSON format optimized for Python processing
    /// Returns a JSON object with separate arrays for values and index
    pub fn to_json_split(&self) -> serde_json::Value {
        // Convert the 2D array to a nested vector
        let mut values: Vec<Vec<f64>> = Vec::with_capacity(self.rows());
        for i in 0..self.rows() {
            let mut row: Vec<f64> = Vec::with_capacity(self.columns());
            for j in 0..self.columns() {
                row.push(self.data[[i, j]]);
            }
            values.push(row);
        }
        
        // Convert timestamps to ISO 8601 strings
        let index: Vec<String> = self.index
            .iter()
            .map(|dt| dt.to_rfc3339())
            .collect();
        
        json!({
            "values": values,
            "index": index,
            "columns": self.columns.clone(),
        })
    }
}

impl DataValidation {
    /// Validate dataset comprehensively
    pub fn validate(dataset: &Dataset) -> Result<Self> {
        let mut issues = Vec::new();
        let warnings = Vec::new();
        
        let total_values = dataset.rows() * dataset.columns();
        let mut missing_values = 0;
        let mut invalid_values = 0;
        let mut suspicious_values = 0;
        
        // Check each value
        for ((i, j), &value) in dataset.data.indexed_iter() {
            if value.is_nan() {
                missing_values += 1;
            } else if let Some(var) = dataset.variables.get(j) {
                // Check bounds
                if let Some(min) = var.min_valid {
                    if value < min {
                        issues.push(ValidationIssue {
                            severity: Severity::High,
                            issue_type: IssueType::OutOfBounds,
                            location: Some((i, j)),
                            message: format!("Value {} below minimum {} for {}", value, min, var.name),
                        });
                        invalid_values += 1;
                    }
                }
                
                if let Some(max) = var.max_valid {
                    if value > max {
                        issues.push(ValidationIssue {
                            severity: Severity::High,
                            issue_type: IssueType::OutOfBounds,
                            location: Some((i, j)),
                            message: format!("Value {} above maximum {} for {}", value, max, var.name),
                        });
                        invalid_values += 1;
                    }
                }
                
                // Check for negative values in non-negative variables
                if value < 0.0 && var.name.contains("concentration") {
                    issues.push(ValidationIssue {
                        severity: Severity::Medium,
                        issue_type: IssueType::NegativeValue,
                        location: Some((i, j)),
                        message: format!("Negative value {} for {}", value, var.name),
                    });
                    suspicious_values += 1;
                }
            }
        }
        
        // Check temporal consistency
        for i in 1..dataset.index.len() {
            if dataset.index[i] <= dataset.index[i - 1] {
                issues.push(ValidationIssue {
                    severity: Severity::Critical,
                    issue_type: IssueType::NonMonotonic,
                    location: Some((i, 0)),
                    message: format!("Non-monotonic timestamp at row {}", i),
                });
            }
        }
        
        // Calculate summary
        let completeness_percentage = 
            ((total_values - missing_values) as f64 / total_values as f64) * 100.0;
        
        let quality_score = 
            ((total_values - missing_values - invalid_values - suspicious_values) as f64 
            / total_values as f64) * 100.0;
        
        let summary = ValidationSummary {
            total_values,
            missing_values,
            invalid_values,
            suspicious_values,
            completeness_percentage,
            quality_score,
        };
        
        Ok(DataValidation {
            is_valid: issues.is_empty(),
            issues,
            warnings,
            summary,
        })
    }
}

impl DataStatistics {
    /// Calculate comprehensive statistics
    pub fn calculate(dataset: &Dataset) -> Result<Self> {
        // Basic statistics
        let basic_stats = Self::calculate_basic_stats(dataset)?;
        
        // Missing statistics
        let missing_stats = Self::calculate_missing_stats(dataset)?;
        
        // Temporal statistics
        let temporal_stats = Self::calculate_temporal_stats(dataset)?;
        
        // Spatial statistics (if applicable)
        let spatial_stats = if dataset.stations.is_some() {
            Some(Self::calculate_spatial_stats(dataset)?)
        } else {
            None
        };
        
        // Distribution statistics
        let distribution_stats = Self::calculate_distribution_stats(dataset)?;
        
        // Correlation matrix
        let correlation_matrix = Self::calculate_correlation_matrix(dataset)?;
        
        let total_rows = dataset.rows();
        let total_columns = dataset.columns();
        let total_missing = dataset.count_missing();
        let total_values = total_rows * total_columns;
        let missing_percentage = if total_values > 0 {
            (total_missing as f64 / total_values as f64) * 100.0
        } else {
            0.0
        };
        
        Ok(DataStatistics {
            basic_stats,
            missing_stats,
            temporal_stats,
            spatial_stats,
            distribution_stats,
            correlation_matrix,
            total_rows,
            total_columns,
            total_missing,
            missing_percentage,
        })
    }
    
    fn calculate_basic_stats(dataset: &Dataset) -> Result<BasicStatistics> {
        let mut count = HashMap::new();
        let mut mean = HashMap::new();
        let mut std = HashMap::new();
        let mut min = HashMap::new();
        let mut max = HashMap::new();
        let mut quartiles = HashMap::new();
        
        for (j, col_name) in dataset.columns.iter().enumerate() {
            let column = dataset.data.column(j);
            let valid_values: Vec<f64> = column
                .iter()
                .filter(|&&x| !x.is_nan())
                .copied()
                .collect();
            
            if !valid_values.is_empty() {
                count.insert(col_name.clone(), valid_values.len());
                
                // Calculate statistics
                let sum: f64 = valid_values.iter().sum();
                let mean_val = sum / valid_values.len() as f64;
                mean.insert(col_name.clone(), mean_val);
                
                let variance: f64 = valid_values
                    .iter()
                    .map(|&x| (x - mean_val).powi(2))
                    .sum::<f64>()
                    / valid_values.len() as f64;
                std.insert(col_name.clone(), variance.sqrt());
                
                let mut sorted = valid_values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                min.insert(col_name.clone(), sorted[0]);
                max.insert(col_name.clone(), sorted[sorted.len() - 1]);
                
                // Calculate quartiles
                let q1_idx = sorted.len() / 4;
                let q2_idx = sorted.len() / 2;
                let q3_idx = 3 * sorted.len() / 4;
                
                quartiles.insert(
                    col_name.clone(),
                    [sorted[q1_idx], sorted[q2_idx], sorted[q3_idx]],
                );
            }
        }
        
        Ok(BasicStatistics {
            count,
            mean,
            std,
            min,
            max,
            quartiles,
        })
    }
    
    fn calculate_missing_stats(dataset: &Dataset) -> Result<MissingStatistics> {
        let total_values = dataset.rows() * dataset.columns();
        let total_missing = dataset.count_missing();
        let missing_percentage = (total_missing as f64 / total_values as f64) * 100.0;
        
        // Missing by column
        let mut missing_by_column = HashMap::new();
        for (j, col_name) in dataset.columns.iter().enumerate() {
            let missing_count = dataset
                .data
                .column(j)
                .iter()
                .filter(|&&x| x.is_nan())
                .count();
            missing_by_column.insert(col_name.clone(), missing_count);
        }
        
        // Missing by row
        let missing_by_row: Vec<usize> = (0..dataset.rows())
            .map(|i| {
                dataset
                    .data
                    .row(i)
                    .iter()
                    .filter(|&&x| x.is_nan())
                    .count()
            })
            .collect();
        
        // Gap analysis (simplified)
        let gap_lengths = vec![]; // Would calculate actual gap lengths
        let longest_gap = gap_lengths.iter().max().copied().unwrap_or(0);
        
        // Pattern detection (simplified)
        let pattern_type = MissingPatternType::Mixed;
        
        Ok(MissingStatistics {
            total_missing,
            missing_percentage,
            missing_by_column,
            missing_by_row,
            gap_lengths,
            longest_gap,
            pattern_type,
        })
    }
    
    fn calculate_temporal_stats(dataset: &Dataset) -> Result<TemporalStatistics> {
        let time_range = (
            dataset.index.first().copied().unwrap_or_else(Utc::now),
            dataset.index.last().copied().unwrap_or_else(Utc::now),
        );
        
        // Simplified implementations
        let sampling_frequency = "hourly".to_string();
        let regular_sampling = true;
        let trend_components = HashMap::new();
        let seasonality = HashMap::new();
        let autocorrelation = HashMap::new();
        
        Ok(TemporalStatistics {
            time_range,
            sampling_frequency,
            regular_sampling,
            trend_components,
            seasonality,
            autocorrelation,
        })
    }
    
    fn calculate_spatial_stats(dataset: &Dataset) -> Result<SpatialStatistics> {
        // Simplified implementation
        Ok(SpatialStatistics {
            spatial_autocorrelation: 0.0,
            morans_i: 0.0,
            gearys_c: 0.0,
            hotspots: vec![],
            spatial_variance: 0.0,
        })
    }
    
    fn calculate_distribution_stats(dataset: &Dataset) -> Result<DistributionStatistics> {
        let skewness = HashMap::new();
        let kurtosis = HashMap::new();
        let normality_test = HashMap::new();
        let outliers = HashMap::new();
        
        Ok(DistributionStatistics {
            skewness,
            kurtosis,
            normality_test,
            outliers,
        })
    }
    
    fn calculate_correlation_matrix(dataset: &Dataset) -> Result<Array2<f64>> {
        let n_cols = dataset.columns();
        let mut corr_matrix = Array2::eye(n_cols);
        
        // Calculate correlations between columns
        for i in 0..n_cols {
            for j in (i + 1)..n_cols {
                let col_i = dataset.data.column(i);
                let col_j = dataset.data.column(j);
                
                // Calculate Pearson correlation (simplified)
                let correlation = 0.0; // Would calculate actual correlation
                
                corr_matrix[[i, j]] = correlation;
                corr_matrix[[j, i]] = correlation;
            }
        }
        
        Ok(corr_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    
    #[test]
    fn test_dataset_creation() {
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let columns = vec!["A".to_string(), "B".to_string()];
        let index = vec![Utc::now(), Utc::now()];
        
        let dataset = Dataset::new("Test".to_string(), data, columns, index);
        
        assert_eq!(dataset.name, "Test");
        assert_eq!(dataset.rows(), 2);
        assert_eq!(dataset.columns(), 2);
    }
    
    #[test]
    fn test_missing_count() {
        let data = arr2(&[[1.0, f64::NAN], [3.0, 4.0]]);
        let columns = vec!["A".to_string(), "B".to_string()];
        let index = vec![Utc::now(), Utc::now()];
        
        let dataset = Dataset::new("Test".to_string(), data, columns, index);
        
        assert_eq!(dataset.count_missing(), 1);
    }
    
    #[test]
    fn test_to_json_split() {
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let columns = vec!["A".to_string(), "B".to_string()];
        let now = Utc::now();
        let index = vec![now, now];
        
        let dataset = Dataset::new("Test".to_string(), data, columns.clone(), index.clone());
        let json_result = dataset.to_json_split();
        
        // Check structure
        assert!(json_result.is_object());
        assert!(json_result["values"].is_array());
        assert!(json_result["index"].is_array());
        assert!(json_result["columns"].is_array());
        
        // Check values
        let values = json_result["values"].as_array().unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].as_array().unwrap().len(), 2);
        assert_eq!(values[0][0].as_f64().unwrap(), 1.0);
        assert_eq!(values[0][1].as_f64().unwrap(), 2.0);
        assert_eq!(values[1][0].as_f64().unwrap(), 3.0);
        assert_eq!(values[1][1].as_f64().unwrap(), 4.0);
        
        // Check columns
        let columns_json = json_result["columns"].as_array().unwrap();
        assert_eq!(columns_json.len(), 2);
        assert_eq!(columns_json[0].as_str().unwrap(), "A");
        assert_eq!(columns_json[1].as_str().unwrap(), "B");
    }
}