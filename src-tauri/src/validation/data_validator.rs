// Practical data validation system

use crate::core::data::Dataset;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub summary: DataSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub code: String,
    pub message: String,
    pub location: Option<ErrorLocation>,
    pub severity: ErrorSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub code: String,
    pub message: String,
    pub suggestion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorLocation {
    pub row: Option<usize>,
    pub column: Option<String>,
    pub value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Critical, // Cannot proceed
    Major,    // Will likely cause issues
    Minor,    // May cause issues
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSummary {
    pub total_rows: usize,
    pub total_columns: usize,
    pub numeric_columns: Vec<String>,
    pub datetime_columns: Vec<String>,
    pub text_columns: Vec<String>,
    pub missing_values: HashMap<String, usize>,
    pub missing_percentage: HashMap<String, f32>,
    pub value_ranges: HashMap<String, (f64, f64)>,
    pub detected_frequency: Option<String>,
}

pub struct DataValidator {
    config: ValidationConfig,
}

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub max_missing_percentage: f32,
    pub min_rows: usize,
    pub max_rows: usize,
    pub required_columns: Vec<String>,
    pub check_timestamps: bool,
    pub check_outliers: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_missing_percentage: 90.0,
            min_rows: 10,
            max_rows: 10_000_000,
            required_columns: vec![],
            check_timestamps: true,
            check_outliers: true,
        }
    }
}

impl DataValidator {
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }
    
    /// Validate a dataset
    pub fn validate(&self, dataset: &Dataset) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Get data summary first
        let summary = self.analyze_data(dataset);
        
        // Check basic requirements
        self.check_size(&summary, &mut errors);
        self.check_columns(dataset, &summary, &mut errors, &mut warnings);
        self.check_data_types(dataset, &summary, &mut errors, &mut warnings);
        self.check_missing_values(&summary, &mut errors, &mut warnings);
        
        if self.config.check_timestamps {
            self.check_timestamps(dataset, &summary, &mut errors, &mut warnings);
        }
        
        if self.config.check_outliers {
            self.check_outliers(dataset, &summary, &mut warnings);
        }
        
        let is_valid = errors.iter().all(|e| !matches!(e.severity, ErrorSeverity::Critical));
        
        ValidationResult {
            is_valid,
            errors,
            warnings,
            summary,
        }
    }
    
    fn analyze_data(&self, dataset: &Dataset) -> DataSummary {
        let total_rows = dataset.rows();
        let total_columns = dataset.columns();
        
        let mut numeric_columns = Vec::new();
        let mut datetime_columns = Vec::new();
        let text_columns = Vec::new();
        let mut missing_values = HashMap::new();
        let mut missing_percentage = HashMap::new();
        let mut value_ranges = HashMap::new();
        
        // All columns in Dataset are numeric (Array2<f64>)
        for (j, col_name) in dataset.columns.iter().enumerate() {
            numeric_columns.push(col_name.clone());
            
            // Count missing values and calculate range
            let column = dataset.data.column(j);
            let mut valid_values = Vec::new();
            let mut missing_count = 0;
            
            for &val in column.iter() {
                if val.is_nan() {
                    missing_count += 1;
                } else {
                    valid_values.push(val);
                }
            }
            
            missing_values.insert(col_name.clone(), missing_count);
            missing_percentage.insert(
                col_name.clone(), 
                (missing_count as f32 / total_rows as f32) * 100.0
            );
            
            // Calculate range
            if !valid_values.is_empty() {
                let min_val = valid_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = valid_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                value_ranges.insert(col_name.clone(), (min_val, max_val));
            }
        }
        
        // Dataset has timestamps in the index field
        if !dataset.index.is_empty() {
            datetime_columns.push("_timestamp".to_string());
        }
        
        // Try to detect frequency
        let detected_frequency = if !dataset.index.is_empty() {
            self.detect_frequency_from_timestamps(&dataset.index)
        } else {
            None
        };
        
        DataSummary {
            total_rows,
            total_columns,
            numeric_columns,
            datetime_columns,
            text_columns,
            missing_values,
            missing_percentage,
            value_ranges,
            detected_frequency,
        }
    }
    
    fn check_size(&self, summary: &DataSummary, errors: &mut Vec<ValidationError>) {
        if summary.total_rows < self.config.min_rows {
            errors.push(ValidationError {
                code: "TOO_FEW_ROWS".to_string(),
                message: format!(
                    "Dataset has only {} rows, minimum required is {}",
                    summary.total_rows, self.config.min_rows
                ),
                location: None,
                severity: ErrorSeverity::Critical,
            });
        }
        
        if summary.total_rows > self.config.max_rows {
            errors.push(ValidationError {
                code: "TOO_MANY_ROWS".to_string(),
                message: format!(
                    "Dataset has {} rows, maximum supported is {}",
                    summary.total_rows, self.config.max_rows
                ),
                location: None,
                severity: ErrorSeverity::Major,
            });
        }
        
        if summary.numeric_columns.is_empty() {
            errors.push(ValidationError {
                code: "NO_NUMERIC_DATA".to_string(),
                message: "No numeric columns found for imputation".to_string(),
                location: None,
                severity: ErrorSeverity::Critical,
            });
        }
    }
    
    fn check_columns(
        &self,
        dataset: &Dataset,
        _summary: &DataSummary,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) {
        // Check required columns
        for required in &self.config.required_columns {
            if !dataset.columns.contains(required) {
                errors.push(ValidationError {
                    code: "MISSING_REQUIRED_COLUMN".to_string(),
                    message: format!("Required column '{}' not found", required),
                    location: None,
                    severity: ErrorSeverity::Critical,
                });
            }
        }
        
        // Warn about non-standard column names
        for col_name in &dataset.columns {
            if col_name.contains(" ") {
                warnings.push(ValidationWarning {
                    code: "COLUMN_NAME_SPACES".to_string(),
                    message: format!("Column '{}' contains spaces", col_name),
                    suggestion: "Consider using underscores instead of spaces".to_string(),
                });
            }
            
            if col_name.chars().any(|c| !c.is_alphanumeric() && c != '_' && c != '-') {
                warnings.push(ValidationWarning {
                    code: "SPECIAL_CHARS_IN_COLUMN".to_string(),
                    message: format!("Column '{}' contains special characters", col_name),
                    suggestion: "Use only letters, numbers, underscores, and hyphens".to_string(),
                });
            }
        }
    }
    
    fn check_data_types(
        &self,
        dataset: &Dataset,
        _summary: &DataSummary,
        _errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) {
        // Since Dataset uses Array2<f64>, all columns are already numeric
        // Check for suspicious patterns in the data
        
        for (j, col_name) in dataset.columns.iter().enumerate() {
            let column = dataset.data.column(j);
            let mut integer_count = 0;
            let mut valid_count = 0;
            
            // Sample first 100 values to check if they're all integers
            let sample_size = 100.min(column.len());
            for i in 0..sample_size {
                let val = column[i];
                if !val.is_nan() {
                    valid_count += 1;
                    if val.fract() == 0.0 {
                        integer_count += 1;
                    }
                }
            }
            
            // Check if this might be categorical data stored as numeric
            if valid_count > 0 && integer_count == valid_count {
                // Count unique values
                let mut unique_values = std::collections::HashSet::new();
                for &val in column.iter() {
                    if !val.is_nan() {
                        unique_values.insert(val.to_bits()); // Use to_bits for HashSet
                    }
                }
                
                // If very few unique values relative to size, might be categorical
                if unique_values.len() < 10 && column.len() > 100 {
                    warnings.push(ValidationWarning {
                        code: "POSSIBLE_CATEGORICAL".to_string(),
                        message: format!("Column '{}' has only {} unique values, might be categorical", col_name, unique_values.len()),
                        suggestion: "Consider if this column represents categories rather than continuous values".to_string(),
                    });
                }
            }
        }
    }
    
    fn check_missing_values(
        &self,
        summary: &DataSummary,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) {
        for (col, percentage) in &summary.missing_percentage {
            if *percentage > self.config.max_missing_percentage {
                errors.push(ValidationError {
                    code: "EXCESSIVE_MISSING_DATA".to_string(),
                    message: format!(
                        "Column '{}' has {:.1}% missing values (max allowed: {:.1}%)",
                        col, percentage, self.config.max_missing_percentage
                    ),
                    location: Some(ErrorLocation {
                        row: None,
                        column: Some(col.clone()),
                        value: None,
                    }),
                    severity: ErrorSeverity::Major,
                });
            } else if *percentage > 50.0 {
                warnings.push(ValidationWarning {
                    code: "HIGH_MISSING_DATA".to_string(),
                    message: format!("Column '{}' has {:.1}% missing values", col, percentage),
                    suggestion: "Consider if this column is suitable for imputation".to_string(),
                });
            }
        }
        
        // Check for columns with all missing
        for col in &summary.numeric_columns {
            if let Some(&missing) = summary.missing_values.get(col) {
                if missing == summary.total_rows {
                    errors.push(ValidationError {
                        code: "COLUMN_ALL_MISSING".to_string(),
                        message: format!("Column '{}' has no valid values", col),
                        location: Some(ErrorLocation {
                            row: None,
                            column: Some(col.clone()),
                            value: None,
                        }),
                        severity: ErrorSeverity::Critical,
                    });
                }
            }
        }
    }
    
    fn check_timestamps(
        &self,
        dataset: &Dataset,
        _summary: &DataSummary,
        _errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) {
        if dataset.index.is_empty() {
            warnings.push(ValidationWarning {
                code: "NO_TIMESTAMP_INDEX".to_string(),
                message: "No timestamp index found".to_string(),
                suggestion: "Time series imputation works better with timestamp information".to_string(),
            });
            return;
        }
        
        // Check for duplicates
        let mut timestamp_set = std::collections::HashSet::new();
        for timestamp in &dataset.index {
            if !timestamp_set.insert(timestamp) {
                warnings.push(ValidationWarning {
                    code: "DUPLICATE_TIMESTAMPS".to_string(),
                    message: "Dataset contains duplicate timestamps".to_string(),
                    suggestion: "Consider aggregating data by timestamp".to_string(),
                });
                break;
            }
        }
        
        // Check if sorted
        let is_sorted = dataset.index.windows(2).all(|w| w[0] <= w[1]);
        if !is_sorted {
            warnings.push(ValidationWarning {
                code: "UNSORTED_TIMESTAMPS".to_string(),
                message: "Timestamps are not sorted".to_string(),
                suggestion: "Sort data by timestamp for better performance".to_string(),
            });
        }
        
        // Check for large gaps
        if dataset.index.len() > 1 {
            let mut max_gap = Duration::seconds(0);
            for window in dataset.index.windows(2) {
                let gap = window[1] - window[0];
                if gap > max_gap {
                    max_gap = gap;
                }
            }
            
            // Warn if gap is more than 7 days
            if max_gap > Duration::days(7) {
                warnings.push(ValidationWarning {
                    code: "LARGE_TIME_GAP".to_string(),
                    message: format!("Maximum time gap is {} days", max_gap.num_days()),
                    suggestion: "Large gaps may affect imputation quality".to_string(),
                });
            }
        }
    }
    
    fn check_outliers(
        &self,
        dataset: &Dataset,
        summary: &DataSummary,
        warnings: &mut Vec<ValidationWarning>,
    ) {
        for (j, col_name) in dataset.columns.iter().enumerate() {
            if let Some((min, max)) = summary.value_ranges.get(col_name) {
                let column = dataset.data.column(j);
                let mut valid_values: Vec<f64> = column.iter()
                    .filter(|&&x| !x.is_nan())
                    .copied()
                    .collect();
                
                if valid_values.len() < 4 {
                    continue; // Not enough data for IQR
                }
                
                // Sort for quantile calculation
                valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                // Calculate quartiles
                let q1_idx = valid_values.len() / 4;
                let q3_idx = 3 * valid_values.len() / 4;
                let q1 = valid_values[q1_idx];
                let q3 = valid_values[q3_idx];
                
                let iqr = q3 - q1;
                let lower_bound = q1 - 3.0 * iqr;
                let upper_bound = q3 + 3.0 * iqr;
                
                if *min < lower_bound || *max > upper_bound {
                    warnings.push(ValidationWarning {
                        code: "POSSIBLE_OUTLIERS".to_string(),
                        message: format!(
                            "Column '{}' may contain extreme outliers (range: {:.2} to {:.2})",
                            col_name, min, max
                        ),
                        suggestion: "Review data for errors or consider outlier treatment".to_string(),
                    });
                }
                
                // Also check for specific variable types with known bounds
                if let Some(var) = dataset.variables.get(j) {
                    // Check physical bounds
                    if let (Some(min_valid), Some(max_valid)) = (var.min_valid, var.max_valid) {
                        if *min < min_valid || *max > max_valid {
                            warnings.push(ValidationWarning {
                                code: "PHYSICAL_BOUNDS_VIOLATION".to_string(),
                                message: format!(
                                    "Column '{}' has values outside physical bounds [{:.2}, {:.2}]",
                                    col_name, min_valid, max_valid
                                ),
                                suggestion: "Check data for measurement errors".to_string(),
                            });
                        }
                    }
                }
            }
        }
    }
    
    fn detect_frequency_from_timestamps(&self, timestamps: &[DateTime<Utc>]) -> Option<String> {
        if timestamps.len() < 2 {
            return None;
        }
        
        // Sample first few timestamps
        let sample_size = 10.min(timestamps.len() - 1);
        let mut diffs = Vec::new();
        
        for i in 0..sample_size {
            let diff = timestamps[i + 1] - timestamps[i];
            diffs.push(diff.num_seconds());
        }
        
        if diffs.is_empty() {
            return None;
        }
        
        // Find median difference
        diffs.sort();
        let median_diff = diffs[diffs.len() / 2];
        
        // Convert to frequency string
        match median_diff {
            0..=90 => Some("1min".to_string()),
            91..=450 => Some("5min".to_string()),
            451..=1350 => Some("15min".to_string()),
            1351..=2700 => Some("30min".to_string()),
            2701..=5400 => Some("1h".to_string()),
            5401..=129600 => Some("1d".to_string()),
            129601..=864000 => Some("1w".to_string()),
            _ => Some("irregular".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    
    fn create_test_dataset() -> Dataset {
        let data = arr2(&[
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]);
        let columns = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let now = Utc::now();
        let index = vec![
            now,
            now + Duration::hours(1),
            now + Duration::hours(2),
            now + Duration::hours(3),
        ];
        
        Dataset::new("Test".to_string(), data, columns, index)
    }
    
    #[test]
    fn test_validation_config() {
        let config = ValidationConfig::default();
        assert_eq!(config.max_missing_percentage, 90.0);
        assert_eq!(config.min_rows, 10);
    }
    
    #[test]
    fn test_error_severity() {
        let error = ValidationError {
            code: "TEST".to_string(),
            message: "Test error".to_string(),
            location: None,
            severity: ErrorSeverity::Critical,
        };
        
        match error.severity {
            ErrorSeverity::Critical => assert!(true),
            _ => assert!(false),
        }
    }
    
    #[test]
    fn test_validate_dataset() {
        let dataset = create_test_dataset();
        let validator = DataValidator::new(ValidationConfig::default());
        let result = validator.validate(&dataset);
        
        // Should have warnings about too few rows
        assert!(!result.errors.is_empty());
        assert_eq!(result.summary.total_rows, 4);
        assert_eq!(result.summary.total_columns, 3);
        assert_eq!(result.summary.numeric_columns.len(), 3);
    }
    
    #[test]
    fn test_missing_value_detection() {
        let dataset = create_test_dataset();
        let validator = DataValidator::new(ValidationConfig::default());
        let result = validator.validate(&dataset);
        
        // Column B has 1 missing value
        assert_eq!(result.summary.missing_values.get("B"), Some(&1));
        assert!(result.summary.missing_percentage.get("B").unwrap() > &0.0);
    }
    
    #[test]
    fn test_frequency_detection() {
        let dataset = create_test_dataset();
        let validator = DataValidator::new(ValidationConfig::default());
        let result = validator.validate(&dataset);
        
        // Should detect hourly frequency
        assert_eq!(result.summary.detected_frequency, Some("1h".to_string()));
    }
}