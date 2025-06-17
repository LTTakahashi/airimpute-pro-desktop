pub mod data_validator;

use crate::error::{CommandError, CommandResult};
use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Validation rules for different data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRules {
    /// Pollutant-specific validation rules
    pub pollutant_bounds: HashMap<String, PollutantBounds>,
    /// Date range constraints
    pub date_range: Option<DateRange>,
    /// Allowed file formats
    pub allowed_formats: Vec<String>,
    /// Maximum file size in MB
    pub max_file_size_mb: u64,
    /// Required columns for datasets
    pub required_columns: Vec<String>,
    /// Optional columns
    pub optional_columns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollutantBounds {
    pub min: f64,
    pub max: f64,
    pub unit: String,
    pub warning_threshold: Option<f64>,
    pub critical_threshold: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub statistics: ValidationStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub field: String,
    pub value: Option<String>,
    pub reason: String,
    pub severity: Severity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub field: String,
    pub message: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStatistics {
    pub total_records: usize,
    pub valid_records: usize,
    pub invalid_records: usize,
    pub missing_values: HashMap<String, usize>,
    pub out_of_bounds: HashMap<String, usize>,
    pub suspicious_patterns: Vec<String>,
}

/// Default validation rules for air quality data
impl Default for ValidationRules {
    fn default() -> Self {
        let mut pollutant_bounds = HashMap::new();
        
        // PM2.5 bounds (μg/m³)
        pollutant_bounds.insert("PM2.5".to_string(), PollutantBounds {
            min: 0.0,
            max: 500.0,
            unit: "μg/m³".to_string(),
            warning_threshold: Some(150.0),
            critical_threshold: Some(250.0),
        });
        
        // PM10 bounds (μg/m³)
        pollutant_bounds.insert("PM10".to_string(), PollutantBounds {
            min: 0.0,
            max: 600.0,
            unit: "μg/m³".to_string(),
            warning_threshold: Some(250.0),
            critical_threshold: Some(350.0),
        });
        
        // O3 bounds (μg/m³)
        pollutant_bounds.insert("O3".to_string(), PollutantBounds {
            min: 0.0,
            max: 400.0,
            unit: "μg/m³".to_string(),
            warning_threshold: Some(180.0),
            critical_threshold: Some(240.0),
        });
        
        // NO2 bounds (μg/m³)
        pollutant_bounds.insert("NO2".to_string(), PollutantBounds {
            min: 0.0,
            max: 400.0,
            unit: "μg/m³".to_string(),
            warning_threshold: Some(200.0),
            critical_threshold: Some(300.0),
        });
        
        // SO2 bounds (μg/m³)
        pollutant_bounds.insert("SO2".to_string(), PollutantBounds {
            min: 0.0,
            max: 500.0,
            unit: "μg/m³".to_string(),
            warning_threshold: Some(250.0),
            critical_threshold: Some(350.0),
        });
        
        // CO bounds (mg/m³)
        pollutant_bounds.insert("CO".to_string(), PollutantBounds {
            min: 0.0,
            max: 50.0,
            unit: "mg/m³".to_string(),
            warning_threshold: Some(15.0),
            critical_threshold: Some(30.0),
        });
        
        Self {
            pollutant_bounds,
            date_range: None,
            allowed_formats: vec![
                "csv".to_string(),
                "xlsx".to_string(),
                "xls".to_string(),
                "parquet".to_string(),
                "json".to_string(),
                "nc".to_string(),
                "h5".to_string(),
                "hdf5".to_string(),
            ],
            max_file_size_mb: 500,
            required_columns: vec![
                "timestamp".to_string(),
                "station_id".to_string(),
            ],
            optional_columns: vec![
                "latitude".to_string(),
                "longitude".to_string(),
                "elevation".to_string(),
                "temperature".to_string(),
                "humidity".to_string(),
                "wind_speed".to_string(),
                "wind_direction".to_string(),
                "pressure".to_string(),
            ],
        }
    }
}

/// Rules-based validator for comprehensive validation
pub struct RulesValidator {
    pub rules: ValidationRules,
}

impl RulesValidator {
    pub fn new(rules: ValidationRules) -> Self {
        Self { rules }
    }
    
    pub fn with_default_rules() -> Self {
        Self::new(ValidationRules::default())
    }
    
    /// Validate file path and format
    pub fn validate_file_path(&self, path: &str) -> CommandResult<()> {
        let path = Path::new(path);
        
        // Check if file exists
        if !path.exists() {
            return Err(CommandError::ValidationError {
                reason: format!("File not found: {}", path.display()),
            });
        }
        
        // Check file extension
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| CommandError::ValidationError {
                reason: "File has no extension".to_string(),
            })?;
        
        if !self.rules.allowed_formats.contains(&ext.to_lowercase()) {
            return Err(CommandError::ValidationError {
                reason: format!("Unsupported file format: {}. Allowed formats: {:?}", 
                    ext, self.rules.allowed_formats),
            });
        }
        
        // Check file size
        let metadata = std::fs::metadata(path)
            .map_err(|e| CommandError::from_io_err(e, path.display().to_string()))?;
        
        let size_mb = metadata.len() / (1024 * 1024);
        if size_mb > self.rules.max_file_size_mb {
            return Err(CommandError::ValidationError {
                reason: format!("File too large: {}MB (max: {}MB)", 
                    size_mb, self.rules.max_file_size_mb),
            });
        }
        
        Ok(())
    }
    
    /// Validate pollutant value
    pub fn validate_pollutant_value(
        &self, 
        pollutant: &str, 
        value: f64
    ) -> Result<Vec<ValidationWarning>, ValidationError> {
        let bounds = self.rules.pollutant_bounds.get(pollutant)
            .ok_or_else(|| ValidationError {
                field: pollutant.to_string(),
                value: Some(value.to_string()),
                reason: format!("Unknown pollutant: {}", pollutant),
                severity: Severity::High,
            })?;
        
        // Check bounds
        if value < bounds.min || value > bounds.max {
            return Err(ValidationError {
                field: pollutant.to_string(),
                value: Some(value.to_string()),
                reason: format!("Value out of bounds [{}, {}]", bounds.min, bounds.max),
                severity: Severity::Critical,
            });
        }
        
        let mut warnings = Vec::new();
        
        // Check warning threshold
        if let Some(warning) = bounds.warning_threshold {
            if value > warning {
                warnings.push(ValidationWarning {
                    field: pollutant.to_string(),
                    message: format!("Value {} exceeds warning threshold {}", value, warning),
                    suggestion: Some("Review data quality or environmental conditions".to_string()),
                });
            }
        }
        
        // Check critical threshold
        if let Some(critical) = bounds.critical_threshold {
            if value > critical {
                warnings.push(ValidationWarning {
                    field: pollutant.to_string(),
                    message: format!("Value {} exceeds critical threshold {}", value, critical),
                    suggestion: Some("Immediate investigation required".to_string()),
                });
            }
        }
        
        Ok(warnings)
    }
    
    /// Validate temporal consistency
    pub fn validate_temporal_consistency(
        &self,
        timestamps: &[DateTime<Utc>]
    ) -> Vec<ValidationWarning> {
        let mut warnings = Vec::new();
        
        if timestamps.len() < 2 {
            return warnings;
        }
        
        // Check for duplicates
        let mut seen = std::collections::HashSet::new();
        for ts in timestamps {
            if !seen.insert(ts) {
                warnings.push(ValidationWarning {
                    field: "timestamp".to_string(),
                    message: format!("Duplicate timestamp found: {}", ts),
                    suggestion: Some("Remove duplicate entries".to_string()),
                });
            }
        }
        
        // Check for gaps
        let mut sorted_ts = timestamps.to_vec();
        sorted_ts.sort();
        
        for window in sorted_ts.windows(2) {
            let diff = window[1].signed_duration_since(window[0]);
            if diff.num_hours() > 24 {
                warnings.push(ValidationWarning {
                    field: "timestamp".to_string(),
                    message: format!("Large gap detected: {} hours between {} and {}", 
                        diff.num_hours(), window[0], window[1]),
                    suggestion: Some("Consider interpolation or data collection issues".to_string()),
                });
            }
        }
        
        warnings
    }
    
    /// Validate spatial coordinates
    pub fn validate_coordinates(lat: f64, lon: f64) -> CommandResult<()> {
        if !(-90.0..=90.0).contains(&lat) {
            return Err(CommandError::ValidationError {
                reason: format!("Invalid latitude: {} (must be between -90 and 90)", lat),
            });
        }
        
        if !(-180.0..=180.0).contains(&lon) {
            return Err(CommandError::ValidationError {
                reason: format!("Invalid longitude: {} (must be between -180 and 180)", lon),
            });
        }
        
        Ok(())
    }
    
    /// Validate station ID format
    pub fn validate_station_id(id: &str) -> CommandResult<()> {
        let re = Regex::new(r"^[A-Za-z0-9_-]+$").unwrap();
        if !re.is_match(id) {
            return Err(CommandError::ValidationError {
                reason: format!("Invalid station ID format: {}. Must contain only alphanumeric characters, hyphens, and underscores", id),
            });
        }
        
        if id.len() > 50 {
            return Err(CommandError::ValidationError {
                reason: format!("Station ID too long: {} characters (max: 50)", id.len()),
            });
        }
        
        Ok(())
    }
}

/// Safe numeric validator
pub struct NumericValidator;

impl NumericValidator {
    /// Validate numeric input
    pub fn validate(value: f64) -> CommandResult<f64> {
        if value.is_nan() {
            return Err(CommandError::ValidationError {
                reason: "Invalid numeric value: NaN".to_string(),
            });
        }
        if value.is_infinite() {
            return Err(CommandError::ValidationError {
                reason: "Invalid numeric value: Infinity".to_string(),
            });
        }
        Ok(value)
    }
    
    /// Validate numeric range
    pub fn validate_range(value: f64, min: f64, max: f64) -> CommandResult<f64> {
        let value = Self::validate(value)?;
        if value < min || value > max {
            return Err(CommandError::ValidationError {
                reason: format!("Value {} is outside valid range [{}, {}]", value, min, max),
            });
        }
        Ok(value)
    }
}

// Note: All types are already public in this module, so no re-export needed