// Simplified, practical error handling system for realistic implementation

use serde::{Serialize, Deserialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "details")]
pub enum AppError {
    // User-facing errors with clear messages
    FileNotFound {
        path: String,
        suggestion: String,
    },
    
    InvalidData {
        message: String,
        line: Option<usize>,
        column: Option<String>,
    },
    
    PythonError {
        message: String,
    },
    
    MemoryError {
        required_mb: usize,
        available_mb: usize,
    },
    
    Timeout {
        operation: String,
        seconds: u64,
    },
    
    ConfigError {
        message: String,
    },
    
    OperationCancelled,
    
    // Generic fallback
    Unknown {
        message: String,
    },
}

impl AppError {
    /// Get user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            AppError::FileNotFound { path, suggestion } => {
                format!("Cannot find file: {}\n\n{}", path, suggestion)
            }
            
            AppError::InvalidData { message, line, column } => {
                let mut msg = format!("Invalid data: {}", message);
                if let Some(l) = line {
                    msg.push_str(&format!("\nLine: {}", l));
                }
                if let Some(c) = column {
                    msg.push_str(&format!("\nColumn: {}", c));
                }
                msg.push_str("\n\nTip: Check that your data is properly formatted and contains numeric values.");
                msg
            }
            
            AppError::PythonError { message } => {
                // Simplify Python errors for users
                let simple_msg = if message.contains("ModuleNotFoundError") {
                    "Required Python packages are not installed. Please reinstall the application."
                } else if message.contains("MemoryError") {
                    "Not enough memory to process this data. Try using a smaller dataset."
                } else {
                    "An error occurred during processing. Please try again."
                };
                format!("{}\n\nDetails: {}", simple_msg, message)
            }
            
            AppError::MemoryError { required_mb, available_mb } => {
                format!(
                    "Not enough memory to process this dataset.\n\n\
                    Required: {} MB\n\
                    Available: {} MB\n\n\
                    Suggestions:\n\
                    • Close other applications\n\
                    • Process a smaller time range\n\
                    • Use a simpler imputation method",
                    required_mb, available_mb
                )
            }
            
            AppError::Timeout { operation, seconds } => {
                format!(
                    "The {} operation took too long (>{} seconds).\n\n\
                    This usually happens with:\n\
                    • Very large datasets\n\
                    • Complex imputation methods\n\n\
                    Try:\n\
                    • Using a smaller dataset\n\
                    • Choosing a simpler method\n\
                    • Increasing the timeout in Settings",
                    operation, seconds
                )
            }
            
            AppError::ConfigError { message } => {
                format!("Configuration error: {}\n\nTry resetting to default settings.", message)
            }
            
            AppError::OperationCancelled => {
                "Operation was cancelled.".to_string()
            }
            
            AppError::Unknown { message } => {
                format!("An unexpected error occurred: {}\n\nPlease report this issue.", message)
            }
        }
    }
    
    /// Get error code for logging
    pub fn code(&self) -> &'static str {
        match self {
            AppError::FileNotFound { .. } => "FILE_NOT_FOUND",
            AppError::InvalidData { .. } => "INVALID_DATA",
            AppError::PythonError { .. } => "PYTHON_ERROR",
            AppError::MemoryError { .. } => "MEMORY_ERROR",
            AppError::Timeout { .. } => "TIMEOUT",
            AppError::ConfigError { .. } => "CONFIG_ERROR",
            AppError::OperationCancelled => "CANCELLED",
            AppError::Unknown { .. } => "UNKNOWN",
        }
    }
    
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            AppError::FileNotFound { .. } => true,
            AppError::InvalidData { .. } => true,
            AppError::ConfigError { .. } => true,
            AppError::Timeout { .. } => true,
            AppError::OperationCancelled => true,
            _ => false,
        }
    }
    
    /// Get suggested action for the error
    pub fn suggested_action(&self) -> Option<String> {
        match self {
            AppError::FileNotFound { .. } => {
                Some("Select a different file".to_string())
            }
            AppError::InvalidData { .. } => {
                Some("Review and fix your data".to_string())
            }
            AppError::MemoryError { .. } => {
                Some("Reduce dataset size".to_string())
            }
            AppError::Timeout { .. } => {
                Some("Try with less data".to_string())
            }
            _ => None,
        }
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.user_message())
    }
}

impl std::error::Error for AppError {}

// Implement conversions from common error types

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        match err.kind() {
            std::io::ErrorKind::NotFound => AppError::FileNotFound {
                path: "Unknown".to_string(),
                suggestion: "Please check if the file exists and you have permission to read it.".to_string(),
            },
            std::io::ErrorKind::PermissionDenied => AppError::Unknown {
                message: "Permission denied. Please check file permissions.".to_string(),
            },
            _ => AppError::Unknown {
                message: format!("File error: {}", err),
            },
        }
    }
}

impl From<serde_json::Error> for AppError {
    fn from(err: serde_json::Error) -> Self {
        AppError::InvalidData {
            message: "Invalid JSON format".to_string(),
            line: Some(err.line()),
            column: Some(err.column().to_string()),
        }
    }
}

impl From<csv::Error> for AppError {
    fn from(err: csv::Error) -> Self {
        let message = match err.kind() {
            csv::ErrorKind::Utf8 { .. } => "File contains invalid characters",
            csv::ErrorKind::UnequalLengths { .. } => "Rows have different number of columns",
            csv::ErrorKind::Deserialize { .. } => "Data format doesn't match expected structure",
            _ => "CSV parsing error",
        };
        
        AppError::InvalidData {
            message: message.to_string(),
            line: err.position().map(|p| p.line() as usize),
            column: None,
        }
    }
}

#[cfg(feature = "python-support")]
impl From<pyo3::PyErr> for AppError {
    fn from(err: pyo3::PyErr) -> Self {
        AppError::PythonError {
            message: err.to_string(),
        }
    }
}

// Result type alias
pub type Result<T> = std::result::Result<T, AppError>;

// Helper functions for common error scenarios

pub fn check_memory_available(required_mb: usize) -> Result<()> {
    use sysinfo::System;
    
    let mut sys = System::new();
    sys.refresh_memory();
    
    let available_mb = (sys.available_memory() / 1024 / 1024) as usize;
    
    if available_mb < required_mb {
        Err(AppError::MemoryError {
            required_mb,
            available_mb,
        })
    } else {
        Ok(())
    }
}

pub fn validate_dataframe_columns(columns: &[String], required: &[&str]) -> Result<()> {
    let missing: Vec<&str> = required
        .iter()
        .filter(|&&req| !columns.contains(&req.to_string())).copied()
        .collect();
    
    if !missing.is_empty() {
        Err(AppError::InvalidData {
            message: format!("Missing required columns: {}", missing.join(", ")),
            line: None,
            column: None,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = AppError::FileNotFound {
            path: "/path/to/file.csv".to_string(),
            suggestion: "Check the file path".to_string(),
        };
        
        assert!(err.user_message().contains("/path/to/file.csv"));
        assert_eq!(err.code(), "FILE_NOT_FOUND");
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_memory_error() {
        let err = AppError::MemoryError {
            required_mb: 1024,
            available_mb: 512,
        };
        
        let msg = err.user_message();
        assert!(msg.contains("1024 MB"));
        assert!(msg.contains("512 MB"));
        assert!(msg.contains("Suggestions"));
    }
}