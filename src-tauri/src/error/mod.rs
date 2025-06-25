pub mod academic_error;
pub mod simple_error;

use std::fmt;
use serde::{Deserialize, Serialize};

/// Comprehensive error type for AirImpute Pro commands
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
#[serde(tag = "type", content = "details")]
pub enum CommandError {
    #[error("Dataset not found: {id}")]
    DatasetNotFound { id: String },
    
    #[error("Invalid dataset format: {reason}")]
    InvalidDataset { reason: String },
    
    #[error("Python runtime error: {message}")]
    PythonError { message: String },
    
    #[error("File I/O error: {path} - {message}")]
    IoError { path: String, message: String },
    
    #[error("Invalid parameter: {param} - {reason}")]
    InvalidParameter { param: String, reason: String },
    
    #[error("Validation failed: {reason}")]
    ValidationError { reason: String },
    
    #[error("Insufficient memory: required {required_mb}MB, available {available_mb}MB")]
    InsufficientMemory { required_mb: u64, available_mb: u64 },
    
    #[error("Method not implemented: {method}")]
    NotImplemented { method: String },
    
    #[error("State error: {reason}")]
    StateError { reason: String },
    
    #[error("Serialization error: {reason}")]
    SerializationError { reason: String },
    
    #[error("Authentication required")]
    AuthenticationRequired,
    
    #[error("Permission denied: {action}")]
    PermissionDenied { action: String },
    
    #[error("Configuration error: {reason}")]
    ConfigurationError { reason: String },
    
    #[error("Numerical computation error: {reason}")]
    NumericalError { reason: String },
    
    #[error("Data integrity error: {reason}")]
    DataIntegrityError { reason: String },
    
    #[error("Database error: {reason}")]
    DatabaseError { reason: String },
}

impl CommandError {
    /// Convert from Python errors
    #[cfg(feature = "python-support")]
    pub fn from_py_err(err: pyo3::PyErr) -> Self {
        CommandError::PythonError {
            message: err.to_string(),
        }
    }
    
    /// Convert from IO errors
    pub fn from_io_err(err: std::io::Error, path: impl AsRef<str>) -> Self {
        CommandError::IoError {
            path: path.as_ref().to_string(),
            message: err.to_string(),
        }
    }
    
    /// Convert from serde errors
    pub fn from_serde_err(err: impl fmt::Display) -> Self {
        CommandError::SerializationError {
            reason: err.to_string(),
        }
    }
}

#[cfg(feature = "python-support")]
impl From<pyo3::PyErr> for CommandError {
    fn from(err: pyo3::PyErr) -> Self {
        CommandError::PythonError {
            message: err.to_string(),
        }
    }
}

impl From<rusqlite::Error> for CommandError {
    fn from(err: rusqlite::Error) -> Self {
        CommandError::DatabaseError {
            reason: err.to_string(),
        }
    }
}

/// Type alias for command results
pub type CommandResult<T> = std::result::Result<T, CommandError>;

/// Extension trait for error conversion
pub trait ErrorExt<T> {
    fn map_py_err(self) -> CommandResult<T>;
}

#[cfg(feature = "python-support")]
impl<T> ErrorExt<T> for std::result::Result<T, pyo3::PyErr> {
    fn map_py_err(self) -> CommandResult<T> {
        self.map_err(CommandError::from_py_err)
    }
}

/// Audit log entry for tracking operations
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AuditLogEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub user_id: Option<String>,
    pub action: String,
    pub resource: String,
    pub success: bool,
    pub error: Option<String>,
    pub metadata: serde_json::Value,
}

impl AuditLogEntry {
    pub fn new(action: impl Into<String>, resource: impl Into<String>) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            user_id: None,
            action: action.into(),
            resource: resource.into(),
            success: true,
            error: None,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
        }
    }
    
    pub fn with_error(mut self, error: impl fmt::Display) -> Self {
        self.success = false;
        self.error = Some(error.to_string());
        self
    }
    
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
    
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }
}

/// Result type alias for command operations
pub type Result<T> = std::result::Result<T, CommandError>;