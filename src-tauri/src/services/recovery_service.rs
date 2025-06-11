// Error recovery service for graceful handling of failures

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPoint {
    pub id: String,
    pub operation: String,
    pub timestamp: DateTime<Utc>,
    pub state: RecoveryState,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryState {
    // Data states
    DataLoaded {
        dataset_id: String,
        file_path: String,
        rows_processed: usize,
    },
    DataValidated {
        dataset_id: String,
        validation_result: serde_json::Value,
    },
    PartiallyProcessed {
        dataset_id: String,
        method: String,
        progress_percent: f32,
        last_chunk: usize,
    },
    
    // Operation states  
    ImputationStarted {
        job_id: String,
        dataset_id: String,
        method: String,
        parameters: serde_json::Value,
    },
    ImputationCheckpoint {
        job_id: String,
        chunks_completed: usize,
        temporary_file: String,
    },
    
    // Error states
    ErrorOccurred {
        error_type: String,
        error_message: String,
        recoverable: bool,
        suggested_action: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAction {
    pub action_type: RecoveryActionType,
    pub description: String,
    pub estimated_time_seconds: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryActionType {
    Retry,
    Resume,
    Rollback,
    UseCache,
    ReduceDataSize,
    SimplifyMethod,
    RepairData,
    Manual,
}

pub struct RecoveryService {
    recovery_points: Arc<RwLock<HashMap<String, RecoveryPoint>>>,
    recovery_dir: PathBuf,
    max_recovery_points: usize,
}

impl RecoveryService {
    pub fn new(app_data_dir: PathBuf) -> Self {
        let recovery_dir = app_data_dir.join("recovery");
        std::fs::create_dir_all(&recovery_dir).ok();
        
        Self {
            recovery_points: Arc::new(RwLock::new(HashMap::new())),
            recovery_dir,
            max_recovery_points: 50,
        }
    }
    
    /// Save a recovery point
    pub async fn save_recovery_point(&self, point: RecoveryPoint) -> anyhow::Result<()> {
        // Store in memory
        self.recovery_points.write().await.insert(point.id.clone(), point.clone());
        
        // Persist to disk
        let file_path = self.recovery_dir.join(format!("{}.json", point.id));
        let json = serde_json::to_string_pretty(&point)?;
        tokio::fs::write(&file_path, json).await?;
        
        // Clean up old recovery points
        self.cleanup_old_points().await?;
        
        Ok(())
    }
    
    /// Get recovery options for a specific error
    pub async fn get_recovery_options(&self, error: &crate::error::simple_error::AppError) -> Vec<RecoveryAction> {
        let mut actions = Vec::new();
        
        match error {
            crate::error::simple_error::AppError::MemoryError { required_mb, available_mb } => {
                actions.push(RecoveryAction {
                    action_type: RecoveryActionType::ReduceDataSize,
                    description: format!(
                        "Process data in smaller chunks. Current memory: {}MB, Required: {}MB",
                        available_mb, required_mb
                    ),
                    estimated_time_seconds: Some(60),
                });
                
                actions.push(RecoveryAction {
                    action_type: RecoveryActionType::SimplifyMethod,
                    description: "Use a simpler imputation method that requires less memory".to_string(),
                    estimated_time_seconds: Some(30),
                });
            }
            
            crate::error::simple_error::AppError::Timeout { operation, seconds } => {
                actions.push(RecoveryAction {
                    action_type: RecoveryActionType::Resume,
                    description: format!("Resume {} from last checkpoint", operation),
                    estimated_time_seconds: None,
                });
                
                actions.push(RecoveryAction {
                    action_type: RecoveryActionType::ReduceDataSize,
                    description: "Process a smaller time range or fewer columns".to_string(),
                    estimated_time_seconds: Some(30),
                });
            }
            
            crate::error::simple_error::AppError::InvalidData { message, .. } => {
                actions.push(RecoveryAction {
                    action_type: RecoveryActionType::RepairData,
                    description: format!("Attempt to repair data issues: {}", message),
                    estimated_time_seconds: Some(45),
                });
                
                actions.push(RecoveryAction {
                    action_type: RecoveryActionType::Manual,
                    description: "Review and fix data manually in Data Import".to_string(),
                    estimated_time_seconds: None,
                });
            }
            
            crate::error::simple_error::AppError::PythonError { message } => {
                if message.contains("ModuleNotFoundError") {
                    actions.push(RecoveryAction {
                        action_type: RecoveryActionType::Manual,
                        description: "Reinstall the application to fix missing Python packages".to_string(),
                        estimated_time_seconds: Some(300),
                    });
                } else {
                    actions.push(RecoveryAction {
                        action_type: RecoveryActionType::Retry,
                        description: "Retry the operation".to_string(),
                        estimated_time_seconds: Some(10),
                    });
                }
            }
            
            _ => {
                // Generic recovery options
                actions.push(RecoveryAction {
                    action_type: RecoveryActionType::Retry,
                    description: "Retry the operation".to_string(),
                    estimated_time_seconds: Some(10),
                });
                
                if let Some(checkpoint) = self.find_latest_checkpoint(error.code()).await {
                    actions.push(RecoveryAction {
                        action_type: RecoveryActionType::Rollback,
                        description: format!("Rollback to checkpoint from {}", checkpoint.timestamp.format("%H:%M:%S")),
                        estimated_time_seconds: Some(20),
                    });
                }
            }
        }
        
        actions
    }
    
    /// Attempt automatic recovery
    pub async fn attempt_recovery(&self, 
                                 error: &crate::error::simple_error::AppError,
                                 context: &HashMap<String, serde_json::Value>) -> anyhow::Result<RecoveryResult> {
        tracing::info!("Attempting automatic recovery for error: {}", error.code());
        
        match error {
            crate::error::simple_error::AppError::MemoryError { .. } => {
                // Try to free memory
                self.free_memory().await?;
                
                // Check if we have a checkpoint
                if let Some(checkpoint) = self.find_checkpoint_for_context(context).await {
                    return Ok(RecoveryResult::ResumeFromCheckpoint { checkpoint });
                }
                
                Ok(RecoveryResult::ReduceDataSize { 
                    suggested_chunk_size: 5000 
                })
            }
            
            crate::error::simple_error::AppError::Timeout { .. } => {
                // Look for partial results
                if let Some(checkpoint) = self.find_checkpoint_for_context(context).await {
                    return Ok(RecoveryResult::ResumeFromCheckpoint { checkpoint });
                }
                
                Ok(RecoveryResult::RetryWithTimeout { 
                    new_timeout_seconds: 600 
                })
            }
            
            crate::error::simple_error::AppError::InvalidData { .. } => {
                // Try to repair common data issues
                if let Some(dataset_id) = context.get("dataset_id").and_then(|v| v.as_str()) {
                    return Ok(RecoveryResult::RepairAndRetry { 
                        dataset_id: dataset_id.to_string(),
                        repair_options: vec![
                            "Remove duplicate rows".to_string(),
                            "Fix column types".to_string(),
                            "Handle missing headers".to_string(),
                        ]
                    });
                }
                
                Ok(RecoveryResult::ManualIntervention)
            }
            
            _ => Ok(RecoveryResult::SimpleRetry)
        }
    }
    
    /// Find the latest checkpoint for the current operation
    async fn find_checkpoint_for_context(&self, context: &HashMap<String, serde_json::Value>) -> Option<RecoveryPoint> {
        let points = self.recovery_points.read().await;
        
        // Find matching checkpoints
        let mut checkpoints: Vec<_> = points.values()
            .filter(|p| {
                // Match by dataset_id or job_id
                if let Some(dataset_id) = context.get("dataset_id").and_then(|v| v.as_str()) {
                    match &p.state {
                        RecoveryState::PartiallyProcessed { dataset_id: d_id, .. } |
                        RecoveryState::DataValidated { dataset_id: d_id, .. } => {
                            d_id == dataset_id
                        }
                        _ => false
                    }
                } else {
                    false
                }
            })
            .cloned()
            .collect();
            
        // Sort by timestamp and return latest
        checkpoints.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        checkpoints.into_iter().next()
    }
    
    async fn find_latest_checkpoint(&self, error_code: &str) -> Option<RecoveryPoint> {
        let points = self.recovery_points.read().await;
        
        points.values()
            .filter(|p| matches!(&p.state, RecoveryState::ImputationCheckpoint { .. }))
            .max_by_key(|p| p.timestamp)
            .cloned()
    }
    
    /// Free memory by clearing caches and forcing garbage collection
    async fn free_memory(&self) -> anyhow::Result<()> {
        // Clear internal caches
        // In a real implementation, this would clear various caches
        
        // Suggest garbage collection to Python
        // TODO: Implement Python bridge garbage collection
        // if let Ok(py_bridge) = crate::python::SafePythonBridge::global() {
        //     py_bridge.run_maintenance().await?;
        // }
        
        Ok(())
    }
    
    /// Clean up old recovery points
    async fn cleanup_old_points(&self) -> anyhow::Result<()> {
        let mut points = self.recovery_points.write().await;
        
        // Keep only the most recent N points
        if points.len() > self.max_recovery_points {
            let mut all_points: Vec<_> = points.drain().map(|(_, v)| v).collect();
            all_points.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
            
            // Keep recent ones
            for point in all_points.into_iter().take(self.max_recovery_points) {
                points.insert(point.id.clone(), point);
            }
            
            // Delete old files
            if let Ok(entries) = std::fs::read_dir(&self.recovery_dir) {
                for entry in entries.flatten() {
                    if let Some(name) = entry.file_name().to_str() {
                        if name.ends_with(".json") {
                            let id = name.trim_end_matches(".json");
                            if !points.contains_key(id) {
                                std::fs::remove_file(entry.path()).ok();
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Load recovery points from disk on startup
    pub async fn load_recovery_points(&self) -> anyhow::Result<()> {
        if let Ok(entries) = std::fs::read_dir(&self.recovery_dir) {
            for entry in entries.flatten() {
                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                    if let Ok(point) = serde_json::from_str::<RecoveryPoint>(&content) {
                        self.recovery_points.write().await.insert(point.id.clone(), point);
                    }
                }
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryResult {
    SimpleRetry,
    ResumeFromCheckpoint { checkpoint: RecoveryPoint },
    RetryWithTimeout { new_timeout_seconds: u32 },
    ReduceDataSize { suggested_chunk_size: usize },
    RepairAndRetry { dataset_id: String, repair_options: Vec<String> },
    UseSimplifiedMethod { method: String },
    ManualIntervention,
}

// Helper to create recovery points easily
pub fn create_recovery_point(operation: String, state: RecoveryState) -> RecoveryPoint {
    RecoveryPoint {
        id: uuid::Uuid::new_v4().to_string(),
        operation,
        timestamp: Utc::now(),
        state,
        metadata: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_recovery_service() {
        let temp_dir = TempDir::new().unwrap();
        let service = RecoveryService::new(temp_dir.path().to_path_buf());
        
        // Create a recovery point
        let point = create_recovery_point(
            "test_operation".to_string(),
            RecoveryState::DataLoaded {
                dataset_id: "test123".to_string(),
                file_path: "/tmp/test.csv".to_string(),
                rows_processed: 1000,
            }
        );
        
        // Save it
        service.save_recovery_point(point.clone()).await.unwrap();
        
        // Verify it was saved
        let points = service.recovery_points.read().await;
        assert!(points.contains_key(&point.id));
    }
    
    #[tokio::test]
    async fn test_recovery_options() {
        let temp_dir = TempDir::new().unwrap();
        let service = RecoveryService::new(temp_dir.path().to_path_buf());
        
        // Test memory error recovery
        let error = crate::error::simple_error::AppError::MemoryError {
            required_mb: 1000,
            available_mb: 500,
        };
        
        let options = service.get_recovery_options(&error).await;
        assert!(!options.is_empty());
        assert!(options.iter().any(|o| matches!(o.action_type, RecoveryActionType::ReduceDataSize)));
    }
}