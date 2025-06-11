// Progress tracking service that actually connects to operations

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressUpdate {
    pub id: String,
    pub operation: String,
    pub progress: f32,
    pub message: String,
    pub details: ProgressDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressDetails {
    pub current: usize,
    pub total: usize,
    pub elapsed_seconds: f32,
    pub estimated_remaining_seconds: Option<f32>,
    pub speed: Option<f32>, // items per second
    pub memory_mb: Option<f32>,
}

#[derive(Debug)]
pub struct ProgressTracker {
    id: String,
    operation: String,
    start_time: Instant,
    last_update: Instant,
    current: Arc<RwLock<usize>>,
    total: usize,
    tx: mpsc::UnboundedSender<ProgressUpdate>,
    cancelled: Arc<RwLock<bool>>,
}

impl ProgressTracker {
    pub fn new(operation: String, total: usize, tx: mpsc::UnboundedSender<ProgressUpdate>) -> Self {
        let id = Uuid::new_v4().to_string();
        let now = Instant::now();
        
        let tracker = Self {
            id: id.clone(),
            operation: operation.clone(),
            start_time: now,
            last_update: now,
            current: Arc::new(RwLock::new(0)),
            total,
            tx: tx.clone(),
            cancelled: Arc::new(RwLock::new(false)),
        };
        
        // Send initial update
        let _ = tx.send(ProgressUpdate {
            id,
            operation,
            progress: 0.0,
            message: "Starting...".to_string(),
            details: ProgressDetails {
                current: 0,
                total,
                elapsed_seconds: 0.0,
                estimated_remaining_seconds: None,
                speed: None,
                memory_mb: None,
            },
        });
        
        tracker
    }
    
    pub async fn update(&self, current: usize, message: String) {
        let mut curr = self.current.write().await;
        *curr = current.min(self.total);
        
        let progress = if self.total > 0 {
            (*curr as f32 / self.total as f32) * 100.0
        } else {
            0.0
        };
        
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let speed = if elapsed > 0.0 {
            Some(*curr as f32 / elapsed)
        } else {
            None
        };
        
        let estimated_remaining = if *curr > 0 && progress < 100.0 {
            let rate = elapsed / *curr as f32;
            Some(rate * (self.total - *curr) as f32)
        } else {
            None
        };
        
        // Get memory usage
        let memory_mb = self.get_memory_usage();
        
        let update = ProgressUpdate {
            id: self.id.clone(),
            operation: self.operation.clone(),
            progress,
            message,
            details: ProgressDetails {
                current: *curr,
                total: self.total,
                elapsed_seconds: elapsed,
                estimated_remaining_seconds: estimated_remaining,
                speed,
                memory_mb: Some(memory_mb),
            },
        };
        
        let _ = self.tx.send(update);
    }
    
    pub async fn increment(&self, message: String) {
        let current = {
            let curr = self.current.read().await;
            *curr + 1
        };
        self.update(current, message).await;
    }
    
    pub async fn complete(&self, message: String) {
        self.update(self.total, message).await;
    }
    
    pub async fn is_cancelled(&self) -> bool {
        *self.cancelled.read().await
    }
    
    pub async fn cancel(&self) {
        let mut cancelled = self.cancelled.write().await;
        *cancelled = true;
        
        let _ = self.tx.send(ProgressUpdate {
            id: self.id.clone(),
            operation: self.operation.clone(),
            progress: self.current.read().await.clone() as f32 / self.total as f32 * 100.0,
            message: "Cancelled".to_string(),
            details: ProgressDetails {
                current: self.current.read().await.clone(),
                total: self.total,
                elapsed_seconds: self.start_time.elapsed().as_secs_f32(),
                estimated_remaining_seconds: None,
                speed: None,
                memory_mb: None,
            },
        });
    }
    
    fn get_memory_usage(&self) -> f32 {
        use sysinfo::System;
        
        let mut sys = System::new();
        sys.refresh_processes();
        
        if let Some(process) = sys.process(sysinfo::get_current_pid().unwrap()) {
            process.memory() as f32 / 1024.0 / 1024.0
        } else {
            0.0
        }
    }
    
    // Create a Python-compatible callback
    pub fn create_python_callback(&self) -> impl Fn(f32, &str) + Send + Sync + 'static {
        let current = self.current.clone();
        let total = self.total;
        let tx = self.tx.clone();
        let id = self.id.clone();
        let operation = self.operation.clone();
        let start_time = self.start_time;
        
        move |progress_pct: f32, message: &str| {
            let current_val = (progress_pct * total as f32 / 100.0) as usize;
            
            // Update atomically
            let curr_clone = current.clone();
            let tx_clone = tx.clone();
            let id_clone = id.clone();
            let operation_clone = operation.clone();
            let message_clone = message.to_string();
            
            tokio::spawn(async move {
                let mut curr = curr_clone.write().await;
                *curr = current_val;
                
                let elapsed = start_time.elapsed().as_secs_f32();
                let speed = if elapsed > 0.0 {
                    Some(current_val as f32 / elapsed)
                } else {
                    None
                };
                
                let estimated_remaining = if current_val > 0 && current_val < total {
                    let rate = elapsed / current_val as f32;
                    Some(rate * (total - current_val) as f32)
                } else {
                    None
                };
                
                let _ = tx_clone.send(ProgressUpdate {
                    id: id_clone,
                    operation: operation_clone,
                    progress: progress_pct,
                    message: message_clone,
                    details: ProgressDetails {
                        current: current_val,
                        total,
                        elapsed_seconds: elapsed,
                        estimated_remaining_seconds: estimated_remaining,
                        speed,
                        memory_mb: None,
                    },
                });
            });
        }
    }
}

pub struct ProgressService {
    tx: mpsc::UnboundedSender<ProgressUpdate>,
    rx: Arc<RwLock<mpsc::UnboundedReceiver<ProgressUpdate>>>,
    active_trackers: Arc<RwLock<HashMap<String, Arc<ProgressTracker>>>>,
    app_handle: AppHandle,
}

impl ProgressService {
    pub fn new(app_handle: AppHandle) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        
        let service = Self {
            tx: tx.clone(),
            rx: Arc::new(RwLock::new(rx)),
            active_trackers: Arc::new(RwLock::new(HashMap::new())),
            app_handle: app_handle.clone(),
        };
        
        // Start background task to emit events
        let app_handle_clone = app_handle.clone();
        let rx_clone = service.rx.clone();
        
        tokio::spawn(async move {
            loop {
                let mut rx = rx_clone.write().await;
                if let Some(update) = rx.recv().await {
                    // Emit to frontend
                    let _ = app_handle_clone.emit_all("progress:update", &update);
                    
                    // Log significant updates
                    if update.progress as usize % 10 == 0 || update.progress >= 100.0 {
                        tracing::debug!(
                            "Progress {}: {:.0}% - {}",
                            update.operation,
                            update.progress,
                            update.message
                        );
                    }
                } else {
                    // Channel closed, exit
                    break;
                }
            }
        });
        
        service
    }
    
    pub async fn create_tracker(&self, operation: String, total: usize) -> Arc<ProgressTracker> {
        let tracker = Arc::new(ProgressTracker::new(operation, total, self.tx.clone()));
        
        self.active_trackers.write().await.insert(
            tracker.id.clone(),
            tracker.clone()
        );
        
        tracker
    }
    
    pub async fn get_tracker(&self, id: &str) -> Option<Arc<ProgressTracker>> {
        self.active_trackers.read().await.get(id).cloned()
    }
    
    pub async fn cancel_operation(&self, id: &str) -> bool {
        if let Some(tracker) = self.get_tracker(id).await {
            tracker.cancel().await;
            true
        } else {
            false
        }
    }
    
    pub async fn cleanup_completed(&self) {
        // Remove completed trackers older than 5 minutes
        let mut trackers = self.active_trackers.write().await;
        trackers.retain(|_, tracker| {
            tracker.start_time.elapsed() < Duration::from_secs(300)
        });
    }
    
    pub async fn get_active_operations(&self) -> Vec<String> {
        self.active_trackers.read().await.keys().cloned().collect()
    }
}

// Global progress service instance
lazy_static::lazy_static! {
    static ref PROGRESS_SERVICE: Arc<RwLock<Option<Arc<ProgressService>>>> = Arc::new(RwLock::new(None));
}

pub async fn initialize_progress_service(app_handle: AppHandle) {
    let service = Arc::new(ProgressService::new(app_handle));
    *PROGRESS_SERVICE.write().await = Some(service);
}

pub async fn get_progress_service() -> Option<Arc<ProgressService>> {
    PROGRESS_SERVICE.read().await.clone()
}

// Helper macro for async progress tracking in loops
#[macro_export]
macro_rules! track_progress_async {
    ($tracker:expr, $iter:expr, $msg_fn:expr) => {{
        let mut results = Vec::new();
        let total = $iter.len();
        
        for (idx, item) in $iter.into_iter().enumerate() {
            if $tracker.is_cancelled().await {
                return Err(anyhow::anyhow!("Operation cancelled"));
            }
            
            let message = $msg_fn(&item, idx, total);
            $tracker.update(idx + 1, message).await;
            
            results.push(item);
        }
        
        results
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_progress_tracking() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let tracker = ProgressTracker::new("Test Operation".to_string(), 100, tx);
        
        // Update progress
        tracker.update(50, "Halfway there".to_string()).await;
        
        // Check update received
        if let Some(update) = rx.recv().await {
            assert_eq!(update.progress, 50.0);
            assert_eq!(update.details.current, 50);
        }
    }
    
    #[tokio::test]
    async fn test_time_estimation() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let tracker = ProgressTracker::new("Test Operation".to_string(), 100, tx);
        
        // Simulate some progress
        tokio::time::sleep(Duration::from_millis(100)).await;
        tracker.update(10, "Processing...".to_string()).await;
        
        // Skip initial update
        rx.recv().await;
        
        // Check time estimation
        if let Some(update) = rx.recv().await {
            assert!(update.details.elapsed_seconds > 0.0);
            assert!(update.details.estimated_remaining_seconds.is_some());
        }
    }
}