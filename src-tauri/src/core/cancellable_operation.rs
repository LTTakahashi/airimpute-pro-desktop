// Cancellable operations with comprehensive progress tracking
// Implements cooperative cancellation and hierarchical progress reporting

use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::{broadcast, watch, oneshot};
use tokio::time::interval;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use tracing::{debug, info, warn};

/// Token for cooperative cancellation
#[derive(Clone, Debug)]
pub struct CancellationToken {
    inner: Arc<CancellationTokenInner>,
}

struct CancellationTokenInner {
    is_cancelled: AtomicBool,
    cancel_sender: broadcast::Sender<()>,
    child_tokens: Mutex<Vec<CancellationToken>>,
    parent: Option<CancellationToken>,
}

impl std::fmt::Debug for CancellationTokenInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationTokenInner")
            .field("is_cancelled", &self.is_cancelled.load(Ordering::Relaxed))
            .field("has_parent", &self.parent.is_some())
            .finish()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        let (cancel_sender, _) = broadcast::channel(1);
        Self {
            inner: Arc::new(CancellationTokenInner {
                is_cancelled: AtomicBool::new(false),
                cancel_sender,
                child_tokens: Mutex::new(Vec::new()),
                parent: None,
            }),
        }
    }

    pub fn child_token(&self) -> Self {
        let (cancel_sender, _) = broadcast::channel(1);
        let child = Self {
            inner: Arc::new(CancellationTokenInner {
                is_cancelled: AtomicBool::new(false),
                cancel_sender,
                child_tokens: Mutex::new(Vec::new()),
                parent: Some(self.clone()),
            }),
        };
        
        self.inner.child_tokens.lock().unwrap().push(child.clone());
        child
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled.load(Ordering::Acquire) ||
        self.inner.parent.as_ref().map_or(false, |p| p.is_cancelled())
    }

    pub fn cancel(&self) {
        if !self.inner.is_cancelled.swap(true, Ordering::AcqRel) {
            let _ = self.inner.cancel_sender.send(());
            
            // Cancel all children
            let children = self.inner.child_tokens.lock().unwrap();
            for child in children.iter() {
                child.cancel();
            }
        }
    }

    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }

        let mut receiver = self.inner.cancel_sender.subscribe();
        let _ = receiver.recv().await;
    }

    pub fn check(&self) -> Result<(), OperationCancelled> {
        if self.is_cancelled() {
            Err(OperationCancelled)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone)]
pub struct OperationCancelled;

impl std::fmt::Display for OperationCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Operation was cancelled")
    }
}

impl std::error::Error for OperationCancelled {}

/// Hierarchical progress tracking
#[derive(Clone)]
pub struct ProgressTracker {
    inner: Arc<ProgressTrackerInner>,
}

struct ProgressTrackerInner {
    id: Uuid,
    name: String,
    total_steps: AtomicU64,
    completed_steps: AtomicU64,
    status_sender: watch::Sender<ProgressStatus>,
    children: RwLock<HashMap<Uuid, ProgressTracker>>,
    parent: Option<ProgressTracker>,
    start_time: Instant,
    metadata: RwLock<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressStatus {
    pub id: Uuid,
    pub name: String,
    pub progress: f64,
    pub completed_steps: u64,
    pub total_steps: u64,
    pub message: String,
    pub state: ProgressState,
    pub elapsed_ms: u64,
    pub estimated_remaining_ms: Option<u64>,
    pub throughput: Option<f64>,
    pub children: Vec<ProgressStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgressState {
    NotStarted,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl ProgressTracker {
    pub fn new(name: impl Into<String>, total_steps: u64) -> Self {
        let name = name.into();
        let (status_sender, _) = watch::channel(ProgressStatus {
            id: Uuid::new_v4(),
            name: name.clone(),
            progress: 0.0,
            completed_steps: 0,
            total_steps,
            message: String::new(),
            state: ProgressState::NotStarted,
            elapsed_ms: 0,
            estimated_remaining_ms: None,
            throughput: None,
            children: Vec::new(),
        });

        Self {
            inner: Arc::new(ProgressTrackerInner {
                id: Uuid::new_v4(),
                name,
                total_steps: AtomicU64::new(total_steps),
                completed_steps: AtomicU64::new(0),
                status_sender,
                children: RwLock::new(HashMap::new()),
                parent: None,
                start_time: Instant::now(),
                metadata: RwLock::new(HashMap::new()),
            }),
        }
    }

    pub fn child(&self, name: impl Into<String>, total_steps: u64) -> Self {
        let mut child = Self::new(name, total_steps);
        // Create a new inner with parent set
        let (status_sender, _) = watch::channel(child.inner.status_sender.borrow().clone());
        let new_inner = Arc::new(ProgressTrackerInner {
            id: child.inner.id,
            name: child.inner.name.clone(),
            total_steps: AtomicU64::new(child.inner.total_steps.load(Ordering::Relaxed)),
            completed_steps: AtomicU64::new(child.inner.completed_steps.load(Ordering::Relaxed)),
            status_sender,
            children: RwLock::new(HashMap::new()),
            parent: Some(self.clone()),
            start_time: child.inner.start_time,
            metadata: RwLock::new(HashMap::new()),
        });
        child.inner = new_inner;

        self.inner.children.write().unwrap()
            .insert(child.inner.id, child.clone());
        
        self.update_status();
        child
    }

    pub fn set_total_steps(&self, total: u64) {
        self.inner.total_steps.store(total, Ordering::Release);
        self.update_status();
    }

    pub fn increment(&self, steps: u64) {
        self.inner.completed_steps.fetch_add(steps, Ordering::AcqRel);
        self.update_status();
    }

    pub fn set_progress(&self, completed: u64) {
        self.inner.completed_steps.store(completed, Ordering::Release);
        self.update_status();
    }

    pub fn set_message(&self, message: impl Into<String>) {
        let mut status = self.inner.status_sender.borrow().clone();
        status.message = message.into();
        let _ = self.inner.status_sender.send(status);
    }

    pub fn set_state(&self, state: ProgressState) {
        let mut status = self.inner.status_sender.borrow().clone();
        status.state = state;
        let _ = self.inner.status_sender.send(status);
        
        if let Some(parent) = &self.inner.parent {
            parent.update_status();
        }
    }

    pub fn set_metadata(&self, key: impl Into<String>, value: serde_json::Value) {
        self.inner.metadata.write().unwrap()
            .insert(key.into(), value);
    }

    pub fn get_status(&self) -> ProgressStatus {
        self.inner.status_sender.borrow().clone()
    }

    pub fn subscribe(&self) -> watch::Receiver<ProgressStatus> {
        self.inner.status_sender.subscribe()
    }

    fn update_status(&self) {
        let completed = self.inner.completed_steps.load(Ordering::Acquire);
        let total = self.inner.total_steps.load(Ordering::Acquire);
        let elapsed = self.inner.start_time.elapsed();
        
        let progress = if total > 0 {
            completed as f64 / total as f64
        } else {
            0.0
        };

        let throughput = if elapsed.as_secs() > 0 {
            Some(completed as f64 / elapsed.as_secs_f64())
        } else {
            None
        };

        let estimated_remaining = if completed > 0 && total > completed {
            let remaining_steps = total - completed;
            let rate = completed as f64 / elapsed.as_secs_f64();
            if rate > 0.0 {
                Some(Duration::from_secs_f64(remaining_steps as f64 / rate))
            } else {
                None
            }
        } else {
            None
        };

        let children: Vec<_> = self.inner.children.read().unwrap()
            .values()
            .map(|child| child.get_status())
            .collect();

        let mut status = self.inner.status_sender.borrow().clone();
        status.progress = progress;
        status.completed_steps = completed;
        status.total_steps = total;
        status.elapsed_ms = elapsed.as_millis() as u64;
        status.estimated_remaining_ms = estimated_remaining.map(|d| d.as_millis() as u64);
        status.throughput = throughput;
        status.children = children;
        
        if status.state == ProgressState::Running && progress >= 1.0 {
            status.state = ProgressState::Completed;
        }

        let _ = self.inner.status_sender.send(status);

        // Update parent
        if let Some(parent) = &self.inner.parent {
            parent.update_status();
        }
    }
}

/// Manages all operations in the system
pub struct OperationManager {
    operations: Arc<RwLock<HashMap<Uuid, Operation>>>,
    global_token: CancellationToken,
}

#[derive(Clone)]
pub struct Operation {
    pub id: Uuid,
    pub name: String,
    pub cancellation_token: CancellationToken,
    pub progress_tracker: ProgressTracker,
    pub created_at: Instant,
    pub operation_type: OperationType,
    pub priority: OperationPriority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    DataImport,
    Imputation,
    Validation,
    Export,
    Benchmarking,
    Maintenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OperationPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl OperationManager {
    pub fn new() -> Self {
        Self {
            operations: Arc::new(RwLock::new(HashMap::new())),
            global_token: CancellationToken::new(),
        }
    }

    pub fn create_operation(
        &self,
        name: impl Into<String>,
        operation_type: OperationType,
        total_steps: u64,
    ) -> Operation {
        let name = name.into();
        let operation = Operation {
            id: Uuid::new_v4(),
            name: name.clone(),
            cancellation_token: self.global_token.child_token(),
            progress_tracker: ProgressTracker::new(name, total_steps),
            created_at: Instant::now(),
            operation_type,
            priority: OperationPriority::Normal,
        };

        self.operations.write().unwrap()
            .insert(operation.id, operation.clone());

        operation
    }

    pub fn get_operation(&self, id: Uuid) -> Option<Operation> {
        self.operations.read().unwrap().get(&id).cloned()
    }

    pub fn list_operations(&self) -> Vec<OperationSummary> {
        self.operations.read().unwrap()
            .values()
            .map(|op| OperationSummary {
                id: op.id,
                name: op.name.clone(),
                operation_type: op.operation_type,
                priority: op.priority,
                status: op.progress_tracker.get_status(),
                created_at_ms: op.created_at.elapsed().as_millis() as u64,
                is_cancelled: op.cancellation_token.is_cancelled(),
            })
            .collect()
    }

    pub fn cancel_operation(&self, id: Uuid) -> Result<(), String> {
        if let Some(operation) = self.operations.read().unwrap().get(&id) {
            operation.cancellation_token.cancel();
            operation.progress_tracker.set_state(ProgressState::Cancelled);
            Ok(())
        } else {
            Err(format!("Operation {} not found", id))
        }
    }

    pub fn cancel_all(&self) {
        self.global_token.cancel();
        
        for operation in self.operations.read().unwrap().values() {
            operation.progress_tracker.set_state(ProgressState::Cancelled);
        }
    }

    pub fn cleanup_completed(&self, older_than: Duration) {
        let now = Instant::now();
        let mut operations = self.operations.write().unwrap();
        
        operations.retain(|_, op| {
            let status = op.progress_tracker.get_status();
            let age = now.duration_since(op.created_at);
            
            match status.state {
                ProgressState::Completed | ProgressState::Failed | ProgressState::Cancelled => {
                    age < older_than
                }
                _ => true,
            }
        });
    }

    pub fn get_resource_usage(&self) -> ResourceUsage {
        let operations = self.operations.read().unwrap();
        let active_count = operations.values()
            .filter(|op| {
                let state = op.progress_tracker.get_status().state;
                state == ProgressState::Running || state == ProgressState::Paused
            })
            .count();

        let by_type = operations.values()
            .filter(|op| {
                let state = op.progress_tracker.get_status().state;
                state == ProgressState::Running
            })
            .fold(HashMap::new(), |mut acc, op| {
                *acc.entry(op.operation_type).or_insert(0) += 1;
                acc
            });

        ResourceUsage {
            total_operations: operations.len(),
            active_operations: active_count,
            operations_by_type: by_type,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OperationSummary {
    pub id: Uuid,
    pub name: String,
    pub operation_type: OperationType,
    pub priority: OperationPriority,
    pub status: ProgressStatus,
    pub created_at_ms: u64,
    pub is_cancelled: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub total_operations: usize,
    pub active_operations: usize,
    pub operations_by_type: HashMap<OperationType, usize>,
}

/// Helper for running cancellable async operations
pub async fn run_cancellable<F, T>(
    operation: &Operation,
    future: F,
) -> Result<T, OperationCancelled>
where
    F: Future<Output = T>,
{
    tokio::select! {
        result = future => Ok(result),
        _ = operation.cancellation_token.cancelled() => {
            operation.progress_tracker.set_state(ProgressState::Cancelled);
            Err(OperationCancelled)
        }
    }
}

/// Helper for running cancellable operations with progress updates
pub async fn run_with_progress<F, T, E>(
    operation: &Operation,
    steps: impl IntoIterator<Item = (&'static str, u64)>,
    mut f: F,
) -> Result<T, E>
where
    F: FnMut(&Operation, &'static str) -> Pin<Box<dyn Future<Output = Result<(), E>> + Send>>,
    T: Default,
    E: From<OperationCancelled>,
{
    operation.progress_tracker.set_state(ProgressState::Running);
    
    for (step_name, weight) in steps {
        operation.cancellation_token.check()?;
        operation.progress_tracker.set_message(step_name);
        
        match f(operation, step_name).await {
            Ok(()) => {
                operation.progress_tracker.increment(weight);
            }
            Err(e) => {
                operation.progress_tracker.set_state(ProgressState::Failed);
                return Err(e);
            }
        }
    }
    
    operation.progress_tracker.set_state(ProgressState::Completed);
    Ok(T::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        
        token.cancel();
        assert!(token.is_cancelled());
        
        // Test child cancellation
        let parent = CancellationToken::new();
        let child = parent.child_token();
        
        parent.cancel();
        assert!(child.is_cancelled());
    }

    #[tokio::test]
    async fn test_progress_tracking() {
        let tracker = ProgressTracker::new("Test", 100);
        
        tracker.increment(50);
        let status = tracker.get_status();
        assert_eq!(status.progress, 0.5);
        assert_eq!(status.completed_steps, 50);
        
        tracker.increment(50);
        let status = tracker.get_status();
        assert_eq!(status.progress, 1.0);
        assert_eq!(status.state, ProgressState::Completed);
    }

    #[tokio::test]
    async fn test_hierarchical_progress() {
        let parent = ProgressTracker::new("Parent", 100);
        let child1 = parent.child("Child1", 50);
        let child2 = parent.child("Child2", 50);
        
        child1.increment(25);
        child2.increment(25);
        
        let parent_status = parent.get_status();
        assert_eq!(parent_status.children.len(), 2);
    }

    #[tokio::test]
    async fn test_operation_manager() {
        let manager = OperationManager::new();
        
        let op = manager.create_operation(
            "Test Operation",
            OperationType::Imputation,
            100,
        );
        
        assert_eq!(manager.list_operations().len(), 1);
        
        manager.cancel_operation(op.id).unwrap();
        assert!(op.cancellation_token.is_cancelled());
    }
}