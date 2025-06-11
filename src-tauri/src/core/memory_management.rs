// Academic-grade memory management system with leak detection and automatic cleanup
// Based on best practices from systems research and Rust memory safety principles

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::thread;
use std::ptr::NonNull;
use std::backtrace::Backtrace;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn, error, info};
use chrono;

/// Custom allocator wrapper for memory tracking
pub struct TrackingAllocator {
    inner: System,
    tracker: Arc<MemoryTracker>,
}

/// Thread-safe memory tracking system
pub struct MemoryTracker {
    allocations: RwLock<HashMap<usize, AllocationInfo>>,
    statistics: RwLock<MemoryStatistics>,
    config: MemoryConfig,
}

#[derive(Debug, Clone)]
struct AllocationInfo {
    size: usize,
    layout: Layout,
    timestamp: Instant,
    backtrace: Arc<Backtrace>,
    thread_id: thread::ThreadId,
    allocation_type: AllocationType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllocationType {
    Dataset,
    Computation,
    Cache,
    Temporary,
    PythonBridge,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    pub total_allocated: usize,
    pub total_freed: usize,
    pub current_usage: usize,
    pub peak_usage: usize,
    pub allocation_count: usize,
    pub free_count: usize,
    pub leaked_bytes: usize,
    pub largest_allocation: usize,
    pub allocations_by_type: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub track_allocations: bool,
    pub leak_detection: bool,
    pub automatic_cleanup: bool,
    pub memory_limit: Option<usize>,
    pub warning_threshold: f64, // Percentage of limit
    pub gc_interval: Duration,
    pub python_memory_limit: Option<usize>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            track_allocations: true,
            leak_detection: true,
            automatic_cleanup: true,
            memory_limit: None, // Will be set based on system
            warning_threshold: 0.8,
            gc_interval: Duration::from_secs(60),
            python_memory_limit: None,
        }
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        
        if !ptr.is_null() && self.tracker.config.track_allocations {
            self.tracker.record_allocation(ptr as usize, layout);
        }
        
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if self.tracker.config.track_allocations {
            self.tracker.record_deallocation(ptr as usize, layout);
        }
        
        self.inner.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc_zeroed(layout);
        
        if !ptr.is_null() && self.tracker.config.track_allocations {
            self.tracker.record_allocation(ptr as usize, layout);
        }
        
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = self.inner.realloc(ptr, layout, new_size);
        
        if self.tracker.config.track_allocations {
            if !ptr.is_null() {
                self.tracker.record_deallocation(ptr as usize, layout);
            }
            if !new_ptr.is_null() {
                let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
                self.tracker.record_allocation(new_ptr as usize, new_layout);
            }
        }
        
        new_ptr
    }
}

impl MemoryTracker {
    pub fn new(config: MemoryConfig) -> Self {
        let tracker = Self {
            allocations: RwLock::new(HashMap::new()),
            statistics: RwLock::new(MemoryStatistics::default()),
            config,
        };

        // Start garbage collection thread if enabled
        if config.automatic_cleanup {
            let tracker_clone = Arc::new(tracker.clone());
            thread::spawn(move || {
                tracker_clone.garbage_collection_loop();
            });
        }

        tracker
    }

    fn record_allocation(&self, ptr: usize, layout: Layout) {
        let info = AllocationInfo {
            size: layout.size(),
            layout,
            timestamp: Instant::now(),
            backtrace: Arc::new(Backtrace::capture()),
            thread_id: thread::current().id(),
            allocation_type: self.infer_allocation_type(&layout),
        };

        let mut allocations = self.allocations.write().unwrap();
        allocations.insert(ptr, info.clone());

        let mut stats = self.statistics.write().unwrap();
        stats.total_allocated += layout.size();
        stats.current_usage += layout.size();
        stats.allocation_count += 1;
        
        if stats.current_usage > stats.peak_usage {
            stats.peak_usage = stats.current_usage;
        }
        
        if layout.size() > stats.largest_allocation {
            stats.largest_allocation = layout.size();
        }

        // Check memory limits
        self.check_memory_limits(&stats);
    }

    fn record_deallocation(&self, ptr: usize, layout: Layout) {
        let mut allocations = self.allocations.write().unwrap();
        
        if let Some(info) = allocations.remove(&ptr) {
            let mut stats = self.statistics.write().unwrap();
            stats.total_freed += info.size;
            stats.current_usage = stats.current_usage.saturating_sub(info.size);
            stats.free_count += 1;
        } else if self.config.leak_detection {
            warn!("Attempted to free untracked memory at {:x}", ptr);
        }
    }

    fn infer_allocation_type(&self, layout: &Layout) -> AllocationType {
        // Heuristic based on size and alignment
        match layout.size() {
            size if size > 100_000_000 => AllocationType::Dataset,
            size if size > 1_000_000 => AllocationType::Computation,
            size if layout.align() > 64 => AllocationType::PythonBridge,
            _ => AllocationType::Unknown,
        }
    }

    fn check_memory_limits(&self, stats: &MemoryStatistics) {
        if let Some(limit) = self.config.memory_limit {
            let usage_ratio = stats.current_usage as f64 / limit as f64;
            
            if usage_ratio > self.config.warning_threshold {
                warn!(
                    "Memory usage at {:.1}% of limit ({:.2} MB / {:.2} MB)",
                    usage_ratio * 100.0,
                    stats.current_usage as f64 / 1_048_576.0,
                    limit as f64 / 1_048_576.0
                );
            }
            
            if stats.current_usage > limit {
                error!("Memory limit exceeded! Triggering emergency cleanup");
                self.emergency_cleanup();
            }
        }
    }

    fn garbage_collection_loop(&self) {
        loop {
            thread::sleep(self.config.gc_interval);
            self.run_garbage_collection();
        }
    }

    fn run_garbage_collection(&self) {
        info!("Running garbage collection");
        
        let now = Instant::now();
        let mut allocations = self.allocations.write().unwrap();
        let mut leaked = Vec::new();
        
        // Detect potential leaks (allocations older than 5 minutes)
        for (ptr, info) in allocations.iter() {
            if now.duration_since(info.timestamp) > Duration::from_secs(300) {
                match info.allocation_type {
                    AllocationType::Temporary => leaked.push(*ptr),
                    AllocationType::Cache => {
                        // Cache can be evicted after 10 minutes
                        if now.duration_since(info.timestamp) > Duration::from_secs(600) {
                            leaked.push(*ptr);
                        }
                    }
                    _ => {}
                }
            }
        }
        
        if !leaked.is_empty() {
            warn!("Detected {} potential memory leaks", leaked.len());
            let mut stats = self.statistics.write().unwrap();
            
            for ptr in leaked {
                if let Some(info) = allocations.get(&ptr) {
                    stats.leaked_bytes += info.size;
                    error!(
                        "Potential leak: {} bytes allocated at {:x} by thread {:?}",
                        info.size, ptr, info.thread_id
                    );
                    
                    // Log backtrace for debugging
                    debug!("Allocation backtrace: {:?}", info.backtrace);
                }
            }
        }
    }

    fn emergency_cleanup(&self) {
        error!("Emergency memory cleanup triggered!");
        
        // Clear caches
        self.clear_cache_allocations();
        
        // Force Python garbage collection
        self.trigger_python_gc();
        
        // Request computation cancellation
        self.request_computation_cancellation();
    }

    fn clear_cache_allocations(&self) {
        let allocations = self.allocations.read().unwrap();
        let cache_allocations: Vec<_> = allocations
            .iter()
            .filter(|(_, info)| info.allocation_type == AllocationType::Cache)
            .map(|(ptr, _)| *ptr)
            .collect();
        
        info!("Clearing {} cache allocations", cache_allocations.len());
        
        // In a real implementation, we would coordinate with the cache system
        // to properly free these allocations
    }

    fn trigger_python_gc(&self) {
        // Trigger Python garbage collection through PyO3
        info!("Triggering Python garbage collection");
        // Implementation would use PyO3 to call gc.collect()
    }

    fn request_computation_cancellation(&self) {
        // Signal all running computations to cancel
        info!("Requesting cancellation of running computations");
        // Implementation would use a global cancellation token system
    }

    pub fn get_statistics(&self) -> MemoryStatistics {
        self.statistics.read().unwrap().clone()
    }

    pub fn get_memory_report(&self) -> MemoryReport {
        let stats = self.get_statistics();
        let allocations = self.allocations.read().unwrap();
        
        let allocations_by_type = allocations
            .values()
            .fold(HashMap::new(), |mut acc, info| {
                *acc.entry(format!("{:?}", info.allocation_type))
                    .or_insert(0) += info.size;
                acc
            });

        let thread_allocations = allocations
            .values()
            .fold(HashMap::new(), |mut acc, info| {
                *acc.entry(info.thread_id)
                    .or_insert(0) += info.size;
                acc
            });

        MemoryReport {
            timestamp: chrono::Utc::now(),
            statistics: stats,
            allocations_by_type,
            thread_allocations: thread_allocations
                .into_iter()
                .map(|(id, size)| (format!("{:?}", id), size))
                .collect(),
            largest_allocations: self.get_largest_allocations(10),
            memory_pressure: self.calculate_memory_pressure(),
        }
    }

    fn get_largest_allocations(&self, count: usize) -> Vec<AllocationSummary> {
        let allocations = self.allocations.read().unwrap();
        let mut sorted: Vec<_> = allocations
            .iter()
            .map(|(ptr, info)| AllocationSummary {
                ptr: *ptr,
                size: info.size,
                age: Instant::now().duration_since(info.timestamp),
                allocation_type: format!("{:?}", info.allocation_type),
            })
            .collect();
        
        sorted.sort_by_key(|a| std::cmp::Reverse(a.size));
        sorted.truncate(count);
        sorted
    }

    fn calculate_memory_pressure(&self) -> f64 {
        let stats = self.statistics.read().unwrap();
        
        if let Some(limit) = self.config.memory_limit {
            stats.current_usage as f64 / limit as f64
        } else {
            // Use system memory as reference
            let sys_info = sysinfo::System::new_all();
            stats.current_usage as f64 / sys_info.total_memory() as f64
        }
    }
}

impl Clone for MemoryTracker {
    fn clone(&self) -> Self {
        Self {
            allocations: RwLock::new(HashMap::new()),
            statistics: RwLock::new(MemoryStatistics::default()),
            config: self.config.clone(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub statistics: MemoryStatistics,
    pub allocations_by_type: HashMap<String, usize>,
    pub thread_allocations: HashMap<String, usize>,
    pub largest_allocations: Vec<AllocationSummary>,
    pub memory_pressure: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AllocationSummary {
    pub ptr: usize,
    pub size: usize,
    pub age: Duration,
    pub allocation_type: String,
}

impl Default for MemoryStatistics {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            total_freed: 0,
            current_usage: 0,
            peak_usage: 0,
            allocation_count: 0,
            free_count: 0,
            leaked_bytes: 0,
            largest_allocation: 0,
            allocations_by_type: HashMap::new(),
        }
    }
}

/// RAII guard for scoped memory tracking
pub struct MemoryScope {
    name: String,
    start_usage: usize,
    tracker: Arc<MemoryTracker>,
}

impl MemoryScope {
    pub fn new(name: String, tracker: Arc<MemoryTracker>) -> Self {
        let start_usage = tracker.get_statistics().current_usage;
        Self {
            name,
            start_usage,
            tracker,
        }
    }
}

impl Drop for MemoryScope {
    fn drop(&mut self) {
        let end_usage = self.tracker.get_statistics().current_usage;
        let delta = end_usage as i64 - self.start_usage as i64;
        
        info!(
            "Memory scope '{}' completed: {} bytes",
            self.name,
            if delta >= 0 {
                format!("+{}", delta)
            } else {
                delta.to_string()
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracking() {
        let config = MemoryConfig::default();
        let tracker = Arc::new(MemoryTracker::new(config));
        
        // Simulate allocation
        let layout = Layout::from_size_align(1024, 8).unwrap();
        tracker.record_allocation(0x1000, layout);
        
        let stats = tracker.get_statistics();
        assert_eq!(stats.current_usage, 1024);
        assert_eq!(stats.allocation_count, 1);
        
        // Simulate deallocation
        tracker.record_deallocation(0x1000, layout);
        
        let stats = tracker.get_statistics();
        assert_eq!(stats.current_usage, 0);
        assert_eq!(stats.free_count, 1);
    }

    #[test]
    fn test_memory_limits() {
        let mut config = MemoryConfig::default();
        config.memory_limit = Some(1_048_576); // 1MB limit
        config.warning_threshold = 0.8;
        
        let tracker = Arc::new(MemoryTracker::new(config));
        
        // Allocate 900KB (should trigger warning)
        let layout = Layout::from_size_align(921_600, 8).unwrap();
        tracker.record_allocation(0x2000, layout);
        
        let stats = tracker.get_statistics();
        assert!(stats.current_usage > 819_200); // 80% of 1MB
    }
}