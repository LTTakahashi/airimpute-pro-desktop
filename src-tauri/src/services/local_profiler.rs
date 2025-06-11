// Local performance profiler - no external dependencies
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub operation: String,
    pub start_time: DateTime<Utc>,
    pub duration_ms: u64,
    pub memory_before_mb: f64,
    pub memory_after_mb: f64,
    pub memory_peak_mb: f64,
    pub cpu_usage_percent: f32,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: DateTime<Utc>,
    pub total_operations: usize,
    pub total_duration_ms: u64,
    pub peak_memory_mb: f64,
    pub operations: Vec<PerformanceMetric>,
    pub bottlenecks: Vec<Bottleneck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub operation: String,
    pub issue: String,
    pub impact: String,
    pub suggestion: String,
}

pub struct LocalProfiler {
    metrics: Arc<RwLock<Vec<PerformanceMetric>>>,
    active_operations: Arc<RwLock<HashMap<String, OperationState>>>,
    report_dir: std::path::PathBuf,
}

struct OperationState {
    start_time: Instant,
    start_memory: f64,
    peak_memory: f64,
    metadata: HashMap<String, serde_json::Value>,
}

impl LocalProfiler {
    pub fn new(app_data_dir: &std::path::Path) -> Self {
        let report_dir = app_data_dir.join("performance_reports");
        std::fs::create_dir_all(&report_dir).ok();
        
        Self {
            metrics: Arc::new(RwLock::new(Vec::new())),
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            report_dir,
        }
    }
    
    /// Start profiling an operation
    pub fn start_operation(&self, operation: String) -> OperationHandle {
        let id = format!("{}_{}", operation, uuid::Uuid::new_v4());
        let start_memory = self.get_current_memory_mb();
        
        self.active_operations.write().insert(
            id.clone(),
            OperationState {
                start_time: Instant::now(),
                start_memory,
                peak_memory: start_memory,
                metadata: HashMap::new(),
            }
        );
        
        OperationHandle {
            profiler: self,
            id,
            operation,
        }
    }
    
    /// Record completion of an operation
    fn complete_operation(&self, id: &str, operation: String) {
        if let Some(state) = self.active_operations.write().remove(id) {
            let duration_ms = state.start_time.elapsed().as_millis() as u64;
            let end_memory = self.get_current_memory_mb();
            
            let metric = PerformanceMetric {
                operation,
                start_time: Utc::now() - chrono::Duration::milliseconds(duration_ms as i64),
                duration_ms,
                memory_before_mb: state.start_memory,
                memory_after_mb: end_memory,
                memory_peak_mb: state.peak_memory.max(end_memory),
                cpu_usage_percent: self.get_cpu_usage(),
                metadata: state.metadata,
            };
            
            self.metrics.write().push(metric);
            
            // Auto-generate report if we have enough data
            if self.metrics.read().len() >= 100 {
                self.generate_report();
            }
        }
    }
    
    /// Update metadata for active operation
    fn update_metadata(&self, id: &str, key: String, value: serde_json::Value) {
        if let Some(state) = self.active_operations.write().get_mut(id) {
            state.metadata.insert(key, value);
            
            // Update peak memory
            let current_memory = self.get_current_memory_mb();
            state.peak_memory = state.peak_memory.max(current_memory);
        }
    }
    
    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let metrics = self.metrics.read().clone();
        
        // Calculate totals
        let total_duration_ms: u64 = metrics.iter().map(|m| m.duration_ms).sum();
        let peak_memory_mb = metrics.iter()
            .map(|m| m.memory_peak_mb)
            .fold(0.0, f64::max);
        
        // Identify bottlenecks
        let mut bottlenecks = Vec::new();
        
        // Memory bottlenecks
        let high_memory_ops: Vec<_> = metrics.iter()
            .filter(|m| m.memory_peak_mb - m.memory_before_mb > 500.0)
            .collect();
            
        for op in high_memory_ops {
            bottlenecks.push(Bottleneck {
                operation: op.operation.clone(),
                issue: format!("High memory usage: {:.0}MB increase", 
                    op.memory_peak_mb - op.memory_before_mb),
                impact: "May cause out-of-memory errors on systems with limited RAM".to_string(),
                suggestion: "Consider processing data in smaller chunks".to_string(),
            });
        }
        
        // Slow operations
        let slow_ops: Vec<_> = metrics.iter()
            .filter(|m| m.duration_ms > 5000)
            .collect();
            
        for op in slow_ops {
            bottlenecks.push(Bottleneck {
                operation: op.operation.clone(),
                issue: format!("Slow operation: {:.1}s", op.duration_ms as f64 / 1000.0),
                impact: "Poor user experience, UI may freeze".to_string(),
                suggestion: "Add progress indicators or move to background thread".to_string(),
            });
        }
        
        // Memory leaks
        let potential_leaks: Vec<_> = metrics.windows(5)
            .filter(|window| {
                let memory_increase = window.last().unwrap().memory_after_mb - 
                                    window.first().unwrap().memory_before_mb;
                memory_increase > 100.0
            })
            .collect();
            
        if !potential_leaks.is_empty() {
            bottlenecks.push(Bottleneck {
                operation: "Multiple operations".to_string(),
                issue: "Potential memory leak detected".to_string(),
                impact: "Application memory usage grows over time".to_string(),
                suggestion: "Review memory cleanup in Python bridge and data structures".to_string(),
            });
        }
        
        let report = PerformanceReport {
            timestamp: Utc::now(),
            total_operations: metrics.len(),
            total_duration_ms,
            peak_memory_mb,
            operations: metrics,
            bottlenecks,
        };
        
        // Save report to file
        self.save_report(&report);
        
        report
    }
    
    /// Save report to local file
    fn save_report(&self, report: &PerformanceReport) {
        let filename = format!("performance_{}.json", 
            report.timestamp.format("%Y%m%d_%H%M%S"));
        let path = self.report_dir.join(filename);
        
        if let Ok(json) = serde_json::to_string_pretty(report) {
            if let Ok(mut file) = File::create(path) {
                let _ = file.write_all(json.as_bytes());
            }
        }
        
        // Also create a human-readable summary
        let summary_path = self.report_dir.join("latest_summary.txt");
        if let Ok(mut file) = File::create(summary_path) {
            let _ = writeln!(file, "Performance Report - {}", report.timestamp);
            let _ = writeln!(file, "=====================================");
            let _ = writeln!(file, "Total Operations: {}", report.total_operations);
            let _ = writeln!(file, "Total Time: {:.1}s", report.total_duration_ms as f64 / 1000.0);
            let _ = writeln!(file, "Peak Memory: {:.0}MB", report.peak_memory_mb);
            let _ = writeln!(file, "\nBottlenecks Found:");
            for bottleneck in &report.bottlenecks {
                let _ = writeln!(file, "\n- {}: {}", bottleneck.operation, bottleneck.issue);
                let _ = writeln!(file, "  Impact: {}", bottleneck.impact);
                let _ = writeln!(file, "  Suggestion: {}", bottleneck.suggestion);
            }
        }
    }
    
    /// Get current memory usage in MB
    fn get_current_memory_mb(&self) -> f64 {
        use sysinfo::{System, Pid};
        
        let mut sys = System::new();
        sys.refresh_processes();
        
        let pid = Pid::from(std::process::id() as usize);
        if let Some(process) = sys.process(pid) {
            process.memory() as f64 / 1024.0 / 1024.0
        } else {
            0.0
        }
    }
    
    /// Get current CPU usage
    fn get_cpu_usage(&self) -> f32 {
        use sysinfo::{System, Pid};
        
        let mut sys = System::new();
        sys.refresh_processes();
        
        let pid = Pid::from(std::process::id() as usize);
        if let Some(process) = sys.process(pid) {
            process.cpu_usage()
        } else {
            0.0
        }
    }
    
    /// Clear all metrics
    pub fn clear(&self) {
        self.metrics.write().clear();
    }
    
    /// Get metrics summary
    pub fn get_summary(&self) -> HashMap<String, serde_json::Value> {
        let metrics = self.metrics.read();
        let mut summary = HashMap::new();
        
        // Group by operation type
        let mut op_stats: HashMap<String, (u64, u64, f64)> = HashMap::new(); // (count, total_ms, total_memory)
        
        for metric in metrics.iter() {
            let entry = op_stats.entry(metric.operation.clone()).or_insert((0, 0, 0.0));
            entry.0 += 1;
            entry.1 += metric.duration_ms;
            entry.2 += metric.memory_peak_mb - metric.memory_before_mb;
        }
        
        summary.insert("operation_count".to_string(), 
            serde_json::json!(metrics.len()));
        summary.insert("operation_stats".to_string(), 
            serde_json::json!(op_stats));
        
        summary
    }
}

/// Handle for tracking a specific operation
pub struct OperationHandle<'a> {
    profiler: &'a LocalProfiler,
    id: String,
    operation: String,
}

impl<'a> OperationHandle<'a> {
    pub fn add_metadata(&self, key: &str, value: impl Serialize) {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.profiler.update_metadata(&self.id, key.to_string(), json_value);
        }
    }
}

impl<'a> Drop for OperationHandle<'a> {
    fn drop(&mut self) {
        self.profiler.complete_operation(&self.id, self.operation.clone());
    }
}

// Convenience macro for profiling code blocks
#[macro_export]
macro_rules! profile {
    ($profiler:expr, $operation:expr, $block:block) => {{
        let _handle = $profiler.start_operation($operation.to_string());
        $block
    }};
    
    ($profiler:expr, $operation:expr, $metadata:expr, $block:block) => {{
        let _handle = $profiler.start_operation($operation.to_string());
        for (key, value) in $metadata {
            _handle.add_metadata(key, value);
        }
        $block
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_profiler() {
        let temp_dir = TempDir::new().unwrap();
        let profiler = LocalProfiler::new(temp_dir.path());
        
        // Profile an operation
        {
            let handle = profiler.start_operation("test_operation".to_string());
            handle.add_metadata("rows", 1000);
            std::thread::sleep(Duration::from_millis(10));
        }
        
        let summary = profiler.get_summary();
        assert_eq!(summary["operation_count"], serde_json::json!(1));
    }
}