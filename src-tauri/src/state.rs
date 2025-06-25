use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tauri::AppHandle;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use anyhow::{Result, Context};

use crate::{
    db::Database,
    python::{PythonRuntime, safe_bridge_v2::SafePythonBridge, PythonWorkerPool},
    services::{cache::CacheManager, offline_resources::OfflineResourceManager},
    core::{
        project::Project,
        data::Dataset,
        imputation::ImputationJob,
        imputation_result::ImputationResult,
        progress_tracker::ProgressManager,
    },
    commands::imputation_v3::ImputationJobV3,
};

/// Central application state manager with thread-safe access
#[derive(Clone)]
pub struct AppState {
    /// Database connection pool
    pub db: Arc<Database>,
    
    /// Embedded Python runtime for scientific computing
    pub python_runtime: Arc<PythonRuntime>,
    
    /// Python bridge for safe integration
    pub python_bridge: Arc<SafePythonBridge>,
    
    /// Progress tracking manager
    pub progress_manager: Arc<ProgressManager>,
    
    /// In-memory cache for performance
    pub cache: Arc<CacheManager>,
    
    /// Active projects mapping
    pub projects: Arc<DashMap<Uuid, Project>>,
    
    /// Current active project ID
    pub active_project: Arc<RwLock<Option<Uuid>>>,
    
    /// User preferences and settings
    pub preferences: Arc<RwLock<UserPreferences>>,
    
    /// System metrics and monitoring
    pub metrics: Arc<SystemMetrics>,
    
    /// Background task handles
    pub task_handles: Arc<DashMap<String, tokio::task::JoinHandle<()>>>,
    
    /// Loaded datasets
    pub datasets: Arc<DashMap<Uuid, Arc<Dataset>>>,
    
    /// Active imputation jobs
    pub imputation_jobs: Arc<DashMap<Uuid, Arc<tokio::sync::Mutex<ImputationJob>>>>,
    
    /// Active imputation jobs (v3 architecture)
    pub imputation_jobs_v3: Arc<RwLock<HashMap<Uuid, ImputationJobV3>>>,
    
    /// Imputation results storage
    pub imputation_results: Arc<RwLock<HashMap<Uuid, ImputationResult>>>,
    
    /// Python worker pool for high-performance computing
    pub worker_pool: Arc<RwLock<Option<PythonWorkerPool>>>,
    
    /// Recent project paths
    pub recent_projects: Arc<RwLock<Vec<String>>>,
    
    /// Offline resources manager
    pub offline_resources: Arc<OfflineResourceManager>,
    
    /// Application data directory path
    pub app_data_dir: std::path::PathBuf,
}

impl AppState {
    /// Get a clone of the inner state (for async tasks)
    pub fn inner(&self) -> Arc<AppState> {
        // Since AppState is already Clone, we can create an Arc from self
        Arc::new(self.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub user_mode: UserMode,
    pub theme: Theme,
    pub language: String,
    pub auto_save_interval: u32, // minutes
    pub enable_telemetry: bool,
    pub computation_settings: ComputationSettings,
    pub ui_settings: UISettings,
    pub export_defaults: ExportDefaults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserMode {
    Student,
    Researcher,
    Expert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Theme {
    Light,
    Dark,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationSettings {
    pub max_threads: Option<usize>,
    pub gpu_acceleration: bool,
    pub memory_limit_gb: Option<f32>,
    pub chunk_size: usize,
    pub progress_update_interval_ms: u32,
    pub enable_caching: bool,
    pub validation_level: ValidationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationLevel {
    Minimal,
    Standard,
    Comprehensive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UISettings {
    pub show_tooltips: bool,
    pub animations_enabled: bool,
    pub compact_mode: bool,
    pub show_confidence_intervals: bool,
    pub default_visualization_type: String,
    pub auto_refresh_interval_s: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportDefaults {
    pub csv_delimiter: String,
    pub csv_encoding: String,
    pub include_metadata: bool,
    pub include_confidence_intervals: bool,
    pub latex_template: String,
    pub figure_dpi: u32,
    pub figure_format: String,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            user_mode: UserMode::Student,
            theme: Theme::Auto,
            language: "en".to_string(),
            auto_save_interval: 5,
            enable_telemetry: false,
            computation_settings: ComputationSettings::default(),
            ui_settings: UISettings::default(),
            export_defaults: ExportDefaults::default(),
        }
    }
}

impl Default for ComputationSettings {
    fn default() -> Self {
        Self {
            max_threads: None, // Use all available
            gpu_acceleration: false,
            memory_limit_gb: None, // No limit
            chunk_size: 10_000,
            progress_update_interval_ms: 100,
            enable_caching: true,
            validation_level: ValidationLevel::Standard,
        }
    }
}

impl Default for UISettings {
    fn default() -> Self {
        Self {
            show_tooltips: true,
            animations_enabled: true,
            compact_mode: false,
            show_confidence_intervals: true,
            default_visualization_type: "time_series".to_string(),
            auto_refresh_interval_s: None,
        }
    }
}

impl Default for ExportDefaults {
    fn default() -> Self {
        Self {
            csv_delimiter: ",".to_string(),
            csv_encoding: "UTF-8".to_string(),
            include_metadata: true,
            include_confidence_intervals: true,
            latex_template: "ieee".to_string(),
            figure_dpi: 300,
            figure_format: "png".to_string(),
        }
    }
}

/// System metrics for monitoring application performance
pub struct SystemMetrics {
    inner: Arc<RwLock<SystemMetricsInner>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemMetricsInner {
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f32,
    pub active_computations: usize,
    pub cache_hit_rate: f32,
    pub python_calls_per_minute: u32,
    pub last_updated: DateTime<Utc>,
}

impl SystemMetrics {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(SystemMetricsInner {
                memory_usage_mb: 0.0,
                cpu_usage_percent: 0.0,
                active_computations: 0,
                cache_hit_rate: 0.0,
                python_calls_per_minute: 0,
                last_updated: Utc::now(),
            })),
        }
    }
    
    pub fn update_memory_usage(&self, usage_mb: f64) {
        self.inner.write().memory_usage_mb = usage_mb;
        self.inner.write().last_updated = Utc::now();
    }
    
    pub fn update_cpu_usage(&self, usage_percent: f32) {
        self.inner.write().cpu_usage_percent = usage_percent;
    }
    
    pub fn increment_active_computations(&self) {
        self.inner.write().active_computations += 1;
    }
    
    pub fn decrement_active_computations(&self) {
        let mut metrics = self.inner.write();
        if metrics.active_computations > 0 {
            metrics.active_computations -= 1;
        }
    }
    
    pub fn get_snapshot(&self) -> SystemMetricsInner {
        self.inner.read().clone()
    }
}

/// Application state manager with initialization and lifecycle management
pub struct AppStateManager;

impl AppStateManager {
    pub fn initialize(app_handle: AppHandle) -> Result<Arc<AppState>> {
        // Initialize database
        let app_data_dir = app_handle
            .path_resolver()
            .app_data_dir()
            .ok_or_else(|| anyhow::anyhow!("Failed to resolve app data directory"))?;
        let db_path = app_data_dir.join("airimpute.db");
            
        // Get migrations path - try resource directory first, then fallback to development path
        let migrations_path = if let Some(resource_dir) = app_handle.path_resolver().resource_dir() {
            let resource_migrations = resource_dir.join("migrations");
            if resource_migrations.exists() {
                resource_migrations
            } else {
                // Development mode - use src-tauri/migrations
                std::path::PathBuf::from("src-tauri/migrations")
            }
        } else {
            // Fallback for development
            std::path::PathBuf::from("src-tauri/migrations")
        };
        
        let db = tokio::runtime::Runtime::new()
            .context("Failed to create Tokio runtime")?
            .block_on(Database::new_with_migrations(&db_path, &migrations_path))?;
        
        // Initialize Python runtime with proper production path handling
        #[cfg(feature = "python-support")]
        {
            use crate::python::runtime_init::initialize_python_runtime;
            initialize_python_runtime(&app_handle)?;
        }
        
        // Get Python path for runtime initialization
        let python_path = app_handle
            .path_resolver()
            .resource_dir()
            .map(|p| p.join("python"))
            .unwrap_or_else(|| {
                // Fallback to checking next to executable in production
                std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|d| d.join("python")))
                    .unwrap_or_else(|| std::path::PathBuf::from("python"))
            });
            
        let python_runtime = PythonRuntime::new(&python_path)?;
        
        // Initialize Python bridge using safe_bridge_v2
        let python_bridge = SafePythonBridge::new(Default::default());
        
        // Initialize progress manager
        let progress_manager = ProgressManager::new();
        
        // Initialize cache
        let cache = CacheManager::new(100_000_000); // 100MB cache
        
        // Load user preferences
        let preferences = Self::load_preferences(&app_handle)?;
        
        // Initialize offline resources
        let offline_resources = OfflineResourceManager::new();
        
        // Create state
        let state = Arc::new(AppState {
            db: Arc::new(db),
            python_runtime: Arc::new(python_runtime),
            python_bridge: Arc::new(python_bridge),
            progress_manager: Arc::new(progress_manager),
            cache: Arc::new(cache),
            projects: Arc::new(DashMap::new()),
            active_project: Arc::new(RwLock::new(None)),
            preferences: Arc::new(RwLock::new(preferences)),
            metrics: Arc::new(SystemMetrics::new()),
            task_handles: Arc::new(DashMap::new()),
            datasets: Arc::new(DashMap::new()),
            imputation_jobs: Arc::new(DashMap::new()),
            imputation_jobs_v3: Arc::new(RwLock::new(HashMap::new())),
            imputation_results: Arc::new(RwLock::new(HashMap::new())),
            worker_pool: Arc::new(RwLock::new(None)),
            recent_projects: Arc::new(RwLock::new(Vec::new())),
            offline_resources: Arc::new(offline_resources),
            app_data_dir,
        });
        
        // Initialize background services
        Self::initialize_background_services(&app_handle, &state)?;
        
        Ok(state)
    }
    
    fn load_preferences(app_handle: &AppHandle) -> Result<UserPreferences> {
        let prefs_path = app_handle
            .path_resolver()
            .app_config_dir()
            .ok_or_else(|| anyhow::anyhow!("Failed to resolve config directory"))?
            .join("preferences.json");
            
        if prefs_path.exists() {
            let contents = std::fs::read_to_string(&prefs_path)?;
            Ok(serde_json::from_str(&contents)?)
        } else {
            // Create default preferences
            let prefs = UserPreferences::default();
            
            // Ensure directory exists
            if let Some(parent) = prefs_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            
            // Save default preferences
            std::fs::write(&prefs_path, serde_json::to_string_pretty(&prefs)?)?;
            
            Ok(prefs)
        }
    }
    
    fn initialize_background_services(
        app_handle: &AppHandle,
        state: &Arc<AppState>,
    ) -> Result<()> {
        // Metrics collector
        let state_clone = state.clone();
        let handle = tokio::spawn(async move {
            loop {
                // Update system metrics
                let sys_info = sysinfo::System::new_all();
                let memory_usage = sys_info.used_memory() as f64 / 1_048_576.0; // MB
                let cpu_usage = sys_info.global_cpu_info().cpu_usage();
                
                state_clone.metrics.update_memory_usage(memory_usage);
                state_clone.metrics.update_cpu_usage(cpu_usage);
                
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        });
        
        state.task_handles.insert("metrics_collector".to_string(), handle);
        
        Ok(())
    }
    
    pub async fn save_preferences(
        app_handle: &AppHandle,
        preferences: &UserPreferences,
    ) -> Result<()> {
        let prefs_path = app_handle
            .path_resolver()
            .app_config_dir()
            .ok_or_else(|| anyhow::anyhow!("Failed to resolve config directory"))?
            .join("preferences.json");
            
        let json = serde_json::to_string_pretty(preferences)?;
        tokio::fs::write(&prefs_path, json).await?;
        
        Ok(())
    }
    
    pub async fn shutdown(state: Arc<AppState>) -> Result<()> {
        // Cancel all background tasks
        for entry in state.task_handles.iter() {
            let name = entry.key();
            let handle = entry.value();
            tracing::info!("Cancelling background task: {}", name);
            handle.abort();
        }
        
        // Save any pending data
        if let Some(project_id) = *state.active_project.read() {
            if let Some(project) = state.projects.get(&project_id) {
                project.save(&state.db).await?;
            }
        }
        
        // Shutdown Python runtime
        state.python_runtime.shutdown().await?;
        
        // Close database
        state.db.close().await?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_preferences() {
        let prefs = UserPreferences::default();
        assert!(matches!(prefs.user_mode, UserMode::Student));
        assert_eq!(prefs.auto_save_interval, 5);
        assert!(!prefs.enable_telemetry);
    }
    
    #[test]
    fn test_system_metrics() {
        let metrics = SystemMetrics::new();
        
        metrics.update_memory_usage(1024.0);
        metrics.increment_active_computations();
        
        let snapshot = metrics.get_snapshot();
        assert_eq!(snapshot.memory_usage_mb, 1024.0);
        assert_eq!(snapshot.active_computations, 1);
    }
}