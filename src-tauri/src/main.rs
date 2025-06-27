#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use mimalloc::MiMalloc;
use std::sync::Arc;
use tauri::{
    CustomMenuItem, Manager, SystemTray, SystemTrayEvent, SystemTrayMenu, SystemTrayMenuItem,
    WindowEvent, GlobalWindowEvent, SystemTraySubmenu,
};
use tracing::{info, error, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod commands;
mod core;
mod db;
mod error;
mod python;
mod security;
mod services;
mod state;
mod utils;
mod validation;

use crate::state::{AppState, AppStateManager};

// Use MiMalloc for better memory performance
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    // Initialize tracing for professional logging (must be early for error reporting)
    initialize_logging();
    
    info!("Starting AirImpute Pro Desktop v{}", env!("CARGO_PKG_VERSION"));
    
    // Initialize the application
    match run_app() {
        Ok(_) => info!("Application terminated successfully"),
        Err(e) => {
            error!("Application failed to start: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_app() -> anyhow::Result<()> {
    // Create system tray
    let system_tray = create_system_tray();
    
    // Build the app with conditional command handlers
    #[cfg(debug_assertions)]
    {
        tauri::Builder::default()
            .system_tray(system_tray)
            .on_system_tray_event(handle_system_tray_event)
            .setup(|app| {
                // Initialize DLL security with Python directory
                #[cfg(target_os = "windows")]
                {
                    let python_dir = security::dll_security::get_python_directory(&app.handle());
                    if let Err(e) = security::dll_security::initialize_dll_security(python_dir.as_deref()) {
                        error!("Failed to initialize DLL security: {}", e);
                        // Continue anyway, but log the error
                    }
                }
                
                let state = initialize_app_state(app)?;
                app.manage(state);
                setup_main_window(app)?;
                register_global_shortcuts(app)?;
                initialize_background_services(app)?;
                perform_startup_checks(app)?;
                Ok(())
            })
            .invoke_handler(tauri::generate_handler![
        // Debug Commands (for testing)
        commands::debug::ping,
        commands::debug::check_python_bridge,
        commands::debug::test_numpy,
        
        // Data Management Commands
        commands::data::load_dataset,
        commands::data::save_dataset,
        commands::data::validate_dataset,
        commands::data::get_dataset_statistics,
        commands::data::preview_dataset,
        commands::data::import_from_multiple_sources,
        
        // Imputation Commands
        commands::imputation::run_imputation,
        commands::imputation::run_batch_imputation,
        commands::imputation::get_available_methods,
        commands::imputation::estimate_processing_time,
        commands::imputation::validate_imputation_results,
        commands::imputation::compare_methods,
        commands::imputation::get_imputation_method_documentation,
        
        // Imputation V2 Commands (New Integration)
        commands::imputation_v2::get_imputation_methods,
        commands::imputation_v2::validate_imputation_data,
        commands::imputation_v2::run_imputation_v2,
        commands::imputation_v2::get_imputation_status,
        commands::imputation_v2::get_imputation_result,
        commands::imputation_v2::cancel_imputation,
        commands::imputation_v2::estimate_imputation_time,
        
        // Imputation V3 Commands (Arrow-based High Performance)
        commands::imputation_v3::initialize_worker_pool,
        commands::imputation_v3::run_imputation_v3,
        commands::imputation_v3::get_imputation_status_v3,
        commands::imputation_v3::cancel_imputation_v3,
        commands::imputation_v3::get_imputation_methods_v3,
        commands::imputation_v3::check_worker_health,
        
        // Analysis Commands  
        commands::analysis::compute_missing_patterns,
        commands::analysis::analyze_temporal_patterns,
        commands::analysis::analyze_spatial_correlations,
        commands::analysis::generate_quality_report,
        commands::analysis::perform_sensitivity_analysis,
        
        // Export Commands
        commands::export::export_to_csv,
        commands::export::export_to_excel,
        commands::export::export_to_netcdf,
        commands::export::export_to_hdf5,
        commands::export::generate_latex_report,
        commands::export::generate_publication_package,
        
        // Visualization Commands
        commands::visualization::generate_missing_pattern_plot,
        commands::visualization::generate_time_series_plot,
        commands::visualization::generate_correlation_matrix,
        commands::visualization::generate_uncertainty_bands,
        commands::visualization::create_interactive_dashboard,
        
        // Settings Commands
        commands::settings::get_user_preferences,
        commands::settings::update_user_preferences,
        commands::settings::get_computation_settings,
        commands::settings::update_computation_settings,
        commands::settings::reset_to_defaults,
        
        // System Commands
        commands::system::get_system_info,
        commands::system::check_python_runtime,
        commands::system::get_memory_usage,
        commands::system::clear_cache,
        commands::system::run_diagnostics,
        
        // Project Management
        commands::project::create_project,
        commands::project::open_project,
        commands::project::save_project,
        commands::project::get_recent_projects,
        commands::project::archive_project,
        
        // Benchmark Commands
        commands::benchmark::get_benchmark_datasets,
        commands::benchmark::run_benchmark,
        commands::benchmark::get_benchmark_results,
        commands::benchmark::export_benchmark_results,
        commands::benchmark::generate_reproducibility_certificate,
        
        // Publication Commands
        commands::publication::save_report,
        commands::publication::load_report,
        commands::publication::list_reports,
        commands::publication::render_latex,
        commands::publication::export_report,
        commands::publication::import_bibtex,
        commands::publication::format_citation,
        commands::publication::generate_bibliography,
        commands::publication::get_report_templates,
        
        // Help Commands (Offline)
        commands::help::search_help,
        commands::help::get_method_documentation,
        commands::help::list_tutorials,
        commands::help::get_tutorial,
        commands::help::get_offline_status,
        commands::help::get_quick_start_guide,
        commands::help::get_sample_datasets,
    ])
    .on_window_event(handle_window_event)
    .run(tauri::generate_context!())
    .map_err(|e| anyhow::anyhow!("Failed to run application: {}", e))?;
    }
    
    #[cfg(not(debug_assertions))]
    {
        tauri::Builder::default()
            .system_tray(system_tray)
            .on_system_tray_event(handle_system_tray_event)
            .setup(|app| {
                // Initialize DLL security with Python directory
                #[cfg(target_os = "windows")]
                {
                    let python_dir = security::dll_security::get_python_directory(&app.handle());
                    if let Err(e) = security::dll_security::initialize_dll_security(python_dir.as_deref()) {
                        error!("Failed to initialize DLL security: {}", e);
                        // Continue anyway, but log the error
                    }
                }
                
                let state = initialize_app_state(app)?;
                app.manage(state);
                setup_main_window(app)?;
                register_global_shortcuts(app)?;
                initialize_background_services(app)?;
                perform_startup_checks(app)?;
                Ok(())
            })
            .invoke_handler(tauri::generate_handler![
        // Data Management Commands
        commands::data::load_dataset,
        commands::data::save_dataset,
        commands::data::validate_dataset,
        commands::data::get_dataset_statistics,
        commands::data::preview_dataset,
        commands::data::import_from_multiple_sources,
        
        // Imputation Commands
        commands::imputation::run_imputation,
        commands::imputation::run_batch_imputation,
        commands::imputation::get_available_methods,
        commands::imputation::estimate_processing_time,
        commands::imputation::validate_imputation_results,
        commands::imputation::compare_methods,
        commands::imputation::get_imputation_method_documentation,
        
        // Imputation V2 Commands (New Integration)
        commands::imputation_v2::get_imputation_methods,
        commands::imputation_v2::validate_imputation_data,
        commands::imputation_v2::run_imputation_v2,
        commands::imputation_v2::get_imputation_status,
        commands::imputation_v2::get_imputation_result,
        commands::imputation_v2::cancel_imputation,
        commands::imputation_v2::estimate_imputation_time,
        
        // Imputation V3 Commands (Arrow-based High Performance)
        commands::imputation_v3::initialize_worker_pool,
        commands::imputation_v3::run_imputation_v3,
        commands::imputation_v3::get_imputation_status_v3,
        commands::imputation_v3::cancel_imputation_v3,
        commands::imputation_v3::get_imputation_methods_v3,
        commands::imputation_v3::check_worker_health,
        
        // Analysis Commands  
        commands::analysis::compute_missing_patterns,
        commands::analysis::analyze_temporal_patterns,
        commands::analysis::analyze_spatial_correlations,
        commands::analysis::generate_quality_report,
        commands::analysis::perform_sensitivity_analysis,
        
        // Export Commands
        commands::export::export_to_csv,
        commands::export::export_to_excel,
        commands::export::export_to_netcdf,
        commands::export::export_to_hdf5,
        commands::export::generate_latex_report,
        commands::export::generate_publication_package,
        
        // Visualization Commands
        commands::visualization::generate_missing_pattern_plot,
        commands::visualization::generate_time_series_plot,
        commands::visualization::generate_correlation_matrix,
        commands::visualization::generate_uncertainty_bands,
        commands::visualization::create_interactive_dashboard,
        
        // Settings Commands
        commands::settings::get_user_preferences,
        commands::settings::update_user_preferences,
        commands::settings::get_computation_settings,
        commands::settings::update_computation_settings,
        commands::settings::reset_to_defaults,
        
        // System Commands
        commands::system::get_system_info,
        commands::system::check_python_runtime,
        commands::system::get_memory_usage,
        commands::system::clear_cache,
        commands::system::run_diagnostics,
        
        // Project Management
        commands::project::create_project,
        commands::project::open_project,
        commands::project::save_project,
        commands::project::get_recent_projects,
        commands::project::archive_project,
        
        // Benchmark Commands
        commands::benchmark::get_benchmark_datasets,
        commands::benchmark::run_benchmark,
        commands::benchmark::get_benchmark_results,
        commands::benchmark::export_benchmark_results,
        commands::benchmark::generate_reproducibility_certificate,
        
        // Publication Commands
        commands::publication::save_report,
        commands::publication::load_report,
        commands::publication::list_reports,
        commands::publication::render_latex,
        commands::publication::export_report,
        commands::publication::import_bibtex,
        commands::publication::format_citation,
        commands::publication::generate_bibliography,
        commands::publication::get_report_templates,
        
        // Help Commands (Offline)
        commands::help::search_help,
        commands::help::get_method_documentation,
        commands::help::list_tutorials,
        commands::help::get_tutorial,
        commands::help::get_offline_status,
        commands::help::get_quick_start_guide,
        commands::help::get_sample_datasets,
    ])
    .on_window_event(handle_window_event)
    .run(tauri::generate_context!())
    .map_err(|e| anyhow::anyhow!("Failed to run application: {}", e))?;
    }
    
    Ok(())
}

fn initialize_logging() {
    let file_appender = tracing_appender::rolling::daily("logs", "airimpute-pro.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "airimpute_pro=debug,tauri=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .json()
                .with_current_span(true)
                .with_span_list(true),
        )
        .init();
}

fn create_system_tray() -> SystemTray {
    let menu = SystemTrayMenu::new()
        .add_item(CustomMenuItem::new("show", "Show AirImpute Pro"))
        .add_native_item(SystemTrayMenuItem::Separator)
        .add_item(CustomMenuItem::new("quick_import", "Quick Import"))
        .add_item(CustomMenuItem::new("recent", "Recent Projects").disabled())
        .add_submenu(SystemTraySubmenu::new(
            "Processing",
            SystemTrayMenu::new()
                .add_item(CustomMenuItem::new("pause", "Pause All"))
                .add_item(CustomMenuItem::new("resume", "Resume All"))
                .add_item(CustomMenuItem::new("cancel", "Cancel All")),
        ))
        .add_native_item(SystemTrayMenuItem::Separator)
        .add_item(CustomMenuItem::new("preferences", "Preferences"))
        .add_item(CustomMenuItem::new("help", "Help"))
        .add_native_item(SystemTrayMenuItem::Separator)
        .add_item(CustomMenuItem::new("quit", "Quit"));
    
    SystemTray::new().with_menu(menu)
}

fn handle_system_tray_event(app: &tauri::AppHandle, event: SystemTrayEvent) {
    match event {
        SystemTrayEvent::LeftClick { .. } => {
            if let Some(window) = app.get_window("main") {
                if let Err(e) = window.show() {
                    error!("Failed to show window: {}", e);
                }
                if let Err(e) = window.set_focus() {
                    error!("Failed to focus window: {}", e);
                }
            }
        }
        SystemTrayEvent::MenuItemClick { id, .. } => match id.as_str() {
            "quit" => {
                info!("Quit requested from system tray");
                app.exit(0);
            }
            "show" => {
                if let Some(window) = app.get_window("main") {
                    if let Err(e) = window.show() {
                        error!("Failed to show window: {}", e);
                    }
                    if let Err(e) = window.unminimize() {
                        error!("Failed to unminimize window: {}", e);
                    }
                    if let Err(e) = window.set_focus() {
                        error!("Failed to focus window: {}", e);
                    }
                }
            }
            "quick_import" => {
                app.emit_all("quick-import", ()).unwrap();
            }
            "preferences" => {
                app.emit_all("open-preferences", ()).unwrap();
            }
            "help" => {
                app.emit_all("open-help", ()).unwrap();
            }
            _ => {}
        },
        _ => {}
    }
}

fn initialize_app_state(app: &tauri::App) -> anyhow::Result<Arc<AppState>> {
    let app_handle = app.handle();
    let state = AppStateManager::initialize(app_handle)?;
    
    info!("Application state initialized successfully");
    Ok(state)
}

fn setup_main_window(app: &tauri::App) -> anyhow::Result<()> {
    let window = app
        .get_window("main")
        .ok_or_else(|| anyhow::anyhow!("Main window not found"))?;
    
    // Platform-specific window setup
    #[cfg(target_os = "macos")]
    {
        use tauri::TitleBarStyle;
        window.set_title_bar_style(TitleBarStyle::Transparent)?;
    }
    
    // Set minimum window size
    window.set_min_size(Some(tauri::LogicalSize::new(1024.0, 768.0)))?;
    
    // Center window on screen
    if let Ok(Some(monitor)) = window.current_monitor() {
        let screen_size = monitor.size();
        let window_size = window.outer_size()?;
        
        let x = (screen_size.width - window_size.width) / 2;
        let y = (screen_size.height - window_size.height) / 2;
        
        window.set_position(tauri::LogicalPosition::new(x as f64, y as f64))?;
    }
    
    info!("Main window configured successfully");
    Ok(())
}

fn register_global_shortcuts(_app: &tauri::App) -> anyhow::Result<()> {
    // Global shortcuts removed for security reasons
    // All shortcuts should be handled through the frontend
    info!("Global shortcuts disabled for security");
    Ok(())
}

fn initialize_background_services(app: &tauri::App) -> anyhow::Result<()> {
    let app_handle = app.handle();
    let state = app.state::<Arc<AppState>>();
    
    // Start auto-save service
    services::auto_save::start_auto_save_service(app_handle.clone(), state.inner().clone());
    
    // Start memory monitor
    services::memory_monitor::start_memory_monitor(app_handle.clone());
    
    // Start update checker
    services::update_checker::start_update_checker(app_handle.clone());
    
    info!("Background services initialized");
    Ok(())
}

fn perform_startup_checks(app: &tauri::App) -> anyhow::Result<()> {
    let state = app.state::<Arc<AppState>>();
    
    // Check Python runtime
    match state.python_runtime.check_health() {
        Ok(_) => info!("Python runtime healthy"),
        Err(e) => {
            error!("Python runtime check failed: {}", e);
            // Show user-friendly error dialog
            tauri::api::dialog::message(
                Some(&app.get_window("main").unwrap()),
                "Python Runtime Error",
                format!("Failed to initialize Python runtime: {}. Some features may be unavailable.", e),
            );
        }
    }
    
    // Check available memory
    let sys_info = sysinfo::System::new_all();
    let available_memory = sys_info.available_memory();
    let total_memory = sys_info.total_memory();
    
    info!(
        "System memory: {:.2} GB available of {:.2} GB total",
        available_memory as f64 / 1_073_741_824.0,
        total_memory as f64 / 1_073_741_824.0
    );
    
    if available_memory < 2_147_483_648 {
        // Less than 2GB available
        warn!("Low memory warning: Less than 2GB available");
        app.emit_all("low-memory-warning", available_memory).unwrap();
    }
    
    Ok(())
}

fn handle_window_event(event: GlobalWindowEvent) {
    match event.event() {
        WindowEvent::CloseRequested { api, .. } => {
            // Prevent close, minimize to tray instead
            #[cfg(not(target_os = "macos"))]
            {
                api.prevent_close();
                if let Err(e) = event.window().hide() {
                    error!("Failed to hide window: {}", e);
                }
            }
        }
        WindowEvent::Resized(size) => {
            info!("Window resized to {:?}", size);
        }
        WindowEvent::FileDrop(_event) => {
            info!("Files dropped");
            // TODO: Update for new Tauri API
            // event.window().emit("files-dropped", event.paths()).unwrap();
        }
        _ => {}
    }
}