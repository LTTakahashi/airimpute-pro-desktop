use tauri::AppHandle;
use tracing::info;

/// Offline-only update checker (disabled for complete offline operation)
pub fn start_update_checker(_app_handle: AppHandle) {
    info!("Update checker disabled - app runs completely offline");
    // No network calls or update checks - app is fully offline
}