use std::sync::Arc;
use tauri::AppHandle;
use crate::state::AppState;

pub fn start_auto_save_service(app_handle: AppHandle, state: Arc<AppState>) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(300)).await;
            // Auto-save implementation would go here
        }
    });
}