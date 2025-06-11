use tauri::command;
use std::sync::Arc;
use tauri::State;

use crate::state::{AppState, UserPreferences};

#[command]
pub async fn get_user_preferences(
    state: State<'_, Arc<AppState>>,
) -> Result<UserPreferences, String> {
    Ok(state.preferences.read().clone())
}

#[command]
pub async fn update_user_preferences(
    state: State<'_, Arc<AppState>>,
    preferences: UserPreferences,
) -> Result<(), String> {
    *state.preferences.write() = preferences;
    Ok(())
}

#[command]
pub async fn get_computation_settings(
    state: State<'_, Arc<AppState>>,
) -> Result<serde_json::Value, String> {
    let prefs = state.preferences.read();
    serde_json::to_value(&prefs.computation_settings)
        .map_err(|e| format!("Failed to serialize computation settings: {}", e))
}

#[command]
pub async fn update_computation_settings(
    state: State<'_, Arc<AppState>>,
    settings: serde_json::Value,
) -> Result<(), String> {
    // Implementation would update computation settings
    Ok(())
}

#[command]
pub async fn reset_to_defaults(
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    *state.preferences.write() = UserPreferences::default();
    Ok(())
}