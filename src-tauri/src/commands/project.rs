use tauri::command;
use std::sync::Arc;
use tauri::State;
use serde::Serialize;
use tracing::info;

use crate::state::AppState;
use crate::core::project::Project;

#[derive(Debug, Clone, Serialize)]
pub struct ProjectInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub created_at: String,
    pub modified_at: String,
}

#[command]
pub async fn create_project(
    state: State<'_, Arc<AppState>>,
    name: String,
    description: String,
) -> Result<ProjectInfo, String> {
    let project = Project::new(name, description);
    let info = ProjectInfo {
        id: project.id.to_string(),
        name: project.name.clone(),
        description: project.description.clone(),
        created_at: project.created_at.to_rfc3339(),
        modified_at: project.modified_at.to_rfc3339(),
    };
    
    state.projects.insert(project.id, project);
    Ok(info)
}

#[command]
pub async fn open_project(
    state: State<'_, Arc<AppState>>,
    project_path: String,
) -> Result<ProjectInfo, String> {
    use std::fs::File;
    use std::io::BufReader;
    use std::path::PathBuf;
    
    info!("Opening project from: {}", project_path);
    
    let path = PathBuf::from(&project_path);
    
    // Check if file exists
    if !path.exists() {
        return Err(format!("Project file not found: {}", project_path));
    }
    
    // Read project file
    let file = File::open(&path)
        .map_err(|e| format!("Failed to open project file: {}", e))?;
    
    let reader = BufReader::new(file);
    
    // Deserialize project
    let mut project: Project = serde_json::from_reader(reader)
        .map_err(|e| format!("Failed to parse project file: {}", e))?;
    
    // Update modified time
    project.modified_at = chrono::Utc::now();
    
    // Create project info
    let info = ProjectInfo {
        id: project.id.to_string(),
        name: project.name.clone(),
        description: project.description.clone(),
        created_at: project.created_at.to_rfc3339(),
        modified_at: project.modified_at.to_rfc3339(),
    };
    
    // Store in state
    state.projects.insert(project.id, project);
    
    // Update recent projects
    let mut recent = state.recent_projects.write();
    recent.retain(|p| p != &project_path);
    recent.insert(0, project_path);
    if recent.len() > 10 {
        recent.truncate(10);
    }
    
    info!("Successfully opened project: {}", info.name);
    Ok(info)
}

#[command]
pub async fn save_project(
    state: State<'_, Arc<AppState>>,
    project_id: String,
) -> Result<String, String> {
    use std::fs::{self, File};
    use std::io::BufWriter;
    use std::path::PathBuf;
    
    info!("Saving project: {}", project_id);
    
    // Parse project ID
    let id = uuid::Uuid::parse_str(&project_id)
        .map_err(|e| format!("Invalid project ID: {}", e))?;
    
    // Get project
    let mut project = state.projects.get_mut(&id)
        .ok_or_else(|| "Project not found".to_string())?;
    
    // Update modified time
    project.modified_at = chrono::Utc::now();
    
    // Create save directory
    let save_dir = dirs::document_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("AirImpute Projects");
    
    fs::create_dir_all(&save_dir)
        .map_err(|e| format!("Failed to create save directory: {}", e))?;
    
    // Generate filename
    let filename = format!("{}.aiproject", project.name.replace(" ", "_"));
    let save_path = save_dir.join(&filename);
    
    // Create backup if file exists
    if save_path.exists() {
        let backup_path = save_path.with_extension("aiproject.bak");
        fs::copy(&save_path, &backup_path)
            .map_err(|e| format!("Failed to create backup: {}", e))?;
    }
    
    // Serialize project
    let file = File::create(&save_path)
        .map_err(|e| format!("Failed to create project file: {}", e))?;
    
    let writer = BufWriter::new(file);
    
    serde_json::to_writer_pretty(writer, &*project)
        .map_err(|e| format!("Failed to write project file: {}", e))?;
    
    // Update recent projects
    let path_str = save_path.to_string_lossy().to_string();
    let mut recent = state.recent_projects.write();
    recent.retain(|p| p != &path_str);
    recent.insert(0, path_str.clone());
    if recent.len() > 10 {
        recent.truncate(10);
    }
    
    info!("Project saved to: {}", path_str);
    Ok(path_str)
}

#[command]
pub async fn get_recent_projects(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<crate::db::models::RecentProject>, String> {
    let recent_paths = state.recent_projects.read().clone();
    let mut recent_projects = Vec::new();
    
    for path in recent_paths {
        let path_buf = std::path::PathBuf::from(&path);
        let exists = path_buf.exists();
        
        let name = if exists {
            // Try to read project name
            match std::fs::File::open(&path_buf) {
                Ok(file) => {
                    let reader = std::io::BufReader::new(file);
                    match serde_json::from_reader::<_, serde_json::Value>(reader) {
                        Ok(json) => json["name"].as_str()
                            .unwrap_or("Unknown")
                            .to_string(),
                        Err(_) => path_buf.file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Unknown")
                            .to_string(),
                    }
                }
                Err(_) => "Unknown".to_string(),
            }
        } else {
            path_buf.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Unknown")
                .to_string()
        };
        
        recent_projects.push(crate::db::models::RecentProject {
            path: path.clone(),
            name,
            last_opened: "Recently".to_string(), // Would track actual time
            exists,
        });
    }
    
    Ok(recent_projects)
}

#[command]
pub async fn archive_project(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    output_path: String,
) -> Result<String, String> {
    use std::fs::{self, File};
    use std::path::PathBuf;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use tar::Builder;
    
    info!("Archiving project: {} to {}", project_id, output_path);
    
    // Parse project ID
    let id = uuid::Uuid::parse_str(&project_id)
        .map_err(|e| format!("Invalid project ID: {}", e))?;
    
    // Get project
    let project = state.projects.get(&id)
        .ok_or_else(|| "Project not found".to_string())?;
    
    // Create temporary directory for archive contents
    let temp_dir = std::env::temp_dir()
        .join(format!("airimpute_archive_{}", project_id));
    
    fs::create_dir_all(&temp_dir)
        .map_err(|e| format!("Failed to create temp directory: {}", e))?;
    
    // Save project file
    let project_file = temp_dir.join("project.json");
    let file = File::create(&project_file)
        .map_err(|e| format!("Failed to create project file: {}", e))?;
    
    serde_json::to_writer_pretty(file, &*project)
        .map_err(|e| format!("Failed to write project data: {}", e))?;
    
    // Create data directory
    let data_dir = temp_dir.join("data");
    fs::create_dir_all(&data_dir)
        .map_err(|e| format!("Failed to create data directory: {}", e))?;
    
    // Collect datasets first to avoid holding locks across await points
    let datasets_to_export: Vec<_> = state.datasets.iter()
        .map(|entry| (*entry.key(), entry.value().clone()))
        .collect();
    
    // Export all associated datasets
    for (dataset_id, dataset) in datasets_to_export {
        // Check if dataset belongs to this project
        // (In real implementation, would have proper project-dataset association)
        
        // Save as CSV
        let csv_path = data_dir.join(format!("{}.csv", dataset.name));
        crate::commands::data::save_csv(&dataset, &csv_path).await
            .map_err(|e| format!("Failed to save dataset: {}", e))?;
        
        // Save metadata
        let meta_path = data_dir.join(format!("{}.meta.json", dataset.name));
        let meta_file = File::create(&meta_path)
            .map_err(|e| format!("Failed to create metadata file: {}", e))?;
        
        let metadata = serde_json::json!({
            "id": dataset_id.to_string(),
            "name": dataset.name,
            "rows": dataset.rows(),
            "columns": dataset.columns(),
            "variables": dataset.variables,
            "created_at": dataset.created_at,
            "modified_at": dataset.modified_at,
        });
        
        serde_json::to_writer_pretty(meta_file, &metadata)
            .map_err(|e| format!("Failed to write metadata: {}", e))?;
    }
    
    // Create results directory and save imputation results
    let results_dir = temp_dir.join("results");
    fs::create_dir_all(&results_dir)
        .map_err(|e| format!("Failed to create results directory: {}", e))?;
    
    // Collect job IDs first
    let job_ids: Vec<_> = state.imputation_jobs.iter()
        .map(|entry| *entry.key())
        .collect();
    
    for job_id in job_ids {
        if let Some(job_mutex) = state.imputation_jobs.get(&job_id) {
            // Lock the mutex to get the job data
            let job_data = job_mutex.lock().await;
            
            let job_file = results_dir.join(format!("job_{}.json", job_id));
            let file = File::create(&job_file)
                .map_err(|e| format!("Failed to create job file: {}", e))?;
            
            serde_json::to_writer_pretty(file, &*job_data)
                .map_err(|e| format!("Failed to write job data: {}", e))?;
        }
    }
    
    // Create archive info
    let archive_info = serde_json::json!({
        "project_id": project_id,
        "project_name": project.name,
        "archived_at": chrono::Utc::now(),
        "archive_version": "1.0",
        "airimpute_version": env!("CARGO_PKG_VERSION"),
    });
    
    let info_file = temp_dir.join("archive_info.json");
    let file = File::create(&info_file)
        .map_err(|e| format!("Failed to create info file: {}", e))?;
    
    serde_json::to_writer_pretty(file, &archive_info)
        .map_err(|e| format!("Failed to write archive info: {}", e))?;
    
    // Create tar.gz archive
    let output_path_buf = PathBuf::from(&output_path);
    let archive_path = if output_path_buf.extension() == Some(std::ffi::OsStr::new("gz")) {
        output_path_buf
    } else {
        output_path_buf.with_extension("tar.gz")
    };
    
    let tar_gz = File::create(&archive_path)
        .map_err(|e| format!("Failed to create archive file: {}", e))?;
    
    let enc = GzEncoder::new(tar_gz, Compression::default());
    let mut tar = Builder::new(enc);
    
    // Add all files to archive
    tar.append_dir_all(
        format!("airimpute_project_{}", project.name.replace(" ", "_")),
        &temp_dir
    ).map_err(|e| format!("Failed to create archive: {}", e))?;
    
    tar.finish()
        .map_err(|e| format!("Failed to finish archive: {}", e))?;
    
    // Clean up temp directory
    fs::remove_dir_all(&temp_dir)
        .map_err(|e| format!("Failed to clean up temp directory: {}", e))?;
    
    info!("Project archived to: {}", archive_path.display());
    Ok(archive_path.to_string_lossy().to_string())
}