use std::path::{Path, PathBuf};
use anyhow::{Result, Context};
use tracing::{info, warn};

/// Initialize Python runtime with proper path resolution for production
pub fn initialize_python_runtime(app_handle: &tauri::AppHandle) -> Result<()> {
    info!("Initializing Python runtime for production");
    
    // Get the executable directory
    let exe_path = std::env::current_exe()
        .context("Failed to get executable path")?;
    let exe_dir = exe_path.parent()
        .context("Failed to get executable directory")?;
    
    // Determine Python location based on build type
    let python_dir = if cfg!(debug_assertions) {
        // Development: Python is in python-dist at project root
        exe_dir.parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .map(|p| p.join("python-dist"))
            .unwrap_or_else(|| {
                // Fallback to old location
                exe_dir.join("python")
            })
    } else {
        // Production: Python is bundled as a resource
        app_handle.path_resolver()
            .resource_dir()
            .map(|p| p.join("python"))
            .unwrap_or_else(|| {
                // Fallback to checking next to executable
                exe_dir.join("python")
            })
    };
    
    info!("Looking for Python in: {:?}", python_dir);
    
    // Verify Python installation
    let python_exe = if cfg!(target_os = "windows") {
        python_dir.join("python.exe")
    } else {
        python_dir.join("bin").join("python3")
    };
    
    if !python_exe.exists() {
        return Err(anyhow::anyhow!(
            "Python executable not found at: {:?}", python_exe
        ));
    }
    
    // Verify Python DLL on Windows with version flexibility
    #[cfg(target_os = "windows")]
    {
        // Try to detect Python version from available DLLs
        let possible_versions = vec!["311", "310", "39", "38"];
        let mut found_dll = None;
        let mut found_version = None;
        
        for version in &possible_versions {
            let dll_name = format!("python{}.dll", version);
            let python_dll = python_dir.join(&dll_name);
            
            if python_dll.exists() {
                info!("Found {} at: {:?}", dll_name, python_dll);
                found_dll = Some(python_dll);
                found_version = Some(version.to_string());
                break;
            }
            
            // Also check in DLLs subdirectory
            let dll_subdir = python_dir.join("DLLs").join(&dll_name);
            if dll_subdir.exists() {
                info!("Found {} in DLLs directory", dll_name);
                // Copy to root for easier access
                if let Err(e) = std::fs::copy(&dll_subdir, &python_dll) {
                    warn!("Failed to copy {} to root: {}", dll_name, e);
                }
                found_dll = Some(python_dll);
                found_version = Some(version.to_string());
                break;
            }
            
            // Check executable directory
            let exe_dir_dll = exe_dir.join(&dll_name);
            if exe_dir_dll.exists() {
                info!("Found {} in executable directory", dll_name);
                found_dll = Some(exe_dir_dll);
                found_version = Some(version.to_string());
                break;
            }
        }
        
        if found_dll.is_none() {
            return Err(anyhow::anyhow!(
                "No Python DLL found. Searched for versions: {:?}", 
                possible_versions
            ));
        }
        
        // Set environment variable for detected Python version
        if let Some(version) = found_version {
            std::env::set_var("PYTHON_DLL_VERSION", version);
        }
    }
    
    // Set environment variables for Python
    std::env::set_var("PYTHONHOME", &python_dir);
    
    // Set PYTHONPATH with multiple directories
    let site_packages = python_dir.join("Lib").join("site-packages");
    let pythonpath = if cfg!(target_os = "windows") {
        format!("{};{}", python_dir.display(), site_packages.display())
    } else {
        format!("{}:{}", python_dir.display(), site_packages.display())
    };
    std::env::set_var("PYTHONPATH", pythonpath);
    
    // Add Python to PATH with all necessary directories
    let path = std::env::var("PATH").unwrap_or_default();
    let new_path = if cfg!(target_os = "windows") {
        let scripts = python_dir.join("Scripts");
        let dlls = python_dir.join("DLLs");
        format!("{};{};{};{}", 
            python_dir.display(), 
            scripts.display(),
            dlls.display(),
            path
        )
    } else {
        format!("{}:{}", python_dir.join("bin").display(), path)
    };
    std::env::set_var("PATH", new_path);
    
    // Set additional Python-specific environment variables
    std::env::set_var("PYTHONDONTWRITEBYTECODE", "1"); // Don't create .pyc files
    std::env::set_var("PYTHONIOENCODING", "utf-8"); // Force UTF-8 encoding
    
    info!("Python runtime initialized successfully");
    info!("PYTHONHOME: {:?}", python_dir);
    
    Ok(())
}

/// Get the Python home directory for the current environment
pub fn get_python_home(app_handle: &tauri::AppHandle) -> Result<PathBuf> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path.parent()
        .context("Failed to get executable directory")?;
    
    let python_dir = if cfg!(debug_assertions) {
        exe_dir.join("python")
    } else {
        app_handle.path_resolver()
            .resource_dir()
            .map(|p| p.join("python"))
            .unwrap_or_else(|| exe_dir.join("python"))
    };
    
    Ok(python_dir)
}