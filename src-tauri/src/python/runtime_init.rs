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
        // Development: Python is in src-tauri/python
        exe_dir.join("python")
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
    
    // Verify python311.dll on Windows
    #[cfg(target_os = "windows")]
    {
        let python_dll = python_dir.join("python311.dll");
        if !python_dll.exists() {
            // Try to find it in the system
            warn!("python311.dll not found at {:?}, checking system paths", python_dll);
            
            // Check if it's in the same directory as the executable
            let exe_dir_dll = exe_dir.join("python311.dll");
            if exe_dir_dll.exists() {
                info!("Found python311.dll in executable directory");
            } else {
                return Err(anyhow::anyhow!(
                    "python311.dll not found. Expected at: {:?} or {:?}", 
                    python_dll, exe_dir_dll
                ));
            }
        } else {
            info!("Found python311.dll at: {:?}", python_dll);
        }
    }
    
    // Set environment variables for Python
    std::env::set_var("PYTHONHOME", &python_dir);
    std::env::set_var("PYTHONPATH", python_dir.join("Lib").join("site-packages"));
    
    // Add Python to PATH
    let path = std::env::var("PATH").unwrap_or_default();
    let new_path = if cfg!(target_os = "windows") {
        format!("{};{}", python_dir.display(), path)
    } else {
        format!("{}:{}", python_dir.join("bin").display(), path)
    };
    std::env::set_var("PATH", new_path);
    
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