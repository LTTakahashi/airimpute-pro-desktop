use std::ffi::OsString;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use tracing::{info, warn, error};

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
extern "system" {
    fn SetDefaultDllDirectories(flags: u32) -> i32;
    fn AddDllDirectory(path: *const u16) -> *mut std::ffi::c_void;
    fn SetDllDirectoryW(path: *const u16) -> i32;
}

#[cfg(target_os = "windows")]
const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x00000800;

/// Initialize secure DLL loading with Python support
#[cfg(target_os = "windows")]
pub fn initialize_dll_security(python_dir: Option<&Path>) -> anyhow::Result<()> {
    unsafe {
        // First, set default DLL directories to prevent DLL hijacking
        // This restricts DLL loading to system32 only
        let result = SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32);
        if result == 0 {
            return Err(anyhow::anyhow!("Failed to set default DLL directories"));
        }
        info!("Set default DLL directories for security");

        // If Python directory is provided, add it to the secure search path
        if let Some(python_path) = python_dir {
            if python_path.exists() {
                // Convert path to wide string for Windows API
                let wide_path = path_to_wide_string(python_path);
                
                // Add the Python directory to the DLL search path
                let handle = AddDllDirectory(wide_path.as_ptr());
                if handle.is_null() {
                    error!("Failed to add Python directory to DLL search path: {:?}", python_path);
                    
                    // Fallback: Try SetDllDirectory (less secure but might work)
                    warn!("Falling back to SetDllDirectory (less secure)");
                    let result = SetDllDirectoryW(wide_path.as_ptr());
                    if result == 0 {
                        return Err(anyhow::anyhow!(
                            "Failed to add Python directory to DLL search path: {:?}", 
                            python_path
                        ));
                    }
                }
                
                info!("Added Python directory to DLL search path: {:?}", python_path);
                
                // Also add the DLLs subdirectory if it exists
                let dlls_path = python_path.join("DLLs");
                if dlls_path.exists() {
                    let wide_dlls_path = path_to_wide_string(&dlls_path);
                    let dlls_handle = AddDllDirectory(wide_dlls_path.as_ptr());
                    if !dlls_handle.is_null() {
                        info!("Added Python DLLs directory to search path: {:?}", dlls_path);
                    }
                }
            } else {
                warn!("Python directory does not exist: {:?}", python_path);
            }
        }
    }
    
    Ok(())
}

/// Initialize DLL security without Python (non-Windows platforms)
#[cfg(not(target_os = "windows"))]
pub fn initialize_dll_security(_python_dir: Option<&Path>) -> anyhow::Result<()> {
    // No-op on non-Windows platforms
    Ok(())
}

/// Convert a Path to a null-terminated wide string for Windows API
#[cfg(target_os = "windows")]
fn path_to_wide_string(path: &Path) -> Vec<u16> {
    let os_str = path.as_os_str();
    let mut wide: Vec<u16> = os_str.encode_wide().collect();
    wide.push(0); // Null terminator
    wide
}

/// Get the Python directory based on the build configuration
pub fn get_python_directory(app_handle: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    // In development, use the python directory next to src-tauri
    #[cfg(debug_assertions)]
    {
        let dev_python_dir = app_handle
            .path_resolver()
            .app_dir()
            .ok()?
            .parent()?
            .parent()?
            .join("python-dist");
        
        if dev_python_dir.exists() {
            info!("Using development Python directory: {:?}", dev_python_dir);
            return Some(dev_python_dir);
        }
    }
    
    // In production, use the bundled Python from resources
    if let Ok(resource_dir) = app_handle.path_resolver().resource_dir() {
        // Try python-dist first (matches the resource configuration)
        let python_dist_dir = resource_dir.join("python-dist");
        if python_dist_dir.exists() {
            info!("Using bundled Python directory: {:?}", python_dist_dir);
            return Some(python_dist_dir);
        }
        
        // Fallback to python directory
        let python_dir = resource_dir.join("python");
        if python_dir.exists() {
            info!("Using bundled Python directory: {:?}", python_dir);
            return Some(python_dir);
        }
    }
    
    // Final fallback: Check next to the executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let python_dir = exe_dir.join("python");
            if python_dir.exists() {
                info!("Using Python directory next to executable: {:?}", python_dir);
                return Some(python_dir);
            }
            
            let python_dist_dir = exe_dir.join("python-dist");
            if python_dist_dir.exists() {
                info!("Using Python directory next to executable: {:?}", python_dist_dir);
                return Some(python_dist_dir);
            }
        }
    }
    
    warn!("Could not locate Python directory");
    None
}