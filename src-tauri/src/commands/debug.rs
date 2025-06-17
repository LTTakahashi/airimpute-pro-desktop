// Debug commands for testing Tauri and PyO3 bridges
use tauri::command;
use pyo3::prelude::*;

#[command]
pub fn ping() -> &'static str {
    "pong"
}

#[command]
pub fn check_python_bridge() -> Result<String, String> {
    // Ensure Python interpreter is ready for multithreading
    pyo3::prepare_freethreaded_python();
    
    Python::with_gil(|py| {
        println!("Attempting to check Python version...");
        
        // Simple Python version check without importing external modules
        let sys = py.import("sys").map_err(|e| {
            format!("Failed to import sys module: {}", e)
        })?;
        
        let version = sys.getattr("version").map_err(|e| {
            format!("Failed to get Python version: {}", e)
        })?;
        
        let version_str: String = version.extract().map_err(|e| {
            format!("Failed to extract version string: {}", e)
        })?;
        
        println!("Python version retrieved: {}", version_str);
        
        Ok(format!("Hello from Python: {}", version_str))
    })
}

#[command]
pub fn test_numpy() -> Result<String, String> {
    pyo3::prepare_freethreaded_python();
    
    Python::with_gil(|py| {
        println!("Testing numpy import...");
        
        let numpy = py.import("numpy").map_err(|e| {
            format!("Failed to import numpy: {}", e)
        })?;
        
        let version = numpy.getattr("__version__").map_err(|e| {
            format!("Failed to get numpy version: {}", e)
        })?;
        
        let version_str: String = version.extract().map_err(|e| {
            format!("Failed to extract numpy version: {}", e)
        })?;
        
        println!("Numpy version: {}", version_str);
        
        Ok(format!("Numpy version: {}", version_str))
    })
}