fn main() {
    // Standard Tauri build
    tauri_build::build();
    
    // Configure PyO3 only if python-support feature is enabled
    #[cfg(feature = "python-support")]
    {
        // Get the manifest directory for absolute paths
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let manifest_path = std::path::Path::new(&manifest_dir);
        
        // Windows-specific Python configuration
        #[cfg(target_os = "windows")]
        {
            // Check if we have an embedded Python distribution
            let python_dir = manifest_path.join("python");
            let python_exe = python_dir.join("python.exe");
            
            if python_exe.exists() {
                // Set environment variables for PyO3
                println!("cargo:rustc-env=PYO3_PYTHON={}", python_exe.display());
                println!("cargo:rustc-env=PYO3_NO_PYTHON=1"); // Don't use system Python
                println!("cargo:warning=Using embedded Python from {}", python_dir.display());
                
                // Also set PYTHONHOME for runtime
                println!("cargo:rustc-env=PYTHONHOME={}", python_dir.display());
            } else {
                println!("cargo:warning=No embedded Python found at {}", python_dir.display());
                println!("cargo:warning=Python features will be disabled");
                // Don't configure PyO3 if Python is missing
                return;
            }
        }
        
        // Configure PyO3 build
        pyo3_build_config::use_pyo3_cfgs();
    }
}