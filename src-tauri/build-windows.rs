// Windows-specific build configuration
// Handles Python embedding and security settings

fn main() {
    // Standard Tauri build
    tauri_build::build();
    
    // Windows-specific Python configuration
    #[cfg(target_os = "windows")]
    {
        // Check if we have an embedded Python distribution
        let python_dir = std::path::Path::new("python");
        if python_dir.exists() && python_dir.join("python.exe").exists() {
            println!("cargo:rustc-env=PYO3_PYTHON=./python/python.exe");
            println!("cargo:warning=Using embedded Python from ./python/");
        } else {
            println!("cargo:warning=No embedded Python found at ./python/");
            println!("cargo:warning=The application will require system Python at runtime");
            println!("cargo:warning=Download python-build-standalone and extract to src-tauri/python/");
        }
    }
    
    // Configure PyO3 for all platforms
    pyo3_build_config::use_pyo3_cfgs();
}