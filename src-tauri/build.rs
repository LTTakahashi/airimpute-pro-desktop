fn main() {
    // Tell cargo to invalidate the built crate whenever the build script itself changes
    println!("cargo:rerun-if-changed=build.rs");
    
    // Watch only the Python distribution directory itself (not its contents)
    // This prevents cargo from watching thousands of individual files
    println!("cargo:rerun-if-changed=../python-dist");
    
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
            // Python distribution is now at project root level
            let python_dir = manifest_path.join("../python-dist");
            let python_exe = python_dir.join("python.exe");
            
            if python_exe.exists() {
                // Handle Python DLL compatibility issue
                let python311_dll = python_dir.join("python311.dll");
                let python310_dll = python_dir.join("python310.dll");
                
                // Create python310.dll from python311.dll if needed
                if python311_dll.exists() && !python310_dll.exists() {
                    println!("cargo:warning=Creating python310.dll from python311.dll for compatibility");
                    if let Err(e) = std::fs::copy(&python311_dll, &python310_dll) {
                        println!("cargo:warning=Failed to create python310.dll: {}", e);
                    }
                }
                
                // Set environment variables for PyO3
                println!("cargo:rustc-env=PYO3_PYTHON={}", python_exe.display());
                println!("cargo:rustc-env=PYO3_NO_PYTHON=1"); // Don't use system Python
                println!("cargo:warning=Using embedded Python from {}", python_dir.display());
                
                // Also set PYTHONHOME for runtime
                println!("cargo:rustc-env=PYTHONHOME={}", python_dir.display());
                
                // Copy DLLs to output directory
                if let Ok(out_dir) = std::env::var("OUT_DIR") {
                    let out_path = std::path::Path::new(&out_dir);
                    if let Some(target_dir) = out_path.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
                        for dll in &["python310.dll", "python311.dll", "python3.dll"] {
                            let src = python_dir.join(dll);
                            if src.exists() {
                                let dst = target_dir.join(dll);
                                if let Err(e) = std::fs::copy(&src, &dst) {
                                    println!("cargo:warning=Failed to copy {} to target: {}", dll, e);
                                }
                            }
                        }
                    }
                }
            } else {
                println!("cargo:warning=No embedded Python found at {}", python_dir.display());
                println!("cargo:warning=Python features will be disabled");
                // Don't configure PyO3 if Python is missing
                return;
            }
        }
        
        // Linux-specific WebKit and libsoup configuration
        #[cfg(target_os = "linux")]
        {
            // Configure Python for Linux - use system Python
            println!("cargo:warning=Configuring PyO3 for Linux with system Python");
            
            // Set pkg-config environment to use our wrapper script
            println!("cargo:rustc-env=PKG_CONFIG=./pkg-config-wrapper.sh");
            
            // Add library search paths for libsoup2
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
            println!("cargo:rustc-link-search=native=/usr/lib");
            
            // Force linking with libsoup-2.4 instead of libsoup3
            println!("cargo:rustc-link-lib=soup-2.4");
            
            // Set environment variables for WebKit compatibility
            println!("cargo:rustc-env=WEBKIT_DISABLE_COMPOSITING_MODE=1");
            println!("cargo:rustc-env=WEBKIT_DISABLE_SANDBOX=1");
            
            // Create webkit compatibility directory if needed
            let webkit_fix_dir = manifest_path.join("webkit-fix");
            if !webkit_fix_dir.exists() {
                if let Err(e) = std::fs::create_dir_all(&webkit_fix_dir) {
                    println!("cargo:warning=Failed to create webkit-fix directory: {}", e);
                }
            }
            
            // Create a marker file to indicate Linux-specific build
            let marker_file = webkit_fix_dir.join(".linux-build");
            if let Err(e) = std::fs::write(&marker_file, "libsoup2-build") {
                println!("cargo:warning=Failed to create Linux build marker: {}", e);
            }
            
            println!("cargo:warning=Configured Linux build for libsoup2 compatibility");
        }
        
        // Configure PyO3 build
        pyo3_build_config::use_pyo3_cfgs();
    }
}