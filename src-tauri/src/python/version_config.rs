use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use anyhow::{Result, Context};

#[derive(Debug, Serialize, Deserialize)]
pub struct PythonVersionConfig {
    pub version: String,
    pub major: String,
    pub minor: String,
    pub patch: String,
    pub dll_name: String,
    pub lib_name: String,
    pub executable: HashMap<String, String>,
    pub embeddable_url: String,
    pub choco_package: String,
    pub notes: String,
}

impl PythonVersionConfig {
    /// Load Python version configuration from the JSON file
    pub fn load() -> Result<Self> {
        let config_path = if cfg!(debug_assertions) {
            // In development, look for the config file relative to the project root
            Path::new("../python-version.json")
        } else {
            // In production, it should be bundled as a resource
            Path::new("python-version.json")
        };
        
        let contents = fs::read_to_string(config_path)
            .context("Failed to read python-version.json")?;
        
        let config: PythonVersionConfig = serde_json::from_str(&contents)
            .context("Failed to parse python-version.json")?;
        
        Ok(config)
    }
    
    /// Get the Python executable name for the current platform
    pub fn get_executable(&self) -> &str {
        let platform = if cfg!(target_os = "windows") {
            "windows"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else {
            "linux"
        };
        
        self.executable.get(platform)
            .map(|s| s.as_str())
            .unwrap_or("python3")
    }
    
    /// Get the expected DLL name for Windows
    #[cfg(target_os = "windows")]
    pub fn get_dll_name(&self) -> &str {
        &self.dll_name
    }
    
    /// Get the expected lib name for Windows
    #[cfg(target_os = "windows")]
    pub fn get_lib_name(&self) -> &str {
        &self.lib_name
    }
}

/// Get the default Python version configuration
pub fn get_default_config() -> PythonVersionConfig {
    PythonVersionConfig {
        version: "3.11.9".to_string(),
        major: "3".to_string(),
        minor: "11".to_string(),
        patch: "9".to_string(),
        dll_name: "python311.dll".to_string(),
        lib_name: "python311.lib".to_string(),
        executable: {
            let mut map = HashMap::new();
            map.insert("windows".to_string(), "python.exe".to_string());
            map.insert("linux".to_string(), "python3".to_string());
            map.insert("macos".to_string(), "python3".to_string());
            map
        },
        embeddable_url: "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip".to_string(),
        choco_package: "python311".to_string(),
        notes: "Default fallback configuration".to_string(),
    }
}

/// Load Python version configuration with fallback to defaults
pub fn load_python_config() -> PythonVersionConfig {
    PythonVersionConfig::load()
        .unwrap_or_else(|_| get_default_config())
}