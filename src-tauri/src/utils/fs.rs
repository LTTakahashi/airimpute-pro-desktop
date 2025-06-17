use std::path::{Path, PathBuf};
use anyhow::{Result, Context};

/// Sanitize and validate file paths for security
pub fn sanitize_path(path: &str) -> Result<PathBuf> {
    let path = PathBuf::from(path);
    
    // Check for path traversal attempts
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                return Err(anyhow::anyhow!("Path traversal detected"));
            }
            std::path::Component::Normal(os_str) => {
                if let Some(s) = os_str.to_str() {
                    if s.starts_with('.') && s != "." {
                        return Err(anyhow::anyhow!("Hidden file access attempted"));
                    }
                }
            }
            _ => {}
        }
    }
    
    // Resolve to absolute path
    let absolute_path = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .context("Failed to get current directory")?
            .join(path)
    };
    
    // Canonicalize path (resolves symlinks)
    let canonical_path = absolute_path
        .canonicalize()
        .or_else(|_| {
            // If file doesn't exist yet, canonicalize parent
            absolute_path
                .parent()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid path"))?
                .canonicalize()
                .map(|parent| parent.join(absolute_path.file_name().unwrap()))
        })
        .context("Failed to canonicalize path")?;
    
    Ok(canonical_path)
}

/// Ensure directory exists, creating it if necessary
#[allow(dead_code)]
pub fn ensure_dir_exists(path: &Path) -> Result<()> {
    if !path.exists() {
        std::fs::create_dir_all(path)
            .with_context(|| format!("Failed to create directory: {:?}", path))?;
    } else if !path.is_dir() {
        return Err(anyhow::anyhow!("Path exists but is not a directory: {:?}", path));
    }
    Ok(())
}

/// Get file size in bytes
pub fn get_file_size(path: &Path) -> Result<u64> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("Failed to get metadata for: {:?}", path))?;
    Ok(metadata.len())
}

/// Check if path is within allowed directory
#[allow(dead_code)]
pub fn is_path_allowed(path: &Path, allowed_dirs: &[PathBuf]) -> bool {
    for allowed_dir in allowed_dirs {
        if path.starts_with(allowed_dir) {
            return true;
        }
    }
    false
}

/// Get file extension
pub fn get_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_sanitize_path_valid() {
        let result = sanitize_path("test.csv");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_sanitize_path_traversal() {
        let result = sanitize_path("../../../etc/passwd");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_ensure_dir_exists() {
        let temp_dir = tempdir().unwrap();
        let new_dir = temp_dir.path().join("test_dir");
        
        assert!(!new_dir.exists());
        ensure_dir_exists(&new_dir).unwrap();
        assert!(new_dir.exists());
        assert!(new_dir.is_dir());
    }
    
    #[test]
    fn test_get_extension() {
        assert_eq!(get_extension(Path::new("file.csv")), Some("csv".to_string()));
        assert_eq!(get_extension(Path::new("file.CSV")), Some("csv".to_string()));
        assert_eq!(get_extension(Path::new("file")), None);
    }
}