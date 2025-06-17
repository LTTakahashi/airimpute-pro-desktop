use std::path::{Path, PathBuf};
use anyhow::{Context, Result, bail};
use tauri::api::path::{home_dir, document_dir, download_dir, data_dir};
use once_cell::sync::Lazy;
use regex::Regex;
use sha2::{Sha256, Digest};
use std::collections::HashSet;
use tracing::{warn, error};

/// Security module for AirImpute Pro Desktop
/// Implements comprehensive security measures for file operations, path validation,
/// and input sanitization to prevent common vulnerabilities.

/// Allowed base directories for file operations
static ALLOWED_DIRECTORIES: Lazy<Vec<PathBuf>> = Lazy::new(|| {
    let mut dirs = Vec::new();
    
    // Add safe directories
    if let Some(home) = home_dir() {
        dirs.push(home.join("AirImpute"));
        dirs.push(home.join("Documents").join("AirImpute"));
    }
    
    if let Some(docs) = document_dir() {
        dirs.push(docs.join("AirImpute"));
    }
    
    if let Some(downloads) = download_dir() {
        dirs.push(downloads);
    }
    
    if let Some(data) = data_dir() {
        dirs.push(data.join("AirImpute"));
    }
    
    dirs
});

/// Allowed Python modules and functions for safe execution
static ALLOWED_PYTHON_OPERATIONS: Lazy<HashSet<(&'static str, &'static str)>> = Lazy::new(|| {
    let mut ops = HashSet::new();
    
    // Analysis operations
    ops.insert(("airimpute.analysis", "analyze_missing_patterns"));
    ops.insert(("airimpute.analysis", "analyze_temporal_patterns"));
    ops.insert(("airimpute.analysis", "analyze_spatial_correlations"));
    ops.insert(("airimpute.analysis", "compute_quality_metrics"));
    
    // Imputation operations
    ops.insert(("airimpute.imputation", "run_custom_imputation"));
    ops.insert(("airimpute.imputation", "run_mean_imputation"));
    ops.insert(("airimpute.imputation", "run_knn_imputation"));
    ops.insert(("airimpute.imputation", "run_mice_imputation"));
    ops.insert(("airimpute.imputation", "run_random_forest_imputation"));
    
    // V3 Arrow-based operations
    ops.insert(("airimpute.arrow_worker", "process_imputation"));
    ops.insert(("airimpute.arrow_worker", "validate_data"));
    ops.insert(("airimpute.dispatcher", "dispatch_imputation"));
    ops.insert(("airimpute.gnn_imputation", "run_gnn_imputation"));
    ops.insert(("airimpute.transformer_imputation", "run_transformer_imputation"));
    
    // Validation operations
    ops.insert(("airimpute.validation", "validate_imputation_results"));
    ops.insert(("airimpute.validation", "compute_rmse"));
    ops.insert(("airimpute.validation", "compute_mae"));
    
    // Benchmark operations
    ops.insert(("airimpute.benchmarking", "run_benchmark"));
    ops.insert(("airimpute.benchmarking", "compare_methods"));
    
    // IO operations (secured with path validation)
    ops.insert(("airimpute.io", "load_from_file"));
    ops.insert(("airimpute.io", "read_csv"));
    ops.insert(("airimpute.io", "read_excel"));
    ops.insert(("airimpute.io", "save_to_file"));
    ops.insert(("airimpute.io", "write_csv"));
    ops.insert(("airimpute.io", "write_excel"));
    
    ops
});

/// Validate and sanitize a file path for reading operations
pub fn validate_read_path(path_str: &str) -> Result<PathBuf> {
    validate_path_internal(path_str, false)
}

/// Validate and sanitize a file path for writing operations
pub fn validate_write_path(path_str: &str) -> Result<PathBuf> {
    validate_path_internal(path_str, true)
}

/// Internal path validation logic
fn validate_path_internal(path_str: &str, is_write: bool) -> Result<PathBuf> {
    // Check for null bytes
    if path_str.contains('\0') {
        bail!("Path contains null bytes");
    }
    
    // Check for suspicious patterns
    if path_str.contains("..") || path_str.contains("~") {
        warn!("Suspicious path pattern detected: {}", path_str);
    }
    
    let path = PathBuf::from(path_str);
    
    // For write operations, ensure parent directory exists
    if is_write {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)
                    .context("Failed to create parent directory")?;
            }
        }
    }
    
    // Canonicalize the path to resolve all symbolic links and relative components
    let canonical_path = if path.exists() {
        path.canonicalize()
            .context("Failed to canonicalize existing path")?
    } else if is_write {
        // For new files, canonicalize the parent and append the filename
        if let Some(parent) = path.parent() {
            let canonical_parent = parent.canonicalize()
                .context("Failed to canonicalize parent directory")?;
            if let Some(filename) = path.file_name() {
                canonical_parent.join(filename)
            } else {
                bail!("Invalid filename in path");
            }
        } else {
            bail!("Path has no parent directory");
        }
    } else {
        bail!("Path does not exist and is not a write operation");
    };
    
    // Check if the path is within allowed directories
    let is_allowed = ALLOWED_DIRECTORIES.iter().any(|allowed_dir| {
        canonical_path.starts_with(allowed_dir)
    });
    
    if !is_allowed {
        error!("Security violation: Path '{}' is outside allowed directories", path_str);
        bail!(
            "Security violation: Path is outside allowed directories. \
             Please save files within your Documents or Downloads folder."
        );
    }
    
    // Additional checks for write operations
    if is_write {
        // Check file extension for known dangerous types
        if let Some(extension) = canonical_path.extension() {
            let ext_str = extension.to_string_lossy().to_lowercase();
            if matches!(ext_str.as_str(), "exe" | "dll" | "so" | "dylib" | "bat" | "sh" | "ps1") {
                bail!("Cannot write executable files");
            }
        }
        
        // Check if we're overwriting system files
        let path_str_lower = canonical_path.to_string_lossy().to_lowercase();
        if path_str_lower.contains(".ssh") || 
           path_str_lower.contains(".gnupg") ||
           path_str_lower.contains(".bashrc") ||
           path_str_lower.contains(".profile") {
            bail!("Cannot overwrite system configuration files");
        }
    }
    
    Ok(canonical_path)
}

/// Validate Python operation against allow-list
pub fn validate_python_operation(module: &str, function: &str) -> Result<()> {
    // Disallow any characters that suggest a complex path or call
    if function.contains('.') || function.contains('(') || function.contains('[') {
        error!(
            "Security violation: Complex Python function path attempted: {}.{}",
            module, function
        );
        bail!(
            "Security violation: The requested operation '{}' is not a valid function name",
            function
        );
    }
    
    // Validate module name format
    if !module.starts_with("airimpute.") || module.contains("..") {
        error!(
            "Security violation: Invalid module name: {}",
            module
        );
        bail!(
            "Security violation: Only airimpute modules are allowed"
        );
    }
    
    if !ALLOWED_PYTHON_OPERATIONS.contains(&(module, function)) {
        error!(
            "Disallowed Python operation attempted: {}.{}",
            module, function
        );
        bail!(
            "Security violation: The requested operation '{}.{}' is not allowed",
            module, function
        );
    }
    Ok(())
}

/// Escape special characters for LaTeX to prevent injection attacks
pub fn escape_latex(text: &str) -> String {
    // Order matters! Backslash must be escaped first
    text.replace('\\', "\\textbackslash{}")
        .replace('&', "\\&")
        .replace('%', "\\%")
        .replace('$', "\\$")
        .replace('#', "\\#")
        .replace('_', "\\_")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace('~', "\\textasciitilde{}")
        .replace('^', "\\textasciicircum{}")
        .replace('<', "\\textless{}")
        .replace('>', "\\textgreater{}")
        .replace('|', "\\textbar{}")
}

/// Sanitize dataset name to prevent injection attacks
pub fn sanitize_dataset_name(name: &str) -> String {
    // Remove or replace potentially dangerous characters
    let safe_chars: Regex = Regex::new(r"[^a-zA-Z0-9\s\-_.]").unwrap();
    let sanitized = safe_chars.replace_all(name, "_");
    
    // Limit length
    let mut result = sanitized.to_string();
    if result.len() > 100 {
        result.truncate(100);
    }
    
    // Ensure it's not empty
    if result.trim().is_empty() {
        result = "Unnamed_Dataset".to_string();
    }
    
    result
}

/// Generate a secure filename with timestamp
pub fn generate_secure_filename(base_name: &str, extension: &str) -> String {
    let sanitized_base = sanitize_dataset_name(base_name);
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    format!("{}_{}.{}", sanitized_base, timestamp, extension)
}

/// Compute SHA256 hash of a file for integrity verification
pub async fn compute_file_hash(path: &Path) -> Result<String> {
    use tokio::io::AsyncReadExt;
    
    let mut file = tokio::fs::File::open(path).await
        .context("Failed to open file for hashing")?;
    
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 8192];
    
    loop {
        let bytes_read = file.read(&mut buffer).await
            .context("Failed to read file for hashing")?;
        
        if bytes_read == 0 {
            break;
        }
        
        hasher.update(&buffer[..bytes_read]);
    }
    
    Ok(format!("{:x}", hasher.finalize()))
}

/// Validate CSV content for potential security issues
pub fn validate_csv_content(content: &str, max_size_mb: usize) -> Result<()> {
    // Check size
    let size_mb = content.len() / (1024 * 1024);
    if size_mb > max_size_mb {
        bail!("CSV file too large: {} MB (max: {} MB)", size_mb, max_size_mb);
    }
    
    // Check for potential CSV injection patterns
    let dangerous_patterns = vec![
        "=SYSTEM(",
        "=CMD(",
        "@SUM(",
        "+SUM(",
        "-SUM(",
        "=IMPORTXML(",
        "=WEBSERVICE(",
    ];
    
    for pattern in dangerous_patterns {
        if content.contains(pattern) {
            warn!("Potential CSV injection pattern detected: {}", pattern);
            bail!("CSV contains potentially dangerous formulas");
        }
    }
    
    Ok(())
}

/// Create a safe temporary directory for processing
pub fn create_safe_temp_dir() -> Result<PathBuf> {
    let temp_base = std::env::temp_dir().join("airimpute");
    std::fs::create_dir_all(&temp_base)
        .context("Failed to create temp directory")?;
    
    let session_id = uuid::Uuid::new_v4();
    let temp_dir = temp_base.join(session_id.to_string());
    std::fs::create_dir(&temp_dir)
        .context("Failed to create session temp directory")?;
    
    Ok(temp_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_path_traversal() {
        // Test path traversal attempts
        assert!(validate_read_path("../../../etc/passwd").is_err());
        assert!(validate_read_path("/etc/passwd").is_err());
        assert!(validate_write_path("~/.ssh/id_rsa").is_err());
    }
    
    #[test]
    fn test_escape_latex() {
        let input = "Test & 100% $cost #1 {group} \\command";
        let escaped = escape_latex(input);
        assert!(escaped.contains("\\&"));
        assert!(escaped.contains("\\%"));
        assert!(escaped.contains("\\$"));
        assert!(escaped.contains("\\#"));
        assert!(escaped.contains("\\{"));
        assert!(escaped.contains("\\}"));
        assert!(escaped.contains("\\textbackslash{}"));
    }
    
    #[test]
    fn test_sanitize_dataset_name() {
        assert_eq!(sanitize_dataset_name("My Dataset!@#"), "My_Dataset___");
        assert_eq!(sanitize_dataset_name(""), "Unnamed_Dataset");
        assert_eq!(
            sanitize_dataset_name(&"x".repeat(200)).len(),
            100
        );
    }
    
    #[test]
    fn test_validate_python_operation() {
        assert!(validate_python_operation("airimpute.analysis", "analyze_missing_patterns").is_ok());
        assert!(validate_python_operation("os", "system").is_err());
        assert!(validate_python_operation("airimpute.analysis", "evil_function").is_err());
    }
    
    #[test]
    fn test_csv_injection_detection() {
        assert!(validate_csv_content("name,value\ntest,123", 100).is_ok());
        assert!(validate_csv_content("=SYSTEM('calc.exe')", 100).is_err());
        assert!(validate_csv_content("name,=CMD('format c:')", 100).is_err());
    }
}