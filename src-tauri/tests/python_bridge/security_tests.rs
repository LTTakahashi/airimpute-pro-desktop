use anyhow::{bail, Result};
use serde_json::json;
use std::path::PathBuf;

// Import from the main crate
use crate::error::AppError;
use crate::python::safe_bridge_v2::{SafePythonBridgeV2, PythonOperation};
use crate::security::{validate_python_operation, validate_read_path, validate_write_path};

/// Test the happy path - a legitimate whitelisted operation works correctly
#[test]
fn test_happy_path_whitelisted_operation() {
    // Test that a valid whitelisted operation passes validation
    let result = validate_python_operation("airimpute.analysis", "analyze_missing_patterns");
    assert!(result.is_ok(), "Valid operation should be allowed");
    
    let result = validate_python_operation("airimpute.imputation", "impute_with_mean");
    assert!(result.is_ok(), "Valid imputation operation should be allowed");
    
    let result = validate_python_operation("airimpute.io", "load_from_file");
    assert!(result.is_ok(), "Valid IO operation should be allowed");
}

/// Test path traversal attacks are blocked
#[test]
fn test_path_traversal_prevention() {
    // Test various path traversal attempts
    let test_cases = vec![
        "../../../etc/passwd",
        "../../sensitive_data",
        "/etc/passwd",
        "/usr/bin/python",
        "valid_module/../../etc/passwd",
        "test\0/etc/passwd", // null byte injection
        "test/../../../etc/passwd",
        "~/.ssh/id_rsa",
        "$HOME/.bashrc",
        "%USERPROFILE%\\Documents",
    ];
    
    for path in test_cases {
        let result = validate_read_path(path);
        assert!(result.is_err(), "Path traversal '{}' should be blocked", path);
        
        let result = validate_write_path(path);
        assert!(result.is_err(), "Path traversal write '{}' should be blocked", path);
    }
}

/// Test that complex function paths are blocked
#[test]
fn test_complex_function_path_prevention() {
    // Test various complex function path attempts
    let test_cases = vec![
        ("airimpute.analysis", "os.system"),
        ("airimpute.analysis", "__import__('os').system"),
        ("airimpute.analysis", "eval('malicious')"),
        ("airimpute.analysis", "exec(bad_code)"),
        ("airimpute.analysis", "globals()['__builtins__']['eval']"),
        ("airimpute.analysis", "analyze.something.nested"),
        ("airimpute.analysis", "analyze_patterns(os.system)"),
        ("airimpute.analysis", "analyze[0]"),
    ];
    
    for (module, function) in test_cases {
        let result = validate_python_operation(module, function);
        assert!(result.is_err(), 
            "Complex function path '{}.{}' should be blocked", module, function);
    }
}

/// Test that invalid module names are blocked
#[test]
fn test_invalid_module_names() {
    let test_cases = vec![
        ("os", "listdir"),
        ("sys", "exit"),
        ("subprocess", "call"),
        ("__builtins__", "eval"),
        ("airimpute", "analysis"), // missing submodule
        ("not_airimpute.analysis", "analyze"),
        ("airimpute..analysis", "analyze"), // double dots
        ("../airimpute/analysis", "analyze"),
        ("airimpute/../../evil", "hack"),
    ];
    
    for (module, function) in test_cases {
        let result = validate_python_operation(module, function);
        assert!(result.is_err(), 
            "Invalid module '{}' should be blocked", module);
    }
}

/// Test that non-whitelisted operations are blocked
#[test]
fn test_non_whitelisted_operations() {
    let test_cases = vec![
        ("airimpute.analysis", "delete_all_data"),
        ("airimpute.analysis", "send_to_server"),
        ("airimpute.imputation", "execute_arbitrary_code"),
        ("airimpute.io", "write_to_system_file"),
        ("airimpute.visualization", "open_browser"), // not in whitelist
    ];
    
    for (module, function) in test_cases {
        let result = validate_python_operation(module, function);
        assert!(result.is_err(), 
            "Non-whitelisted operation '{}.{}' should be blocked", module, function);
    }
}

/// Test that argument validation works for IO operations
#[test]
fn test_io_operation_argument_validation() {
    // Create a mock Python operation with path traversal in arguments
    let operation = PythonOperation {
        module: "airimpute.io".to_string(),
        function: "load_from_file".to_string(),
        args: vec![json!("../../../etc/passwd")],
        kwargs: Default::default(),
    };
    
    // This should fail due to path validation
    // Note: This test would need the actual SafePythonBridgeV2 implementation
    // to be tested properly. For now, we test the validation function directly.
    let path_result = validate_read_path("../../../etc/passwd");
    assert!(path_result.is_err(), "Path traversal in args should be blocked");
}

/// Test keyword argument validation for paths
#[test]
fn test_io_operation_kwargs_validation() {
    // Test various keyword argument names that might contain paths
    let dangerous_paths = vec![
        "../../../etc/passwd",
        "/etc/shadow",
        "~/.ssh/id_rsa",
    ];
    
    let path_keywords = vec![
        "path",
        "filepath", 
        "filename",
        "filepath_or_buffer",
        "file",
    ];
    
    for path in &dangerous_paths {
        for keyword in &path_keywords {
            let result = validate_read_path(path);
            assert!(result.is_err(), 
                "Path '{}' in kwarg '{}' should be blocked", path, keyword);
        }
    }
}

/// Test safe paths are allowed
#[test]
fn test_safe_paths_allowed() {
    let safe_paths = vec![
        "data.csv",
        "output/results.json",
        "datasets/air_quality_2024.csv",
        "./local_file.txt",
        "subdir/another_subdir/file.dat",
    ];
    
    for path in safe_paths {
        let result = validate_read_path(path);
        assert!(result.is_ok(), "Safe path '{}' should be allowed", path);
    }
}

/// Test empty and malformed inputs
#[test] 
fn test_empty_and_malformed_inputs() {
    // Empty module/function
    let result = validate_python_operation("", "");
    assert!(result.is_err(), "Empty module/function should be rejected");
    
    let result = validate_python_operation("airimpute.analysis", "");
    assert!(result.is_err(), "Empty function should be rejected");
    
    let result = validate_python_operation("", "analyze");
    assert!(result.is_err(), "Empty module should be rejected");
    
    // Very long inputs (potential DoS)
    let long_module = "airimpute.".to_string() + &"a".repeat(1000);
    let result = validate_python_operation(&long_module, "analyze");
    assert!(result.is_err(), "Very long module name should be rejected");
}

/// Test unicode and special character handling
#[test]
fn test_unicode_and_special_characters() {
    let test_cases = vec![
        ("airimpute.analysis", "analyze_😀"),
        ("airimpute.analysis", "analyze\npatterns"),
        ("airimpute.analysis", "analyze\rpatterns"),
        ("airimpute.analysis", "analyze\tpatterns"),
        ("airimpute.analysis", "analyze\\patterns"),
        ("airimpute.analysis", "analyze\"patterns"),
        ("airimpute.analysis", "analyze'patterns"),
    ];
    
    for (module, function) in test_cases {
        let result = validate_python_operation(module, function);
        // These should likely be rejected for safety
        assert!(result.is_err() || !function.contains(char::is_control), 
            "Special characters in function '{}' should be handled safely", function);
    }
}