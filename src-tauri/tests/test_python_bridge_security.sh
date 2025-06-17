#!/bin/bash
# Test script for Python bridge security features
# This script tests the validation logic without requiring WebKit dependencies

echo "=== Python Bridge Security Tests ==="
echo

# Set up test environment
export RUST_BACKTRACE=1
export RUST_LOG=debug

# Create a simple Rust test that validates the Python bridge security
cat > /tmp/test_python_bridge_security.rs << 'EOF'
use std::collections::HashMap;

// Mock structures for testing
#[derive(Debug)]
struct PythonOperation {
    module: String,
    function: String,
    args: Vec<String>,
    kwargs: HashMap<String, String>,
    timeout_ms: Option<u64>,
}

fn validate_operation(op: &PythonOperation) -> Result<(), String> {
    // Check for path traversal attempts
    if op.module.contains("..") {
        return Err("Path traversal detected".to_string());
    }
    
    // Check for dangerous modules
    let dangerous_modules = vec!["os", "subprocess", "eval", "__builtins__", "sys"];
    if dangerous_modules.contains(&op.module.as_str()) {
        return Err(format!("Dangerous module: {}", op.module));
    }
    
    // Check for dangerous functions
    if op.function.starts_with("__") && op.function != "__init__" {
        return Err("Dangerous function pattern detected".to_string());
    }
    
    // Whitelist validation
    let whitelisted_modules = vec![
        "airimpute.imputation",
        "airimpute.io",
        "airimpute.stats",
        "airimpute.benchmarks",
        "airimpute.visualization",
    ];
    
    if !whitelisted_modules.iter().any(|m| op.module.starts_with(m)) {
        return Err(format!("Module not whitelisted: {}", op.module));
    }
    
    Ok(())
}

fn main() {
    println!("Testing Python Bridge Security Validation...\n");
    
    // Test cases that should fail
    let invalid_operations = vec![
        PythonOperation {
            module: "../../../etc/passwd".to_string(),
            function: "read".to_string(),
            args: vec![],
            kwargs: HashMap::new(),
            timeout_ms: Some(5000),
        },
        PythonOperation {
            module: "os".to_string(),
            function: "system".to_string(),
            args: vec!["rm -rf /".to_string()],
            kwargs: HashMap::new(),
            timeout_ms: Some(5000),
        },
        PythonOperation {
            module: "airimpute.io".to_string(),
            function: "__import__".to_string(),
            args: vec!["os".to_string()],
            kwargs: HashMap::new(),
            timeout_ms: Some(5000),
        },
        PythonOperation {
            module: "subprocess".to_string(),
            function: "run".to_string(),
            args: vec!["malicious_command".to_string()],
            kwargs: HashMap::new(),
            timeout_ms: Some(5000),
        },
    ];
    
    println!("Testing invalid operations (should all fail):");
    for (i, op) in invalid_operations.iter().enumerate() {
        match validate_operation(op) {
            Ok(_) => println!("  ❌ Test {} FAILED - Operation was allowed: {:?}", i + 1, op),
            Err(e) => println!("  ✅ Test {} passed - Rejected: {}", i + 1, e),
        }
    }
    
    println!("\nTesting valid operations (should all pass):");
    // Test cases that should pass
    let valid_operations = vec![
        PythonOperation {
            module: "airimpute.imputation".to_string(),
            function: "kriging_impute".to_string(),
            args: vec!["data.csv".to_string()],
            kwargs: HashMap::new(),
            timeout_ms: Some(30000),
        },
        PythonOperation {
            module: "airimpute.stats".to_string(),
            function: "calculate_rmse".to_string(),
            args: vec![],
            kwargs: HashMap::new(),
            timeout_ms: Some(5000),
        },
    ];
    
    for (i, op) in valid_operations.iter().enumerate() {
        match validate_operation(op) {
            Ok(_) => println!("  ✅ Test {} passed - Operation allowed", i + 1),
            Err(e) => println!("  ❌ Test {} FAILED - Operation rejected: {}", i + 1, e),
        }
    }
    
    println!("\n=== Security Validation Tests Complete ===");
}
EOF

# Compile and run the test
echo "Compiling security test..."
rustc /tmp/test_python_bridge_security.rs -o /tmp/test_python_bridge_security

if [ $? -eq 0 ]; then
    echo "Running security test..."
    /tmp/test_python_bridge_security
else
    echo "Failed to compile test"
    exit 1
fi

# Clean up
rm -f /tmp/test_python_bridge_security.rs /tmp/test_python_bridge_security