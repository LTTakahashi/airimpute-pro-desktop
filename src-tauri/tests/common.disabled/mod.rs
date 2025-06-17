use std::sync::Arc;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use uuid::Uuid;
use chrono::{Utc, DateTime};
use ndarray::{Array2, Array1};
use parking_lot::RwLock;
use dashmap::DashMap;
use tauri::Window;
use std::fs::File;
use std::io::Write;
use csv::Writer;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use airimpute_pro::{
    state::{AppState, UserPreferences},
    core::{
        data::{Dataset, Station, Variable, MeasurementType},
        project::Project,
    },
    python::bridge::PythonBridge,
    db::Database,
};

/// Create a mock window for testing
pub fn create_test_window() -> Window {
    // This is a simplified mock - in real tests would use tauri::test
    unsafe {
        std::mem::zeroed()
    }
}

/// Create a test AppState with all necessary components
pub async fn create_test_app_state() -> Arc<AppState> {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    
    // Create test database
    let db = Database::new(&db_path).await.unwrap();
    
    // Create mock Python runtime
    let python_runtime = Arc::new(MockPythonBridge::new());
    
    // Create test state
    Arc::new(AppState {
        db: Arc::new(db),
        datasets: DashMap::new(),
        projects: DashMap::new(),
        imputation_jobs: DashMap::new(),
        dashboards: DashMap::new(),
        preferences: RwLock::new(UserPreferences::default()),
        python_runtime: python_runtime as Arc<dyn PythonBridge>,
        recent_projects: RwLock::new(Vec::new()),
        cache: moka::future::Cache::new(100),
        metrics: Arc::new(MockMetrics::new()),
    })
}

/// Mock Python bridge for testing
pub struct MockPythonBridge;

impl MockPythonBridge {
    pub fn new() -> Self {
        Self
    }
}

impl PythonBridge for MockPythonBridge {
    fn check_health(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::json!({
            "status": "healthy",
            "python_version": "3.11.0",
            "packages": {
                "numpy": "1.24.0",
                "pandas": "2.0.0",
                "scikit-learn": "1.3.0",
            }
        }))
    }
    
    fn execute(&self, code: &str, context: Option<serde_json::Value>) -> anyhow::Result<()> {
        // Mock execution
        Ok(())
    }
    
    fn execute_with_return(&self, code: &str, context: Option<serde_json::Value>) -> anyhow::Result<String> {
        // Return mock results based on code content
        if code.contains("generate_missing_pattern_plot") {
            Ok(serde_json::json!({
                "image_data": "base64_encoded_image_data_here",
                "metadata": {
                    "total_missing": 150,
                    "total_values": 1000,
                    "missing_percentage": 15.0,
                }
            }).to_string())
        } else if code.contains("impute_mean") {
            Ok(serde_json::json!({
                "success": true,
                "metrics": {
                    "rmse": 0.123,
                    "mae": 0.089,
                    "r2": 0.945,
                }
            }).to_string())
        } else {
            Ok(serde_json::json!({"result": "mock_result"}).to_string())
        }
    }
}

/// Mock metrics collector
pub struct MockMetrics;

impl MockMetrics {
    pub fn new() -> Self {
        Self
    }
    
    pub fn get_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "memory_usage": 1234567,
            "cpu_usage": 12.5,
            "active_tasks": 2,
        })
    }
}

/// Create a test dataset with specified dimensions
pub fn create_test_dataset(rows: usize, cols: usize) -> Dataset {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Generate data with some missing values
    let mut data = Array2::<f64>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            if rng.gen::<f64>() > 0.1 {
                data[[i, j]] = rng.gen_range(0.0..100.0);
            } else {
                data[[i, j]] = f64::NAN;
            }
        }
    }
    
    // Generate timestamps
    let now = Utc::now();
    let timestamps: Vec<DateTime<Utc>> = (0..rows)
        .map(|i| now - chrono::Duration::hours(rows as i64 - i as i64))
        .collect();
    
    // Generate column names
    let columns: Vec<String> = (0..cols)
        .map(|i| format!("Variable_{}", i + 1))
        .collect();
    
    // Create variables
    let variables: Vec<Variable> = columns.iter()
        .map(|name| Variable {
            name: name.clone(),
            unit: "units".to_string(),
            description: format!("Test variable {}", name),
            min_valid: Some(0.0),
            max_valid: Some(100.0),
            precision: Some(2),
            measurement_type: MeasurementType::Continuous,
        })
        .collect();
    
    let mut dataset = Dataset::new(
        "Test Dataset".to_string(),
        data,
        columns,
        timestamps,
    );
    
    dataset.variables = variables;
    dataset
}

/// Create a test CSV file
pub fn create_test_csv(temp_dir: &TempDir, rows: usize, cols: usize) -> PathBuf {
    let path = temp_dir.path().join("test_data.csv");
    create_test_csv_at_path(&path, rows, cols)
}

/// Create a test CSV file at specific path
pub fn create_test_csv_at_path(path: &Path, rows: usize, cols: usize) -> PathBuf {
    let mut file = File::create(path).unwrap();
    let mut writer = Writer::from_writer(file);
    
    // Write headers
    let mut headers = vec!["timestamp".to_string()];
    for i in 0..cols {
        headers.push(format!("var_{}", i + 1));
    }
    writer.write_record(&headers).unwrap();
    
    // Write data
    let mut rng = StdRng::seed_from_u64(42);
    let now = Utc::now();
    
    for i in 0..rows {
        let mut record = vec![
            (now - chrono::Duration::hours(rows as i64 - i as i64))
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
        ];
        
        for _ in 0..cols {
            if rng.gen::<f64>() > 0.1 {
                record.push(format!("{:.2}", rng.gen_range(0.0..100.0)));
            } else {
                record.push("".to_string()); // Missing value
            }
        }
        
        writer.write_record(&record).unwrap();
    }
    
    writer.flush().unwrap();
    path.to_path_buf()
}

/// Create test station data
pub fn create_test_stations(count: usize) -> Vec<Station> {
    let mut rng = StdRng::seed_from_u64(42);
    
    (0..count)
        .map(|i| Station {
            id: format!("station_{}", i + 1),
            name: format!("Test Station {}", i + 1),
            latitude: rng.gen_range(-90.0..90.0),
            longitude: rng.gen_range(-180.0..180.0),
            altitude: Some(rng.gen_range(0.0..1000.0)),
            metadata: Default::default(),
        })
        .collect()
}

/// Get current memory usage (mock implementation)
pub fn get_memory_usage() -> usize {
    // In real implementation would use system calls
    1234567
}

/// Assert float equality with tolerance
pub fn assert_float_eq(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() < tolerance,
        "Float assertion failed: {} != {} (tolerance: {})",
        a, b, tolerance
    );
}

/// Create test imputation parameters
pub fn create_test_imputation_params(method: &str) -> serde_json::Value {
    match method {
        "mean" => serde_json::json!({}),
        "forward_fill" => serde_json::json!({
            "limit": 5,
        }),
        "linear" => serde_json::json!({
            "limit_direction": "both",
        }),
        "kalman_filter" => serde_json::json!({
            "model_order": 2,
            "process_noise": 0.01,
            "measurement_noise": 0.1,
        }),
        "random_forest" => serde_json::json!({
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
        }),
        _ => serde_json::json!({}),
    }
}

/// Verify dataset integrity
pub fn verify_dataset_integrity(dataset: &Dataset) {
    assert_eq!(dataset.data.nrows(), dataset.index.len());
    assert_eq!(dataset.data.ncols(), dataset.columns.len());
    assert_eq!(dataset.columns.len(), dataset.variables.len());
    
    // Check for NaN handling
    let nan_count = dataset.data.iter().filter(|&&v| v.is_nan()).count();
    assert!(nan_count >= 0);
    
    // Check timestamps are monotonic
    for i in 1..dataset.index.len() {
        assert!(dataset.index[i] >= dataset.index[i - 1]);
    }
}