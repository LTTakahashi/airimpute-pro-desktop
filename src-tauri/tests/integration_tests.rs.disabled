use airimpute_pro::*;
use tempfile::TempDir;
use std::path::PathBuf;
use uuid::Uuid;
use chrono::Utc;
use serde_json::json;

mod common;
use common::*;

#[tokio::test]
async fn test_full_data_pipeline() {
    // Create test environment
    let temp_dir = TempDir::new().unwrap();
    let app_state = create_test_app_state().await;
    
    // Test data import
    let csv_path = create_test_csv(&temp_dir, 100, 5);
    let import_result = commands::data::load_dataset(
        create_test_window(),
        tauri::State::new(app_state.clone()),
        csv_path.to_string_lossy().to_string(),
        commands::data::ImportOptions {
            delimiter: Some(",".to_string()),
            encoding: None,
            has_header: true,
            date_column: Some("timestamp".to_string()),
            parse_dates: true,
            na_values: None,
            skip_rows: None,
            use_cols: None,
        },
    )
    .await;
    
    assert!(import_result.is_ok());
    let dataset_response = import_result.unwrap();
    assert_eq!(dataset_response.rows, 100);
    assert_eq!(dataset_response.columns, 4); // 5 columns minus timestamp
    
    // Test data validation
    let validation_result = commands::data::validate_dataset(
        tauri::State::new(app_state.clone()),
        dataset_response.id.clone(),
    )
    .await;
    
    assert!(validation_result.is_ok());
    let validation = validation_result.unwrap();
    assert!(validation.is_valid);
    
    // Test imputation
    let imputation_result = commands::imputation::run_imputation(
        create_test_window(),
        tauri::State::new(app_state.clone()),
        dataset_response.id.clone(),
        "kalman_filter".to_string(),
        json!({
            "model_order": 2,
            "process_noise": 0.01,
            "measurement_noise": 0.1,
        }),
    )
    .await;
    
    assert!(imputation_result.is_ok());
    let job_info = imputation_result.unwrap();
    assert_eq!(job_info.status, "running");
    
    // Wait for completion (in real tests, would poll)
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    
    // Test export
    let export_path = temp_dir.path().join("export.csv");
    let export_result = commands::export::export_to_csv(
        create_test_window(),
        tauri::State::new(app_state.clone()),
        dataset_response.id.clone(),
        export_path.to_string_lossy().to_string(),
        commands::export::ExportOptions {
            include_metadata: Some(true),
            include_quality_flags: Some(false),
            date_format: Some("%Y-%m-%d %H:%M:%S".to_string()),
            delimiter: Some(",".to_string()),
            encoding: None,
            precision: Some(6),
            na_representation: Some("NA".to_string()),
            compression: None,
        },
    )
    .await;
    
    assert!(export_result.is_ok());
    assert!(export_path.exists());
}

#[tokio::test]
async fn test_project_lifecycle() {
    let app_state = create_test_app_state().await;
    
    // Create project
    let create_result = commands::project::create_project(
        tauri::State::new(app_state.clone()),
        "Test Project".to_string(),
        "A test project for integration testing".to_string(),
    )
    .await;
    
    assert!(create_result.is_ok());
    let project_info = create_result.unwrap();
    
    // Save project
    let save_result = commands::project::save_project(
        tauri::State::new(app_state.clone()),
        project_info.id.clone(),
    )
    .await;
    
    assert!(save_result.is_ok());
    let saved_path = save_result.unwrap();
    assert!(PathBuf::from(&saved_path).exists());
    
    // Archive project
    let temp_dir = TempDir::new().unwrap();
    let archive_path = temp_dir.path().join("project_archive");
    let archive_result = commands::project::archive_project(
        tauri::State::new(app_state.clone()),
        project_info.id.clone(),
        archive_path.to_string_lossy().to_string(),
    )
    .await;
    
    assert!(archive_result.is_ok());
    let archived = archive_result.unwrap();
    assert!(PathBuf::from(&archived).exists());
}

#[tokio::test]
async fn test_visualization_generation() {
    let temp_dir = TempDir::new().unwrap();
    let app_state = create_test_app_state().await;
    
    // Create and load test dataset
    let dataset = create_test_dataset(50, 3);
    let dataset_id = dataset.id;
    app_state.datasets.insert(dataset_id, dataset);
    
    // Test missing pattern plot
    let missing_plot_result = commands::visualization::generate_missing_pattern_plot(
        create_test_window(),
        tauri::State::new(app_state.clone()),
        dataset_id.to_string(),
        commands::visualization::PlotOptions {
            width: Some(800),
            height: Some(600),
            dpi: Some(100),
            theme: Some("seaborn-v0_8-darkgrid".to_string()),
            color_scheme: Some("viridis".to_string()),
            title: Some("Test Missing Pattern".to_string()),
            save_path: Some(temp_dir.path().join("missing.png").to_string_lossy().to_string()),
            format: Some("png".to_string()),
        },
    )
    .await;
    
    assert!(missing_plot_result.is_ok());
    let plot_result = missing_plot_result.unwrap();
    assert!(!plot_result.image_data.is_empty());
    assert!(temp_dir.path().join("missing.png").exists());
    
    // Test time series plot
    let ts_plot_result = commands::visualization::generate_time_series_plot(
        create_test_window(),
        tauri::State::new(app_state.clone()),
        dataset_id.to_string(),
        vec!["Variable_1".to_string(), "Variable_2".to_string()],
        commands::visualization::PlotOptions {
            width: Some(1200),
            height: Some(800),
            dpi: Some(100),
            theme: Some("seaborn-v0_8-whitegrid".to_string()),
            color_scheme: None,
            title: Some("Test Time Series".to_string()),
            save_path: None,
            format: Some("png".to_string()),
        },
    )
    .await;
    
    assert!(ts_plot_result.is_ok());
}

#[tokio::test]
async fn test_concurrent_operations() {
    let app_state = create_test_app_state().await;
    let temp_dir = TempDir::new().unwrap();
    
    // Create multiple datasets concurrently
    let mut handles = vec![];
    
    for i in 0..5 {
        let app_state_clone = app_state.clone();
        let temp_dir_clone = temp_dir.path().to_path_buf();
        
        let handle = tokio::spawn(async move {
            let csv_path = create_test_csv_at_path(
                &temp_dir_clone.join(format!("data_{}.csv", i)),
                50,
                3,
            );
            
            commands::data::load_dataset(
                create_test_window(),
                tauri::State::new(app_state_clone),
                csv_path.to_string_lossy().to_string(),
                commands::data::ImportOptions {
                    delimiter: Some(",".to_string()),
                    encoding: None,
                    has_header: true,
                    date_column: None,
                    parse_dates: false,
                    na_values: None,
                    skip_rows: None,
                    use_cols: None,
                },
            )
            .await
        });
        
        handles.push(handle);
    }
    
    // Wait for all to complete
    let results = futures::future::join_all(handles).await;
    
    // Verify all succeeded
    for result in results {
        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }
    
    // Verify all datasets are in state
    assert_eq!(app_state.datasets.len(), 5);
}

#[tokio::test]
async fn test_error_handling() {
    let app_state = create_test_app_state().await;
    
    // Test loading non-existent file
    let load_result = commands::data::load_dataset(
        create_test_window(),
        tauri::State::new(app_state.clone()),
        "/non/existent/file.csv".to_string(),
        Default::default(),
    )
    .await;
    
    assert!(load_result.is_err());
    assert!(load_result.unwrap_err().contains("not found"));
    
    // Test invalid dataset ID
    let validate_result = commands::data::validate_dataset(
        tauri::State::new(app_state.clone()),
        "invalid-uuid".to_string(),
    )
    .await;
    
    assert!(validate_result.is_err());
    assert!(validate_result.unwrap_err().contains("Invalid dataset ID"));
    
    // Test imputation on non-existent dataset
    let impute_result = commands::imputation::run_imputation(
        create_test_window(),
        tauri::State::new(app_state.clone()),
        Uuid::new_v4().to_string(),
        "kalman_filter".to_string(),
        json!({}),
    )
    .await;
    
    assert!(impute_result.is_err());
    assert!(impute_result.unwrap_err().contains("Dataset not found"));
}

#[tokio::test]
async fn test_memory_management() {
    let app_state = create_test_app_state().await;
    let temp_dir = TempDir::new().unwrap();
    
    // Load large dataset
    let large_csv = create_test_csv(&temp_dir, 10000, 20);
    
    let initial_memory = get_memory_usage();
    
    let load_result = commands::data::load_dataset(
        create_test_window(),
        tauri::State::new(app_state.clone()),
        large_csv.to_string_lossy().to_string(),
        Default::default(),
    )
    .await;
    
    assert!(load_result.is_ok());
    let dataset_id = load_result.unwrap().id;
    
    let after_load_memory = get_memory_usage();
    let memory_increase = after_load_memory - initial_memory;
    
    // Memory increase should be reasonable (less than 100MB for 10k x 20 dataset)
    assert!(memory_increase < 100_000_000);
    
    // Remove dataset
    app_state.datasets.remove(&Uuid::parse_str(&dataset_id).unwrap());
    
    // Force garbage collection (in real Rust, would rely on RAII)
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    let after_remove_memory = get_memory_usage();
    
    // Memory should be mostly reclaimed
    assert!(after_remove_memory < after_load_memory);
}

#[tokio::test] 
async fn test_python_bridge_integration() {
    let app_state = create_test_app_state().await;
    
    // Test Python runtime health check
    let health_result = commands::system::check_python_runtime(
        tauri::State::new(app_state.clone()),
    )
    .await;
    
    assert!(health_result.is_ok());
    let health = health_result.unwrap();
    assert_eq!(health["status"], "healthy");
    assert!(health["python_version"].is_string());
    
    // Test Python code execution through imputation
    let dataset = create_test_dataset(100, 5);
    let dataset_id = dataset.id;
    app_state.datasets.insert(dataset_id, dataset);
    
    let impute_result = commands::imputation::run_imputation(
        create_test_window(),
        tauri::State::new(app_state.clone()),
        dataset_id.to_string(),
        "mean".to_string(), // Simple method for testing
        json!({}),
    )
    .await;
    
    assert!(impute_result.is_ok());
}