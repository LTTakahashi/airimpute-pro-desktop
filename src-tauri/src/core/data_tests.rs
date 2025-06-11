#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use chrono::TimeZone;
    
    fn create_test_dataset() -> Dataset {
        let data = arr2(&[
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [7.0, 8.0, f64::NAN],
        ]);
        
        let columns = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let index = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 2, 0, 0).unwrap(),
        ];
        
        Dataset::new("Test".to_string(), data, columns, index)
    }
    
    #[test]
    fn test_dataset_creation() {
        let dataset = create_test_dataset();
        
        assert_eq!(dataset.name, "Test");
        assert_eq!(dataset.rows(), 3);
        assert_eq!(dataset.columns(), 3);
        assert_eq!(dataset.count_missing(), 2);
    }
    
    #[test]
    fn test_dataset_preview() {
        let dataset = create_test_dataset();
        let preview = dataset.preview(2, None).unwrap();
        
        assert_eq!(preview["preview_rows"], 2);
        assert_eq!(preview["total_rows"], 3);
        
        let data = preview["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
    }
    
    #[test]
    fn test_dataset_preview_with_columns() {
        let dataset = create_test_dataset();
        let preview = dataset.preview(10, Some(&["A".to_string(), "C".to_string()])).unwrap();
        
        let data = preview["data"].as_array().unwrap();
        let first_row = data[0].as_object().unwrap();
        
        assert!(first_row.contains_key("A"));
        assert!(!first_row.contains_key("B"));
        assert!(first_row.contains_key("C"));
    }
    
    #[test]
    fn test_data_validation() {
        let mut dataset = create_test_dataset();
        
        // Add variable constraints
        dataset.variables[0].min_valid = Some(0.0);
        dataset.variables[0].max_valid = Some(10.0);
        
        let validation = DataValidation::validate(&dataset).unwrap();
        
        assert!(validation.is_valid);
        assert_eq!(validation.summary.total_values, 9);
        assert_eq!(validation.summary.missing_values, 2);
        assert!((validation.summary.missing_percentage - 22.22).abs() < 0.01);
    }
    
    #[test]
    fn test_data_validation_out_of_bounds() {
        let mut dataset = create_test_dataset();
        
        // Add constraint that will fail
        dataset.variables[0].min_valid = Some(5.0);
        dataset.variables[0].max_valid = Some(10.0);
        
        let validation = DataValidation::validate(&dataset).unwrap();
        
        assert!(!validation.is_valid);
        assert!(!validation.issues.is_empty());
        
        let out_of_bounds = validation.issues.iter()
            .filter(|i| matches!(i.issue_type, IssueType::OutOfBounds))
            .count();
        assert!(out_of_bounds > 0);
    }
    
    #[test]
    fn test_data_statistics_basic() {
        let dataset = create_test_dataset();
        let stats = DataStatistics::calculate(&dataset).unwrap();
        
        // Check basic statistics
        assert_eq!(stats.basic_stats.count["A"], 3);
        assert_eq!(stats.basic_stats.count["B"], 2);
        assert_eq!(stats.basic_stats.count["C"], 2);
        
        assert_eq!(stats.basic_stats.mean["A"], 4.0);
        assert_eq!(stats.basic_stats.mean["B"], 5.0);
        assert_eq!(stats.basic_stats.mean["C"], 4.5);
    }
    
    #[test]
    fn test_missing_statistics() {
        let dataset = create_test_dataset();
        let stats = DataStatistics::calculate(&dataset).unwrap();
        
        assert_eq!(stats.missing_stats.total_missing, 2);
        assert!((stats.missing_stats.missing_percentage - 22.22).abs() < 0.01);
        assert_eq!(stats.missing_stats.missing_by_column["A"], 0);
        assert_eq!(stats.missing_stats.missing_by_column["B"], 1);
        assert_eq!(stats.missing_stats.missing_by_column["C"], 1);
    }
    
    #[test]
    fn test_temporal_statistics() {
        let dataset = create_test_dataset();
        let stats = DataStatistics::calculate(&dataset).unwrap();
        
        let (start, end) = stats.temporal_stats.time_range;
        assert_eq!(start, dataset.index[0]);
        assert_eq!(end, dataset.index[2]);
        assert_eq!(stats.temporal_stats.sampling_frequency, "hourly");
        assert!(stats.temporal_stats.regular_sampling);
    }
    
    #[test]
    fn test_quality_flags() {
        use QualityFlag::*;
        
        assert_eq!(Valid as u8, 0);
        assert_eq!(Missing as u8, 1);
        assert_eq!(Imputed as u8, 7);
    }
    
    #[test]
    fn test_station_metadata() {
        let station = Station {
            id: "S001".to_string(),
            name: "Test Station".to_string(),
            latitude: -23.5505,
            longitude: -46.6333,
            altitude: Some(760.0),
            metadata: [
                ("city".to_string(), "São Paulo".to_string()),
                ("type".to_string(), "urban".to_string()),
            ].into_iter().collect(),
        };
        
        assert_eq!(station.id, "S001");
        assert_eq!(station.latitude, -23.5505);
        assert_eq!(station.metadata.get("city"), Some(&"São Paulo".to_string()));
    }
    
    #[test]
    fn test_dataset_with_stations() {
        let mut dataset = create_test_dataset();
        
        dataset.stations = Some(vec![
            Station {
                id: "S001".to_string(),
                name: "Station 1".to_string(),
                latitude: -23.5,
                longitude: -46.6,
                altitude: None,
                metadata: Default::default(),
            },
            Station {
                id: "S002".to_string(),
                name: "Station 2".to_string(),
                latitude: -23.6,
                longitude: -46.7,
                altitude: Some(800.0),
                metadata: Default::default(),
            },
        ]);
        
        let station_names = dataset.get_station_names();
        assert_eq!(station_names.len(), 2);
        assert_eq!(station_names[0], "Station 1");
        
        let coords = dataset.get_coordinates().unwrap();
        assert_eq!(coords.len(), 2);
        assert_eq!(coords[0], (-23.5, -46.6));
    }
    
    #[test]
    fn test_physical_bounds() {
        let mut dataset = create_test_dataset();
        
        dataset.variables[0].min_valid = Some(0.0);
        dataset.variables[0].max_valid = Some(100.0);
        dataset.variables[1].min_valid = Some(-10.0);
        dataset.variables[1].max_valid = Some(10.0);
        
        let bounds = dataset.get_physical_bounds();
        
        assert_eq!(bounds.len(), 2);
        assert_eq!(bounds["A"], (0.0, 100.0));
        assert_eq!(bounds["B"], (-10.0, 10.0));
        assert!(!bounds.contains_key("C"));
    }
    
    #[test]
    fn test_correlation_matrix() {
        let dataset = create_test_dataset();
        let stats = DataStatistics::calculate(&dataset).unwrap();
        
        let corr = &stats.correlation_matrix;
        assert_eq!(corr.nrows(), 3);
        assert_eq!(corr.ncols(), 3);
        
        // Diagonal should be 1.0
        for i in 0..3 {
            assert_eq!(corr[[i, i]], 1.0);
        }
        
        // Matrix should be symmetric
        for i in 0..3 {
            for j in (i+1)..3 {
                assert_eq!(corr[[i, j]], corr[[j, i]]);
            }
        }
    }
    
    #[test]
    fn test_measurement_types() {
        let continuous = MeasurementType::Continuous;
        let discrete = MeasurementType::Discrete;
        
        match continuous {
            MeasurementType::Continuous => assert!(true),
            _ => assert!(false),
        }
        
        match discrete {
            MeasurementType::Discrete => assert!(true),
            _ => assert!(false),
        }
    }
}