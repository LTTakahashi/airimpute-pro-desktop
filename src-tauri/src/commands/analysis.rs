use tauri::command;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tauri::State;
use ndarray::Array2;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use tracing::info;
use uuid::Uuid;

use crate::state::AppState;
use crate::error::{CommandError, CommandResult, AuditLogEntry};
// use crate::python::bridge::{PythonBridge, MissingPattern as BridgeMissingPattern};

/// Comprehensive missing data pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingPatternAnalysis {
    pub pattern_type: MissingPatternType,
    pub statistics: MissingStatistics,
    pub temporal_analysis: TemporalMissingAnalysis,
    pub spatial_analysis: Option<SpatialMissingAnalysis>,
    pub recommendations: Vec<MethodRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingPatternType {
    CompletelyRandom,      // MCAR - Missing Completely At Random
    AtRandom,             // MAR - Missing At Random
    NotAtRandom,          // MNAR - Missing Not At Random
    Systematic,           // Systematic patterns (e.g., sensor failures)
    Mixed,               // Combination of patterns
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingStatistics {
    pub total_observations: usize,
    pub missing_count: usize,
    pub missing_percentage: f64,
    pub completeness_by_variable: HashMap<String, f64>,
    pub completeness_by_station: HashMap<String, f64>,
    pub consecutive_missing_stats: ConsecutiveMissingStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsecutiveMissingStats {
    pub max_consecutive_missing: usize,
    pub avg_consecutive_missing: f64,
    pub gap_distribution: HashMap<usize, usize>, // gap_length -> count
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMissingAnalysis {
    pub hourly_pattern: Vec<f64>,      // 24 hours
    pub daily_pattern: Vec<f64>,       // 7 days
    pub monthly_pattern: Vec<f64>,     // 12 months
    #[serde(rename = "trend")]
    pub trend_analysis: TrendAnalysis,
    pub seasonality_detected: bool,
    pub periodicity: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub trend_coefficient: f64,
    pub p_value: f64,
    pub trend_type: String, // "increasing", "decreasing", "stable"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialMissingAnalysis {
    #[serde(rename = "morans_i")]
    pub spatial_autocorrelation: f64,   // Moran's I
    pub hotspots: Vec<SpatialHotspot>,
    pub correlation_range: f64,         // Distance at which correlation drops
    pub anisotropy: Option<Anisotropy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialHotspot {
    pub center: (f64, f64),  // lat, lon
    pub radius_km: f64,
    pub missing_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anisotropy {
    pub major_axis_direction: f64,  // degrees from north
    pub ratio: f64,                 // major/minor axis ratio
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodRecommendation {
    pub method: String,
    pub suitability_score: f64,
    pub reasoning: String,
    pub expected_performance: ExpectedPerformance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedPerformance {
    pub rmse_range: (f64, f64),
    pub mae_range: (f64, f64),
    pub computation_time: String,
    pub uncertainty_handling: String,
}

/// Temporal pattern analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPatternAnalysis {
    pub periodicity: Vec<PeriodicComponent>,
    pub trend: TrendComponent,
    pub seasonality: SeasonalityComponent,
    pub anomalies: Vec<TemporalAnomaly>,
    pub autocorrelation: AutocorrelationAnalysis,
    pub stationarity_tests: StationarityTests,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicComponent {
    pub period_hours: f64,
    pub amplitude: f64,
    pub phase: f64,
    pub contribution: f64,  // percentage of variance explained
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendComponent {
    pub polynomial_order: usize,
    pub coefficients: Vec<f64>,
    pub r_squared: f64,
    pub change_points: Vec<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityComponent {
    pub seasonal_periods: Vec<usize>,
    pub seasonal_strength: f64,
    pub seasonal_patterns: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnomaly {
    pub timestamp: DateTime<Utc>,
    pub severity: f64,
    pub anomaly_type: String,
    pub affected_variables: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationAnalysis {
    pub acf: Vec<f64>,
    pub pacf: Vec<f64>,
    pub significant_lags: Vec<usize>,
    pub suggested_ar_order: usize,
    pub suggested_ma_order: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationarityTests {
    pub adf_test: StatisticalTest,
    pub kpss_test: StatisticalTest,
    pub pp_test: StatisticalTest,
    pub is_stationary: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    pub statistic: f64,
    pub p_value: f64,
    pub critical_values: HashMap<String, f64>,
}

/// Spatial correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialCorrelationAnalysis {
    pub global_indicators: GlobalSpatialIndicators,
    pub local_indicators: Vec<LocalSpatialIndicator>,
    pub variogram: Variogram,
    pub kriging_parameters: KrigingParameters,
    pub network_analysis: Option<NetworkAnalysis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalSpatialIndicators {
    pub morans_i: f64,
    pub gearys_c: f64,
    pub getis_ord_g: f64,
    pub p_values: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSpatialIndicator {
    pub station_id: String,
    pub location: (f64, f64),
    pub lisa_value: f64,  // Local Indicator of Spatial Association
    pub cluster_type: String,  // "high-high", "low-low", "high-low", "low-high"
    pub significance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variogram {
    pub model_type: String,  // "spherical", "exponential", "gaussian"
    pub nugget: f64,
    pub sill: f64,
    pub range: f64,
    pub fitted_values: Vec<(f64, f64)>,  // (distance, semivariance)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KrigingParameters {
    pub optimal_neighbors: usize,
    pub search_radius: f64,
    pub anisotropy_ratio: f64,
    pub anisotropy_angle: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAnalysis {
    pub network_density: f64,
    pub clustering_coefficient: f64,
    pub central_stations: Vec<String>,
    pub communities: Vec<Vec<String>>,
}

/// Comprehensive quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    #[serde(default)]
    pub metadata: QualityMetadata,
    pub data_quality_metrics: DataQualityMetrics,
    pub statistical_summary: StatisticalSummary,
    pub integrity_checks: IntegrityChecks,
    pub recommendations: Vec<QualityRecommendation>,
    pub quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetadata {
    pub dataset_id: String,
    pub analysis_timestamp: DateTime<Utc>,
    pub data_period: (DateTime<Utc>, DateTime<Utc>),
    pub stations_count: usize,
    pub variables_count: usize,
}

impl Default for QualityMetadata {
    fn default() -> Self {
        Self {
            dataset_id: String::new(),
            analysis_timestamp: Utc::now(),
            data_period: (Utc::now(), Utc::now()),
            stations_count: 0,
            variables_count: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityMetrics {
    pub completeness: f64,
    pub consistency: f64,
    pub timeliness: f64,
    pub validity: f64,
    pub accuracy: Option<f64>,
    pub uniqueness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub variable_statistics: HashMap<String, VariableStatistics>,
    pub correlation_matrix: Array2<f64>,
    pub outlier_summary: OutlierSummary,
    pub distribution_tests: HashMap<String, DistributionTest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableStatistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: HashMap<String, f64>,
    pub skewness: f64,
    pub kurtosis: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierSummary {
    pub total_outliers: usize,
    pub outlier_percentage: f64,
    pub outliers_by_variable: HashMap<String, usize>,
    pub extreme_values: Vec<ExtremeValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtremeValue {
    pub variable: String,
    pub value: f64,
    pub timestamp: DateTime<Utc>,
    pub station_id: String,
    pub z_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionTest {
    pub test_name: String,
    pub statistic: f64,
    pub p_value: f64,
    pub distribution_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityChecks {
    pub duplicate_records: usize,
    pub invalid_timestamps: usize,
    pub out_of_bounds_values: HashMap<String, usize>,
    pub referential_integrity: bool,
    pub consistency_violations: Vec<ConsistencyViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyViolation {
    pub violation_type: String,
    pub description: String,
    pub affected_records: usize,
    pub severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRecommendation {
    pub issue: String,
    pub impact: String,
    pub recommendation: String,
    pub priority: String,
}

/// Sensitivity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    pub parameter_sensitivity: HashMap<String, ParameterSensitivity>,
    pub method_comparison: MethodComparison,
    pub robustness_metrics: RobustnessMetrics,
    pub uncertainty_propagation: UncertaintyPropagation,
    pub recommendations: Vec<SensitivityRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSensitivity {
    pub parameter_name: String,
    pub sensitivity_index: f64,
    pub sobol_indices: SobolIndices,
    pub optimal_range: (f64, f64),
    pub interaction_effects: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SobolIndices {
    pub first_order: f64,
    pub total_order: f64,
    pub interaction_indices: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodComparison {
    pub methods_tested: Vec<String>,
    pub performance_metrics: HashMap<String, MethodPerformance>,
    pub stability_analysis: HashMap<String, f64>,
    pub computational_efficiency: HashMap<String, ComputationalMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodPerformance {
    pub rmse: f64,
    pub mae: f64,
    pub r_squared: f64,
    pub bias: f64,
    pub consistency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalMetrics {
    pub avg_time_ms: f64,
    pub memory_usage_mb: f64,
    pub scalability_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessMetrics {
    pub noise_tolerance: f64,
    pub missing_data_tolerance: f64,
    pub outlier_resistance: f64,
    pub stability_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyPropagation {
    pub input_uncertainty: f64,
    pub model_uncertainty: f64,
    pub output_uncertainty: f64,
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityRecommendation {
    pub parameter: String,
    pub recommendation: String,
    pub expected_improvement: f64,
}

#[command]
pub async fn compute_missing_patterns(
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
) -> Result<MissingPatternAnalysis, String> {
    info!("Computing missing patterns for dataset: {}", dataset_id);
    
    // Create audit log entry
    let audit_entry = AuditLogEntry::new(
        "compute_missing_patterns",
        &dataset_id
    );
    
    let result = compute_missing_patterns_internal(&state, dataset_id).await;
    
    // Update audit log
    // TODO: Implement audit logging
    match &result {
        Ok(_) => {
            // state.audit_logger.log(audit_entry);
        }
        Err(_e) => {
            // state.audit_logger.log(audit_entry.with_error(e));
        }
    }
    
    result.map_err(|e| e.to_string())
}

async fn compute_missing_patterns_internal(
    state: &Arc<AppState>,
    dataset_id: String,
) -> CommandResult<MissingPatternAnalysis> {
    // Get dataset from state - Uuid::parse_str provides sufficient validation
    let sanitized_id = dataset_id;
    
    // Get dataset from state
    let dataset_uuid = Uuid::parse_str(&sanitized_id)
        .map_err(|_| CommandError::DatasetNotFound { id: sanitized_id.clone() })?;
    let dataset = state.datasets.get(&dataset_uuid)
        .ok_or_else(|| CommandError::DatasetNotFound { id: sanitized_id.clone() })?;
    
    // Get Python bridge - it's already an Arc, so just clone it
    let bridge = state.python_bridge.clone();
    
    // Run missing pattern analysis
    let analysis_result = bridge.run_analysis(
        &dataset.value().data,
        "missing_patterns"
    )?;
    
    // Parse results
    let pattern_type = match analysis_result["pattern_type"].as_str() {
        Some("mcar") => MissingPatternType::CompletelyRandom,
        Some("mar") => MissingPatternType::AtRandom,
        Some("mnar") => MissingPatternType::NotAtRandom,
        Some("systematic") => MissingPatternType::Systematic,
        _ => MissingPatternType::Mixed,
    };
    
    // Extract statistics
    let stats = &analysis_result["statistics"];
    let missing_stats = MissingStatistics {
        total_observations: stats["total_observations"].as_u64().unwrap_or(0) as usize,
        missing_count: stats["missing_count"].as_u64().unwrap_or(0) as usize,
        missing_percentage: stats["missing_percentage"].as_f64().unwrap_or(0.0),
        completeness_by_variable: serde_json::from_value(
            stats["completeness_by_variable"].clone()
        ).unwrap_or_default(),
        completeness_by_station: serde_json::from_value(
            stats["completeness_by_station"].clone()
        ).unwrap_or_default(),
        consecutive_missing_stats: serde_json::from_value(
            stats["consecutive_missing_stats"].clone()
        ).unwrap_or(ConsecutiveMissingStats {
            max_consecutive_missing: 0,
            avg_consecutive_missing: 0.0,
            gap_distribution: HashMap::new(),
        }),
    };
    
    // Extract temporal analysis
    let temporal = &analysis_result["temporal_analysis"];
    let temporal_analysis = TemporalMissingAnalysis {
        hourly_pattern: serde_json::from_value(
            temporal["hourly_pattern"].clone()
        ).unwrap_or_else(|_| vec![0.0; 24]),
        daily_pattern: serde_json::from_value(
            temporal["daily_pattern"].clone()
        ).unwrap_or_else(|_| vec![0.0; 7]),
        monthly_pattern: serde_json::from_value(
            temporal["monthly_pattern"].clone()
        ).unwrap_or_else(|_| vec![0.0; 12]),
        trend_analysis: TrendAnalysis {
            trend_coefficient: temporal["trend"]["coefficient"].as_f64().unwrap_or(0.0),
            p_value: temporal["trend"]["p_value"].as_f64().unwrap_or(1.0),
            trend_type: temporal["trend"]["type"].as_str().unwrap_or("stable").to_string(),
        },
        seasonality_detected: temporal["seasonality_detected"].as_bool().unwrap_or(false),
        periodicity: temporal["periodicity"].as_u64().map(|p| p as usize),
    };
    
    // Extract spatial analysis if available
    let spatial_analysis = analysis_result.get("spatial_analysis").map(|spatial| SpatialMissingAnalysis {
            spatial_autocorrelation: spatial["morans_i"].as_f64().unwrap_or(0.0),
            hotspots: serde_json::from_value(
                spatial["hotspots"].clone()
            ).unwrap_or_default(),
            correlation_range: spatial["correlation_range"].as_f64().unwrap_or(0.0),
            anisotropy: serde_json::from_value(
                spatial["anisotropy"].clone()
            ).ok(),
        });
    
    // Generate method recommendations
    let recommendations = generate_method_recommendations(
        &pattern_type,
        &missing_stats,
        &temporal_analysis,
        &spatial_analysis
    );
    
    Ok(MissingPatternAnalysis {
        pattern_type,
        statistics: missing_stats,
        temporal_analysis,
        spatial_analysis,
        recommendations,
    })
}

fn generate_method_recommendations(
    pattern_type: &MissingPatternType,
    stats: &MissingStatistics,
    temporal: &TemporalMissingAnalysis,
    spatial: &Option<SpatialMissingAnalysis>,
) -> Vec<MethodRecommendation> {
    let mut recommendations = Vec::new();
    
    // RAH method is always recommended for its robustness
    recommendations.push(MethodRecommendation {
        method: "Robust Adaptive Hybrid (RAH)".to_string(),
        suitability_score: 0.95,
        reasoning: "RAH adapts to both spatial and temporal patterns, providing robust imputation with uncertainty quantification".to_string(),
        expected_performance: ExpectedPerformance {
            rmse_range: (0.8, 1.2),
            mae_range: (0.6, 0.9),
            computation_time: "moderate".to_string(),
            uncertainty_handling: "excellent".to_string(),
        },
    });
    
    // Temporal methods
    if temporal.seasonality_detected || temporal.periodicity.is_some() {
        recommendations.push(MethodRecommendation {
            method: "Seasonal Decomposition".to_string(),
            suitability_score: 0.85,
            reasoning: "Strong seasonal patterns detected in missing data".to_string(),
            expected_performance: ExpectedPerformance {
                rmse_range: (0.9, 1.3),
                mae_range: (0.7, 1.0),
                computation_time: "fast".to_string(),
                uncertainty_handling: "good".to_string(),
            },
        });
    }
    
    // Spatial methods
    if let Some(spatial_data) = spatial {
        if spatial_data.spatial_autocorrelation > 0.3 {
            recommendations.push(MethodRecommendation {
                method: "Spatial Kriging".to_string(),
                suitability_score: 0.80,
                reasoning: format!("High spatial autocorrelation ({:.2}) suggests spatial interpolation will be effective", 
                    spatial_data.spatial_autocorrelation).to_string(),
                expected_performance: ExpectedPerformance {
                    rmse_range: (0.85, 1.25),
                    mae_range: (0.65, 0.95),
                    computation_time: "slow".to_string(),
                    uncertainty_handling: "excellent".to_string(),
                },
            });
        }
    }
    
    // Matrix factorization for high missing rates
    if stats.missing_percentage > 30.0 {
        recommendations.push(MethodRecommendation {
            method: "Matrix Factorization".to_string(),
            suitability_score: 0.75,
            reasoning: format!("High missing rate ({:.1}%) - matrix factorization can leverage global patterns", 
                stats.missing_percentage).to_string(),
            expected_performance: ExpectedPerformance {
                rmse_range: (1.0, 1.5),
                mae_range: (0.8, 1.2),
                computation_time: "moderate".to_string(),
                uncertainty_handling: "moderate".to_string(),
            },
        });
    }
    
    recommendations.sort_by(|a, b| 
        b.suitability_score.partial_cmp(&a.suitability_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    );
    
    recommendations
}

#[command]
pub async fn analyze_temporal_patterns(
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
) -> Result<TemporalPatternAnalysis, String> {
    info!("Analyzing temporal patterns for dataset: {}", dataset_id);
    
    let result = analyze_temporal_patterns_internal(&state, dataset_id).await;
    result.map_err(|e| e.to_string())
}

async fn analyze_temporal_patterns_internal(
    state: &Arc<AppState>,
    dataset_id: String,
) -> CommandResult<TemporalPatternAnalysis> {
    // Get dataset from state - Uuid::parse_str provides sufficient validation
    let sanitized_id = dataset_id;
    
    // Parse UUID and get dataset from DashMap
    let dataset_uuid = Uuid::parse_str(&sanitized_id)
        .map_err(|_| CommandError::DatasetNotFound { id: sanitized_id.clone() })?;
    let dataset = state.datasets.get(&dataset_uuid)
        .ok_or_else(|| CommandError::DatasetNotFound { id: sanitized_id.clone() })?;
    
    let bridge = state.python_bridge.clone();
    
    let analysis_result = bridge.run_analysis(
        &dataset.value().data,
        "temporal_patterns"
    )?;
    
    // Parse complex temporal analysis results
    let temporal_analysis: TemporalPatternAnalysis = serde_json::from_value(analysis_result)
        .map_err(|e| CommandError::SerializationError {
            reason: format!("Failed to parse temporal analysis: {}", e)
        })?;
    
    Ok(temporal_analysis)
}

#[command]
pub async fn analyze_spatial_correlations(
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
) -> Result<SpatialCorrelationAnalysis, String> {
    info!("Analyzing spatial correlations for dataset: {}", dataset_id);
    
    let result = analyze_spatial_correlations_internal(&state, dataset_id).await;
    result.map_err(|e| e.to_string())
}

async fn analyze_spatial_correlations_internal(
    state: &Arc<AppState>,
    dataset_id: String,
) -> CommandResult<SpatialCorrelationAnalysis> {
    // Get dataset from state - Uuid::parse_str provides sufficient validation
    let sanitized_id = dataset_id;
    
    // Parse UUID and get dataset from DashMap
    let dataset_uuid = Uuid::parse_str(&sanitized_id)
        .map_err(|_| CommandError::DatasetNotFound { id: sanitized_id.clone() })?;
    let dataset = state.datasets.get(&dataset_uuid)
        .ok_or_else(|| CommandError::DatasetNotFound { id: sanitized_id.clone() })?;
    
    // Check if dataset has spatial information
    if dataset.value().stations.is_none() {
        return Err(CommandError::ValidationError {
            reason: "Dataset does not contain spatial coordinate information".to_string()
        });
    }
    
    let bridge = state.python_bridge.clone();
    
    let analysis_result = bridge.run_analysis(
        &dataset.value().data,
        "spatial_correlations"
    )?;
    
    let spatial_analysis: SpatialCorrelationAnalysis = serde_json::from_value(analysis_result)
        .map_err(|e| CommandError::SerializationError {
            reason: format!("Failed to parse spatial analysis: {}", e)
        })?;
    
    Ok(spatial_analysis)
}

#[command]
pub async fn generate_quality_report(
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
) -> Result<QualityReport, String> {
    info!("Generating quality report for dataset: {}", dataset_id);
    
    let result = generate_quality_report_internal(&state, dataset_id).await;
    result.map_err(|e| e.to_string())
}

async fn generate_quality_report_internal(
    state: &Arc<AppState>,
    dataset_id: String,
) -> CommandResult<QualityReport> {
    // Get dataset from state - Uuid::parse_str provides sufficient validation
    let sanitized_id = dataset_id;
    
    // Parse UUID and get dataset from DashMap
    let dataset_uuid = Uuid::parse_str(&sanitized_id)
        .map_err(|_| CommandError::DatasetNotFound { id: sanitized_id.clone() })?;
    let dataset = state.datasets.get(&dataset_uuid)
        .ok_or_else(|| CommandError::DatasetNotFound { id: sanitized_id.clone() })?;
    
    let bridge = state.python_bridge.clone();
    
    let analysis_result = bridge.run_analysis(
        &dataset.value().data,
        "quality_report"
    )?;
    
    // Build comprehensive quality report
    let metadata = QualityMetadata {
        dataset_id: sanitized_id.clone(),
        analysis_timestamp: Utc::now(),
        data_period: (
            dataset.value().index.first().cloned().unwrap_or_else(Utc::now),
            dataset.value().index.last().cloned().unwrap_or_else(Utc::now)
        ),
        stations_count: dataset.value().stations.as_ref().map(|s| s.len()).unwrap_or(0),
        variables_count: dataset.value().columns.len(),
    };
    
    let quality_report: QualityReport = serde_json::from_value(analysis_result)
        .map_err(|e| CommandError::SerializationError {
            reason: format!("Failed to parse quality report: {}", e)
        })?;
    
    Ok(QualityReport {
        metadata,
        ..quality_report
    })
}

#[command]
pub async fn perform_sensitivity_analysis(
    state: State<'_, Arc<AppState>>,
    job_id: String,
) -> Result<SensitivityAnalysis, String> {
    info!("Performing sensitivity analysis for job: {}", job_id);
    
    let result = perform_sensitivity_analysis_internal(&state, job_id).await;
    result.map_err(|e| e.to_string())
}

async fn perform_sensitivity_analysis_internal(
    state: &Arc<AppState>,
    job_id: String,
) -> CommandResult<SensitivityAnalysis> {
    // Parse the job ID as UUID - Uuid::parse_str provides sufficient validation
    let job_uuid = Uuid::parse_str(&job_id)
        .map_err(|_| CommandError::InvalidParameter { 
            param: "job_id".to_string(),
            reason: "Invalid UUID format".to_string()
        })?;
    
    // Get imputation job results from DashMap
    let job_entry = state.imputation_jobs.get(&job_uuid)
        .ok_or_else(|| CommandError::DatasetNotFound { id: job_id.clone() })?;
    let job = job_entry.lock().await;
    
    let bridge = state.python_bridge.clone();
    
    // Ensure job has completed results
    let result = job.result.as_ref()
        .ok_or_else(|| CommandError::ValidationError {
            reason: "Job has no results yet".to_string()
        })?;
    
    // Prepare sensitivity analysis parameters
    let analysis_params = serde_json::json!({
        "method": &job.method,
        "parameters": &job.parameters,
        "original_data": &job.original_data,
        "imputed_data": &result.imputed_data,
    });
    
    let analysis_result = bridge.run_analysis(
        &job.original_data,
        "sensitivity_analysis"
    )?;
    
    let sensitivity_analysis: SensitivityAnalysis = serde_json::from_value(analysis_result)
        .map_err(|e| CommandError::SerializationError {
            reason: format!("Failed to parse sensitivity analysis: {}", e)
        })?;
    
    Ok(sensitivity_analysis)
}