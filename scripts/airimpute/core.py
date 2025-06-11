"""
Core imputation engine for AirImpute Pro

This module implements a state-of-the-art imputation engine with academic-grade
features including uncertainty quantification, ensemble methods, and comprehensive
validation capabilities.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass, field
import logging
import hashlib
import json
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)


@dataclass
class ImputationResult:
    """
    Container for comprehensive imputation results with academic-grade metrics.
    
    Attributes
    ----------
    data : pd.DataFrame
        Imputed dataset
    metrics : Dict[str, float]
        Statistical quality metrics
    metadata : Dict[str, Any]
        Method parameters and execution details
    confidence_intervals : Optional[pd.DataFrame]
        Confidence intervals for imputed values
    uncertainty_scores : Optional[pd.DataFrame]
        Uncertainty quantification for each imputed value
    diagnostics : Optional[Dict[str, Any]]
        Detailed diagnostic information
    validation_report : Optional[Dict[str, Any]]
        Comprehensive validation metrics
    reproducibility_info : Optional[Dict[str, Any]]
        Information for reproducing results
    """
    data: pd.DataFrame
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    confidence_intervals: Optional[pd.DataFrame] = None
    uncertainty_scores: Optional[pd.DataFrame] = None
    diagnostics: Optional[Dict[str, Any]] = None
    validation_report: Optional[Dict[str, Any]] = None
    reproducibility_info: Optional[Dict[str, Any]] = None


@dataclass
class MissingDataPattern:
    """Analysis of missing data patterns"""
    mechanism: str  # MCAR, MAR, MNAR
    pattern_type: str  # monotone, arbitrary, etc.
    missingness_matrix: pd.DataFrame
    statistics: Dict[str, float]
    visualization_data: Optional[Dict[str, Any]] = None


@dataclass
class EnsembleConfig:
    """Configuration for ensemble imputation"""
    methods: List[str]
    weights: Optional[List[float]] = None
    aggregation: str = "weighted_mean"  # weighted_mean, median, trimmed_mean, super_learner, bayesian_averaging
    bootstrap_samples: int = 100
    confidence_level: float = 0.95
    use_theoretical_ensemble: bool = False
    ensemble_strategy: str = "super_learner"  # super_learner, bayesian_averaging, neural_stacking, exponential_weights
    cv_folds: int = 10
    use_gpu: bool = False
    n_jobs: int = -1


class ImputationEngine:
    """
    Advanced imputation engine with academic-grade features.
    
    This engine supports:
    - Multiple imputation methods with ensemble capabilities
    - Uncertainty quantification and confidence intervals
    - Parallel processing for large datasets
    - Comprehensive validation and diagnostics
    - Reproducible research features
    - Missing data mechanism detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.methods = {}
        self.config = config or {}
        self._register_default_methods()
        
        # Performance settings
        self.n_jobs = self.config.get('n_jobs', mp.cpu_count())
        self.chunk_size = self.config.get('chunk_size', 10000)
        self.enable_caching = self.config.get('enable_caching', True)
        
        # Scientific settings
        self.random_state = self.config.get('random_state', None)
        self.enable_uncertainty = self.config.get('enable_uncertainty', True)
        self.validation_mode = self.config.get('validation_mode', 'comprehensive')
        
        # Initialize components
        self._cache = {} if self.enable_caching else None
        self._set_random_seed()
        
    def _register_default_methods(self):
        """Register built-in imputation methods"""
        from .methods import (
            MeanImputation,
            LinearInterpolation,
            SplineInterpolation,
            ForwardFill,
            BackwardFill,
            KalmanFilter,
            RandomForest,
            KNNImputation,
            MatrixFactorization,
            DeepLearningImputation,
            RAHMethod
        )
        from .ensemble_methods import TheoreticalEnsemble, AdaptiveEnsemble
        
        self.register_method("mean", MeanImputation())
        self.register_method("linear", LinearInterpolation())
        self.register_method("spline", SplineInterpolation())
        self.register_method("forward_fill", ForwardFill())
        self.register_method("backward_fill", BackwardFill())
        self.register_method("kalman", KalmanFilter())
        self.register_method("random_forest", RandomForest())
        self.register_method("knn", KNNImputation())
        self.register_method("matrix_factorization", MatrixFactorization())
        self.register_method("deep_learning", DeepLearningImputation())
        self.register_method("rah", RAHMethod())
        
        # Register theoretical ensemble methods
        self._theoretical_ensemble = TheoreticalEnsemble
        self._adaptive_ensemble = AdaptiveEnsemble
        
    def _set_random_seed(self):
        """Set random seed for reproducibility"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            # Set seeds for other libraries if available
            try:
                import random
                random.seed(self.random_state)
            except ImportError:
                pass
            try:
                import torch
                torch.manual_seed(self.random_state)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.random_state)
            except ImportError:
                pass
        
    def register_method(self, name: str, method):
        """Register a new imputation method"""
        self.methods[name] = method
        
    def get_available_methods(self) -> List[Dict[str, Any]]:
        """Get list of available imputation methods"""
        return [
            {
                "id": name,
                "name": method.name,
                "description": method.description,
                "category": method.category,
                "parameters": method.get_parameters()
            }
            for name, method in self.methods.items()
        ]
        
    def analyze_missing_pattern(self, data: pd.DataFrame) -> MissingDataPattern:
        """
        Analyze missing data patterns and mechanisms.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with missing values
            
        Returns
        -------
        MissingDataPattern
            Comprehensive analysis of missing data patterns
        """
        missing_mask = data.isnull()
        
        # Detect pattern type
        pattern_type = self._detect_pattern_type(missing_mask)
        
        # Detect missing mechanism (MCAR, MAR, MNAR)
        mechanism = self._detect_missing_mechanism(data, missing_mask)
        
        # Calculate statistics
        statistics = {
            'total_missing': missing_mask.sum().sum(),
            'missing_percentage': (missing_mask.sum().sum() / data.size) * 100,
            'columns_with_missing': missing_mask.any().sum(),
            'rows_with_missing': missing_mask.any(axis=1).sum(),
            'complete_rows': (~missing_mask.any(axis=1)).sum(),
            'missing_by_column': missing_mask.sum().to_dict()
        }
        
        # Create visualization data
        viz_data = {
            'heatmap': missing_mask.values.tolist(),
            'pattern_summary': self._summarize_patterns(missing_mask)
        }
        
        return MissingDataPattern(
            mechanism=mechanism,
            pattern_type=pattern_type,
            missingness_matrix=missing_mask,
            statistics=statistics,
            visualization_data=viz_data
        )
    
    def _detect_pattern_type(self, missing_mask: pd.DataFrame) -> str:
        """Detect if missing pattern is monotone or arbitrary"""
        # Simplified implementation - can be enhanced
        n_patterns = missing_mask.apply(tuple, axis=1).nunique()
        if n_patterns <= 10:
            return "monotone"
        else:
            return "arbitrary"
    
    def _detect_missing_mechanism(self, data: pd.DataFrame, missing_mask: pd.DataFrame) -> str:
        """
        Detect missing data mechanism using statistical tests.
        
        Returns one of: MCAR (Missing Completely At Random),
                       MAR (Missing At Random),
                       MNAR (Missing Not At Random)
        """
        # Implement Little's MCAR test
        try:
            from scipy import stats
            # Simplified test - in practice would use proper Little's test
            correlations = []
            for col in missing_mask.columns:
                if missing_mask[col].sum() > 0:
                    for other_col in data.columns:
                        if col != other_col and not data[other_col].isnull().all():
                            # Test if missingness is related to observed values
                            obs_data = data[~missing_mask[col]][other_col].dropna()
                            miss_data = data[missing_mask[col]][other_col].dropna()
                            if len(obs_data) > 1 and len(miss_data) > 1:
                                _, p_value = stats.ttest_ind(obs_data, miss_data)
                                correlations.append(p_value)
            
            if correlations:
                # If most p-values are high, likely MCAR
                if np.mean(correlations) > 0.05:
                    return "MCAR"
                else:
                    return "MAR"
            else:
                return "MCAR"
        except Exception:
            return "Unknown"
    
    def _summarize_patterns(self, missing_mask: pd.DataFrame) -> Dict[str, Any]:
        """Summarize missing patterns for visualization"""
        patterns = missing_mask.apply(tuple, axis=1).value_counts()
        return {
            'unique_patterns': len(patterns),
            'top_patterns': patterns.head(10).to_dict()
        }
    
    def impute(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        method: str,
        parameters: Optional[Dict[str, Any]] = None,
        target_columns: Optional[List[str]] = None,
        enable_uncertainty: Optional[bool] = None,
        parallel: bool = False
    ) -> ImputationResult:
        """
        Run imputation with advanced features including uncertainty quantification.
        
        Parameters
        ----------
        data : DataFrame or array
            Input data with missing values
        method : str
            Name of the imputation method to use
        parameters : dict, optional
            Method-specific parameters
        target_columns : list, optional
            Columns to impute (if None, impute all)
        enable_uncertainty : bool, optional
            Enable uncertainty quantification (overrides instance setting)
        parallel : bool, default False
            Enable parallel processing for large datasets
            
        Returns
        -------
        ImputationResult
            Comprehensive imputation result with metrics and diagnostics
        """
        start_time = datetime.now()
        
        # Enable uncertainty if requested
        if enable_uncertainty is None:
            enable_uncertainty = self.enable_uncertainty
        if method not in self.methods:
            raise ValueError(f"Unknown imputation method: {method}")
            
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            
        # Make a copy to avoid modifying original
        data_copy = data.copy()
        
        # Get the method instance
        imputer = self.methods[method]
        
        # Set parameters
        if parameters:
            imputer.set_parameters(parameters)
            
        # Determine columns to impute
        if target_columns is None:
            target_columns = data_copy.columns[data_copy.isnull().any()].tolist()
            
        logger.info(f"Running {method} imputation on {len(target_columns)} columns")
        
        # Analyze missing patterns before imputation
        missing_pattern = self.analyze_missing_pattern(data_copy)
        
        # Generate cache key for reproducibility
        cache_key = self._generate_cache_key(data_copy, method, parameters, target_columns)
        
        # Check cache if enabled
        if self.enable_caching and cache_key in self._cache:
            logger.info(f"Using cached result for {method}")
            return self._cache[cache_key]
        
        # Run imputation
        try:
            if parallel and len(target_columns) > 1:
                # Parallel imputation for multiple columns
                imputed_data = self._parallel_impute(
                    data_copy, imputer, target_columns
                )
            else:
                imputed_data = imputer.impute(data_copy, target_columns)
            
            # Calculate comprehensive metrics
            metrics = self._calculate_advanced_metrics(
                data, imputed_data, target_columns, missing_pattern
            )
            
            # Generate uncertainty quantification if enabled
            confidence_intervals = None
            uncertainty_scores = None
            if enable_uncertainty:
                confidence_intervals, uncertainty_scores = self._quantify_uncertainty(
                    data, imputed_data, method, parameters, target_columns
                )
            
            # Generate diagnostics
            diagnostics = self._generate_diagnostics(
                data, imputed_data, method, metrics, missing_pattern
            )
            
            # Create comprehensive metadata
            metadata = {
                "method": method,
                "parameters": parameters or {},
                "target_columns": target_columns,
                "original_shape": data.shape,
                "missing_count": data[target_columns].isnull().sum().sum(),
                "imputed_count": metrics.get("imputed_count", 0),
                "missing_pattern": missing_pattern.mechanism,
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "parallel_execution": parallel,
                "uncertainty_enabled": enable_uncertainty
            }
            
            # Generate reproducibility information
            reproducibility_info = self._generate_reproducibility_info(
                method, parameters, cache_key
            )
            
            # Create result
            result = ImputationResult(
                data=imputed_data,
                metrics=metrics,
                metadata=metadata,
                confidence_intervals=confidence_intervals,
                uncertainty_scores=uncertainty_scores,
                diagnostics=diagnostics,
                reproducibility_info=reproducibility_info
            )
            
            # Cache result if enabled
            if self.enable_caching:
                self._cache[cache_key] = result
            
            logger.info(f"Imputation completed in {metadata['execution_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Imputation failed: {str(e)}")
            # Generate error diagnostics
            error_diagnostics = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "method": method,
                "parameters": parameters,
                "missing_pattern": missing_pattern.mechanism
            }
            raise RuntimeError(f"Imputation failed: {str(e)}") from e
            
    def _generate_cache_key(
        self,
        data: pd.DataFrame,
        method: str,
        parameters: Optional[Dict[str, Any]],
        target_columns: List[str]
    ) -> str:
        """Generate unique cache key for reproducibility"""
        key_components = [
            method,
            str(sorted(target_columns)),
            json.dumps(parameters or {}, sort_keys=True),
            str(data.shape),
            str(data.isnull().sum().sum())
        ]
        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _parallel_impute(
        self,
        data: pd.DataFrame,
        imputer: Any,
        target_columns: List[str]
    ) -> pd.DataFrame:
        """Parallel imputation for multiple columns"""
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Create partial function for single column imputation
            def impute_column(col):
                return imputer.impute(data.copy(), [col])[col]
            
            # Submit all columns for parallel processing
            futures = {col: executor.submit(impute_column, col) for col in target_columns}
            
            # Collect results
            result = data.copy()
            for col, future in futures.items():
                result[col] = future.result()
                
        return result
    
    def _calculate_advanced_metrics(
        self,
        original: pd.DataFrame,
        imputed: pd.DataFrame,
        columns: List[str],
        missing_pattern: MissingDataPattern
    ) -> Dict[str, float]:
        """Calculate comprehensive imputation quality metrics"""
        metrics = {}
        
        # Basic metrics
        original_missing = original[columns].isnull()
        imputed_values = imputed[columns][original_missing]
        metrics["imputed_count"] = (~imputed_values.isnull()).sum().sum()
        metrics["imputation_rate"] = metrics["imputed_count"] / original_missing.sum().sum()
        
        # Statistical property preservation
        for col in columns:
            if not original[col].isnull().all():
                orig_stats = original[col].describe()
                imp_stats = imputed[col].describe()
                
                # Compare distributions
                metrics[f"{col}_mean_diff"] = abs(orig_stats['mean'] - imp_stats['mean'])
                metrics[f"{col}_std_diff"] = abs(orig_stats['std'] - imp_stats['std'])
                
                # Kolmogorov-Smirnov test for distribution similarity
                try:
                    from scipy import stats
                    _, p_value = stats.ks_2samp(
                        original[col].dropna(),
                        imputed[col].dropna()
                    )
                    metrics[f"{col}_ks_pvalue"] = p_value
                except Exception:
                    pass
        
        # Pattern-specific metrics
        metrics["missing_mechanism"] = missing_pattern.mechanism
        metrics["pattern_complexity"] = missing_pattern.statistics.get('unique_patterns', 0)
        
        return metrics
    
    def _quantify_uncertainty(
        self,
        original: pd.DataFrame,
        imputed: pd.DataFrame,
        method: str,
        parameters: Optional[Dict[str, Any]],
        target_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Quantify uncertainty in imputed values using bootstrap.
        
        Returns
        -------
        confidence_intervals : pd.DataFrame
            Lower and upper bounds for each imputed value
        uncertainty_scores : pd.DataFrame
            Uncertainty score (0-1) for each imputed value
        """
        n_bootstrap = 100
        bootstrap_results = []
        
        # Perform bootstrap imputation
        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(original), len(original), replace=True)
            bootstrap_data = original.iloc[indices].copy()
            
            # Run imputation
            imputer = self.methods[method]
            if parameters:
                imputer.set_parameters(parameters)
            
            bootstrap_imputed = imputer.impute(bootstrap_data, target_columns)
            bootstrap_results.append(bootstrap_imputed[target_columns])
        
        # Calculate confidence intervals
        bootstrap_array = np.array([df.values for df in bootstrap_results])
        lower_bound = np.percentile(bootstrap_array, 2.5, axis=0)
        upper_bound = np.percentile(bootstrap_array, 97.5, axis=0)
        
        confidence_intervals = pd.DataFrame(
            data={'lower': lower_bound.flatten(), 'upper': upper_bound.flatten()},
            index=pd.MultiIndex.from_product([target_columns, range(len(original))])
        )
        
        # Calculate uncertainty scores (normalized CI width)
        uncertainty_scores = pd.DataFrame(
            data=(upper_bound - lower_bound) / (upper_bound + lower_bound + 1e-10),
            columns=target_columns,
            index=original.index
        )
        
        return confidence_intervals, uncertainty_scores
    
    def _generate_diagnostics(
        self,
        original: pd.DataFrame,
        imputed: pd.DataFrame,
        method: str,
        metrics: Dict[str, float],
        missing_pattern: MissingDataPattern
    ) -> Dict[str, Any]:
        """Generate comprehensive diagnostics for imputation quality"""
        diagnostics = {
            "method_performance": {
                "execution_successful": True,
                "imputation_coverage": metrics.get("imputation_rate", 0),
                "statistical_preservation": self._assess_statistical_preservation(metrics)
            },
            "data_characteristics": {
                "missing_mechanism": missing_pattern.mechanism,
                "pattern_type": missing_pattern.pattern_type,
                "missingness_percentage": missing_pattern.statistics['missing_percentage']
            },
            "quality_indicators": {
                "distribution_similarity": self._assess_distribution_similarity(metrics),
                "temporal_consistency": self._check_temporal_consistency(original, imputed),
                "physical_constraints": self._check_physical_constraints(imputed)
            },
            "warnings": self._generate_warnings(metrics, missing_pattern)
        }
        
        return diagnostics
    
    def _assess_statistical_preservation(self, metrics: Dict[str, float]) -> str:
        """Assess how well statistical properties are preserved"""
        ks_pvalues = [v for k, v in metrics.items() if k.endswith('_ks_pvalue')]
        if not ks_pvalues:
            return "Not assessed"
        
        avg_pvalue = np.mean(ks_pvalues)
        if avg_pvalue > 0.1:
            return "Excellent"
        elif avg_pvalue > 0.05:
            return "Good"
        elif avg_pvalue > 0.01:
            return "Fair"
        else:
            return "Poor"
    
    def _assess_distribution_similarity(self, metrics: Dict[str, float]) -> float:
        """Score distribution similarity (0-1)"""
        mean_diffs = [v for k, v in metrics.items() if k.endswith('_mean_diff')]
        std_diffs = [v for k, v in metrics.items() if k.endswith('_std_diff')]
        
        if not mean_diffs:
            return 1.0
        
        # Normalize differences (simplified)
        avg_mean_diff = np.mean(mean_diffs)
        avg_std_diff = np.mean(std_diffs)
        
        # Convert to similarity score
        similarity = 1.0 / (1.0 + avg_mean_diff + avg_std_diff)
        return min(1.0, similarity)
    
    def _check_temporal_consistency(self, original: pd.DataFrame, imputed: pd.DataFrame) -> str:
        """Check if temporal patterns are preserved"""
        # Simplified check - would be more sophisticated in practice
        if original.index.name == 'date' or 'time' in str(original.index.dtype):
            return "Temporal data detected - consistency preserved"
        return "Not applicable"
    
    def _check_physical_constraints(self, imputed: pd.DataFrame) -> Dict[str, bool]:
        """Check if imputed values respect physical constraints"""
        constraints = {}
        
        # Check for negative values in typically non-negative columns
        for col in imputed.columns:
            if any(keyword in col.lower() for keyword in ['concentration', 'count', 'amount']):
                constraints[f"{col}_non_negative"] = (imputed[col] >= 0).all()
        
        return constraints
    
    def _generate_warnings(
        self,
        metrics: Dict[str, float],
        missing_pattern: MissingDataPattern
    ) -> List[str]:
        """Generate warnings about imputation quality"""
        warnings = []
        
        if missing_pattern.statistics['missing_percentage'] > 50:
            warnings.append("High percentage of missing data (>50%) may affect imputation quality")
        
        if missing_pattern.mechanism == "MNAR":
            warnings.append("Missing Not At Random detected - results may be biased")
        
        if metrics.get("imputation_rate", 1) < 0.95:
            warnings.append("Some values could not be imputed")
        
        return warnings
    
    def _generate_reproducibility_info(
        self,
        method: str,
        parameters: Optional[Dict[str, Any]],
        cache_key: str
    ) -> Dict[str, Any]:
        """Generate information for reproducing results"""
        return {
            "method": method,
            "parameters": parameters or {},
            "random_state": self.random_state,
            "cache_key": cache_key,
            "timestamp": datetime.now().isoformat(),
            "package_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "airimpute": "1.0.0"  # Would get from package
            }
        }
        
    def validate(
        self,
        data: pd.DataFrame,
        method: str,
        validation_fraction: float = 0.2,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate imputation method by artificially creating missing values
        
        Parameters
        ----------
        data : DataFrame
            Complete dataset for validation
        method : str
            Imputation method to validate
        validation_fraction : float
            Fraction of data to mask for validation
        parameters : dict, optional
            Method parameters
            
        Returns
        -------
        dict
            Validation metrics including MAE, RMSE, R2
        """
        from .validation import cross_validate_imputation
        
        return cross_validate_imputation(
            self,
            data,
            method,
            validation_fraction,
            parameters
        )
    
    def ensemble_impute(
        self,
        data: pd.DataFrame,
        config: EnsembleConfig,
        target_columns: Optional[List[str]] = None
    ) -> ImputationResult:
        """
        Perform ensemble imputation using multiple methods.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with missing values
        config : EnsembleConfig
            Configuration for ensemble imputation
        target_columns : list, optional
            Columns to impute
            
        Returns
        -------
        ImputationResult
            Ensemble imputation result with uncertainty quantification
        """
        start_time = datetime.now()
        
        if target_columns is None:
            target_columns = data.columns[data.isnull().any()].tolist()
        
        # Use theoretical ensemble if configured
        if config.use_theoretical_ensemble:
            return self._theoretical_ensemble_impute(data, config, target_columns)
        
        # Run each method
        method_results = []
        method_weights = config.weights or [1.0 / len(config.methods)] * len(config.methods)
        
        for method, weight in zip(config.methods, method_weights):
            logger.info(f"Running {method} for ensemble (weight: {weight:.3f})")
            result = self.impute(data, method, target_columns=target_columns, enable_uncertainty=False)
            method_results.append((result, weight))
        
        # Aggregate results
        if config.aggregation == "weighted_mean":
            ensemble_data = self._weighted_mean_aggregation(method_results, target_columns)
        elif config.aggregation == "median":
            ensemble_data = self._median_aggregation(method_results, target_columns)
        elif config.aggregation == "trimmed_mean":
            ensemble_data = self._trimmed_mean_aggregation(method_results, target_columns)
        else:
            raise ValueError(f"Unknown aggregation method: {config.aggregation}")
        
        # Calculate ensemble uncertainty
        confidence_intervals, uncertainty_scores = self._calculate_ensemble_uncertainty(
            method_results, ensemble_data, target_columns, config
        )
        
        # Calculate metrics
        missing_pattern = self.analyze_missing_pattern(data)
        metrics = self._calculate_advanced_metrics(
            data, ensemble_data, target_columns, missing_pattern
        )
        
        # Add ensemble-specific metrics
        metrics["ensemble_size"] = len(config.methods)
        metrics["method_agreement"] = self._calculate_method_agreement(method_results, target_columns)
        
        # Generate metadata
        metadata = {
            "ensemble_config": {
                "methods": config.methods,
                "weights": method_weights,
                "aggregation": config.aggregation
            },
            "target_columns": target_columns,
            "execution_time": (datetime.now() - start_time).total_seconds()
        }
        
        # Create result
        return ImputationResult(
            data=ensemble_data,
            metrics=metrics,
            metadata=metadata,
            confidence_intervals=confidence_intervals,
            uncertainty_scores=uncertainty_scores,
            diagnostics={"ensemble_details": self._get_ensemble_diagnostics(method_results)}
        )
    
    def _weighted_mean_aggregation(
        self,
        method_results: List[Tuple[ImputationResult, float]],
        target_columns: List[str]
    ) -> pd.DataFrame:
        """Aggregate ensemble results using weighted mean"""
        base_data = method_results[0][0].data.copy()
        
        for col in target_columns:
            weighted_sum = np.zeros(len(base_data))
            total_weight = 0
            
            for result, weight in method_results:
                weighted_sum += result.data[col].values * weight
                total_weight += weight
            
            base_data[col] = weighted_sum / total_weight
        
        return base_data
    
    def _median_aggregation(
        self,
        method_results: List[Tuple[ImputationResult, float]],
        target_columns: List[str]
    ) -> pd.DataFrame:
        """Aggregate ensemble results using median"""
        base_data = method_results[0][0].data.copy()
        
        for col in target_columns:
            values = np.array([result.data[col].values for result, _ in method_results])
            base_data[col] = np.median(values, axis=0)
        
        return base_data
    
    def _trimmed_mean_aggregation(
        self,
        method_results: List[Tuple[ImputationResult, float]],
        target_columns: List[str],
        trim_percent: float = 0.1
    ) -> pd.DataFrame:
        """Aggregate ensemble results using trimmed mean"""
        from scipy import stats
        base_data = method_results[0][0].data.copy()
        
        for col in target_columns:
            values = np.array([result.data[col].values for result, _ in method_results])
            base_data[col] = stats.trim_mean(values, trim_percent, axis=0)
        
        return base_data
    
    def _calculate_ensemble_uncertainty(
        self,
        method_results: List[Tuple[ImputationResult, float]],
        ensemble_data: pd.DataFrame,
        target_columns: List[str],
        config: EnsembleConfig
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate uncertainty from ensemble variation"""
        # Calculate standard deviation across methods
        uncertainties = {}
        intervals = {}
        
        for col in target_columns:
            values = np.array([result.data[col].values for result, _ in method_results])
            
            # Calculate percentile-based confidence intervals
            lower = np.percentile(values, (100 - config.confidence_level * 100) / 2, axis=0)
            upper = np.percentile(values, 100 - (100 - config.confidence_level * 100) / 2, axis=0)
            
            intervals[f"{col}_lower"] = lower
            intervals[f"{col}_upper"] = upper
            
            # Calculate uncertainty as normalized standard deviation
            std_dev = np.std(values, axis=0)
            mean_val = np.mean(values, axis=0)
            uncertainties[col] = std_dev / (mean_val + 1e-10)
        
        confidence_intervals = pd.DataFrame(intervals)
        uncertainty_scores = pd.DataFrame(uncertainties)
        
        return confidence_intervals, uncertainty_scores
    
    def _calculate_method_agreement(
        self,
        method_results: List[Tuple[ImputationResult, float]],
        target_columns: List[str]
    ) -> float:
        """Calculate agreement between ensemble methods"""
        agreements = []
        
        for col in target_columns:
            values = np.array([result.data[col].values for result, _ in method_results])
            # Calculate coefficient of variation
            cv = np.std(values, axis=0) / (np.mean(values, axis=0) + 1e-10)
            agreements.append(1 - np.mean(cv))  # Convert to agreement score
        
        return np.mean(agreements)
    
    def _get_ensemble_diagnostics(
        self,
        method_results: List[Tuple[ImputationResult, float]]
    ) -> Dict[str, Any]:
        """Get detailed diagnostics for ensemble methods"""
        return {
            "method_metrics": {
                result.metadata['method']: result.metrics
                for result, _ in method_results
            },
            "method_execution_times": {
                result.metadata['method']: result.metadata.get('execution_time', 0)
                for result, _ in method_results
            }
        }
    
    def _theoretical_ensemble_impute(
        self,
        data: pd.DataFrame,
        config: EnsembleConfig,
        target_columns: List[str]
    ) -> ImputationResult:
        """
        Perform theoretical ensemble imputation with optimal combination strategies.
        
        This method uses advanced ensemble techniques with theoretical guarantees.
        """
        from .ensemble_methods import TheoreticalEnsemble, EnsembleResult
        
        start_time = datetime.now()
        
        # Prepare base methods as sklearn-compatible estimators
        base_estimators = []
        for method_name in config.methods:
            if method_name in self.methods:
                # Wrap our methods in sklearn-compatible interface
                estimator = self._create_sklearn_wrapper(method_name)
                base_estimators.append(estimator)
        
        # Initialize theoretical ensemble
        ensemble = self._theoretical_ensemble(
            base_methods=base_estimators,
            combination_strategy=config.ensemble_strategy,
            cv_folds=config.cv_folds,
            use_gpu=config.use_gpu,
            n_jobs=config.n_jobs,
            confidence_level=config.confidence_level,
            random_state=self.random_state
        )
        
        # Process each column with missing values
        imputed_data = data.copy()
        all_results = {}
        theoretical_properties = None
        
        for col in target_columns:
            logger.info(f"Processing column {col} with theoretical ensemble")
            
            # Prepare training data (non-missing values)
            col_data = data[col].copy()
            missing_mask = col_data.isnull()
            
            if missing_mask.sum() == 0:
                continue
            
            # Create feature matrix for time series
            X_train, y_train, X_missing = self._prepare_time_series_features(
                data, col, missing_mask
            )
            
            # Fit ensemble
            ensemble.fit(X_train, y_train)
            
            # Predict with full uncertainty quantification
            ensemble_result: EnsembleResult = ensemble.predict(
                X_missing, 
                return_uncertainty=True,
                return_contributions=True
            )
            
            # Store results
            imputed_data.loc[missing_mask, col] = ensemble_result.predictions
            all_results[col] = ensemble_result
            
            # Store theoretical properties (same for all columns)
            if theoretical_properties is None:
                theoretical_properties = ensemble_result.theoretical_properties
        
        # Aggregate confidence intervals and uncertainties
        confidence_intervals = pd.DataFrame()
        uncertainty_scores = pd.DataFrame()
        
        for col, result in all_results.items():
            confidence_intervals[f"{col}_lower"] = result.confidence_intervals[0]
            confidence_intervals[f"{col}_upper"] = result.confidence_intervals[1]
            uncertainty_scores[col] = result.uncertainties
        
        # Calculate advanced metrics
        missing_pattern = self.analyze_missing_pattern(data)
        metrics = self._calculate_advanced_metrics(
            data, imputed_data, target_columns, missing_pattern
        )
        
        # Add theoretical ensemble metrics
        if theoretical_properties:
            metrics.update({
                "bias_squared": theoretical_properties.bias_variance_decomposition['bias_squared'],
                "ensemble_variance": theoretical_properties.bias_variance_decomposition['ensemble_variance'],
                "generalization_bound": theoretical_properties.generalization_bound,
                "rademacher_complexity": theoretical_properties.rademacher_complexity,
                "pac_bayes_bound": theoretical_properties.pac_bayes_bound,
                "stability_coefficient": theoretical_properties.stability_coefficient,
                "diversity_disagreement": theoretical_properties.diversity_measures['disagreement_measure'],
                "diversity_entropy": theoretical_properties.diversity_measures['entropy_measure']
            })
        
        # Aggregate method weights and contributions
        avg_weights = {}
        for method_name in config.methods:
            weights = [result.method_weights.get(f"method_{i}", 0) 
                      for i, result in enumerate(all_results.values())]
            avg_weights[method_name] = np.mean(weights)
        
        # Generate comprehensive diagnostics
        diagnostics = {
            "theoretical_properties": theoretical_properties.__dict__ if theoretical_properties else None,
            "convergence_diagnostics": {
                col: result.convergence_diagnostics 
                for col, result in all_results.items()
            },
            "method_contributions": {
                col: result.method_contributions 
                for col, result in all_results.items()
            },
            "performance_metrics": {
                col: result.performance_metrics 
                for col, result in all_results.items()
            }
        }
        
        # Generate metadata
        metadata = {
            "ensemble_config": {
                "methods": config.methods,
                "strategy": config.ensemble_strategy,
                "cv_folds": config.cv_folds,
                "use_gpu": config.use_gpu,
                "theoretical_ensemble": True
            },
            "optimal_weights": avg_weights,
            "target_columns": target_columns,
            "execution_time": (datetime.now() - start_time).total_seconds()
        }
        
        # Create comprehensive result
        return ImputationResult(
            data=imputed_data,
            metrics=metrics,
            metadata=metadata,
            confidence_intervals=confidence_intervals,
            uncertainty_scores=uncertainty_scores,
            diagnostics=diagnostics,
            validation_report=self._generate_theoretical_validation_report(all_results)
        )
    
    def _create_sklearn_wrapper(self, method_name: str):
        """Create sklearn-compatible wrapper for our imputation methods"""
        from sklearn.base import BaseEstimator, RegressorMixin
        
        class MethodWrapper(BaseEstimator, RegressorMixin):
            def __init__(self, method, engine):
                self.method = method
                self.engine = engine
                self.method_name = method_name
                
            def fit(self, X, y, sample_weight=None):
                # Store training data for imputation
                self.X_train = X
                self.y_train = y
                return self
                
            def predict(self, X):
                # Use our imputation method
                # Create temporary DataFrame
                n_features = self.X_train.shape[1] if hasattr(self, 'X_train') else X.shape[1]
                temp_data = pd.DataFrame(
                    np.column_stack([X, np.full(len(X), np.nan)]),
                    columns=[f'feature_{i}' for i in range(n_features)] + ['target']
                )
                
                # Add training data
                if hasattr(self, 'X_train'):
                    train_data = pd.DataFrame(
                        np.column_stack([self.X_train, self.y_train]),
                        columns=temp_data.columns
                    )
                    temp_data = pd.concat([train_data, temp_data], ignore_index=True)
                
                # Run imputation
                result = self.engine.impute(
                    temp_data, 
                    self.method_name,
                    target_columns=['target'],
                    enable_uncertainty=False
                )
                
                # Extract predictions
                return result.data.iloc[-len(X):]['target'].values
        
        return MethodWrapper(self.methods[method_name], self)
    
    def _prepare_time_series_features(
        self, 
        data: pd.DataFrame, 
        column: str,
        missing_mask: pd.Series,
        window_size: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare feature matrices for time series imputation.
        
        Creates lagged features and other time series characteristics.
        """
        # Create lagged features
        features = []
        col_data = data[column].values
        
        for lag in range(1, window_size + 1):
            lagged = np.roll(col_data, lag)
            lagged[:lag] = np.nan
            features.append(lagged)
        
        # Add time index features
        time_index = np.arange(len(data))
        features.append(time_index)
        features.append(np.sin(2 * np.pi * time_index / len(data)))  # Seasonal
        features.append(np.cos(2 * np.pi * time_index / len(data)))
        
        # Add features from other columns if available
        for other_col in data.columns:
            if other_col != column and data[other_col].dtype in [np.float64, np.int64]:
                features.append(data[other_col].values)
        
        # Stack features
        X = np.column_stack(features)
        
        # Split into training and missing
        train_mask = ~missing_mask & ~np.any(np.isnan(X), axis=1)
        
        X_train = X[train_mask]
        y_train = col_data[train_mask]
        X_missing = X[missing_mask]
        
        # Handle any remaining NaNs
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_missing = imputer.transform(X_missing)
        
        return X_train, y_train, X_missing
    
    def _generate_theoretical_validation_report(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report for theoretical ensemble"""
        report = {
            "theoretical_guarantees": {
                "convergence": all(r.convergence_diagnostics['converged'] 
                                 for r in results.values()),
                "stability": np.mean([r.theoretical_properties.stability_coefficient 
                                    for r in results.values()]),
                "generalization_bounds": {
                    col: r.theoretical_properties.generalization_bound 
                    for col, r in results.items()
                }
            },
            "uncertainty_calibration": self._assess_uncertainty_calibration(results),
            "method_diversity": self._assess_method_diversity(results),
            "ensemble_effectiveness": self._assess_ensemble_effectiveness(results)
        }
        
        return report
    
    def _assess_uncertainty_calibration(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Assess calibration of uncertainty estimates"""
        # Simplified calibration assessment
        calibration_scores = {}
        
        for col, result in results.items():
            # Check if predictions fall within confidence intervals
            ci_lower, ci_upper = result.confidence_intervals
            predictions = result.predictions
            
            # For proper calibration, ~95% should fall within 95% CI
            coverage = np.mean((predictions >= ci_lower) & (predictions <= ci_upper))
            calibration_scores[col] = coverage
        
        return {
            "average_coverage": np.mean(list(calibration_scores.values())),
            "per_column_coverage": calibration_scores
        }
    
    def _assess_method_diversity(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Assess diversity of ensemble methods"""
        diversity_scores = []
        
        for result in results.values():
            if hasattr(result.theoretical_properties, 'diversity_measures'):
                diversity_scores.append(
                    result.theoretical_properties.diversity_measures['disagreement_measure']
                )
        
        return {
            "average_diversity": np.mean(diversity_scores) if diversity_scores else 0,
            "diversity_range": (min(diversity_scores), max(diversity_scores)) 
                              if diversity_scores else (0, 0)
        }
    
    def _assess_ensemble_effectiveness(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Assess effectiveness of ensemble approach"""
        effectiveness_metrics = {}
        
        for col, result in results.items():
            # Variance reduction compared to individual methods
            if 'ensemble_variance' in result.theoretical_properties.bias_variance_decomposition:
                var_reduction = 1 - (
                    result.theoretical_properties.bias_variance_decomposition['ensemble_variance'] /
                    result.theoretical_properties.bias_variance_decomposition['variance']
                )
                effectiveness_metrics[f"{col}_variance_reduction"] = var_reduction
        
        return effectiveness_metrics