"""
Advanced Benchmarking System with Statistical Significance Testing
Implements comprehensive performance evaluation with rigorous statistical analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import psutil
import GPUtil
import torch
import warnings
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tracemalloc
import cProfile
import pstats
import io
from contextlib import contextmanager
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    # Data splitting
    n_splits: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1
    
    # Cross-validation
    cv_strategy: str = "time_series"  # "time_series", "random", "blocked"
    gap_size: int = 0  # Gap between train and test in time series
    
    # Missing data simulation
    missing_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    missing_patterns: List[str] = field(default_factory=lambda: ["MCAR", "MAR", "MNAR"])
    gap_sizes: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 24])
    
    # Performance metrics
    metrics: List[str] = field(default_factory=lambda: [
        "rmse", "mae", "mape", "smape", "r2", "mase", "coverage", "sharpness"
    ])
    
    # Statistical tests
    significance_level: float = 0.05
    multiple_testing_correction: str = "bonferroni"  # "bonferroni", "holm", "fdr"
    
    # Computational resources
    n_jobs: int = -1  # -1 for all CPUs
    memory_limit: Optional[int] = None  # Bytes
    time_limit: Optional[float] = None  # Seconds per method
    
    # Profiling
    enable_profiling: bool = True
    profile_memory: bool = True
    profile_cpu: bool = True
    profile_gpu: bool = True
    
    # Output
    save_intermediate: bool = True
    output_dir: Path = Path("benchmark_results")
    random_seed: int = 42


@dataclass
class MethodResult:
    """Results for a single method on a single dataset"""
    method_name: str
    dataset_id: str
    fold: int
    
    # Performance metrics
    metrics: Dict[str, float]
    
    # Computational resources
    execution_time: float
    peak_memory: int
    cpu_usage: float
    gpu_memory: Optional[int] = None
    
    # Profiling data
    cpu_profile: Optional[str] = None
    memory_profile: Optional[Dict[str, Any]] = None
    
    # Error information
    success: bool = True
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Predictions for further analysis
    predictions: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark results"""
    config: BenchmarkConfig
    start_time: datetime
    end_time: datetime
    
    # Results by method and dataset
    results: Dict[str, Dict[str, List[MethodResult]]]
    
    # Statistical analysis
    statistical_tests: Dict[str, Any]
    pairwise_comparisons: Dict[str, Any]
    
    # Rankings
    rankings: Dict[str, pd.DataFrame]
    
    # Computational analysis
    computational_summary: Dict[str, Any]
    
    # Metadata
    system_info: Dict[str, Any]
    dataset_info: Dict[str, Any]


class PerformanceMetrics:
    """Comprehensive performance metrics for imputation evaluation"""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        mask = denominator != 0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, 
             y_train: Optional[np.ndarray] = None) -> float:
        """Mean Absolute Scaled Error"""
        if y_train is None:
            # Use naive forecast (previous value)
            naive_errors = np.abs(np.diff(y_true))
            scale = np.mean(naive_errors)
        else:
            naive_errors = np.abs(np.diff(y_train))
            scale = np.mean(naive_errors)
        
        if scale == 0:
            return np.inf
        
        return np.mean(np.abs(y_true - y_pred)) / scale
    
    @staticmethod
    def coverage(y_true: np.ndarray, y_pred: np.ndarray,
                 lower: Optional[np.ndarray] = None,
                 upper: Optional[np.ndarray] = None,
                 confidence: float = 0.95) -> float:
        """Coverage of prediction intervals"""
        if lower is None or upper is None:
            # Assume normal distribution
            std = np.std(y_true - y_pred)
            z_score = stats.norm.ppf((1 + confidence) / 2)
            lower = y_pred - z_score * std
            upper = y_pred + z_score * std
        
        within_interval = (y_true >= lower) & (y_true <= upper)
        return np.mean(within_interval)
    
    @staticmethod
    def sharpness(lower: np.ndarray, upper: np.ndarray) -> float:
        """Average width of prediction intervals"""
        return np.mean(upper - lower)
    
    @staticmethod
    def interval_score(y_true: np.ndarray, y_pred: np.ndarray,
                      lower: np.ndarray, upper: np.ndarray,
                      alpha: float = 0.05) -> float:
        """Interval score for probabilistic predictions"""
        width = upper - lower
        lower_penalty = 2 / alpha * (lower - y_true) * (y_true < lower)
        upper_penalty = 2 / alpha * (y_true - upper) * (y_true > upper)
        return np.mean(width + lower_penalty + upper_penalty)


class MissingDataSimulator:
    """Simulate realistic missing data patterns"""
    
    def __init__(self, random_state: Optional[int] = None):
        self.rng = np.random.RandomState(random_state)
    
    def generate_missing_mask(self, 
                            shape: Tuple[int, ...],
                            missing_rate: float,
                            pattern: str = "MCAR",
                            **kwargs) -> np.ndarray:
        """
        Generate missing data mask
        
        Parameters:
        -----------
        shape : tuple
            Shape of data
        missing_rate : float
            Proportion of missing values
        pattern : str
            Missing pattern: "MCAR", "MAR", "MNAR", "temporal", "spatial"
        **kwargs : dict
            Pattern-specific parameters
        
        Returns:
        --------
        np.ndarray
            Boolean mask (True = missing)
        """
        if pattern == "MCAR":
            return self._generate_mcar(shape, missing_rate)
        elif pattern == "MAR":
            return self._generate_mar(shape, missing_rate, **kwargs)
        elif pattern == "MNAR":
            return self._generate_mnar(shape, missing_rate, **kwargs)
        elif pattern == "temporal":
            return self._generate_temporal(shape, missing_rate, **kwargs)
        elif pattern == "spatial":
            return self._generate_spatial(shape, missing_rate, **kwargs)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    def _generate_mcar(self, shape: Tuple[int, ...], rate: float) -> np.ndarray:
        """Missing Completely At Random"""
        return self.rng.random(shape) < rate
    
    def _generate_mar(self, shape: Tuple[int, ...], rate: float,
                     dependency_strength: float = 0.7) -> np.ndarray:
        """Missing At Random - depends on observed variables"""
        mask = np.zeros(shape, dtype=bool)
        
        # Generate dependency variable
        dependency = self.rng.randn(*shape)
        
        # Make missingness depend on dependency variable
        threshold = np.percentile(dependency, (1 - rate) * 100)
        base_mask = dependency > threshold
        
        # Add some randomness
        random_mask = self.rng.random(shape) < (1 - dependency_strength) * rate
        
        mask = base_mask | random_mask
        
        # Ensure correct missing rate
        current_rate = mask.mean()
        if current_rate > 0:
            mask = mask & (self.rng.random(shape) < rate / current_rate)
        
        return mask
    
    def _generate_mnar(self, shape: Tuple[int, ...], rate: float,
                      threshold_percentile: float = 80) -> np.ndarray:
        """Missing Not At Random - depends on unobserved values"""
        # This requires the actual data values
        # For simulation, we'll create a pattern where high values are more likely missing
        mask = np.zeros(shape, dtype=bool)
        
        # Simulate that high values are more likely to be missing
        synthetic_data = self.rng.randn(*shape)
        threshold = np.percentile(synthetic_data, threshold_percentile)
        
        # High values have higher missing probability
        high_value_mask = synthetic_data > threshold
        high_value_missing_rate = min(0.8, rate * 3)
        mask[high_value_mask] = self.rng.random(np.sum(high_value_mask)) < high_value_missing_rate
        
        # Low values have lower missing probability
        low_value_mask = ~high_value_mask
        remaining_rate = (rate * np.prod(shape) - np.sum(mask)) / np.sum(low_value_mask)
        remaining_rate = max(0, min(1, remaining_rate))
        mask[low_value_mask] = self.rng.random(np.sum(low_value_mask)) < remaining_rate
        
        return mask
    
    def _generate_temporal(self, shape: Tuple[int, ...], rate: float,
                          gap_size_mean: int = 6,
                          gap_size_std: int = 3) -> np.ndarray:
        """Temporal gaps pattern"""
        mask = np.zeros(shape, dtype=bool)
        n_samples = shape[0]
        
        # Calculate number of gaps needed
        avg_gap_size = gap_size_mean
        n_gaps = int(n_samples * rate / avg_gap_size)
        
        for _ in range(n_gaps):
            # Random gap size
            gap_size = max(1, int(self.rng.normal(gap_size_mean, gap_size_std)))
            
            # Random start position
            max_start = n_samples - gap_size
            if max_start > 0:
                start = self.rng.randint(0, max_start)
                mask[start:start + gap_size] = True
        
        return mask
    
    def _generate_spatial(self, shape: Tuple[int, ...], rate: float,
                         correlation_length: float = 0.3) -> np.ndarray:
        """Spatial correlation pattern for multivariate data"""
        if len(shape) < 2:
            return self._generate_mcar(shape, rate)
        
        mask = np.zeros(shape, dtype=bool)
        n_features = shape[1] if len(shape) > 1 else 1
        
        # Generate correlated missing patterns across features
        for i in range(shape[0]):
            if self.rng.random() < rate:
                # Make nearby features also missing
                center_feature = self.rng.randint(0, n_features)
                for j in range(n_features):
                    distance = abs(j - center_feature) / n_features
                    missing_prob = rate * np.exp(-distance / correlation_length)
                    if self.rng.random() < missing_prob:
                        mask[i, j] = True
        
        return mask


class StatisticalTester:
    """Rigorous statistical testing for benchmark results"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def compare_methods(self, 
                       results: Dict[str, List[float]],
                       correction: str = "bonferroni") -> Dict[str, Any]:
        """
        Compare multiple methods with appropriate statistical tests
        
        Parameters:
        -----------
        results : dict
            Method name -> list of performance scores
        correction : str
            Multiple testing correction method
        
        Returns:
        --------
        dict
            Statistical test results
        """
        method_names = list(results.keys())
        n_methods = len(method_names)
        
        if n_methods < 2:
            return {"error": "Need at least 2 methods to compare"}
        
        # Prepare data matrix
        data_matrix = []
        max_len = max(len(scores) for scores in results.values())
        
        for method in method_names:
            scores = results[method]
            if len(scores) < max_len:
                # Pad with NaN if necessary
                scores = scores + [np.nan] * (max_len - len(scores))
            data_matrix.append(scores)
        
        data_matrix = np.array(data_matrix).T
        
        # Remove rows with any NaN
        valid_mask = ~np.any(np.isnan(data_matrix), axis=1)
        data_matrix = data_matrix[valid_mask]
        
        analysis = {
            "n_methods": n_methods,
            "n_samples": len(data_matrix),
            "method_names": method_names
        }
        
        # Normality test
        normality_results = self._test_normality(data_matrix, method_names)
        analysis["normality"] = normality_results
        
        # Choose appropriate test based on normality
        if normality_results["all_normal"]:
            # Use parametric tests
            if n_methods == 2:
                analysis["omnibus"] = self._paired_t_test(
                    data_matrix[:, 0], data_matrix[:, 1]
                )
            else:
                analysis["omnibus"] = self._repeated_measures_anova(
                    data_matrix, method_names
                )
        else:
            # Use non-parametric tests
            if n_methods == 2:
                analysis["omnibus"] = self._wilcoxon_test(
                    data_matrix[:, 0], data_matrix[:, 1]
                )
            else:
                analysis["omnibus"] = self._friedman_test(
                    data_matrix, method_names
                )
        
        # Post-hoc pairwise comparisons if omnibus is significant
        if analysis["omnibus"]["p_value"] < self.alpha:
            analysis["pairwise"] = self._pairwise_comparisons(
                data_matrix, method_names, 
                parametric=normality_results["all_normal"],
                correction=correction
            )
        else:
            analysis["pairwise"] = {
                "performed": False,
                "reason": "Omnibus test not significant"
            }
        
        # Effect sizes
        analysis["effect_sizes"] = self._calculate_effect_sizes(
            data_matrix, method_names
        )
        
        # Rankings
        analysis["rankings"] = self._calculate_rankings(data_matrix, method_names)
        
        return analysis
    
    def _test_normality(self, data: np.ndarray, 
                       method_names: List[str]) -> Dict[str, Any]:
        """Test normality for each method"""
        results = {
            "shapiro_wilk": {},
            "all_normal": True
        }
        
        for i, method in enumerate(method_names):
            if len(data[:, i]) < 3:
                results["shapiro_wilk"][method] = {
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "normal": False
                }
                results["all_normal"] = False
            else:
                stat, p_value = stats.shapiro(data[:, i])
                is_normal = p_value > self.alpha
                
                results["shapiro_wilk"][method] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "normal": is_normal
                }
                
                if not is_normal:
                    results["all_normal"] = False
        
        return results
    
    def _paired_t_test(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Paired t-test for two methods"""
        stat, p_value = stats.ttest_rel(x, y)
        
        # Calculate effect size (Cohen's d)
        diff = x - y
        d = np.mean(diff) / np.std(diff, ddof=1)
        
        return {
            "test": "Paired t-test",
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "effect_size": d,
            "interpretation": self._interpret_cohens_d(d)
        }
    
    def _wilcoxon_test(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Wilcoxon signed-rank test for two methods"""
        stat, p_value = wilcoxon(x, y)
        
        # Calculate effect size (r = Z / sqrt(N))
        n = len(x)
        z = stats.norm.ppf(1 - p_value / 2)  # Approximate Z-score
        r = z / np.sqrt(n)
        
        return {
            "test": "Wilcoxon signed-rank test",
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "effect_size": r,
            "interpretation": self._interpret_effect_size_r(r)
        }
    
    def _repeated_measures_anova(self, data: np.ndarray, 
                                method_names: List[str]) -> Dict[str, Any]:
        """Repeated measures ANOVA for multiple methods"""
        # Simplified version - for full implementation use statsmodels
        # This uses one-way ANOVA as approximation
        f_stat, p_value = stats.f_oneway(*[data[:, i] for i in range(data.shape[1])])
        
        # Calculate eta squared
        ss_between = np.sum([len(data) * (np.mean(data[:, i]) - np.mean(data))**2 
                            for i in range(data.shape[1])])
        ss_total = np.sum((data - np.mean(data))**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            "test": "One-way ANOVA (repeated measures approximation)",
            "statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "effect_size": eta_squared,
            "interpretation": self._interpret_eta_squared(eta_squared)
        }
    
    def _friedman_test(self, data: np.ndarray, 
                      method_names: List[str]) -> Dict[str, Any]:
        """Friedman test for multiple methods"""
        stat, p_value = friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])
        
        # Calculate Kendall's W (concordance coefficient)
        n_samples, n_methods = data.shape
        ranks = np.array([rankdata(row) for row in data])
        mean_ranks = np.mean(ranks, axis=0)
        
        ss_total = n_samples * np.sum((mean_ranks - np.mean(mean_ranks))**2)
        w = 12 * ss_total / (n_samples**2 * (n_methods**3 - n_methods))
        
        return {
            "test": "Friedman test",
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "kendalls_w": w,
            "interpretation": self._interpret_kendalls_w(w)
        }
    
    def _pairwise_comparisons(self, data: np.ndarray, method_names: List[str],
                             parametric: bool = True,
                             correction: str = "bonferroni") -> Dict[str, Any]:
        """Perform pairwise comparisons with correction"""
        n_methods = len(method_names)
        n_comparisons = n_methods * (n_methods - 1) // 2
        
        comparisons = []
        p_values = []
        
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                if parametric:
                    stat, p_val = stats.ttest_rel(data[:, i], data[:, j])
                    test_name = "Paired t-test"
                else:
                    stat, p_val = wilcoxon(data[:, i], data[:, j])
                    test_name = "Wilcoxon signed-rank"
                
                comparisons.append({
                    "method1": method_names[i],
                    "method2": method_names[j],
                    "test": test_name,
                    "statistic": stat,
                    "p_value": p_val
                })
                p_values.append(p_val)
        
        # Apply multiple testing correction
        corrected_p_values = self._correct_p_values(p_values, correction)
        
        # Update results with corrected p-values
        for comp, corrected_p in zip(comparisons, corrected_p_values):
            comp["p_value_corrected"] = corrected_p
            comp["significant"] = corrected_p < self.alpha
        
        return {
            "comparisons": comparisons,
            "correction_method": correction,
            "n_comparisons": n_comparisons
        }
    
    def _correct_p_values(self, p_values: List[float], 
                         method: str = "bonferroni") -> List[float]:
        """Apply multiple testing correction"""
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == "bonferroni":
            return np.minimum(p_values * n, 1.0)
        
        elif method == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected = np.zeros_like(p_values)
            for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
                corrected[idx] = min(p * (n - i), 1.0)
            
            # Ensure monotonicity
            for i in range(1, n):
                if corrected[sorted_indices[i]] < corrected[sorted_indices[i-1]]:
                    corrected[sorted_indices[i]] = corrected[sorted_indices[i-1]]
            
            return corrected
        
        elif method == "fdr":
            # Benjamini-Hochberg FDR
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected = np.zeros_like(p_values)
            for i in range(n):
                corrected[sorted_indices[i]] = min(
                    sorted_p[i] * n / (i + 1), 1.0
                )
            
            # Ensure monotonicity
            for i in range(n - 2, -1, -1):
                if corrected[sorted_indices[i]] > corrected[sorted_indices[i+1]]:
                    corrected[sorted_indices[i]] = corrected[sorted_indices[i+1]]
            
            return corrected
        
        else:
            raise ValueError(f"Unknown correction method: {method}")
    
    def _calculate_effect_sizes(self, data: np.ndarray, 
                               method_names: List[str]) -> Dict[str, Any]:
        """Calculate various effect size measures"""
        effect_sizes = {}
        
        # Mean performance
        means = {method: np.mean(data[:, i]) 
                for i, method in enumerate(method_names)}
        
        # Standard deviations
        stds = {method: np.std(data[:, i], ddof=1) 
               for i, method in enumerate(method_names)}
        
        effect_sizes["summary"] = {
            "means": means,
            "stds": stds
        }
        
        # Pairwise effect sizes
        pairwise_effects = {}
        
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                method1, method2 = method_names[i], method_names[j]
                
                # Cohen's d
                mean_diff = means[method1] - means[method2]
                pooled_std = np.sqrt((stds[method1]**2 + stds[method2]**2) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                # Probability of superiority
                prob_superiority = np.mean(data[:, i] > data[:, j])
                
                pairwise_effects[f"{method1}_vs_{method2}"] = {
                    "cohens_d": cohens_d,
                    "interpretation": self._interpret_cohens_d(cohens_d),
                    "probability_of_superiority": prob_superiority
                }
        
        effect_sizes["pairwise"] = pairwise_effects
        
        return effect_sizes
    
    def _calculate_rankings(self, data: np.ndarray, 
                           method_names: List[str]) -> Dict[str, Any]:
        """Calculate method rankings"""
        # Rank methods for each sample (lower is better)
        ranks = np.array([rankdata(row) for row in data])
        
        # Average ranks
        avg_ranks = {method: np.mean(ranks[:, i]) 
                    for i, method in enumerate(method_names)}
        
        # Rank distribution
        rank_distribution = {}
        for i, method in enumerate(method_names):
            rank_counts = np.bincount(ranks[:, i].astype(int), 
                                     minlength=len(method_names) + 1)[1:]
            rank_distribution[method] = {
                f"rank_{r}": count 
                for r, count in enumerate(rank_counts, 1)
            }
        
        # Critical difference for Nemenyi test
        n_methods = len(method_names)
        n_samples = len(data)
        
        q_alpha = self._get_nemenyi_critical_value(n_methods, self.alpha)
        cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_samples))
        
        return {
            "average_ranks": avg_ranks,
            "rank_distribution": rank_distribution,
            "critical_difference": cd,
            "interpretation": f"Methods with rank difference > {cd:.3f} are significantly different"
        }
    
    def _get_nemenyi_critical_value(self, k: int, alpha: float) -> float:
        """Get critical value for Nemenyi test"""
        # Simplified - use pre-computed values
        # For full implementation, use statistical tables
        critical_values = {
            (2, 0.05): 1.960,
            (3, 0.05): 2.343,
            (4, 0.05): 2.569,
            (5, 0.05): 2.728,
            (6, 0.05): 2.850,
            (7, 0.05): 2.949,
            (8, 0.05): 3.031,
            (9, 0.05): 3.102,
            (10, 0.05): 3.164
        }
        
        return critical_values.get((k, alpha), 2.576)  # Default to normal approximation
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_effect_size_r(self, r: float) -> str:
        """Interpret correlation-based effect size"""
        r_abs = abs(r)
        if r_abs < 0.1:
            return "negligible"
        elif r_abs < 0.3:
            return "small"
        elif r_abs < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta: float) -> str:
        """Interpret eta squared effect size"""
        if eta < 0.01:
            return "negligible"
        elif eta < 0.06:
            return "small"
        elif eta < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_kendalls_w(self, w: float) -> str:
        """Interpret Kendall's W"""
        if w < 0.1:
            return "very weak agreement"
        elif w < 0.3:
            return "weak agreement"
        elif w < 0.5:
            return "moderate agreement"
        elif w < 0.7:
            return "strong agreement"
        else:
            return "very strong agreement"


class AdvancedBenchmarkRunner:
    """Main benchmark runner with comprehensive evaluation"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self.missing_simulator = MissingDataSimulator(config.random_seed)
        self.statistical_tester = StatisticalTester(config.significance_level)
        
        # Set random seeds
        np.random.seed(config.random_seed)
        if torch.cuda.is_available():
            torch.manual_seed(config.random_seed)
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize process pool
        n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=n_jobs)
    
    def _setup_logging(self):
        """Setup logging for benchmark"""
        log_file = self.config.output_dir / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_benchmark(self,
                     methods: Dict[str, Callable],
                     datasets: Dict[str, pd.DataFrame]) -> BenchmarkResult:
        """
        Run comprehensive benchmark
        
        Parameters:
        -----------
        methods : dict
            Method name -> imputation function
        datasets : dict
            Dataset name -> DataFrame
        
        Returns:
        --------
        BenchmarkResult
            Complete benchmark results
        """
        logger.info(f"Starting benchmark with {len(methods)} methods on {len(datasets)} datasets")
        
        start_time = datetime.now()
        
        # Collect system information
        system_info = self._get_system_info()
        
        # Prepare dataset information
        dataset_info = self._analyze_datasets(datasets)
        
        # Initialize results storage
        results = {method: {dataset: [] for dataset in datasets} 
                  for method in methods}
        
        # Run benchmark for each dataset
        for dataset_name, data in datasets.items():
            logger.info(f"Benchmarking on dataset: {dataset_name}")
            
            # Run cross-validation
            cv_results = self._run_cross_validation(
                methods, data, dataset_name
            )
            
            # Store results
            for method_name, method_results in cv_results.items():
                results[method_name][dataset_name] = method_results
        
        # Statistical analysis
        logger.info("Performing statistical analysis")
        statistical_tests = self._run_statistical_tests(results)
        
        # Pairwise comparisons
        pairwise_comparisons = self._run_pairwise_comparisons(results)
        
        # Calculate rankings
        rankings = self._calculate_final_rankings(results)
        
        # Computational summary
        computational_summary = self._summarize_computational_resources(results)
        
        end_time = datetime.now()
        
        # Create final result
        benchmark_result = BenchmarkResult(
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            results=results,
            statistical_tests=statistical_tests,
            pairwise_comparisons=pairwise_comparisons,
            rankings=rankings,
            computational_summary=computational_summary,
            system_info=system_info,
            dataset_info=dataset_info
        )
        
        # Save results
        self._save_results(benchmark_result)
        
        # Generate report
        self._generate_report(benchmark_result)
        
        logger.info(f"Benchmark completed in {end_time - start_time}")
        
        return benchmark_result
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        info = {
            "platform": {
                "python_version": sys.version,
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "scipy_version": stats.__version__,
            },
            "hardware": {
                "cpu_count": mp.cpu_count(),
                "total_memory": psutil.virtual_memory().total,
                "available_memory": psutil.virtual_memory().available,
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # GPU information
        if torch.cuda.is_available():
            info["hardware"]["gpu"] = {
                "device_count": torch.cuda.device_count(),
                "devices": []
            }
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["hardware"]["gpu"]["devices"].append({
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "compute_capability": f"{props.major}.{props.minor}"
                })
        
        return info
    
    def _analyze_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze dataset characteristics"""
        info = {}
        
        for name, data in datasets.items():
            info[name] = {
                "shape": data.shape,
                "columns": data.columns.tolist(),
                "dtypes": data.dtypes.to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "missing_percentage": (data.isnull().sum() / len(data) * 100).to_dict(),
                "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
                "memory_usage": data.memory_usage(deep=True).sum()
            }
            
            # Basic statistics for numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                info[name]["statistics"] = {
                    "mean": numeric_data.mean().to_dict(),
                    "std": numeric_data.std().to_dict(),
                    "min": numeric_data.min().to_dict(),
                    "max": numeric_data.max().to_dict()
                }
        
        return info
    
    def _run_cross_validation(self,
                            methods: Dict[str, Callable],
                            data: pd.DataFrame,
                            dataset_name: str) -> Dict[str, List[MethodResult]]:
        """Run cross-validation for all methods"""
        results = {method: [] for method in methods}
        
        # Setup cross-validation splitter
        if self.config.cv_strategy == "time_series":
            cv = TimeSeriesSplit(
                n_splits=self.config.n_splits,
                gap=self.config.gap_size
            )
        else:
            cv = KFold(
                n_splits=self.config.n_splits,
                shuffle=True,
                random_state=self.config.random_seed
            )
        
        # Get numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Run for each fold
        for fold, (train_idx, test_idx) in enumerate(cv.split(numeric_data)):
            logger.info(f"Processing fold {fold + 1}/{self.config.n_splits}")
            
            # Split data
            train_data = numeric_data.iloc[train_idx].copy()
            test_data = numeric_data.iloc[test_idx].copy()
            
            # For each missing pattern and rate
            for pattern in self.config.missing_patterns:
                for missing_rate in self.config.missing_rates:
                    # Generate missing mask
                    missing_mask = self.missing_simulator.generate_missing_mask(
                        test_data.shape,
                        missing_rate,
                        pattern
                    )
                    
                    # Apply mask to create test set with missing values
                    test_missing = test_data.copy()
                    test_missing[missing_mask] = np.nan
                    
                    # Ground truth (values that were masked)
                    ground_truth = test_data[missing_mask]
                    
                    # Run each method
                    for method_name, method_func in methods.items():
                        result = self._evaluate_method(
                            method_func,
                            train_data,
                            test_missing,
                            ground_truth,
                            missing_mask,
                            method_name,
                            dataset_name,
                            fold,
                            pattern,
                            missing_rate
                        )
                        results[method_name].append(result)
        
        return results
    
    @contextmanager
    def _resource_monitor(self):
        """Context manager for monitoring resource usage"""
        # CPU and memory monitoring
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss
        
        # GPU monitoring if available
        gpu_start = None
        if torch.cuda.is_available() and self.config.profile_gpu:
            torch.cuda.synchronize()
            gpu_start = torch.cuda.memory_allocated()
        
        # Memory profiling
        if self.config.profile_memory:
            tracemalloc.start()
        
        # CPU profiling
        profiler = None
        if self.config.profile_cpu:
            profiler = cProfile.Profile()
            profiler.enable()
        
        try:
            yield
        finally:
            # Calculate resource usage
            end_time = time.time()
            end_memory = process.memory_info().rss
            
            execution_time = end_time - start_time
            peak_memory = end_memory - start_memory
            
            # CPU usage
            cpu_usage = process.cpu_percent(interval=0.1)
            
            # GPU memory
            gpu_memory = None
            if gpu_start is not None:
                torch.cuda.synchronize()
                gpu_memory = torch.cuda.memory_allocated() - gpu_start
            
            # Memory profile
            memory_profile = None
            if self.config.profile_memory:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')[:10]
                memory_profile = {
                    "top_allocations": [
                        {
                            "file": stat.traceback.format()[0],
                            "size": stat.size,
                            "count": stat.count
                        }
                        for stat in top_stats
                    ]
                }
                tracemalloc.stop()
            
            # CPU profile
            cpu_profile = None
            if profiler is not None:
                profiler.disable()
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(10)
                cpu_profile = s.getvalue()
            
            # Store results
            self._last_resource_usage = {
                "execution_time": execution_time,
                "peak_memory": peak_memory,
                "cpu_usage": cpu_usage,
                "gpu_memory": gpu_memory,
                "memory_profile": memory_profile,
                "cpu_profile": cpu_profile
            }
    
    def _evaluate_method(self,
                        method_func: Callable,
                        train_data: pd.DataFrame,
                        test_missing: pd.DataFrame,
                        ground_truth: pd.DataFrame,
                        missing_mask: np.ndarray,
                        method_name: str,
                        dataset_name: str,
                        fold: int,
                        pattern: str,
                        missing_rate: float) -> MethodResult:
        """Evaluate a single method"""
        logger.debug(f"Evaluating {method_name} on fold {fold}, "
                    f"pattern={pattern}, rate={missing_rate}")
        
        try:
            # Monitor resources
            with self._resource_monitor():
                # Combine train and test data for imputation
                combined_data = pd.concat([train_data, test_missing])
                
                # Run imputation
                imputed_data = method_func(combined_data)
                
                # Extract imputed values
                imputed_values = imputed_data.iloc[len(train_data):][missing_mask]
            
            # Get resource usage
            resource_usage = self._last_resource_usage
            
            # Calculate metrics
            metrics = {}
            for metric_name in self.config.metrics:
                metric_func = getattr(self.metrics, metric_name, None)
                if metric_func:
                    try:
                        if metric_name in ["coverage", "sharpness", "interval_score"]:
                            # These need uncertainty estimates
                            # Skip if method doesn't provide them
                            continue
                        
                        score = metric_func(ground_truth.values, imputed_values.values)
                        metrics[metric_name] = score
                    except Exception as e:
                        logger.warning(f"Failed to calculate {metric_name}: {e}")
                        metrics[metric_name] = np.nan
            
            # Create result
            result = MethodResult(
                method_name=method_name,
                dataset_id=f"{dataset_name}_{pattern}_{missing_rate}",
                fold=fold,
                metrics=metrics,
                execution_time=resource_usage["execution_time"],
                peak_memory=resource_usage["peak_memory"],
                cpu_usage=resource_usage["cpu_usage"],
                gpu_memory=resource_usage["gpu_memory"],
                cpu_profile=resource_usage["cpu_profile"],
                memory_profile=resource_usage["memory_profile"],
                success=True,
                predictions=imputed_values.values
            )
            
        except Exception as e:
            logger.error(f"Method {method_name} failed: {e}")
            
            # Create failure result
            result = MethodResult(
                method_name=method_name,
                dataset_id=f"{dataset_name}_{pattern}_{missing_rate}",
                fold=fold,
                metrics={metric: np.nan for metric in self.config.metrics},
                execution_time=0,
                peak_memory=0,
                cpu_usage=0,
                success=False,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
        
        return result
    
    def _run_statistical_tests(self, 
                             results: Dict[str, Dict[str, List[MethodResult]]]) -> Dict[str, Any]:
        """Run statistical tests on results"""
        statistical_tests = {}
        
        # Aggregate results by metric
        for metric in self.config.metrics:
            metric_results = {}
            
            for method_name, method_results in results.items():
                scores = []
                for dataset_results in method_results.values():
                    for result in dataset_results:
                        if result.success and metric in result.metrics:
                            score = result.metrics[metric]
                            if not np.isnan(score):
                                scores.append(score)
                
                if scores:
                    metric_results[method_name] = scores
            
            # Run statistical comparison
            if len(metric_results) >= 2:
                statistical_tests[metric] = self.statistical_tester.compare_methods(
                    metric_results,
                    correction=self.config.multiple_testing_correction
                )
        
        return statistical_tests
    
    def _run_pairwise_comparisons(self,
                                 results: Dict[str, Dict[str, List[MethodResult]]]) -> Dict[str, Any]:
        """Run pairwise method comparisons"""
        comparisons = {}
        
        # Compare methods for each dataset and condition
        for dataset_name in next(iter(results.values())).keys():
            dataset_comparisons = {}
            
            # Group by pattern and missing rate
            for pattern in self.config.missing_patterns:
                for rate in self.config.missing_rates:
                    condition = f"{pattern}_{rate}"
                    condition_results = {}
                    
                    for method_name, method_results in results.items():
                        scores = []
                        for result in method_results[dataset_name]:
                            if (result.success and 
                                pattern in result.dataset_id and 
                                str(rate) in result.dataset_id):
                                # Use primary metric (first in list)
                                primary_metric = self.config.metrics[0]
                                if primary_metric in result.metrics:
                                    scores.append(result.metrics[primary_metric])
                        
                        if scores:
                            condition_results[method_name] = scores
                    
                    if len(condition_results) >= 2:
                        dataset_comparisons[condition] = self.statistical_tester.compare_methods(
                            condition_results,
                            correction=self.config.multiple_testing_correction
                        )
            
            comparisons[dataset_name] = dataset_comparisons
        
        return comparisons
    
    def _calculate_final_rankings(self,
                                 results: Dict[str, Dict[str, List[MethodResult]]]) -> Dict[str, pd.DataFrame]:
        """Calculate final method rankings"""
        rankings = {}
        
        # Overall ranking
        overall_scores = {}
        
        for method_name, method_results in results.items():
            all_scores = []
            for dataset_results in method_results.values():
                for result in dataset_results:
                    if result.success:
                        # Use primary metric
                        primary_metric = self.config.metrics[0]
                        if primary_metric in result.metrics:
                            score = result.metrics[primary_metric]
                            if not np.isnan(score):
                                all_scores.append(score)
            
            if all_scores:
                overall_scores[method_name] = {
                    "mean": np.mean(all_scores),
                    "std": np.std(all_scores),
                    "median": np.median(all_scores),
                    "count": len(all_scores)
                }
        
        # Convert to DataFrame
        overall_df = pd.DataFrame(overall_scores).T
        overall_df["rank"] = overall_df["mean"].rank()
        rankings["overall"] = overall_df.sort_values("rank")
        
        # Rankings by condition
        for pattern in self.config.missing_patterns:
            pattern_scores = {}
            
            for method_name, method_results in results.items():
                scores = []
                for dataset_results in method_results.values():
                    for result in dataset_results:
                        if result.success and pattern in result.dataset_id:
                            primary_metric = self.config.metrics[0]
                            if primary_metric in result.metrics:
                                score = result.metrics[primary_metric]
                                if not np.isnan(score):
                                    scores.append(score)
                
                if scores:
                    pattern_scores[method_name] = {
                        "mean": np.mean(scores),
                        "std": np.std(scores),
                        "count": len(scores)
                    }
            
            if pattern_scores:
                pattern_df = pd.DataFrame(pattern_scores).T
                pattern_df["rank"] = pattern_df["mean"].rank()
                rankings[f"pattern_{pattern}"] = pattern_df.sort_values("rank")
        
        return rankings
    
    def _summarize_computational_resources(self,
                                         results: Dict[str, Dict[str, List[MethodResult]]]) -> Dict[str, Any]:
        """Summarize computational resource usage"""
        summary = {}
        
        for method_name, method_results in results.items():
            times = []
            memories = []
            cpu_usages = []
            gpu_memories = []
            failures = 0
            
            for dataset_results in method_results.values():
                for result in dataset_results:
                    if result.success:
                        times.append(result.execution_time)
                        memories.append(result.peak_memory)
                        cpu_usages.append(result.cpu_usage)
                        if result.gpu_memory is not None:
                            gpu_memories.append(result.gpu_memory)
                    else:
                        failures += 1
            
            summary[method_name] = {
                "execution_time": {
                    "mean": np.mean(times) if times else np.nan,
                    "std": np.std(times) if times else np.nan,
                    "min": np.min(times) if times else np.nan,
                    "max": np.max(times) if times else np.nan
                },
                "memory_usage": {
                    "mean": np.mean(memories) if memories else np.nan,
                    "std": np.std(memories) if memories else np.nan,
                    "max": np.max(memories) if memories else np.nan
                },
                "cpu_usage": {
                    "mean": np.mean(cpu_usages) if cpu_usages else np.nan
                },
                "gpu_memory": {
                    "mean": np.mean(gpu_memories) if gpu_memories else np.nan,
                    "max": np.max(gpu_memories) if gpu_memories else np.nan
                } if gpu_memories else None,
                "failure_rate": failures / (len(times) + failures) if (times or failures) else 1.0
            }
        
        return summary
    
    def _save_results(self, results: BenchmarkResult):
        """Save benchmark results"""
        # Save as JSON
        json_path = self.config.output_dir / "benchmark_results.json"
        
        # Convert to serializable format
        results_dict = {
            "config": self.config.__dict__,
            "start_time": results.start_time.isoformat(),
            "end_time": results.end_time.isoformat(),
            "system_info": results.system_info,
            "dataset_info": results.dataset_info,
            "statistical_tests": results.statistical_tests,
            "pairwise_comparisons": results.pairwise_comparisons,
            "computational_summary": results.computational_summary
        }
        
        # Convert method results
        results_dict["results"] = {}
        for method_name, method_results in results.results.items():
            results_dict["results"][method_name] = {}
            for dataset_name, dataset_results in method_results.items():
                results_dict["results"][method_name][dataset_name] = [
                    {
                        "dataset_id": r.dataset_id,
                        "fold": r.fold,
                        "metrics": r.metrics,
                        "execution_time": r.execution_time,
                        "peak_memory": r.peak_memory,
                        "success": r.success,
                        "error_message": r.error_message
                    }
                    for r in dataset_results
                ]
        
        # Convert rankings DataFrames
        results_dict["rankings"] = {
            name: df.to_dict()
            for name, df in results.rankings.items()
        }
        
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save rankings as CSV
        for name, df in results.rankings.items():
            csv_path = self.config.output_dir / f"rankings_{name}.csv"
            df.to_csv(csv_path)
        
        logger.info(f"Results saved to {self.config.output_dir}")
    
    def _generate_report(self, results: BenchmarkResult):
        """Generate comprehensive benchmark report"""
        # Create report directory
        report_dir = self.config.output_dir / "report"
        report_dir.mkdir(exist_ok=True)
        
        # Generate plots
        self._generate_plots(results, report_dir)
        
        # Generate LaTeX report
        self._generate_latex_report(results, report_dir)
        
        # Generate markdown summary
        self._generate_markdown_summary(results, report_dir)
    
    def _generate_plots(self, results: BenchmarkResult, output_dir: Path):
        """Generate visualization plots"""
        # Set style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        # 1. Method comparison boxplot
        self._plot_method_comparison(results, output_dir)
        
        # 2. Rankings heatmap
        self._plot_rankings_heatmap(results, output_dir)
        
        # 3. Computational resources
        self._plot_computational_resources(results, output_dir)
        
        # 4. Critical difference diagram
        self._plot_critical_difference(results, output_dir)
    
    def _plot_method_comparison(self, results: BenchmarkResult, output_dir: Path):
        """Plot method comparison boxplots"""
        # Aggregate scores by method
        method_scores = {}
        primary_metric = self.config.metrics[0]
        
        for method_name, method_results in results.results.items():
            scores = []
            for dataset_results in method_results.values():
                for result in dataset_results:
                    if result.success and primary_metric in result.metrics:
                        scores.append(result.metrics[primary_metric])
            method_scores[method_name] = scores
        
        # Create boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = [scores for scores in method_scores.values()]
        labels = list(method_scores.keys())
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Customize colors
        colors = sns.color_palette("husl", len(labels))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel(primary_metric.upper())
        ax.set_title(f"Method Comparison - {primary_metric.upper()}")
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(output_dir / "method_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "method_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rankings_heatmap(self, results: BenchmarkResult, output_dir: Path):
        """Plot rankings heatmap"""
        # Prepare data for heatmap
        ranking_data = []
        
        for condition, df in results.rankings.items():
            if condition != "overall":
                for method in df.index:
                    ranking_data.append({
                        "Method": method,
                        "Condition": condition,
                        "Rank": df.loc[method, "rank"]
                    })
        
        if ranking_data:
            ranking_df = pd.DataFrame(ranking_data)
            pivot_df = ranking_df.pivot(index="Method", columns="Condition", values="Rank")
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="RdYlGn_r",
                       cbar_kws={"label": "Rank"}, ax=ax)
            
            ax.set_title("Method Rankings by Condition")
            plt.tight_layout()
            
            plt.savefig(output_dir / "rankings_heatmap.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / "rankings_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_computational_resources(self, results: BenchmarkResult, output_dir: Path):
        """Plot computational resource usage"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        methods = list(results.computational_summary.keys())
        
        # Execution time
        times = [results.computational_summary[m]["execution_time"]["mean"] 
                for m in methods]
        axes[0, 0].bar(methods, times)
        axes[0, 0].set_ylabel("Time (seconds)")
        axes[0, 0].set_title("Average Execution Time")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage
        memories = [results.computational_summary[m]["memory_usage"]["mean"] / 1e6
                   for m in methods]
        axes[0, 1].bar(methods, memories)
        axes[0, 1].set_ylabel("Memory (MB)")
        axes[0, 1].set_title("Average Memory Usage")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # CPU usage
        cpu_usages = [results.computational_summary[m]["cpu_usage"]["mean"]
                     for m in methods]
        axes[1, 0].bar(methods, cpu_usages)
        axes[1, 0].set_ylabel("CPU Usage (%)")
        axes[1, 0].set_title("Average CPU Usage")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Failure rate
        failure_rates = [results.computational_summary[m]["failure_rate"] * 100
                        for m in methods]
        axes[1, 1].bar(methods, failure_rates)
        axes[1, 1].set_ylabel("Failure Rate (%)")
        axes[1, 1].set_title("Method Failure Rate")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        plt.savefig(output_dir / "computational_resources.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "computational_resources.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_critical_difference(self, results: BenchmarkResult, output_dir: Path):
        """Plot critical difference diagram"""
        # Get average ranks
        overall_rankings = results.rankings.get("overall")
        if overall_rankings is None:
            return
        
        # Sort by rank
        sorted_methods = overall_rankings.sort_values("rank").index.tolist()
        ranks = overall_rankings.loc[sorted_methods, "rank"].values
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot method ranks
        y_positions = np.arange(len(sorted_methods))
        ax.scatter(ranks, y_positions, s=100, zorder=3)
        
        # Add method names
        for i, (method, rank) in enumerate(zip(sorted_methods, ranks)):
            ax.text(rank - 0.1, i, method, ha='right', va='center')
        
        # Add critical difference line
        primary_metric = self.config.metrics[0]
        if primary_metric in results.statistical_tests:
            stats_result = results.statistical_tests[primary_metric]
            if "rankings" in stats_result:
                cd = stats_result["rankings"]["critical_difference"]
                
                # Draw CD line
                ax.axhline(y=-1, xmin=0.1, xmax=0.9, color='red', 
                          linestyle='--', label=f'CD = {cd:.3f}')
        
        ax.set_xlabel("Average Rank")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([""] * len(sorted_methods))
        ax.set_title("Critical Difference Diagram")
        ax.grid(True, axis='x', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        plt.savefig(output_dir / "critical_difference.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "critical_difference.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_latex_report(self, results: BenchmarkResult, output_dir: Path):
        """Generate LaTeX report"""
        template = r"""
\documentclass[11pt]{article}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}

\title{Benchmark Report: Air Quality Imputation Methods}
\author{Generated by AirImpute Pro}
\date{\today}

\begin{document}
\maketitle

\section{Executive Summary}
This report presents comprehensive benchmark results for air quality imputation methods.
The evaluation included %d methods across %d datasets with %d different missing patterns
and %d missing rates.

\section{Statistical Analysis}
%s

\section{Rankings}
\begin{table}[h]
\centering
\caption{Overall Method Rankings}
%s
\end{table}

\section{Computational Performance}
%s

\section{Figures}
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{method_comparison.pdf}
\caption{Method comparison boxplots}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{rankings_heatmap.pdf}
\caption{Rankings by condition}
\end{figure}

\end{document}
"""
        
        # Fill in template
        n_methods = len(results.results)
        n_datasets = len(next(iter(results.results.values())))
        n_patterns = len(self.config.missing_patterns)
        n_rates = len(self.config.missing_rates)
        
        # Statistical summary
        stat_summary = self._format_statistical_summary(results)
        
        # Rankings table
        rankings_table = self._format_rankings_table(results.rankings["overall"])
        
        # Computational summary
        comp_summary = self._format_computational_summary(results)
        
        report_content = template % (
            n_methods, n_datasets, n_patterns, n_rates,
            stat_summary, rankings_table, comp_summary
        )
        
        # Save report
        with open(output_dir / "benchmark_report.tex", 'w') as f:
            f.write(report_content)
    
    def _format_statistical_summary(self, results: BenchmarkResult) -> str:
        """Format statistical summary for LaTeX"""
        summary = ""
        
        primary_metric = self.config.metrics[0]
        if primary_metric in results.statistical_tests:
            test_results = results.statistical_tests[primary_metric]
            
            if "omnibus" in test_results:
                omnibus = test_results["omnibus"]
                summary += f"""
The {omnibus['test']} revealed {'significant' if omnibus['significant'] else 'no significant'} 
differences between methods (p = {omnibus['p_value']:.4f}).
"""
            
            if "pairwise" in test_results and test_results["pairwise"].get("performed", False):
                n_sig = sum(1 for comp in test_results["pairwise"]["comparisons"] 
                           if comp["significant"])
                n_total = test_results["pairwise"]["n_comparisons"]
                summary += f"""
Pairwise comparisons ({test_results['pairwise']['correction_method']} correction) 
revealed {n_sig} significant differences out of {n_total} comparisons.
"""
        
        return summary
    
    def _format_rankings_table(self, rankings_df: pd.DataFrame) -> str:
        """Format rankings table for LaTeX"""
        # Convert to LaTeX table
        latex_table = rankings_df[["mean", "std", "rank"]].to_latex(
            float_format="%.3f",
            bold_rows=True,
            caption="Overall method rankings based on primary metric"
        )
        
        return latex_table
    
    def _format_computational_summary(self, results: BenchmarkResult) -> str:
        """Format computational summary for LaTeX"""
        summary = "\\subsection{Resource Usage}\n\n"
        
        # Find most and least efficient methods
        comp_data = results.computational_summary
        
        # Fastest method
        fastest = min(comp_data.items(), 
                     key=lambda x: x[1]["execution_time"]["mean"])
        summary += f"Fastest method: {fastest[0]} ({fastest[1]['execution_time']['mean']:.2f}s average)\n\n"
        
        # Most memory efficient
        least_memory = min(comp_data.items(),
                          key=lambda x: x[1]["memory_usage"]["mean"])
        summary += f"Most memory efficient: {least_memory[0]} ({least_memory[1]['memory_usage']['mean']/1e6:.1f}MB average)\n\n"
        
        return summary
    
    def _generate_markdown_summary(self, results: BenchmarkResult, output_dir: Path):
        """Generate markdown summary"""
        summary = f"""# Benchmark Summary

## Overview
- **Methods evaluated**: {len(results.results)}
- **Datasets**: {len(next(iter(results.results.values())))}
- **Total duration**: {results.end_time - results.start_time}

## Top Performers

### Overall Rankings
"""
        
        # Add rankings
        overall_rankings = results.rankings["overall"].head(5)
        summary += overall_rankings[["mean", "std", "rank"]].to_markdown()
        
        # Add statistical significance
        summary += "\n\n## Statistical Significance\n"
        primary_metric = self.config.metrics[0]
        
        if primary_metric in results.statistical_tests:
            test_results = results.statistical_tests[primary_metric]
            if "omnibus" in test_results:
                omnibus = test_results["omnibus"]
                summary += f"\n{omnibus['test']}: p-value = {omnibus['p_value']:.4e}\n"
        
        # Save summary
        with open(output_dir / "summary.md", 'w') as f:
            f.write(summary)
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Example usage
if __name__ == "__main__":
    # Configuration
    config = BenchmarkConfig(
        n_splits=5,
        missing_rates=[0.1, 0.3, 0.5],
        missing_patterns=["MCAR", "MAR"],
        metrics=["rmse", "mae", "mape"],
        significance_level=0.05,
        enable_profiling=True,
        output_dir=Path("benchmark_results")
    )
    
    # Initialize runner
    runner = AdvancedBenchmarkRunner(config)
    
    # Example methods (would be actual imputation methods)
    def mean_imputation(data):
        return data.fillna(data.mean())
    
    def forward_fill(data):
        return data.fillna(method='ffill').fillna(method='bfill')
    
    methods = {
        "mean": mean_imputation,
        "forward_fill": forward_fill
    }
    
    # Example datasets
    np.random.seed(42)
    datasets = {
        "synthetic": pd.DataFrame({
            "value": np.sin(np.linspace(0, 4*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
        })
    }
    
    # Run benchmark
    results = runner.run_benchmark(methods, datasets)
    
    print(f"Benchmark completed. Results saved to {config.output_dir}")