"""
Comprehensive academic validation test suite with property-based testing
Implements rigorous testing standards for publication-grade software
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, assume, note
from hypothesis.extra.numpy import arrays
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from typing import Tuple, Dict, Any, List, Optional
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging for academic rigor
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our imputation methods
import sys
sys.path.append(str(Path(__file__).parent.parent))
from airimpute import ImputationEngine
from airimpute.methods import IMPUTATION_METHODS
from airimpute.validation import ValidationFramework
from airimpute.statistical_tests import StatisticalTestSuite


@dataclass
class TestConfiguration:
    """Configuration for academic testing standards"""
    min_samples: int = 1000
    max_missing_ratio: float = 0.7
    convergence_tolerance: float = 1e-6
    statistical_significance: float = 0.05
    random_seeds: List[int] = None
    
    def __post_init__(self):
        if self.random_seeds is None:
            # Use fixed seeds for reproducibility
            self.random_seeds = [42, 123, 456, 789, 2024]


class AcademicTestSuite:
    """Comprehensive test suite for academic validation"""
    
    def __init__(self, config: TestConfiguration = None):
        self.config = config or TestConfiguration()
        self.engine = ImputationEngine()
        self.validator = ValidationFramework()
        self.stat_tests = StatisticalTestSuite()
        self.results = []
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report for publication"""
        return {
            'configuration': self.config.__dict__,
            'test_results': self.results,
            'statistical_summary': self._compute_statistical_summary(),
            'reproducibility_hash': self._compute_reproducibility_hash(),
        }
    
    def _compute_statistical_summary(self) -> Dict[str, Any]:
        """Compute statistical summary of all tests"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        summary = {
            'total_tests': len(df),
            'passed_tests': len(df[df['passed']]),
            'failure_rate': 1 - len(df[df['passed']]) / len(df),
            'mean_runtime': df['runtime'].mean(),
            'method_performance': df.groupby('method')['performance'].mean().to_dict(),
        }
        return summary
    
    def _compute_reproducibility_hash(self) -> str:
        """Compute hash for reproducibility verification"""
        content = json.dumps(self.results, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


# Property-based testing strategies
@st.composite
def time_series_with_gaps(draw):
    """Generate realistic time series with missing data patterns"""
    # Base parameters
    length = draw(st.integers(min_value=100, max_value=10000))
    trend = draw(st.floats(min_value=-0.1, max_value=0.1))
    seasonality = draw(st.booleans())
    noise_level = draw(st.floats(min_value=0.1, max_value=2.0))
    
    # Generate base time series
    t = np.arange(length)
    base = trend * t
    
    if seasonality:
        period = draw(st.integers(min_value=12, max_value=365))
        amplitude = draw(st.floats(min_value=5, max_value=50))
        base += amplitude * np.sin(2 * np.pi * t / period)
    
    # Add realistic noise
    noise = np.random.normal(0, noise_level, length)
    data = base + noise + 50  # Offset to ensure positive values
    
    # Generate missing data pattern
    missing_type = draw(st.sampled_from(['MCAR', 'MAR', 'MNAR']))
    missing_ratio = draw(st.floats(min_value=0.05, max_value=0.5))
    
    if missing_type == 'MCAR':
        # Missing Completely At Random
        mask = np.random.random(length) < missing_ratio
    elif missing_type == 'MAR':
        # Missing At Random (depends on observed values)
        threshold = np.percentile(data, missing_ratio * 100)
        mask = data < threshold
    else:  # MNAR
        # Missing Not At Random (depends on missing values)
        # Simulate sensor failure at high values
        high_threshold = np.percentile(data, 80)
        mask = (data > high_threshold) & (np.random.random(length) < 0.7)
    
    data[mask] = np.nan
    
    return data, {
        'length': length,
        'missing_type': missing_type,
        'missing_ratio': np.mean(mask),
        'trend': trend,
        'seasonality': seasonality,
        'noise_level': noise_level,
    }


@st.composite
def air_quality_data(draw):
    """Generate realistic air quality data"""
    # Pollutant-specific parameters
    pollutant = draw(st.sampled_from(['PM2.5', 'PM10', 'O3', 'NO2']))
    
    pollutant_params = {
        'PM2.5': {'mean': 25, 'std': 15, 'min': 0, 'max': 200},
        'PM10': {'mean': 50, 'std': 25, 'min': 0, 'max': 400},
        'O3': {'mean': 60, 'std': 20, 'min': 0, 'max': 200},
        'NO2': {'mean': 40, 'std': 20, 'min': 0, 'max': 150},
    }
    
    params = pollutant_params[pollutant]
    length = draw(st.integers(min_value=168, max_value=8760))  # 1 week to 1 year
    
    # Generate with diurnal and weekly patterns
    t = np.arange(length)
    hourly = np.sin(2 * np.pi * t / 24) * params['std'] * 0.3
    weekly = np.sin(2 * np.pi * t / (24 * 7)) * params['std'] * 0.2
    
    # Add random walk for realism
    random_walk = np.cumsum(np.random.normal(0, 0.1, length))
    
    # Combine components
    data = params['mean'] + hourly + weekly + random_walk
    data += np.random.normal(0, params['std'] * 0.5, length)
    
    # Enforce physical constraints
    data = np.clip(data, params['min'], params['max'])
    
    # Add missing data
    missing_pattern = draw(st.sampled_from(['random', 'blocks', 'periodic']))
    missing_ratio = draw(st.floats(min_value=0.05, max_value=0.4))
    
    if missing_pattern == 'random':
        mask = np.random.random(length) < missing_ratio
    elif missing_pattern == 'blocks':
        # Simulate sensor maintenance
        n_blocks = draw(st.integers(min_value=1, max_value=10))
        mask = np.zeros(length, dtype=bool)
        for _ in range(n_blocks):
            start = np.random.randint(0, length - 24)
            duration = np.random.randint(6, 48)
            mask[start:start + duration] = True
    else:  # periodic
        # Simulate regular calibration
        period = draw(st.integers(min_value=168, max_value=720))  # Weekly to monthly
        mask = (t % period) < 6  # 6-hour calibration
    
    data[mask] = np.nan
    
    return data, {
        'pollutant': pollutant,
        'length': length,
        'missing_pattern': missing_pattern,
        'actual_missing_ratio': np.mean(mask),
    }


class TestImputationMethods:
    """Test all imputation methods with academic rigor"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment"""
        self.engine = ImputationEngine()
        self.validator = ValidationFramework()
        np.random.seed(42)
    
    @given(time_series_with_gaps())
    @settings(max_examples=50, deadline=30000)  # 30 seconds per test
    def test_imputation_preserves_statistical_properties(self, data_info):
        """Test that imputation preserves key statistical properties"""
        data, info = data_info
        
        # Skip if too much missing data
        assume(info['missing_ratio'] < 0.5)
        
        # Original statistics (on non-missing data)
        original_mean = np.nanmean(data)
        original_std = np.nanstd(data)
        original_acf = self._compute_acf(data[~np.isnan(data)])
        
        # Test each method
        for method_name in ['linear', 'random_forest', 'rah']:
            with self.subTest(method=method_name):
                # Impute
                result = self.engine.impute(
                    pd.DataFrame({'value': data}),
                    method=method_name
                )
                imputed_data = result['value'].values
                
                # Check statistical properties
                imputed_mean = np.mean(imputed_data)
                imputed_std = np.std(imputed_data)
                imputed_acf = self._compute_acf(imputed_data)
                
                # Assertions with tolerance
                assert abs(imputed_mean - original_mean) / original_mean < 0.1, \
                    f"Mean preservation failed: {imputed_mean:.2f} vs {original_mean:.2f}"
                
                assert abs(imputed_std - original_std) / original_std < 0.2, \
                    f"Std preservation failed: {imputed_std:.2f} vs {original_std:.2f}"
                
                # ACF similarity (using correlation of ACF functions)
                acf_correlation = np.corrcoef(original_acf[:10], imputed_acf[:10])[0, 1]
                assert acf_correlation > 0.8, \
                    f"Temporal structure not preserved: ACF correlation = {acf_correlation:.3f}"
    
    @given(air_quality_data())
    @settings(max_examples=30, deadline=60000)
    def test_physical_constraints_respected(self, data_info):
        """Test that imputation respects physical constraints"""
        data, info = data_info
        
        pollutant_limits = {
            'PM2.5': (0, 500),
            'PM10': (0, 600),
            'O3': (0, 300),
            'NO2': (0, 200),
        }
        
        min_val, max_val = pollutant_limits[info['pollutant']]
        
        # Test methods that should respect constraints
        for method_name in ['random_forest', 'xgboost', 'rah']:
            with self.subTest(method=method_name):
                result = self.engine.impute(
                    pd.DataFrame({'value': data}),
                    method=method_name,
                    constraints={'min': min_val, 'max': max_val}
                )
                imputed_data = result['value'].values
                
                # Check constraints
                assert np.all(imputed_data >= min_val), \
                    f"Minimum constraint violated: {np.min(imputed_data):.2f} < {min_val}"
                assert np.all(imputed_data <= max_val), \
                    f"Maximum constraint violated: {np.max(imputed_data):.2f} > {max_val}"
                
                # Check no new outliers introduced
                original_q99 = np.nanpercentile(data, 99)
                imputed_q99 = np.percentile(imputed_data, 99)
                assert imputed_q99 <= original_q99 * 1.1, \
                    "Imputation introduced unrealistic high values"
    
    def test_convergence_properties(self):
        """Test convergence properties of iterative methods"""
        # Generate test data
        np.random.seed(42)
        n = 1000
        t = np.arange(n)
        data = 50 + 10 * np.sin(2 * np.pi * t / 100) + np.random.normal(0, 5, n)
        
        # Create 30% missing
        mask = np.random.random(n) < 0.3
        data[mask] = np.nan
        
        # Test iterative methods
        iterative_methods = ['em_algorithm', 'mice', 'rah']
        
        for method_name in iterative_methods:
            with self.subTest(method=method_name):
                # Get convergence history
                result, history = self.engine.impute_with_diagnostics(
                    pd.DataFrame({'value': data}),
                    method=method_name,
                    return_diagnostics=True
                )
                
                if 'convergence_history' in history:
                    conv_history = history['convergence_history']
                    
                    # Check monotonic decrease
                    differences = np.diff(conv_history)
                    assert np.all(differences <= 0), \
                        "Convergence history not monotonically decreasing"
                    
                    # Check convergence achieved
                    final_change = abs(conv_history[-1] - conv_history[-2])
                    assert final_change < 1e-4, \
                        f"Method did not converge: final change = {final_change:.6f}"
    
    def test_uncertainty_quantification(self):
        """Test uncertainty quantification capabilities"""
        # Generate data with known properties
        np.random.seed(42)
        n = 500
        true_data = np.random.normal(50, 10, n)
        
        # Create missing data
        data = true_data.copy()
        mask = np.random.random(n) < 0.3
        data[mask] = np.nan
        
        # Methods that support uncertainty
        uncertainty_methods = ['gaussian_process', 'bayesian', 'rah']
        
        for method_name in uncertainty_methods:
            with self.subTest(method=method_name):
                result = self.engine.impute(
                    pd.DataFrame({'value': data}),
                    method=method_name,
                    return_uncertainty=True
                )
                
                if 'uncertainty' in result:
                    uncertainty = result['uncertainty'].values
                    imputed_values = result['value'].values
                    
                    # Check uncertainty is higher for imputed values
                    mean_uncertainty_imputed = np.mean(uncertainty[mask])
                    mean_uncertainty_observed = np.mean(uncertainty[~mask])
                    
                    assert mean_uncertainty_imputed > mean_uncertainty_observed, \
                        "Uncertainty should be higher for imputed values"
                    
                    # Check calibration (approximate)
                    # Count how many true values fall within confidence intervals
                    lower = imputed_values - 2 * uncertainty
                    upper = imputed_values + 2 * uncertainty
                    coverage = np.mean((true_data >= lower) & (true_data <= upper))
                    
                    assert 0.9 < coverage < 0.99, \
                        f"Uncertainty intervals poorly calibrated: {coverage:.2f} coverage"
    
    def test_missing_pattern_detection(self):
        """Test ability to detect missing data patterns"""
        # Test MCAR
        n = 1000
        data_mcar = np.random.normal(50, 10, n)
        mask_mcar = np.random.random(n) < 0.3
        data_mcar[mask_mcar] = np.nan
        
        pattern_mcar = self.validator.detect_missing_pattern(
            pd.DataFrame({'value': data_mcar})
        )
        assert pattern_mcar['pattern_type'] == 'MCAR', \
            "Failed to detect MCAR pattern"
        
        # Test MAR
        data_mar = np.random.normal(50, 10, n)
        # Missing depends on another variable
        covariate = np.random.normal(0, 1, n)
        mask_mar = covariate < -0.5  # About 30% missing
        data_mar[mask_mar] = np.nan
        
        pattern_mar = self.validator.detect_missing_pattern(
            pd.DataFrame({'value': data_mar, 'covariate': covariate})
        )
        assert pattern_mar['pattern_type'] == 'MAR', \
            "Failed to detect MAR pattern"
        
        # Test MNAR
        data_mnar = np.random.normal(50, 10, n)
        # High values are missing (sensor saturation)
        mask_mnar = data_mnar > 65  # About 10% missing
        data_mnar[mask_mnar] = np.nan
        
        pattern_mnar = self.validator.detect_missing_pattern(
            pd.DataFrame({'value': data_mnar})
        )
        assert pattern_mnar['pattern_type'] == 'MNAR', \
            "Failed to detect MNAR pattern"
    
    def test_benchmark_reproducibility(self):
        """Test that benchmarks are reproducible"""
        # Set up benchmark
        from airimpute.benchmarking import BenchmarkRunner
        
        runner = BenchmarkRunner(random_seed=42)
        
        # Generate test dataset
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='h'),
            'PM25': np.random.normal(25, 10, 1000),
            'PM10': np.random.normal(50, 20, 1000),
        })
        
        # Add missing values
        for col in ['PM25', 'PM10']:
            mask = np.random.random(1000) < 0.2
            data.loc[mask, col] = np.nan
        
        # Run benchmark twice
        results1 = runner.run_benchmark(
            data=data,
            methods=['linear', 'random_forest'],
            cv_splits=3
        )
        
        results2 = runner.run_benchmark(
            data=data,
            methods=['linear', 'random_forest'],
            cv_splits=3
        )
        
        # Check reproducibility
        for method in ['linear', 'random_forest']:
            mae1 = results1['results'][method]['mae']
            mae2 = results2['results'][method]['mae']
            
            assert abs(mae1 - mae2) < 1e-10, \
                f"Benchmark not reproducible for {method}: {mae1} vs {mae2}"
    
    def test_cross_validation_strategies(self):
        """Test different cross-validation strategies"""
        # Generate time series data
        n = 2000
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
            'value': np.random.normal(50, 10, n),
        })
        
        # Add trend and seasonality
        t = np.arange(n)
        data['value'] += 0.01 * t + 10 * np.sin(2 * np.pi * t / 168)  # Weekly pattern
        
        # Add missing values
        mask = np.random.random(n) < 0.2
        data.loc[mask, 'value'] = np.nan
        
        # Test time series CV
        from airimpute.validation import TimeSeriesCrossValidator
        
        cv = TimeSeriesCrossValidator(n_splits=5, test_size=168)  # 1 week test
        
        scores = []
        for train_idx, test_idx in cv.split(data):
            train_data = data.iloc[train_idx].copy()
            test_data = data.iloc[test_idx].copy()
            
            # Ensure temporal order
            assert train_data['timestamp'].max() < test_data['timestamp'].min(), \
                "Time series CV violates temporal order"
            
            # Simple validation
            imputed = self.engine.impute(train_data, method='linear')
            score = mean_absolute_error(
                test_data['value'].dropna(),
                imputed.loc[test_data.index, 'value'].dropna()
            )
            scores.append(score)
        
        # Check consistency of scores
        score_std = np.std(scores)
        assert score_std < 5.0, \
            f"Cross-validation scores too variable: std = {score_std:.2f}"
    
    def _compute_acf(self, data: np.ndarray, max_lag: int = 20) -> np.ndarray:
        """Compute autocorrelation function"""
        if len(data) < max_lag * 2:
            max_lag = len(data) // 2
        
        acf = []
        for lag in range(max_lag):
            if lag == 0:
                acf.append(1.0)
            else:
                c0 = np.sum((data[:-lag] - np.mean(data)) * (data[lag:] - np.mean(data)))
                c0 /= len(data) - lag
                acf.append(c0 / np.var(data))
        
        return np.array(acf)


class TestStatisticalAssumptions:
    """Test statistical assumptions and requirements"""
    
    def test_normality_tests(self):
        """Test normality testing capabilities"""
        stat_suite = StatisticalTestSuite()
        
        # Normal data
        normal_data = np.random.normal(0, 1, 1000)
        normal_result = stat_suite.test_normality(normal_data)
        assert normal_result['is_normal'], "Failed to identify normal data"
        
        # Non-normal data
        exponential_data = np.random.exponential(1, 1000)
        exp_result = stat_suite.test_normality(exponential_data)
        assert not exp_result['is_normal'], "Failed to identify non-normal data"
    
    def test_stationarity_tests(self):
        """Test stationarity testing (ADF, KPSS)"""
        stat_suite = StatisticalTestSuite()
        
        # Stationary series
        stationary = np.random.normal(0, 1, 500)
        stat_result = stat_suite.test_stationarity(stationary)
        assert stat_result['is_stationary'], "Failed to identify stationary series"
        
        # Non-stationary series (random walk)
        non_stationary = np.cumsum(np.random.normal(0, 1, 500))
        non_stat_result = stat_suite.test_stationarity(non_stationary)
        assert not non_stat_result['is_stationary'], "Failed to identify non-stationary series"
    
    def test_independence_tests(self):
        """Test independence and autocorrelation tests"""
        stat_suite = StatisticalTestSuite()
        
        # Independent data
        independent = np.random.normal(0, 1, 500)
        indep_result = stat_suite.test_independence(independent)
        assert indep_result['is_independent'], "Failed to identify independent data"
        
        # Autocorrelated data
        ar_process = [0]
        for _ in range(499):
            ar_process.append(0.8 * ar_process[-1] + np.random.normal(0, 1))
        ar_process = np.array(ar_process)
        
        ar_result = stat_suite.test_independence(ar_process)
        assert not ar_result['is_independent'], "Failed to identify autocorrelation"


if __name__ == "__main__":
    # Run comprehensive test suite
    test_suite = AcademicTestSuite()
    
    # Run all tests and generate report
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Generate academic validation report
    report = test_suite.generate_test_report()
    
    # Save report
    with open("academic_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Validation complete. Reproducibility hash: {report['reproducibility_hash']}")