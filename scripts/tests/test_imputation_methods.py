"""
Comprehensive tests for air quality data imputation methods.
Following IEEE/ACM standards for scientific software testing.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import warnings
from scipy import stats

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airimpute.methods import (
    SimpleImputer, InterpolationImputer, StatisticalImputer,
    MachineLearningImputer, RAHImputer
)
from airimpute.validation import CrossValidator, ValidationMetrics
from airimpute.core import ImputationResult


class TestDataGenerator:
    """Generate synthetic test data with known properties."""
    
    @staticmethod
    def generate_time_series(
        n_samples: int = 1000,
        n_features: int = 5,
        missing_rate: float = 0.2,
        pattern: str = 'random',
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic time series with missing values."""
        np.random.seed(seed)
        
        # Create timestamps
        timestamps = pd.date_range(
            start='2024-01-01',
            periods=n_samples,
            freq='H'
        )
        
        # Generate base signals
        t = np.arange(n_samples)
        data = {}
        
        for i in range(n_features):
            # Different patterns for each feature
            if i == 0:  # Sinusoidal
                signal = 50 + 20 * np.sin(2 * np.pi * t / 168)  # Weekly pattern
            elif i == 1:  # Trend + seasonal
                signal = 30 + 0.01 * t + 10 * np.sin(2 * np.pi * t / 24)  # Daily pattern
            elif i == 2:  # Random walk
                signal = 40 + np.cumsum(np.random.randn(n_samples) * 0.5)
            else:  # AR process
                signal = np.zeros(n_samples)
                signal[0] = 50
                for j in range(1, n_samples):
                    signal[j] = 0.8 * signal[j-1] + np.random.randn() * 5 + 10
            
            # Add noise
            signal += np.random.randn(n_samples) * 2
            data[f'var_{i+1}'] = signal
        
        # Create complete dataframe
        df_complete = pd.DataFrame(data, index=timestamps)
        
        # Create missing data
        df_missing = df_complete.copy()
        
        if pattern == 'random':
            # Random missing
            mask = np.random.random((n_samples, n_features)) < missing_rate
            df_missing[mask] = np.nan
            
        elif pattern == 'blocks':
            # Block missing (sensor failures)
            for _ in range(int(n_samples * missing_rate / 50)):
                start = np.random.randint(0, n_samples - 50)
                col = np.random.randint(0, n_features)
                df_missing.iloc[start:start+50, col] = np.nan
                
        elif pattern == 'systematic':
            # Systematic missing (e.g., maintenance)
            for i in range(0, n_samples, 168):  # Weekly
                df_missing.iloc[i:i+12, :] = np.nan
        
        return df_complete, df_missing
    
    @staticmethod
    def generate_spatial_data(
        n_stations: int = 10,
        n_timesteps: int = 100,
        n_variables: int = 3,
        missing_rate: float = 0.2,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Generate spatial-temporal data."""
        np.random.seed(seed)
        
        # Generate station coordinates
        coords = np.random.uniform(-50, -20, size=(n_stations, 2))  # São Paulo region
        
        # Generate spatially correlated data
        data_complete = []
        
        for t in range(n_timesteps):
            values = []
            for v in range(n_variables):
                # Spatial Gaussian process
                distances = np.sqrt(((coords[:, None] - coords) ** 2).sum(axis=2))
                cov_matrix = np.exp(-distances / 10)  # Spatial correlation
                mean = 50 + 10 * np.sin(2 * np.pi * t / 24)  # Temporal pattern
                station_values = np.random.multivariate_normal(
                    mean * np.ones(n_stations), 
                    cov_matrix * 25
                )
                values.append(station_values)
            data_complete.append(np.array(values).T)
        
        data_complete = np.array(data_complete)  # Shape: (timesteps, stations, variables)
        
        # Flatten for DataFrame
        timestamps = pd.date_range('2024-01-01', periods=n_timesteps, freq='H')
        columns = []
        data_dict = {}
        
        for s in range(n_stations):
            for v in range(n_variables):
                col_name = f'station_{s}_var_{v}'
                columns.append(col_name)
                data_dict[col_name] = data_complete[:, s, v]
        
        df_complete = pd.DataFrame(data_dict, index=timestamps)
        
        # Add missing values
        df_missing = df_complete.copy()
        mask = np.random.random(df_missing.shape) < missing_rate
        df_missing[mask] = np.nan
        
        return df_complete, df_missing, coords


class TestSimpleImputer:
    """Test simple imputation methods."""
    
    def test_mean_imputation(self):
        """Test mean imputation."""
        df_complete, df_missing = TestDataGenerator.generate_time_series(
            n_samples=100, missing_rate=0.3
        )
        
        imputer = SimpleImputer(method='mean')
        result = imputer.impute(df_missing.values)
        
        # Check no missing values
        assert not np.isnan(result.imputed_data).any()
        
        # Check mean preservation
        for col in range(df_missing.shape[1]):
            original_mean = np.nanmean(df_missing.iloc[:, col])
            imputed_mean = np.mean(result.imputed_data[:, col])
            assert np.abs(original_mean - imputed_mean) < 0.1
    
    def test_forward_fill(self):
        """Test forward fill imputation."""
        df_complete, df_missing = TestDataGenerator.generate_time_series(
            n_samples=100, missing_rate=0.2
        )
        
        imputer = SimpleImputer(method='forward_fill', limit=5)
        result = imputer.impute(df_missing.values)
        
        # Check that gaps larger than limit are not filled
        for col in range(df_missing.shape[1]):
            missing_mask = np.isnan(df_missing.iloc[:, col])
            gap_lengths = []
            current_gap = 0
            
            for is_missing in missing_mask:
                if is_missing:
                    current_gap += 1
                else:
                    if current_gap > 0:
                        gap_lengths.append(current_gap)
                    current_gap = 0
            
            # Check remaining NaNs correspond to large gaps
            remaining_nans = np.isnan(result.imputed_data[:, col])
            if remaining_nans.any():
                assert max(gap_lengths) > 5
    
    def test_backward_fill(self):
        """Test backward fill imputation."""
        df_complete, df_missing = TestDataGenerator.generate_time_series()
        
        imputer = SimpleImputer(method='backward_fill')
        result = imputer.impute(df_missing.values)
        
        # Check propagation direction
        data = df_missing.values.copy()
        for col in range(data.shape[1]):
            col_data = data[:, col]
            for i in range(len(col_data) - 2, -1, -1):
                if np.isnan(col_data[i]) and not np.isnan(col_data[i + 1]):
                    assert result.imputed_data[i, col] == result.imputed_data[i + 1, col]


class TestInterpolationImputer:
    """Test interpolation-based imputation methods."""
    
    def test_linear_interpolation(self):
        """Test linear interpolation."""
        # Create simple test case
        data = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        df = pd.DataFrame({'value': data})
        
        imputer = InterpolationImputer(method='linear')
        result = imputer.impute(df.values)
        
        # Check interpolated values
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(result.imputed_data[:, 0], expected)
    
    def test_spline_interpolation(self):
        """Test spline interpolation."""
        df_complete, df_missing = TestDataGenerator.generate_time_series(
            n_samples=200, missing_rate=0.1
        )
        
        imputer = InterpolationImputer(method='spline', order=3)
        result = imputer.impute(df_missing.values)
        
        # Check smoothness (second derivative continuity)
        for col in range(result.imputed_data.shape[1]):
            second_diff = np.diff(result.imputed_data[:, col], n=2)
            # Spline should have smooth second derivative
            assert np.std(second_diff) < np.std(np.diff(df_missing.values[:, col], n=2))
    
    def test_seasonal_decomposition(self):
        """Test seasonal decomposition interpolation."""
        # Generate data with clear seasonality
        t = np.arange(168 * 4)  # 4 weeks of hourly data
        seasonal = 50 + 20 * np.sin(2 * np.pi * t / 168)  # Weekly pattern
        trend = 0.01 * t
        noise = np.random.randn(len(t)) * 2
        data = seasonal + trend + noise
        
        # Add missing values
        missing_mask = np.random.random(len(data)) < 0.2
        data[missing_mask] = np.nan
        
        df = pd.DataFrame({'value': data}, 
                         index=pd.date_range('2024-01-01', periods=len(data), freq='H'))
        
        imputer = InterpolationImputer(method='seasonal', period=168)
        result = imputer.impute(df.values)
        
        # Check that seasonal pattern is preserved
        imputed_fft = np.fft.fft(result.imputed_data[:, 0])
        peak_freq = np.argmax(np.abs(imputed_fft[1:len(imputed_fft)//2])) + 1
        expected_freq = len(data) / 168  # Weekly frequency
        
        assert abs(peak_freq - expected_freq) <= 1


class TestStatisticalImputer:
    """Test statistical imputation methods."""
    
    def test_kalman_filter(self):
        """Test Kalman filter imputation."""
        # Generate AR(1) process
        n = 200
        phi = 0.8
        sigma = 1.0
        
        data = np.zeros(n)
        data[0] = np.random.randn()
        for i in range(1, n):
            data[i] = phi * data[i-1] + sigma * np.random.randn()
        
        # Add missing values
        missing_mask = np.random.random(n) < 0.3
        data_missing = data.copy()
        data_missing[missing_mask] = np.nan
        
        df = pd.DataFrame({'value': data_missing})
        
        imputer = StatisticalImputer(
            method='kalman_filter',
            state_dim=1,
            observation_dim=1
        )
        result = imputer.impute(df.values)
        
        # Check that AR structure is preserved
        imputed_data = result.imputed_data[:, 0]
        
        # Estimate AR coefficient from imputed data
        valid_idx = ~missing_mask
        X = imputed_data[:-1]
        y = imputed_data[1:]
        phi_est = np.corrcoef(X[valid_idx[1:]], y[valid_idx[1:]])[0, 1]
        
        assert abs(phi_est - phi) < 0.1
    
    def test_em_algorithm(self):
        """Test EM algorithm imputation."""
        # Generate multivariate normal data
        n = 500
        mean = [10, 20, 30]
        cov = [[1, 0.5, 0.3],
               [0.5, 1, 0.4],
               [0.3, 0.4, 1]]
        
        data = np.random.multivariate_normal(mean, cov, size=n)
        
        # Add missing values (MCAR)
        missing_mask = np.random.random((n, 3)) < 0.2
        data[missing_mask] = np.nan
        
        df = pd.DataFrame(data, columns=['A', 'B', 'C'])
        
        imputer = StatisticalImputer(method='em_algorithm', max_iter=100)
        result = imputer.impute(df.values)
        
        # Check covariance structure is preserved
        imputed_cov = np.cov(result.imputed_data.T)
        
        for i in range(3):
            for j in range(3):
                assert abs(imputed_cov[i, j] - cov[i][j]) < 0.2


class TestMachineLearningImputer:
    """Test machine learning imputation methods."""
    
    def test_random_forest(self):
        """Test Random Forest imputation."""
        df_complete, df_missing = TestDataGenerator.generate_time_series(
            n_samples=500, n_features=5, missing_rate=0.2
        )
        
        imputer = MachineLearningImputer(
            method='random_forest',
            n_estimators=50,
            max_depth=10
        )
        
        # Add temporal features
        result = imputer.impute(
            df_missing.values,
            timestamps=df_missing.index.to_numpy()
        )
        
        # Evaluate performance
        mask = np.isnan(df_missing.values)
        mae = np.mean(np.abs(
            result.imputed_data[mask] - df_complete.values[mask]
        ))
        
        # Should achieve reasonable accuracy
        assert mae < 10.0  # Adjust based on data scale
        
        # Check feature importance
        assert 'feature_importance' in result.metadata
        assert len(result.metadata['feature_importance']) > 0
    
    def test_neural_network(self):
        """Test neural network imputation."""
        df_complete, df_missing = TestDataGenerator.generate_time_series(
            n_samples=1000, n_features=3, missing_rate=0.15
        )
        
        imputer = MachineLearningImputer(
            method='neural_network',
            hidden_sizes=(50, 30, 50),
            learning_rate=0.001,
            max_iter=100
        )
        
        result = imputer.impute(df_missing.values)
        
        # Check convergence
        assert result.metadata['converged']
        
        # Check reconstruction quality
        mask = np.isnan(df_missing.values)
        mse = np.mean((result.imputed_data[mask] - df_complete.values[mask]) ** 2)
        assert mse < 50.0  # Adjust threshold as needed
    
    @pytest.mark.slow
    def test_deep_ar(self):
        """Test DeepAR imputation (if available)."""
        try:
            from airimpute.deep_learning_models import DeepARImputer
        except ImportError:
            pytest.skip("DeepAR not available")
        
        df_complete, df_missing = TestDataGenerator.generate_time_series(
            n_samples=2000, n_features=5, missing_rate=0.2
        )
        
        imputer = DeepARImputer(
            hidden_dim=40,
            num_layers=2,
            dropout=0.1,
            learning_rate=0.001
        )
        
        result = imputer.impute(
            df_missing.values,
            timestamps=df_missing.index
        )
        
        # Check uncertainty quantification
        assert 'prediction_intervals' in result.metadata
        lower, upper = result.metadata['prediction_intervals']
        
        # Check calibration
        mask = np.isnan(df_missing.values)
        coverage = np.mean(
            (df_complete.values[mask] >= lower[mask]) & 
            (df_complete.values[mask] <= upper[mask])
        )
        
        # Should be close to 95% for 95% prediction interval
        assert 0.90 <= coverage <= 0.98


class TestRAHImputer:
    """Test Regularized Anchor Huber (RAH) imputation."""
    
    def test_rah_basic(self):
        """Test basic RAH functionality."""
        # Generate correlated data
        n = 500
        p = 10
        
        # Create correlation structure
        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.7 ** abs(i - j)
        
        data = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
        
        # Add outliers
        outlier_idx = np.random.choice(n, size=int(0.05 * n), replace=False)
        data[outlier_idx] += np.random.randn(len(outlier_idx), p) * 10
        
        # Add missing values
        missing_mask = np.random.random((n, p)) < 0.2
        data[missing_mask] = np.nan
        
        imputer = RAHImputer(
            anchor_strength=0.1,
            huber_delta=1.5,
            max_iter=100
        )
        
        result = imputer.impute(data)
        
        # Check robustness to outliers
        imputed_cov = np.cov(result.imputed_data.T)
        
        # Correlation structure should be preserved despite outliers
        for i in range(p):
            for j in range(i+1, p):
                expected_corr = 0.7 ** abs(i - j)
                actual_corr = imputed_cov[i, j] / np.sqrt(imputed_cov[i, i] * imputed_cov[j, j])
                assert abs(actual_corr - expected_corr) < 0.2
    
    def test_rah_convergence(self):
        """Test RAH convergence properties."""
        df_complete, df_missing = TestDataGenerator.generate_time_series(
            n_samples=300, n_features=5, missing_rate=0.25
        )
        
        imputer = RAHImputer(
            anchor_strength=0.05,
            convergence_threshold=1e-4,
            max_iter=200
        )
        
        result = imputer.impute(df_missing.values)
        
        # Check convergence
        assert result.metadata['converged']
        assert result.metadata['n_iterations'] < 200
        
        # Check objective function decrease
        obj_values = result.metadata['objective_values']
        assert all(obj_values[i] >= obj_values[i+1] for i in range(len(obj_values)-1))


class TestValidation:
    """Test validation and cross-validation procedures."""
    
    def test_cross_validation(self):
        """Test k-fold cross-validation."""
        df_complete, df_missing = TestDataGenerator.generate_time_series(
            n_samples=1000, n_features=5, missing_rate=0.2
        )
        
        cv = CrossValidator(n_folds=5, random_state=42)
        
        # Test multiple imputers
        imputers = [
            SimpleImputer(method='mean'),
            InterpolationImputer(method='linear'),
            StatisticalImputer(method='kalman_filter'),
        ]
        
        results = {}
        for imputer in imputers:
            cv_results = cv.cross_validate(
                imputer,
                df_complete.values,
                df_missing.values
            )
            results[imputer.__class__.__name__] = cv_results
        
        # Check that we get metrics for each fold
        for name, cv_results in results.items():
            assert len(cv_results['rmse']) == 5
            assert len(cv_results['mae']) == 5
            assert all(score > 0 for score in cv_results['rmse'])
    
    def test_validation_metrics(self):
        """Test calculation of validation metrics."""
        true_values = np.array([1, 2, 3, 4, 5])
        predicted_values = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
        
        metrics = ValidationMetrics.calculate_metrics(true_values, predicted_values)
        
        # Check metric values
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics
        
        # Verify calculations
        expected_rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
        assert abs(metrics['rmse'] - expected_rmse) < 1e-6
        
        expected_mae = np.mean(np.abs(true_values - predicted_values))
        assert abs(metrics['mae'] - expected_mae) < 1e-6
    
    def test_pattern_specific_evaluation(self):
        """Test evaluation for different missing patterns."""
        patterns = ['random', 'blocks', 'systematic']
        
        for pattern in patterns:
            df_complete, df_missing = TestDataGenerator.generate_time_series(
                n_samples=500,
                missing_rate=0.2,
                pattern=pattern
            )
            
            imputer = InterpolationImputer(method='linear')
            result = imputer.impute(df_missing.values)
            
            # Evaluate based on pattern
            mask = np.isnan(df_missing.values)
            
            if pattern == 'random':
                # Should work well for random missing
                mae = np.mean(np.abs(
                    result.imputed_data[mask] - df_complete.values[mask]
                ))
                assert mae < 5.0
                
            elif pattern == 'blocks':
                # Longer gaps should have higher error
                gap_errors = []
                for col in range(df_missing.shape[1]):
                    col_mask = mask[:, col]
                    if col_mask.any():
                        errors = np.abs(
                            result.imputed_data[col_mask, col] - 
                            df_complete.values[col_mask, col]
                        )
                        gap_errors.extend(errors)
                
                assert len(gap_errors) > 0
                assert np.mean(gap_errors) > 3.0  # Higher error for blocks


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_all_missing_column(self):
        """Test handling of completely missing columns."""
        data = np.random.randn(100, 5)
        data[:, 2] = np.nan  # Entire column missing
        
        imputer = SimpleImputer(method='mean')
        
        with pytest.warns(UserWarning, match="completely missing"):
            result = imputer.impute(data)
        
        # Should fill with 0 or handle gracefully
        assert not np.isnan(result.imputed_data).all()
    
    def test_single_value_imputation(self):
        """Test imputation with very few observed values."""
        data = np.full((10, 3), np.nan)
        data[0, 0] = 5.0
        data[5, 1] = 10.0
        data[9, 2] = 15.0
        
        imputer = SimpleImputer(method='mean')
        result = imputer.impute(data)
        
        # Should use the single values appropriately
        assert result.imputed_data[0, 0] == 5.0
        assert np.all(result.imputed_data[:, 0] == 5.0)
    
    def test_monotonic_constraints(self):
        """Test imputation with monotonic constraints."""
        # Create monotonically increasing data
        data = np.arange(100, dtype=float)
        missing_mask = np.random.random(100) < 0.3
        data[missing_mask] = np.nan
        
        df = pd.DataFrame({'value': data})
        
        imputer = InterpolationImputer(
            method='linear',
            limit_direction='both'
        )
        
        result = imputer.impute(df.values)
        
        # Check monotonicity is preserved
        imputed = result.imputed_data[:, 0]
        assert all(imputed[i] <= imputed[i+1] for i in range(len(imputed)-1))


class TestPerformance:
    """Test performance and scalability."""
    
    @pytest.mark.benchmark
    def test_large_dataset_performance(self, benchmark):
        """Benchmark imputation on large datasets."""
        df_complete, df_missing = TestDataGenerator.generate_time_series(
            n_samples=10000,
            n_features=20,
            missing_rate=0.2
        )
        
        imputer = InterpolationImputer(method='linear')
        
        # Benchmark the imputation
        result = benchmark(imputer.impute, df_missing.values)
        
        assert not np.isnan(result.imputed_data).any()
    
    def test_memory_efficiency(self):
        """Test memory usage remains reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Large dataset
        df_complete, df_missing = TestDataGenerator.generate_time_series(
            n_samples=50000,
            n_features=10,
            missing_rate=0.2
        )
        
        imputer = SimpleImputer(method='mean')
        result = imputer.impute(df_missing.values)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory (less than 500MB increase)
        assert memory_increase < 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])