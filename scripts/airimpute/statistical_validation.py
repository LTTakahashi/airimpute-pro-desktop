"""
Comprehensive statistical validation framework for imputation methods
Implements rigorous statistical tests and validation procedures for academic standards
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    normaltest, jarque_bera, shapiro,
    kstest, anderson, chi2_contingency,
    friedmanchisquare, wilcoxon, mannwhitneyu,
    pearsonr, spearmanr, kendalltau
)
from statsmodels.stats.diagnostic import (
    acorr_ljungbox, het_breuschpagan,
    het_white, normal_ad
)
from statsmodels.tsa.stattools import adfuller, kpss, pacf_ols, acf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation test results"""
    test_name: str
    statistic: float
    p_value: float
    passed: bool
    message: str
    metadata: Dict[str, Any]


@dataclass
class DiagnosticResult:
    """Container for diagnostic test results"""
    residuals: np.ndarray
    fitted_values: np.ndarray
    normality_tests: Dict[str, ValidationResult]
    independence_tests: Dict[str, ValidationResult]
    homoscedasticity_tests: Dict[str, ValidationResult]
    overall_validity: bool


class StatisticalValidator(ABC):
    """Abstract base class for statistical validators"""
    
    @abstractmethod
    def validate(self, data: np.ndarray, **kwargs) -> ValidationResult:
        """Perform validation test"""
        pass
    
    @abstractmethod
    def get_assumptions(self) -> List[str]:
        """Return list of assumptions for this validator"""
        pass


class NormalityValidator(StatisticalValidator):
    """Comprehensive normality testing"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def validate(self, data: np.ndarray, method: str = 'all') -> Dict[str, ValidationResult]:
        """
        Test for normality using multiple methods
        
        Parameters:
        -----------
        data : np.ndarray
            Data to test for normality
        method : str
            'all' or specific test name
        
        Returns:
        --------
        Dict[str, ValidationResult]
            Results from each normality test
        """
        results = {}
        
        # Remove NaN values
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 8:
            return {
                'insufficient_data': ValidationResult(
                    test_name='normality',
                    statistic=np.nan,
                    p_value=np.nan,
                    passed=False,
                    message='Insufficient data for normality testing',
                    metadata={'n_samples': len(clean_data)}
                )
            }
        
        # Shapiro-Wilk test (best for small samples)
        if method in ['all', 'shapiro'] and len(clean_data) <= 5000:
            stat, p_value = shapiro(clean_data)
            results['shapiro_wilk'] = ValidationResult(
                test_name='Shapiro-Wilk',
                statistic=stat,
                p_value=p_value,
                passed=p_value > self.alpha,
                message=f"Data {'appears' if p_value > self.alpha else 'does not appear'} to be normally distributed",
                metadata={'n_samples': len(clean_data)}
            )
        
        # Jarque-Bera test
        if method in ['all', 'jarque_bera']:
            stat, p_value = jarque_bera(clean_data)
            results['jarque_bera'] = ValidationResult(
                test_name='Jarque-Bera',
                statistic=stat,
                p_value=p_value,
                passed=p_value > self.alpha,
                message=f"Skewness and kurtosis {'match' if p_value > self.alpha else 'do not match'} normal distribution",
                metadata={
                    'skewness': stats.skew(clean_data),
                    'kurtosis': stats.kurtosis(clean_data)
                }
            )
        
        # D'Agostino-Pearson test
        if method in ['all', 'dagostino'] and len(clean_data) >= 20:
            stat, p_value = normaltest(clean_data)
            results['dagostino_pearson'] = ValidationResult(
                test_name="D'Agostino-Pearson",
                statistic=stat,
                p_value=p_value,
                passed=p_value > self.alpha,
                message=f"Combined skewness/kurtosis test {'passed' if p_value > self.alpha else 'failed'}",
                metadata={'min_samples_required': 20}
            )
        
        # Anderson-Darling test
        if method in ['all', 'anderson']:
            result = anderson(clean_data, dist='norm')
            # Use 5% significance level
            critical_value = result.critical_values[2]
            passed = result.statistic < critical_value
            results['anderson_darling'] = ValidationResult(
                test_name='Anderson-Darling',
                statistic=result.statistic,
                p_value=np.nan,  # A-D doesn't provide p-value
                passed=passed,
                message=f"Test statistic {'below' if passed else 'above'} critical value at 5% level",
                metadata={
                    'critical_values': dict(zip(result.significance_level, result.critical_values)),
                    'significance_level': 5.0
                }
            )
        
        # Kolmogorov-Smirnov test
        if method in ['all', 'ks']:
            # Standardize data for KS test
            standardized = (clean_data - np.mean(clean_data)) / np.std(clean_data)
            stat, p_value = kstest(standardized, 'norm')
            results['kolmogorov_smirnov'] = ValidationResult(
                test_name='Kolmogorov-Smirnov',
                statistic=stat,
                p_value=p_value,
                passed=p_value > self.alpha,
                message=f"Distribution {'matches' if p_value > self.alpha else 'differs from'} standard normal",
                metadata={'standardized': True}
            )
        
        return results
    
    def get_assumptions(self) -> List[str]:
        return [
            "Data should be continuous",
            "Sample should be random",
            "No significant outliers (for some tests)",
            "Sufficient sample size (varies by test)"
        ]


class IndependenceValidator(StatisticalValidator):
    """Test for independence and autocorrelation"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def validate(self, data: np.ndarray, max_lag: Optional[int] = None) -> Dict[str, ValidationResult]:
        """
        Test for independence using multiple methods
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data to test
        max_lag : int, optional
            Maximum lag for autocorrelation tests
        
        Returns:
        --------
        Dict[str, ValidationResult]
            Results from independence tests
        """
        results = {}
        
        # Clean data
        clean_data = data[~np.isnan(data)]
        n = len(clean_data)
        
        if n < 10:
            return {
                'insufficient_data': ValidationResult(
                    test_name='independence',
                    statistic=np.nan,
                    p_value=np.nan,
                    passed=False,
                    message='Insufficient data for independence testing',
                    metadata={'n_samples': n}
                )
            }
        
        # Determine max lag if not specified
        if max_lag is None:
            max_lag = min(int(np.sqrt(n)), n // 4)
        
        # Ljung-Box test
        lb_result = acorr_ljungbox(clean_data, lags=max_lag, return_df=True)
        
        # Get minimum p-value across all lags
        min_p_value = lb_result['lb_pvalue'].min()
        lag_min_p = lb_result['lb_pvalue'].idxmin()
        
        results['ljung_box'] = ValidationResult(
            test_name='Ljung-Box',
            statistic=lb_result.loc[lag_min_p, 'lb_stat'],
            p_value=min_p_value,
            passed=min_p_value > self.alpha,
            message=f"{'No significant' if min_p_value > self.alpha else 'Significant'} autocorrelation detected",
            metadata={
                'max_lag_tested': max_lag,
                'critical_lag': lag_min_p,
                'all_p_values': lb_result['lb_pvalue'].to_dict()
            }
        )
        
        # Runs test for randomness
        median = np.median(clean_data)
        runs, n_pos, n_neg = self._count_runs(clean_data >= median)
        
        if n_pos > 0 and n_neg > 0:
            # Expected runs and variance
            expected_runs = (2 * n_pos * n_neg) / (n_pos + n_neg) + 1
            variance = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)) / \
                       ((n_pos + n_neg) ** 2 * (n_pos + n_neg - 1))
            
            if variance > 0:
                z_stat = (runs - expected_runs) / np.sqrt(variance)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
                results['runs_test'] = ValidationResult(
                    test_name='Runs Test',
                    statistic=z_stat,
                    p_value=p_value,
                    passed=p_value > self.alpha,
                    message=f"Data {'appears' if p_value > self.alpha else 'does not appear'} to be random",
                    metadata={
                        'n_runs': runs,
                        'expected_runs': expected_runs,
                        'n_positive': n_pos,
                        'n_negative': n_neg
                    }
                )
        
        # Durbin-Watson test for first-order autocorrelation
        if len(clean_data) > 1:
            residuals = clean_data[1:] - clean_data[:-1]
            dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
            
            # Rough interpretation (exact critical values depend on regression)
            independence = 1.5 < dw_stat < 2.5
            
            results['durbin_watson'] = ValidationResult(
                test_name='Durbin-Watson',
                statistic=dw_stat,
                p_value=np.nan,  # No direct p-value
                passed=independence,
                message=f"{'No significant' if independence else 'Significant'} first-order autocorrelation",
                metadata={
                    'interpretation': 'Positive autocorrelation' if dw_stat < 1.5 
                                    else 'Negative autocorrelation' if dw_stat > 2.5 
                                    else 'No autocorrelation'
                }
            )
        
        return results
    
    def _count_runs(self, binary_sequence: np.ndarray) -> Tuple[int, int, int]:
        """Count runs in a binary sequence"""
        runs = 1
        for i in range(1, len(binary_sequence)):
            if binary_sequence[i] != binary_sequence[i-1]:
                runs += 1
        
        n_true = np.sum(binary_sequence)
        n_false = len(binary_sequence) - n_true
        
        return runs, n_true, n_false
    
    def get_assumptions(self) -> List[str]:
        return [
            "Observations should be ordered (time series)",
            "No missing values in the sequence",
            "Sufficient number of observations"
        ]


class StationarityValidator(StatisticalValidator):
    """Test for stationarity in time series"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def validate(self, data: np.ndarray) -> Dict[str, ValidationResult]:
        """
        Test for stationarity using multiple methods
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        
        Returns:
        --------
        Dict[str, ValidationResult]
            Results from stationarity tests
        """
        results = {}
        
        # Clean data
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 20:
            return {
                'insufficient_data': ValidationResult(
                    test_name='stationarity',
                    statistic=np.nan,
                    p_value=np.nan,
                    passed=False,
                    message='Insufficient data for stationarity testing',
                    metadata={'n_samples': len(clean_data)}
                )
            }
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(clean_data, autolag='AIC')
        results['adf'] = ValidationResult(
            test_name='Augmented Dickey-Fuller',
            statistic=adf_result[0],
            p_value=adf_result[1],
            passed=adf_result[1] < self.alpha,  # Reject null = stationary
            message=f"Series {'is' if adf_result[1] < self.alpha else 'is not'} stationary",
            metadata={
                'used_lag': adf_result[2],
                'n_obs': adf_result[3],
                'critical_values': adf_result[4]
            }
        )
        
        # KPSS test
        kpss_result = kpss(clean_data, regression='c', nlags='auto')
        results['kpss'] = ValidationResult(
            test_name='KPSS',
            statistic=kpss_result[0],
            p_value=kpss_result[1],
            passed=kpss_result[1] > self.alpha,  # Fail to reject null = stationary
            message=f"Series {'is' if kpss_result[1] > self.alpha else 'is not'} stationary",
            metadata={
                'used_lag': kpss_result[2],
                'critical_values': kpss_result[3]
            }
        )
        
        # Combined interpretation
        adf_stationary = results['adf'].passed
        kpss_stationary = results['kpss'].passed
        
        if adf_stationary and kpss_stationary:
            interpretation = "Both tests indicate stationarity"
        elif not adf_stationary and not kpss_stationary:
            interpretation = "Both tests indicate non-stationarity"
        elif adf_stationary and not kpss_stationary:
            interpretation = "Difference stationary (unit root)"
        else:
            interpretation = "Trend stationary"
        
        results['combined'] = ValidationResult(
            test_name='Combined ADF-KPSS',
            statistic=np.nan,
            p_value=np.nan,
            passed=adf_stationary and kpss_stationary,
            message=interpretation,
            metadata={
                'adf_stationary': adf_stationary,
                'kpss_stationary': kpss_stationary
            }
        )
        
        return results
    
    def get_assumptions(self) -> List[str]:
        return [
            "Time series data with regular intervals",
            "No structural breaks",
            "Sufficient observations (>20)",
            "No missing values"
        ]


class HomoscedasticityValidator(StatisticalValidator):
    """Test for homoscedasticity (constant variance)"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def validate(self, residuals: np.ndarray, 
                fitted_values: Optional[np.ndarray] = None,
                exog: Optional[np.ndarray] = None) -> Dict[str, ValidationResult]:
        """
        Test for homoscedasticity
        
        Parameters:
        -----------
        residuals : np.ndarray
            Residuals from model
        fitted_values : np.ndarray, optional
            Fitted values from model
        exog : np.ndarray, optional
            Exogenous variables for White test
        
        Returns:
        --------
        Dict[str, ValidationResult]
            Results from homoscedasticity tests
        """
        results = {}
        
        # Clean data
        mask = ~np.isnan(residuals)
        clean_residuals = residuals[mask]
        
        if fitted_values is not None:
            clean_fitted = fitted_values[mask]
        else:
            # Use index as proxy for fitted values
            clean_fitted = np.arange(len(clean_residuals))
        
        if len(clean_residuals) < 10:
            return {
                'insufficient_data': ValidationResult(
                    test_name='homoscedasticity',
                    statistic=np.nan,
                    p_value=np.nan,
                    passed=False,
                    message='Insufficient data for homoscedasticity testing',
                    metadata={'n_samples': len(clean_residuals)}
                )
            }
        
        # Breusch-Pagan test
        try:
            # Prepare data for test
            y = clean_residuals ** 2
            X = np.column_stack([np.ones_like(clean_fitted), clean_fitted])
            
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(y, X)
            
            results['breusch_pagan'] = ValidationResult(
                test_name='Breusch-Pagan',
                statistic=bp_stat,
                p_value=bp_pvalue,
                passed=bp_pvalue > self.alpha,
                message=f"{'Constant' if bp_pvalue > self.alpha else 'Non-constant'} variance detected",
                metadata={'method': 'Lagrange multiplier test'}
            )
        except Exception as e:
            logger.warning(f"Breusch-Pagan test failed: {e}")
        
        # Goldfeld-Quandt test
        if len(clean_residuals) >= 20:
            try:
                # Sort by fitted values
                sorted_idx = np.argsort(clean_fitted)
                sorted_residuals = clean_residuals[sorted_idx]
                
                # Split into two groups
                n_drop = len(sorted_residuals) // 10  # Drop middle 10%
                n_group = (len(sorted_residuals) - n_drop) // 2
                
                group1 = sorted_residuals[:n_group]
                group2 = sorted_residuals[-n_group:]
                
                # F-test for equal variances
                var1 = np.var(group1, ddof=1)
                var2 = np.var(group2, ddof=1)
                
                if var1 > 0 and var2 > 0:
                    f_stat = max(var1, var2) / min(var1, var2)
                    df1 = df2 = n_group - 1
                    p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 
                                     1 - stats.f.cdf(f_stat, df1, df2))
                    
                    results['goldfeld_quandt'] = ValidationResult(
                        test_name='Goldfeld-Quandt',
                        statistic=f_stat,
                        p_value=p_value,
                        passed=p_value > self.alpha,
                        message=f"{'Equal' if p_value > self.alpha else 'Unequal'} variances between groups",
                        metadata={
                            'group1_variance': var1,
                            'group2_variance': var2,
                            'n_per_group': n_group
                        }
                    )
            except Exception as e:
                logger.warning(f"Goldfeld-Quandt test failed: {e}")
        
        # White test (if exogenous variables provided)
        if exog is not None and len(clean_residuals) > exog.shape[1] * 3:
            try:
                white_stat, white_pvalue, _, _ = het_white(clean_residuals, exog)
                
                results['white'] = ValidationResult(
                    test_name='White',
                    statistic=white_stat,
                    p_value=white_pvalue,
                    passed=white_pvalue > self.alpha,
                    message=f"{'No' if white_pvalue > self.alpha else 'Significant'} heteroscedasticity detected",
                    metadata={'includes_cross_terms': True}
                )
            except Exception as e:
                logger.warning(f"White test failed: {e}")
        
        return results
    
    def get_assumptions(self) -> List[str]:
        return [
            "Residuals from a fitted model",
            "Independent observations",
            "No perfect multicollinearity (for White test)",
            "Sufficient sample size"
        ]


class MissingDataValidator:
    """Validate and characterize missing data patterns"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def validate_missing_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing data patterns and test MCAR assumption
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with missing values
        
        Returns:
        --------
        Dict[str, Any]
            Analysis of missing data patterns
        """
        results = {
            'summary': self._missing_summary(data),
            'pattern_tests': {},
            'correlations': {},
            'likely_mechanism': None
        }
        
        # Little's MCAR test
        mcar_result = self._little_mcar_test(data)
        if mcar_result:
            results['pattern_tests']['little_mcar'] = mcar_result
        
        # Test for patterns in missingness
        for col in data.columns:
            if data[col].isnull().any() and not data[col].isnull().all():
                # Test if missingness depends on other variables
                results['correlations'][col] = self._test_missing_correlations(
                    data, col
                )
        
        # Determine likely missing mechanism
        results['likely_mechanism'] = self._determine_mechanism(results)
        
        return results
    
    def _missing_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for missing data"""
        n_rows, n_cols = data.shape
        
        summary = {
            'total_cells': n_rows * n_cols,
            'missing_cells': data.isnull().sum().sum(),
            'complete_rows': len(data.dropna()),
            'complete_columns': len(data.dropna(axis=1).columns),
            'by_column': {},
            'by_row': {}
        }
        
        # Column-wise analysis
        for col in data.columns:
            n_missing = data[col].isnull().sum()
            summary['by_column'][col] = {
                'n_missing': n_missing,
                'pct_missing': n_missing / n_rows * 100,
                'longest_gap': self._longest_consecutive_missing(data[col])
            }
        
        # Row-wise analysis
        row_missing = data.isnull().sum(axis=1)
        summary['by_row'] = {
            'mean_missing_per_row': row_missing.mean(),
            'max_missing_per_row': row_missing.max(),
            'rows_with_any_missing': (row_missing > 0).sum()
        }
        
        return summary
    
    def _longest_consecutive_missing(self, series: pd.Series) -> int:
        """Find longest consecutive missing values"""
        is_missing = series.isnull().astype(int)
        
        # Find changes from non-missing to missing
        change_points = is_missing.diff().fillna(0)
        
        # Count consecutive missing
        max_consecutive = 0
        current_consecutive = 0
        
        for val in is_missing:
            if val:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _little_mcar_test(self, data: pd.DataFrame) -> Optional[ValidationResult]:
        """
        Simplified version of Little's MCAR test
        
        Note: Full implementation requires iterative EM algorithm
        This is a simplified chi-square test for MCAR
        """
        # Remove columns with no missing or all missing
        valid_cols = [col for col in data.columns 
                     if data[col].isnull().any() and not data[col].isnull().all()]
        
        if len(valid_cols) < 2:
            return None
        
        # Create missing indicator matrix
        missing_indicators = data[valid_cols].isnull().astype(int)
        
        # Test independence of missing patterns
        # This is a simplified approach
        n_patterns = len(missing_indicators.drop_duplicates())
        n_expected = 2 ** len(valid_cols)
        
        # Chi-square test for uniform distribution of patterns
        pattern_counts = missing_indicators.value_counts()
        expected_count = len(data) / n_patterns
        
        chi2_stat = sum((count - expected_count) ** 2 / expected_count 
                       for count in pattern_counts)
        
        df = n_patterns - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
        return ValidationResult(
            test_name="Little's MCAR Test (Simplified)",
            statistic=chi2_stat,
            p_value=p_value,
            passed=p_value > self.alpha,
            message=f"Missing data {'appears' if p_value > self.alpha else 'does not appear'} to be MCAR",
            metadata={
                'n_patterns': n_patterns,
                'degrees_of_freedom': df,
                'note': 'Simplified implementation'
            }
        )
    
    def _test_missing_correlations(self, data: pd.DataFrame, 
                                  target_col: str) -> Dict[str, float]:
        """Test if missingness in target_col correlates with other variables"""
        correlations = {}
        missing_indicator = data[target_col].isnull().astype(int)
        
        for col in data.columns:
            if col != target_col and not data[col].isnull().all():
                # For numeric columns
                if pd.api.types.is_numeric_dtype(data[col]):
                    # Point-biserial correlation
                    clean_mask = ~data[col].isnull()
                    if clean_mask.sum() > 10:
                        corr, p_value = stats.pointbiserialr(
                            missing_indicator[clean_mask],
                            data[col][clean_mask]
                        )
                        correlations[col] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'significant': p_value < self.alpha
                        }
        
        return correlations
    
    def _determine_mechanism(self, results: Dict[str, Any]) -> str:
        """Determine likely missing data mechanism"""
        
        # Check MCAR test
        if 'little_mcar' in results['pattern_tests']:
            if results['pattern_tests']['little_mcar'].passed:
                return 'MCAR'
        
        # Check for correlations
        significant_correlations = 0
        total_correlations = 0
        
        for col_results in results['correlations'].values():
            for corr_info in col_results.values():
                if isinstance(corr_info, dict) and 'significant' in corr_info:
                    total_correlations += 1
                    if corr_info['significant']:
                        significant_correlations += 1
        
        if total_correlations > 0:
            correlation_ratio = significant_correlations / total_correlations
            
            if correlation_ratio < 0.1:
                return 'MCAR'
            elif correlation_ratio < 0.5:
                return 'MAR'
            else:
                return 'MNAR'
        
        return 'Unknown'


class ComprehensiveValidator:
    """Orchestrates all validation tests"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.normality_validator = NormalityValidator(alpha)
        self.independence_validator = IndependenceValidator(alpha)
        self.stationarity_validator = StationarityValidator(alpha)
        self.homoscedasticity_validator = HomoscedasticityValidator(alpha)
        self.missing_data_validator = MissingDataValidator(alpha)
    
    def validate_imputation_results(self, 
                                  original_data: pd.DataFrame,
                                  imputed_data: pd.DataFrame,
                                  method_name: str) -> Dict[str, Any]:
        """
        Comprehensive validation of imputation results
        
        Parameters:
        -----------
        original_data : pd.DataFrame
            Original data with missing values
        imputed_data : pd.DataFrame
            Imputed data
        method_name : str
            Name of imputation method
        
        Returns:
        --------
        Dict[str, Any]
            Comprehensive validation results
        """
        results = {
            'method': method_name,
            'timestamp': pd.Timestamp.now(),
            'data_characteristics': self._analyze_data_characteristics(original_data),
            'preservation_tests': {},
            'statistical_tests': {},
            'quality_metrics': {},
            'recommendations': []
        }
        
        # Test statistical property preservation
        for col in imputed_data.select_dtypes(include=[np.number]).columns:
            if col in original_data.columns:
                results['preservation_tests'][col] = self._test_preservation(
                    original_data[col].dropna().values,
                    imputed_data[col].values
                )
        
        # Run statistical tests on imputed data
        for col in imputed_data.select_dtypes(include=[np.number]).columns:
            col_results = {}
            
            # Normality tests
            col_results['normality'] = self.normality_validator.validate(
                imputed_data[col].values
            )
            
            # Independence tests (for time series)
            if len(imputed_data) > 20:
                col_results['independence'] = self.independence_validator.validate(
                    imputed_data[col].values
                )
            
            # Stationarity tests (for time series)
            if len(imputed_data) > 50:
                col_results['stationarity'] = self.stationarity_validator.validate(
                    imputed_data[col].values
                )
            
            results['statistical_tests'][col] = col_results
        
        # Calculate quality metrics
        results['quality_metrics'] = self._calculate_quality_metrics(
            original_data, imputed_data
        )
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of the data"""
        return {
            'shape': data.shape,
            'missing_percentage': (data.isnull().sum().sum() / 
                                 (data.shape[0] * data.shape[1]) * 100),
            'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
            'datetime_columns': data.select_dtypes(include=['datetime']).columns.tolist(),
            'missing_pattern': self.missing_data_validator.validate_missing_pattern(data)
        }
    
    def _test_preservation(self, original: np.ndarray, 
                          imputed: np.ndarray) -> Dict[str, Any]:
        """Test if statistical properties are preserved"""
        # Only compare non-imputed values
        mask = ~np.isnan(original)
        
        preservation_results = {
            'mean': {
                'original': np.mean(original),
                'imputed': np.mean(imputed),
                'relative_change': abs(np.mean(imputed) - np.mean(original)) / 
                                 abs(np.mean(original)) if np.mean(original) != 0 else np.inf
            },
            'std': {
                'original': np.std(original),
                'imputed': np.std(imputed),
                'relative_change': abs(np.std(imputed) - np.std(original)) / 
                                 abs(np.std(original)) if np.std(original) != 0 else np.inf
            },
            'skewness': {
                'original': stats.skew(original),
                'imputed': stats.skew(imputed),
                'relative_change': abs(stats.skew(imputed) - stats.skew(original)) / 
                                 abs(stats.skew(original)) if stats.skew(original) != 0 else np.inf
            },
            'kurtosis': {
                'original': stats.kurtosis(original),
                'imputed': stats.kurtosis(imputed),
                'relative_change': abs(stats.kurtosis(imputed) - stats.kurtosis(original)) / 
                                 abs(stats.kurtosis(original)) if stats.kurtosis(original) != 0 else np.inf
            }
        }
        
        # Distribution similarity test
        if len(original) > 10:
            ks_stat, ks_pvalue = stats.ks_2samp(original, imputed)
            preservation_results['distribution_similarity'] = {
                'ks_statistic': ks_stat,
                'p_value': ks_pvalue,
                'similar': ks_pvalue > self.alpha
            }
        
        return preservation_results
    
    def _calculate_quality_metrics(self, original: pd.DataFrame, 
                                 imputed: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall quality metrics"""
        metrics = {}
        
        # Coverage (how many missing values were filled)
        original_missing = original.isnull().sum().sum()
        imputed_missing = imputed.isnull().sum().sum()
        
        metrics['coverage'] = {
            'original_missing': original_missing,
            'imputed_missing': imputed_missing,
            'coverage_rate': (original_missing - imputed_missing) / original_missing 
                           if original_missing > 0 else 1.0
        }
        
        # Plausibility checks
        numeric_cols = imputed.select_dtypes(include=[np.number]).columns
        
        plausibility_issues = []
        for col in numeric_cols:
            # Check for values outside original range
            if col in original.columns:
                orig_min = original[col].min()
                orig_max = original[col].max()
                imp_min = imputed[col].min()
                imp_max = imputed[col].max()
                
                if imp_min < orig_min or imp_max > orig_max:
                    plausibility_issues.append({
                        'column': col,
                        'issue': 'Values outside original range',
                        'original_range': (orig_min, orig_max),
                        'imputed_range': (imp_min, imp_max)
                    })
        
        metrics['plausibility'] = {
            'issues': plausibility_issues,
            'n_issues': len(plausibility_issues)
        }
        
        return metrics
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check missing data mechanism
        mechanism = results['data_characteristics']['missing_pattern']['likely_mechanism']
        if mechanism == 'MNAR':
            recommendations.append(
                "Data appears to be Missing Not At Random (MNAR). "
                "Consider using methods that model the missing mechanism explicitly."
            )
        
        # Check preservation of properties
        for col, preservation in results['preservation_tests'].items():
            if preservation['mean']['relative_change'] > 0.1:
                recommendations.append(
                    f"Column '{col}': Mean changed by >"
                    f"{preservation['mean']['relative_change']:.1%}. "
                    "Consider methods that better preserve central tendency."
                )
            
            if preservation['std']['relative_change'] > 0.2:
                recommendations.append(
                    f"Column '{col}': Standard deviation changed by "
                    f"{preservation['std']['relative_change']:.1%}. "
                    "Consider methods that better preserve variance."
                )
        
        # Check statistical test failures
        for col, tests in results['statistical_tests'].items():
            if 'normality' in tests:
                normal_tests_passed = sum(
                    1 for test in tests['normality'].values() 
                    if hasattr(test, 'passed') and test.passed
                )
                if normal_tests_passed == 0:
                    recommendations.append(
                        f"Column '{col}': Imputed data is not normally distributed. "
                        "If normality is required, consider transformation or "
                        "different imputation method."
                    )
        
        # Check quality metrics
        quality = results['quality_metrics']
        if quality['plausibility']['n_issues'] > 0:
            recommendations.append(
                f"Found {quality['plausibility']['n_issues']} plausibility issues. "
                "Review imputed values for physical constraints."
            )
        
        if not recommendations:
            recommendations.append(
                "Imputation results appear satisfactory. "
                "Consider validating on holdout data for final verification."
            )
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)
    n = 1000
    
    # Create time series with missing values
    t = np.arange(n)
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
        'value': 50 + 10 * np.sin(2 * np.pi * t / 168) + np.random.normal(0, 5, n)
    })
    
    # Add missing values (MAR pattern)
    mask = data['value'] > 55
    data.loc[mask, 'value'] = np.nan
    
    print(f"Created dataset with {data['value'].isnull().sum()} missing values")
    
    # Validate missing pattern
    validator = ComprehensiveValidator()
    missing_analysis = validator.missing_data_validator.validate_missing_pattern(data)
    
    print(f"\nMissing data mechanism: {missing_analysis['likely_mechanism']}")
    
    # Simple imputation for testing
    imputed_data = data.copy()
    imputed_data['value'].fillna(imputed_data['value'].mean(), inplace=True)
    
    # Comprehensive validation
    validation_results = validator.validate_imputation_results(
        data, imputed_data, 'mean_imputation'
    )
    
    print("\nValidation Summary:")
    print(f"Quality metrics: {validation_results['quality_metrics']}")
    print(f"\nRecommendations:")
    for rec in validation_results['recommendations']:
        print(f"- {rec}")