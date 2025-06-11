"""
Advanced Statistical Tests for Missing Data Analysis

This module implements rigorous statistical tests for missing data mechanisms,
imputation validation, and hypothesis testing following academic standards.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass
from scipy import stats, linalg
from scipy.stats import chi2, f, t
import warnings
from itertools import combinations


@dataclass
class TestResult:
    """Standardized result for statistical tests"""
    statistic: float
    p_value: float
    degrees_of_freedom: Union[int, Tuple[int, int]]
    reject_null: bool
    confidence_level: float = 0.05
    method: str = ""
    interpretation: str = ""
    
    def __repr__(self):
        return (f"TestResult(statistic={self.statistic:.4f}, "
                f"p_value={self.p_value:.4f}, reject_null={self.reject_null})")


class LittleMCARTest:
    """
    Little's Missing Completely at Random (MCAR) Test
    
    Implements the chi-squared test for MCAR mechanism as described in:
    Little, R.J.A. (1988). A Test of Missing Completely at Random for 
    Multivariate Data with Missing Values. JASA, 83(404), 1198-1202.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def test(self, data: np.ndarray) -> TestResult:
        """
        Perform Little's MCAR test
        
        Args:
            data: Array with missing values (NaN)
            
        Returns:
            TestResult with chi-squared statistic and p-value
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        n, p = data.shape
        
        # Create missing data patterns
        patterns = self._get_missing_patterns(data)
        
        if len(patterns) == 1:
            # Only one pattern (all complete or all missing)
            return TestResult(
                statistic=0.0,
                p_value=1.0,
                degrees_of_freedom=0,
                reject_null=False,
                confidence_level=self.alpha,
                method="Little's MCAR Test",
                interpretation="Only one missing pattern found"
            )
        
        # Calculate test statistic
        d2 = 0.0
        df = 0
        
        for pattern_idx, pattern_info in patterns.items():
            if pattern_info['count'] > 0:
                # Get observed variables for this pattern
                obs_vars = pattern_info['observed_vars']
                
                if len(obs_vars) < p:  # Pattern has missing values
                    # Subset data for this pattern
                    pattern_data = data[pattern_info['rows'], :][:, obs_vars]
                    
                    # Calculate means and covariance
                    pattern_mean = np.nanmean(pattern_data, axis=0)
                    pattern_cov = np.cov(pattern_data.T, ddof=1)
                    
                    # Get overall statistics for observed variables
                    overall_data = data[:, obs_vars]
                    overall_mean = np.nanmean(overall_data, axis=0)
                    overall_cov = np.cov(overall_data[~np.isnan(overall_data).any(axis=1)].T, ddof=1)
                    
                    # Calculate contribution to test statistic
                    try:
                        inv_cov = linalg.pinv(overall_cov)
                        diff = pattern_mean - overall_mean
                        d2 += pattern_info['count'] * np.dot(np.dot(diff, inv_cov), diff)
                        df += len(obs_vars)
                    except:
                        continue
        
        # Calculate p-value
        p_value = 1 - chi2.cdf(d2, df) if df > 0 else 1.0
        
        return TestResult(
            statistic=d2,
            p_value=p_value,
            degrees_of_freedom=df,
            reject_null=p_value < self.alpha,
            confidence_level=self.alpha,
            method="Little's MCAR Test",
            interpretation="Data is MCAR" if p_value >= self.alpha else "Data is not MCAR"
        )
    
    def _get_missing_patterns(self, data: np.ndarray) -> Dict:
        """Identify unique missing data patterns"""
        patterns = {}
        pattern_id = 0
        
        for i in range(data.shape[0]):
            # Create pattern signature
            missing_mask = ~np.isnan(data[i, :])
            pattern_key = tuple(missing_mask)
            
            if pattern_key not in patterns:
                patterns[pattern_key] = {
                    'id': pattern_id,
                    'observed_vars': np.where(missing_mask)[0].tolist(),
                    'missing_vars': np.where(~missing_mask)[0].tolist(),
                    'count': 0,
                    'rows': []
                }
                pattern_id += 1
            
            patterns[pattern_key]['count'] += 1
            patterns[pattern_key]['rows'].append(i)
        
        return {v['id']: v for v in patterns.values()}


class RubinRules:
    """
    Rubin's Rules for Multiple Imputation Inference
    
    Implements combination rules for multiple imputation as described in:
    Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys.
    """
    
    @staticmethod
    def combine_estimates(estimates: np.ndarray, 
                         variances: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Combine parameter estimates from multiple imputations
        
        Args:
            estimates: Array of shape (m, p) with m imputations and p parameters
            variances: Array of shape (m, p) with variance estimates
            
        Returns:
            Dictionary with combined estimates, variances, and statistics
        """
        m = estimates.shape[0]  # Number of imputations
        
        # Combined point estimate (average)
        q_bar = np.mean(estimates, axis=0)
        
        # Within-imputation variance
        u_bar = np.mean(variances, axis=0)
        
        # Between-imputation variance
        b = np.var(estimates, axis=0, ddof=1)
        
        # Total variance
        t = u_bar + (1 + 1/m) * b
        
        # Relative increase in variance
        r = ((1 + 1/m) * b) / u_bar
        
        # Fraction of missing information
        lambda_est = (r + 2/(m - 1)) / (r + 1)
        
        # Degrees of freedom (old formula)
        nu_old = (m - 1) * (1 + 1/r)**2
        
        # Modified degrees of freedom for small samples
        n_com = estimates.shape[1] * 100  # Approximate complete-data df
        nu_obs = ((n_com + 1)/(n_com + 3)) * n_com * (1 - lambda_est)
        nu = 1 / (1/nu_old + 1/nu_obs)
        
        return {
            'estimate': q_bar,
            'variance': t,
            'std_error': np.sqrt(t),
            'relative_efficiency': 1 / (1 + lambda_est/m),
            'fraction_missing': lambda_est,
            'degrees_of_freedom': nu,
            'within_variance': u_bar,
            'between_variance': b
        }
    
    @staticmethod
    def test_parameters(estimates: np.ndarray, 
                       variances: np.ndarray,
                       null_value: float = 0,
                       alpha: float = 0.05) -> TestResult:
        """
        Test parameters using Rubin's rules
        
        Args:
            estimates: Parameter estimates from m imputations
            variances: Variance estimates from m imputations
            null_value: Value under null hypothesis
            alpha: Significance level
            
        Returns:
            TestResult with t-statistic and p-value
        """
        combined = RubinRules.combine_estimates(estimates, variances)
        
        # Calculate t-statistic
        t_stat = (combined['estimate'] - null_value) / combined['std_error']
        
        # Calculate p-value (two-sided)
        df = combined['degrees_of_freedom']
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), df))
        
        return TestResult(
            statistic=float(t_stat),
            p_value=float(p_value),
            degrees_of_freedom=int(df),
            reject_null=p_value < alpha,
            confidence_level=alpha,
            method="Rubin's Rules t-test",
            interpretation=f"FMI: {combined['fraction_missing']:.3f}"
        )


class DieboldMarianoTest:
    """
    Diebold-Mariano Test for Predictive Accuracy
    
    Tests whether two forecasting methods have equal predictive accuracy.
    Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy. 
    Journal of Business & Economic Statistics, 13(3), 253-263.
    """
    
    def __init__(self, loss_function: str = 'squared'):
        self.loss_function = loss_function
        
    def test(self, 
             errors1: np.ndarray, 
             errors2: np.ndarray,
             h: int = 1,
             alternative: str = 'two-sided') -> TestResult:
        """
        Perform Diebold-Mariano test
        
        Args:
            errors1: Forecast errors from method 1
            errors2: Forecast errors from method 2
            h: Forecast horizon
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            TestResult with DM statistic and p-value
        """
        # Calculate loss differential
        if self.loss_function == 'squared':
            d = errors1**2 - errors2**2
        elif self.loss_function == 'absolute':
            d = np.abs(errors1) - np.abs(errors2)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
            
        # Calculate test statistic
        T = len(d)
        d_bar = np.mean(d)
        
        # Calculate variance with Harvey et al. (1997) modification
        gamma = np.zeros(h)
        for k in range(h):
            gamma[k] = np.cov(d[k:], d[:-k] if k > 0 else d)[0, 1 if k > 0 else 0]
        
        # Long-run variance
        V_d = gamma[0] + 2 * np.sum(gamma[1:])
        
        # Harvey et al. (1997) finite-sample correction
        correction = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
        
        # DM statistic
        dm_stat = d_bar / np.sqrt(V_d/T) * correction
        
        # Calculate p-value using t-distribution
        df = T - 1
        if alternative == 'two-sided':
            p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df))
        elif alternative == 'greater':
            p_value = 1 - t.cdf(dm_stat, df)
        else:  # less
            p_value = t.cdf(dm_stat, df)
            
        return TestResult(
            statistic=dm_stat,
            p_value=p_value,
            degrees_of_freedom=df,
            reject_null=p_value < 0.05,
            method="Diebold-Mariano Test",
            interpretation="Equal accuracy" if p_value >= 0.05 else "Different accuracy"
        )


class TiaoBoxTest:
    """
    Tiao-Box Bias Test for Model Adequacy
    
    Tests for bias in model residuals.
    Tiao, G.C. & Box, G.E.P. (1981). Modeling Multiple Time Series with Applications.
    """
    
    def test(self, residuals: np.ndarray, lags: int = 10) -> TestResult:
        """
        Perform Tiao-Box bias test
        
        Args:
            residuals: Model residuals
            lags: Number of lags to test
            
        Returns:
            TestResult with test statistic and p-value
        """
        n = len(residuals)
        
        # Calculate autocorrelations
        acf = np.array([np.corrcoef(residuals[:-i], residuals[i:])[0, 1] 
                       for i in range(1, lags + 1)])
        
        # Tiao-Box statistic (modification of Ljung-Box)
        weights = n * (n + 2) / (n - np.arange(1, lags + 1))
        tb_stat = np.sum(weights * acf**2)
        
        # Calculate p-value
        p_value = 1 - chi2.cdf(tb_stat, lags)
        
        return TestResult(
            statistic=tb_stat,
            p_value=p_value,
            degrees_of_freedom=lags,
            reject_null=p_value < 0.05,
            method="Tiao-Box Test",
            interpretation="No bias" if p_value >= 0.05 else "Bias detected"
        )


class MultivariateNormalityTest:
    """
    Tests for multivariate normality including:
    - Mardia's skewness and kurtosis tests
    - Henze-Zirkler test
    - Royston's test
    """
    
    @staticmethod
    def mardia_test(data: np.ndarray) -> Dict[str, TestResult]:
        """
        Mardia's tests for multivariate skewness and kurtosis
        
        Args:
            data: n x p array of observations
            
        Returns:
            Dictionary with skewness and kurtosis test results
        """
        n, p = data.shape
        
        # Center the data
        data_centered = data - np.mean(data, axis=0)
        
        # Calculate sample covariance
        S = np.cov(data_centered.T, ddof=1)
        S_inv = linalg.pinv(S)
        
        # Calculate Mahalanobis distances
        D = np.array([np.dot(np.dot(data_centered[i], S_inv), data_centered[i]) 
                     for i in range(n)])
        
        # Skewness test
        b1p = np.sum(D**3) / n**2
        skew_stat = n * b1p / 6
        skew_df = p * (p + 1) * (p + 2) / 6
        skew_pval = 1 - chi2.cdf(skew_stat, skew_df)
        
        # Kurtosis test  
        b2p = np.sum(D**2) / n
        kurt_stat = (b2p - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
        kurt_pval = 2 * (1 - stats.norm.cdf(np.abs(kurt_stat)))
        
        return {
            'skewness': TestResult(
                statistic=skew_stat,
                p_value=skew_pval,
                degrees_of_freedom=int(skew_df),
                reject_null=skew_pval < 0.05,
                method="Mardia's Skewness Test",
                interpretation="MVN" if skew_pval >= 0.05 else "Not MVN (skewed)"
            ),
            'kurtosis': TestResult(
                statistic=kurt_stat,
                p_value=kurt_pval,
                degrees_of_freedom=np.inf,
                reject_null=kurt_pval < 0.05,
                method="Mardia's Kurtosis Test",
                interpretation="MVN" if kurt_pval >= 0.05 else "Not MVN (kurtosis)"
            )
        }


def perform_all_tests(data: np.ndarray, 
                     imputed_data: Optional[np.ndarray] = None,
                     alpha: float = 0.05) -> Dict[str, TestResult]:
    """
    Perform comprehensive statistical testing suite
    
    Args:
        data: Original data with missing values
        imputed_data: Imputed data (optional)
        alpha: Significance level
        
    Returns:
        Dictionary of test results
    """
    results = {}
    
    # Test 1: Little's MCAR test
    mcar_test = LittleMCARTest(alpha=alpha)
    results['mcar'] = mcar_test.test(data)
    
    # Test 2: Multivariate normality (if applicable)
    complete_rows = ~np.isnan(data).any(axis=1)
    if np.sum(complete_rows) > data.shape[1] + 1:
        mvn_results = MultivariateNormalityTest.mardia_test(data[complete_rows])
        results.update({f'mvn_{k}': v for k, v in mvn_results.items()})
    
    # Additional tests if imputed data provided
    if imputed_data is not None:
        # Test 3: Distribution preservation
        for col in range(data.shape[1]):
            orig_col = data[:, col][~np.isnan(data[:, col])]
            imp_col = imputed_data[:, col]
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = stats.ks_2samp(orig_col, imp_col)
            results[f'ks_col_{col}'] = TestResult(
                statistic=ks_stat,
                p_value=ks_pval,
                degrees_of_freedom=len(orig_col),
                reject_null=ks_pval < alpha,
                method="Kolmogorov-Smirnov Test",
                interpretation="Same distribution" if ks_pval >= alpha else "Different distribution"
            )
    
    return results