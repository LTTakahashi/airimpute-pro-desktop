"""
Conformal Prediction Framework for Uncertainty Quantification

Implements distribution-free prediction intervals with finite-sample coverage guarantees
following the conformal prediction framework.

References:
- Vovk et al. (2005). Algorithmic Learning in a Random World
- Lei et al. (2018). Distribution-Free Predictive Inference
- Romano et al. (2019). Conformalized Quantile Regression
"""

import numpy as np
from typing import Tuple, Optional, Callable, Union, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold
from scipy.stats import beta


@dataclass
class PredictionInterval:
    """Container for prediction intervals with coverage information"""
    lower: np.ndarray
    upper: np.ndarray
    coverage_probability: float
    empirical_coverage: Optional[float] = None
    method: str = "conformal"
    
    @property
    def width(self) -> np.ndarray:
        """Average interval width"""
        return self.upper - self.lower
    
    @property
    def mean_width(self) -> float:
        """Average interval width"""
        return np.mean(self.width)


class ConformalPredictor(ABC):
    """Abstract base class for conformal prediction methods"""
    
    def __init__(self, alpha: float = 0.1, symmetric: bool = True):
        """
        Args:
            alpha: Miscoverage level (1-alpha is the coverage)
            symmetric: Whether to use symmetric intervals
        """
        self.alpha = alpha
        self.symmetric = symmetric
        self.calibration_scores = None
        
    @abstractmethod
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calibrate the conformal predictor"""
        pass
    
    @abstractmethod
    def predict_interval(self, X: np.ndarray) -> PredictionInterval:
        """Predict intervals for new data"""
        pass
    
    def get_quantile(self, scores: np.ndarray, alpha: float) -> float:
        """
        Calculate the (1-alpha)-quantile with finite-sample correction
        
        Args:
            scores: Conformity scores
            alpha: Miscoverage level
            
        Returns:
            Corrected quantile value
        """
        n = len(scores)
        q = np.ceil((n + 1) * (1 - alpha)) / n
        return np.quantile(scores, q, method='higher')


class SplitConformalRegressor(ConformalPredictor):
    """
    Split Conformal Prediction for Regression
    
    Splits data into training and calibration sets, providing
    exact finite-sample coverage guarantees.
    """
    
    def __init__(self, 
                 estimator: BaseEstimator,
                 alpha: float = 0.1,
                 train_size: float = 0.5,
                 symmetric: bool = True):
        super().__init__(alpha, symmetric)
        self.estimator = estimator
        self.train_size = train_size
        self.fitted_estimator = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SplitConformalRegressor':
        """
        Fit the model and calibrate conformal predictor
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Self
        """
        # Split data
        n = len(y)
        n_train = int(n * self.train_size)
        indices = np.random.permutation(n)
        
        train_idx = indices[:n_train]
        cal_idx = indices[n_train:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]
        
        # Fit estimator
        self.fitted_estimator = clone(self.estimator)
        self.fitted_estimator.fit(X_train, y_train)
        
        # Calibrate
        self.calibrate(X_cal, y_cal)
        
        return self
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """
        Calculate conformity scores on calibration set
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        # Get predictions
        y_pred = self.fitted_estimator.predict(X_cal)
        
        # Calculate conformity scores (absolute residuals)
        self.calibration_scores = np.abs(y_cal - y_pred)
        
        # Calculate quantile
        self.q_hat = self.get_quantile(self.calibration_scores, self.alpha)
    
    def predict_interval(self, X: np.ndarray) -> PredictionInterval:
        """
        Predict intervals for new data
        
        Args:
            X: Features
            
        Returns:
            PredictionInterval with coverage guarantees
        """
        if self.fitted_estimator is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Get point predictions
        y_pred = self.fitted_estimator.predict(X)
        
        # Construct intervals
        if self.symmetric:
            lower = y_pred - self.q_hat
            upper = y_pred + self.q_hat
        else:
            # Asymmetric intervals (future extension)
            lower = y_pred - self.q_hat
            upper = y_pred + self.q_hat
            
        return PredictionInterval(
            lower=lower,
            upper=upper,
            coverage_probability=1 - self.alpha,
            method="split_conformal"
        )


class ConformizedQuantileRegression(ConformalPredictor):
    """
    Conformalized Quantile Regression (CQR)
    
    Provides adaptive interval widths based on conditional quantiles.
    Romano et al. (2019). Conformalized Quantile Regression.
    """
    
    def __init__(self,
                 quantile_estimator_low: BaseEstimator,
                 quantile_estimator_high: BaseEstimator,
                 alpha: float = 0.1):
        super().__init__(alpha, symmetric=False)
        self.qr_low = quantile_estimator_low
        self.qr_high = quantile_estimator_high
        self.fitted_qr_low = None
        self.fitted_qr_high = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ConformizedQuantileRegression':
        """Fit quantile regressors and calibrate"""
        # Split data
        n = len(y)
        n_train = int(n * 0.5)
        indices = np.random.permutation(n)
        
        train_idx = indices[:n_train]
        cal_idx = indices[n_train:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]
        
        # Fit quantile regressors
        self.fitted_qr_low = clone(self.qr_low)
        self.fitted_qr_high = clone(self.qr_high)
        
        self.fitted_qr_low.fit(X_train, y_train)
        self.fitted_qr_high.fit(X_train, y_train)
        
        # Calibrate
        self.calibrate(X_cal, y_cal)
        
        return self
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calculate conformity scores"""
        # Get quantile predictions
        q_low = self.fitted_qr_low.predict(X_cal)
        q_high = self.fitted_qr_high.predict(X_cal)
        
        # Calculate conformity scores
        scores = np.maximum(q_low - y_cal, y_cal - q_high)
        self.calibration_scores = scores
        
        # Get quantile
        self.q_hat = self.get_quantile(scores, self.alpha)
    
    def predict_interval(self, X: np.ndarray) -> PredictionInterval:
        """Predict adaptive intervals"""
        q_low = self.fitted_qr_low.predict(X)
        q_high = self.fitted_qr_high.predict(X)
        
        lower = q_low - self.q_hat
        upper = q_high + self.q_hat
        
        return PredictionInterval(
            lower=lower,
            upper=upper,
            coverage_probability=1 - self.alpha,
            method="CQR"
        )


class JackknifePlus(ConformalPredictor):
    """
    Jackknife+ for Predictive Inference
    
    Provides intervals without data splitting, using leave-one-out residuals.
    Barber et al. (2021). Predictive Inference with the Jackknife+.
    """
    
    def __init__(self, estimator: BaseEstimator, alpha: float = 0.1):
        super().__init__(alpha, symmetric=True)
        self.estimator = estimator
        self.loo_residuals = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'JackknifePlus':
        """Fit using leave-one-out procedure"""
        n = len(y)
        self.loo_residuals = np.zeros(n)
        
        # Store models for efficiency
        self.models = []
        
        # Leave-one-out training
        for i in range(n):
            # Create LOO dataset
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            
            X_loo = X[mask]
            y_loo = y[mask]
            
            # Fit model
            model_i = clone(self.estimator)
            model_i.fit(X_loo, y_loo)
            self.models.append(model_i)
            
            # Predict on left-out point
            y_pred_i = model_i.predict(X[i:i+1])[0]
            self.loo_residuals[i] = np.abs(y[i] - y_pred_i)
        
        # Fit final model on all data
        self.fitted_estimator = clone(self.estimator)
        self.fitted_estimator.fit(X, y)
        
        return self
    
    def predict_interval(self, X: np.ndarray) -> PredictionInterval:
        """Predict intervals using jackknife+ method"""
        n_train = len(self.loo_residuals)
        n_test = len(X)
        
        # Get predictions from all LOO models
        y_pred_loo = np.zeros((n_test, n_train))
        for i, model in enumerate(self.models):
            y_pred_loo[:, i] = model.predict(X)
        
        # Get prediction from full model
        y_pred_full = self.fitted_estimator.predict(X)
        
        # Calculate intervals
        lower = np.zeros(n_test)
        upper = np.zeros(n_test)
        
        q = self.get_quantile(self.loo_residuals, self.alpha)
        
        for j in range(n_test):
            # Jackknife+ intervals
            residuals_j = np.abs(y_pred_loo[j] - y_pred_full[j]) + self.loo_residuals
            q_j = self.get_quantile(residuals_j, self.alpha)
            
            lower[j] = y_pred_full[j] - q_j
            upper[j] = y_pred_full[j] + q_j
            
        return PredictionInterval(
            lower=lower,
            upper=upper,
            coverage_probability=1 - self.alpha,
            method="jackknife+"
        )


class CVPlus(ConformalPredictor):
    """
    CV+ (Cross-validation+) for Predictive Inference
    
    Extension of Jackknife+ using K-fold cross-validation.
    Barber et al. (2021). Predictive Inference with the Jackknife+.
    """
    
    def __init__(self, 
                 estimator: BaseEstimator, 
                 alpha: float = 0.1,
                 n_folds: int = 10):
        super().__init__(alpha, symmetric=True)
        self.estimator = estimator
        self.n_folds = n_folds
        self.cv_residuals = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CVPlus':
        """Fit using cross-validation"""
        n = len(y)
        self.cv_residuals = np.zeros(n)
        self.fold_models = []
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Fit model on training fold
            model_fold = clone(self.estimator)
            model_fold.fit(X[train_idx], y[train_idx])
            self.fold_models.append(model_fold)
            
            # Predict on validation fold
            y_pred_val = model_fold.predict(X[val_idx])
            self.cv_residuals[val_idx] = np.abs(y[val_idx] - y_pred_val)
        
        # Fit final model
        self.fitted_estimator = clone(self.estimator)
        self.fitted_estimator.fit(X, y)
        
        return self
    
    def predict_interval(self, X: np.ndarray) -> PredictionInterval:
        """Predict intervals using CV+"""
        y_pred = self.fitted_estimator.predict(X)
        q = self.get_quantile(self.cv_residuals, self.alpha)
        
        lower = y_pred - q
        upper = y_pred + q
        
        return PredictionInterval(
            lower=lower,
            upper=upper,
            coverage_probability=1 - self.alpha,
            method="CV+"
        )


class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference for time series and non-exchangeable data
    
    Implements methods for maintaining coverage in presence of distribution shift.
    Gibbs & Candès (2021). Adaptive Conformal Inference Under Distribution Shift.
    """
    
    def __init__(self, 
                 base_predictor: ConformalPredictor,
                 gamma: float = 0.005,
                 target_coverage: float = 0.9):
        self.predictor = base_predictor
        self.gamma = gamma  # Learning rate
        self.target_coverage = target_coverage
        self.alpha_t = 1 - target_coverage  # Initial miscoverage
        self.coverage_history = []
        
    def update_alpha(self, covered: bool) -> None:
        """
        Update miscoverage level based on observed coverage
        
        Args:
            covered: Whether the true value was covered
        """
        # Gradient update
        gradient = 1 if covered else 0
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha_t - gradient)
        
        # Clip to valid range
        self.alpha_t = np.clip(self.alpha_t, 0.001, 0.999)
        
        # Update predictor
        self.predictor.alpha = self.alpha_t
        
        # Track coverage
        self.coverage_history.append(covered)
        
    def predict_interval_adaptive(self, 
                                 X: np.ndarray,
                                 y_true: Optional[np.ndarray] = None) -> PredictionInterval:
        """
        Predict interval with adaptive miscoverage level
        
        Args:
            X: Features
            y_true: True values (for online adaptation)
            
        Returns:
            Adaptive prediction interval
        """
        # Get current interval
        interval = self.predictor.predict_interval(X)
        
        # If true values provided, update alpha
        if y_true is not None:
            covered = (y_true >= interval.lower) & (y_true <= interval.upper)
            for c in covered:
                self.update_alpha(c)
        
        # Add adaptive information
        interval.method = f"adaptive_{interval.method}"
        if len(self.coverage_history) > 0:
            interval.empirical_coverage = np.mean(self.coverage_history[-100:])
            
        return interval


def evaluate_coverage(y_true: np.ndarray, 
                     interval: PredictionInterval,
                     conditional_on: Optional[np.ndarray] = None) -> dict:
    """
    Evaluate coverage properties of prediction intervals
    
    Args:
        y_true: True values
        interval: Prediction intervals
        conditional_on: Features for conditional coverage analysis
        
    Returns:
        Dictionary with coverage metrics
    """
    # Marginal coverage
    covered = (y_true >= interval.lower) & (y_true <= interval.upper)
    marginal_coverage = np.mean(covered)
    
    # Coverage CI using Wilson score interval
    n = len(y_true)
    z = 1.96  # 95% CI
    p_hat = marginal_coverage
    
    ci_lower = (p_hat + z**2/(2*n) - z*np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / (1 + z**2/n)
    ci_upper = (p_hat + z**2/(2*n) + z*np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / (1 + z**2/n)
    
    results = {
        'marginal_coverage': marginal_coverage,
        'coverage_ci': (ci_lower, ci_upper),
        'mean_width': interval.mean_width,
        'width_std': np.std(interval.width),
        'n_samples': n
    }
    
    # Conditional coverage analysis if features provided
    if conditional_on is not None:
        # Bin features and check coverage per bin
        n_bins = min(10, n // 50)
        for col in range(conditional_on.shape[1]):
            bins = np.quantile(conditional_on[:, col], 
                              np.linspace(0, 1, n_bins + 1))
            bin_coverage = []
            
            for i in range(n_bins):
                if i == n_bins - 1:
                    mask = (conditional_on[:, col] >= bins[i])
                else:
                    mask = (conditional_on[:, col] >= bins[i]) & \
                           (conditional_on[:, col] < bins[i+1])
                
                if np.sum(mask) > 10:
                    bin_coverage.append(np.mean(covered[mask]))
                    
            results[f'conditional_coverage_feature_{col}'] = bin_coverage
            
    return results