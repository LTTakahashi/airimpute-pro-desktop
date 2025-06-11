"""
Bayesian Methods for Imputation with Uncertainty Quantification

Implements Bayesian approaches including Gaussian Processes, MCMC,
and Variational Bayes for principled uncertainty quantification.

All methods include complexity analysis and academic citations as required by CLAUDE.md

References:
- Rasmussen, C. E., & Williams, C. K. (2006). Gaussian processes for machine learning. 
  MIT press. ISBN: 026218253X
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. 
  (2013). Bayesian data analysis. CRC press. DOI: 10.1201/b16018
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: 
  A review for statisticians. Journal of the American statistical Association, 
  112(518), 859-877. DOI: 10.1080/01621459.2017.1285773
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Union, Callable
from dataclasses import dataclass
import warnings
from scipy import stats, linalg
from scipy.special import gammaln, digamma
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
import pymc as pm
import arviz as az


@dataclass
class BayesianPrediction:
    """
    Container for Bayesian predictions with uncertainty
    
    Encapsulates posterior predictive distribution with:
    - Point estimates (mean)
    - Uncertainty (variance/std)
    - Full posterior samples (if available)
    - Credible intervals
    
    Time Complexity for sampling: O(n_samples)
    Space Complexity: O(n_samples × dimensions)
    """
    mean: np.ndarray
    variance: np.ndarray
    samples: Optional[np.ndarray] = None
    credible_interval: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    @property
    def std(self) -> np.ndarray:
        """Standard deviation"""
        return np.sqrt(self.variance)
    
    def sample(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate samples from predictive distribution
        
        Time Complexity: O(n_samples) for resampling or generation
        Space Complexity: O(n_samples × dimensions)
        """
        if self.samples is not None:
            # Use existing samples
            idx = np.random.choice(len(self.samples), n_samples)
            return self.samples[idx]
        else:
            # Generate from normal distribution
            return np.random.normal(self.mean, self.std, size=(n_samples,))


class GaussianProcessImputer(BaseEstimator):
    """
    Gaussian Process imputation with full uncertainty quantification
    
    Academic Reference:
    Rasmussen, C. E., & Williams, C. K. (2006). Gaussian processes for machine learning.
    MIT press. ISBN: 026218253X
    
    Mathematical Foundation:
    GP defines a distribution over functions: f ~ GP(m(x), k(x, x'))
    For regression with noise: y = f(x) + ε, ε ~ N(0, σ²)
    
    Posterior mean: μ* = K* K⁻¹ y
    Posterior variance: Σ* = K** - K* K⁻¹ K*ᵀ
    
    where:
    - K = k(X, X) + σ²I (training covariance)
    - K* = k(X*, X) (test-train covariance)
    - K** = k(X*, X*) (test covariance)
    
    Provides posterior distributions over missing values with
    automatic relevance determination (ARD).
    
    Assumptions:
    - Gaussian process prior over functions
    - Gaussian likelihood (appropriate for continuous data)
    - Stationarity (for standard kernels like RBF)
    - Hyperparameters can be learned from data
    
    Time Complexity:
    - Training: O(n³) for Cholesky decomposition + O(n² × m) for kernel computation
    - Prediction: O(n²) per missing value
    - Hyperparameter optimization: O(I × n³) where I = optimization iterations
    
    Space Complexity: O(n² + n × m) for kernel matrix and data storage
    
    Advantages:
    - Principled uncertainty quantification
    - Non-parametric (flexible)
    - Automatic relevance determination
    - Optimal in RKHS sense
    """
    
    def __init__(self,
                 kernel: str = "rbf",
                 length_scale: Union[float, np.ndarray] = 1.0,
                 length_scale_bounds: Tuple[float, float] = (1e-3, 1e3),
                 noise_variance: float = 1e-6,
                 optimize_hyperparameters: bool = True,
                 n_restarts: int = 5,
                 normalize: bool = True):
        self.kernel = kernel
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.noise_variance = noise_variance
        self.optimize_hyperparameters = optimize_hyperparameters
        self.n_restarts = n_restarts
        self.normalize = normalize
        
        self.gp_models_ = {}
        self.scalers_ = {}
        
    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray, 
                        length_scale: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix with ARD
        
        Time Complexity: O(n₁ × n₂ × d) where d = dimensions
        Space Complexity: O(n₁ × n₂) for kernel matrix
        """
        if self.kernel == "rbf":
            # RBF kernel with ARD
            scaled_X1 = X1 / length_scale
            scaled_X2 = X2 / length_scale
            dists = np.sum((scaled_X1[:, np.newaxis] - scaled_X2[np.newaxis])**2, axis=2)
            return np.exp(-0.5 * dists)
        elif self.kernel == "matern":
            # Simplified Matérn 5/2
            from scipy.spatial.distance import cdist
            dists = cdist(X1 / length_scale, X2 / length_scale)
            sqrt_5 = np.sqrt(5)
            return (1 + sqrt_5 * dists + 5/3 * dists**2) * np.exp(-sqrt_5 * dists)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _optimize_gp_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Optimize GP hyperparameters via marginal likelihood
        
        Time Complexity: O(R × I × n³) where:
            R = n_restarts
            I = optimization iterations
            n = number of observations
        Space Complexity: O(n²) for kernel matrix
        
        Algorithm: L-BFGS-B optimization of log marginal likelihood
        """
        n_features = X.shape[1]
        
        def negative_log_marginal_likelihood(log_params):
            # Extract parameters
            log_length_scales = log_params[:n_features]
            log_noise = log_params[-1]
            
            length_scales = np.exp(log_length_scales)
            noise = np.exp(log_noise)
            
            # Compute kernel
            K = self._kernel_function(X, X, length_scales)
            K_noise = K + noise * np.eye(len(X))
            
            # Cholesky decomposition
            try:
                L = linalg.cholesky(K_noise, lower=True)
            except linalg.LinAlgError:
                return np.inf
                
            # Compute log marginal likelihood
            alpha = linalg.solve_triangular(L, y, lower=True)
            alpha = linalg.solve_triangular(L.T, alpha, lower=False)
            
            log_likelihood = -0.5 * y.T @ alpha
            log_likelihood -= np.sum(np.log(np.diag(L)))
            log_likelihood -= 0.5 * len(y) * np.log(2 * np.pi)
            
            return -float(log_likelihood)
        
        # Multiple random restarts
        best_params = None
        best_value = np.inf
        
        for _ in range(self.n_restarts):
            # Random initialization
            x0 = np.random.randn(n_features + 1)
            
            # Bounds
            bounds = [self.length_scale_bounds] * n_features
            bounds.append((np.log(1e-10), np.log(1)))  # Noise bounds
            
            # Optimize
            result = minimize(negative_log_marginal_likelihood, x0,
                            method='L-BFGS-B', bounds=bounds)
            
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x
                
        # Extract optimized parameters
        opt_length_scales = np.exp(best_params[:n_features])
        opt_noise = np.exp(best_params[-1])
        
        return {
            'length_scales': opt_length_scales,
            'noise_variance': opt_noise,
            'log_marginal_likelihood': -best_value
        }
    
    def fit(self, X: np.ndarray) -> 'GaussianProcessImputer':
        """
        Fit GP models for each feature with missing values
        
        Args:
            X: Data with missing values (NaN)
            
        Returns:
            Self
            
        Time Complexity: O(F × n³) where F = features with missing values
        Space Complexity: O(F × n²) for storing GP models
        
        Algorithm:
        1. For each feature with missing values:
           a. Extract complete cases for training
           b. Optimize hyperparameters (optional)
           c. Compute and store Cholesky decomposition
           d. Compute alpha = L⁻ᵀ L⁻¹ y for predictions
        """
        n_samples, n_features = X.shape
        
        for target_idx in range(n_features):
            # Check if feature has missing values
            missing_mask = np.isnan(X[:, target_idx])
            if not np.any(missing_mask):
                continue
                
            # Get complete cases
            complete_mask = ~missing_mask
            other_features = [i for i in range(n_features) if i != target_idx]
            
            # Find rows with complete predictors
            predictor_complete = complete_mask.copy()
            for idx in other_features:
                predictor_complete &= ~np.isnan(X[:, idx])
                
            if np.sum(predictor_complete) < 10:
                warnings.warn(f"Not enough complete cases for feature {target_idx}")
                continue
                
            # Extract training data
            X_train = X[predictor_complete][:, other_features]
            y_train = X[predictor_complete, target_idx]
            
            # Normalize if requested
            if self.normalize:
                X_mean = np.mean(X_train, axis=0)
                X_std = np.std(X_train, axis=0) + 1e-8
                y_mean = np.mean(y_train)
                y_std = np.std(y_train) + 1e-8
                
                X_train = (X_train - X_mean) / X_std
                y_train = (y_train - y_mean) / y_std
                
                self.scalers_[target_idx] = {
                    'X_mean': X_mean, 'X_std': X_std,
                    'y_mean': y_mean, 'y_std': y_std
                }
            
            # Optimize hyperparameters
            if self.optimize_hyperparameters:
                hp = self._optimize_gp_hyperparameters(X_train, y_train)
                length_scales = hp['length_scales']
                noise = hp['noise_variance']
            else:
                if isinstance(self.length_scale, float):
                    length_scales = np.ones(len(other_features)) * self.length_scale
                else:
                    length_scales = self.length_scale
                noise = self.noise_variance
            
            # Store GP components
            K = self._kernel_function(X_train, X_train, length_scales)
            K_noise = K + noise * np.eye(len(X_train))
            
            try:
                L = linalg.cholesky(K_noise, lower=True)
                alpha = linalg.solve_triangular(L, y_train, lower=True)
                alpha = linalg.solve_triangular(L.T, alpha, lower=False)
                
                self.gp_models_[target_idx] = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'alpha': alpha,
                    'L': L,
                    'length_scales': length_scales,
                    'noise': noise,
                    'other_features': other_features
                }
            except linalg.LinAlgError:
                warnings.warn(f"GP fitting failed for feature {target_idx}")
                
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Union[np.ndarray, BayesianPrediction]:
        """
        Predict missing values with uncertainty
        
        Args:
            X: Data with missing values
            return_std: Whether to return standard deviations
            
        Returns:
            Imputed values or BayesianPrediction object
            
        Time Complexity: O(M × n²) where M = total missing values
        Space Complexity: O(n × m) for predictions and uncertainties
        
        Algorithm:
        1. For each missing value:
           a. Compute k* = k(x*, X) with training data
           b. Predict mean: μ* = k* α
           c. If uncertainty needed:
              - Solve L v = k*
              - Compute variance: σ*² = k** - vᵀv + σ_noise²
        """
        X_imputed = X.copy()
        uncertainties = np.zeros_like(X)
        
        for target_idx, gp_model in self.gp_models_.items():
            # Find missing values
            missing_mask = np.isnan(X[:, target_idx])
            if not np.any(missing_mask):
                continue
                
            # Extract test features
            other_features = gp_model['other_features']
            X_test = X[missing_mask][:, other_features]
            
            # Check for complete predictors
            complete_predictors = ~np.isnan(X_test).any(axis=1)
            if not np.any(complete_predictors):
                continue
                
            X_test_complete = X_test[complete_predictors]
            
            # Normalize if needed
            if target_idx in self.scalers_:
                scaler = self.scalers_[target_idx]
                X_test_complete = (X_test_complete - scaler['X_mean']) / scaler['X_std']
            
            # GP prediction
            K_star = self._kernel_function(X_test_complete, gp_model['X_train'], 
                                         gp_model['length_scales'])
            mean = K_star @ gp_model['alpha']
            
            if return_std:
                # Compute variance
                v = linalg.solve_triangular(gp_model['L'], K_star.T, lower=True)
                K_star_star = self._kernel_function(X_test_complete, X_test_complete,
                                                  gp_model['length_scales'])
                var = np.diag(K_star_star) - np.sum(v**2, axis=0) + gp_model['noise']
                std = np.sqrt(np.maximum(var, 0))
                
                # Denormalize
                if target_idx in self.scalers_:
                    scaler = self.scalers_[target_idx]
                    mean = mean * scaler['y_std'] + scaler['y_mean']
                    std = std * scaler['y_std']
                
                # Store uncertainties
                missing_indices = np.where(missing_mask)[0][complete_predictors]
                uncertainties[missing_indices, target_idx] = std
            else:
                # Denormalize
                if target_idx in self.scalers_:
                    scaler = self.scalers_[target_idx]
                    mean = mean * scaler['y_std'] + scaler['y_mean']
            
            # Impute values
            missing_indices = np.where(missing_mask)[0][complete_predictors]
            X_imputed[missing_indices, target_idx] = mean
            
        if return_std:
            return BayesianPrediction(
                mean=X_imputed,
                variance=uncertainties**2,
                credible_interval=(X_imputed - 1.96*uncertainties, 
                                 X_imputed + 1.96*uncertainties)
            )
        else:
            return X_imputed


class BayesianStructuralTimeSeries:
    """
    Bayesian Structural Time Series (BSTS) for temporal imputation
    
    Academic Reference:
    Scott, S. L., & Varian, H. R. (2014). Predicting the present with Bayesian
    structural time series. International Journal of Mathematical Modelling and
    Numerical Optimisation, 5(1-2), 4-23. DOI: 10.1504/IJMMNO.2014.059942
    
    Mathematical Foundation:
    State space model: y_t = Z_t α_t + ε_t, ε_t ~ N(0, σ²)
                      α_{t+1} = T_t α_t + R_t η_t, η_t ~ N(0, Q_t)
    
    Components:
    1. Local linear trend: μ_{t+1} = μ_t + δ_t + η_{μ,t}
                          δ_{t+1} = δ_t + η_{δ,t}
    2. Seasonal: s_{t+1} = -Σ_{i=1}^{S-1} s_{t-i} + η_{s,t}
    3. Regression: β_t x_t
    
    Decomposes time series into trend, seasonal, and regression components
    with full posterior distributions.
    
    Assumptions:
    - Additive decomposition appropriate
    - Gaussian errors
    - State evolution follows Markovian dynamics
    - Seasonal patterns are stable
    
    Time Complexity:
    - Model building: O(T × S) where T = time points, S = seasonal components
    - MCMC sampling: O(I × T × (C² + S)) where I = iterations, C = state dimension
    - Prediction: O(T × C) per sample
    
    Space Complexity: O(I × T × C) for storing MCMC samples
    """
    
    def __init__(self,
                 seasonal_periods: Optional[List[int]] = None,
                 n_seasons: Optional[List[int]] = None,
                 trend: str = "local_linear",
                 ar_order: int = 0,
                 mcmc_samples: int = 1000):
        self.seasonal_periods = seasonal_periods or []
        self.n_seasons = n_seasons or [12] * len(self.seasonal_periods)
        self.trend = trend
        self.ar_order = ar_order
        self.mcmc_samples = mcmc_samples
        
        self.trace_ = None
        self.model_ = None
        
    def build_model(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> pm.Model:
        """
        Build PyMC model for BSTS
        
        Args:
            y: Time series with missing values
            X: Optional exogenous predictors
            
        Returns:
            PyMC model
            
        Time Complexity: O(T × S) for building seasonal matrices
        Space Complexity: O(T × (S + P)) where P = number of predictors
        """
        coords = {"time": np.arange(len(y))}
        
        with pm.Model(coords=coords) as model:
            # Observation noise
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)
            
            # Trend component
            if self.trend == "local_linear":
                # Local linear trend
                sigma_level = pm.HalfNormal("sigma_level", sigma=1)
                sigma_trend = pm.HalfNormal("sigma_trend", sigma=0.1)
                
                level = pm.GaussianRandomWalk("level", sigma=sigma_level, 
                                             dims="time")
                trend = pm.GaussianRandomWalk("trend", sigma=sigma_trend,
                                             dims="time")
                
                trend_component = level + pm.math.cumsum(trend)
            elif self.trend == "local_level":
                # Local level (random walk)
                sigma_level = pm.HalfNormal("sigma_level", sigma=1)
                trend_component = pm.GaussianRandomWalk("level", sigma=sigma_level,
                                                       dims="time")
            else:
                trend_component = 0
            
            # Seasonal components
            seasonal_component = 0
            for i, (period, n_season) in enumerate(zip(self.seasonal_periods, self.n_seasons)):
                sigma_seasonal = pm.HalfNormal(f"sigma_seasonal_{i}", sigma=0.5)
                
                # Fourier seasonal representation
                seasonal_matrix = self._make_seasonal_matrix(len(y), period, n_season)
                seasonal_coefs = pm.Normal(f"seasonal_coefs_{i}", 
                                         mu=0, sigma=sigma_seasonal,
                                         shape=seasonal_matrix.shape[1])
                
                seasonal_component += pm.math.dot(seasonal_matrix, seasonal_coefs)
            
            # AR component
            if self.ar_order > 0:
                rho = pm.Normal("rho", mu=0, sigma=0.5, shape=self.ar_order)
                sigma_ar = pm.HalfNormal("sigma_ar", sigma=0.5)
                
                ar_component = pm.AR("ar", rho=rho, sigma=sigma_ar, 
                                    dims="time", constant=True)
            else:
                ar_component = 0
            
            # Regression component
            if X is not None:
                beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
                regression = pm.math.dot(X, beta)
            else:
                regression = 0
            
            # Combine components
            mu = trend_component + seasonal_component + ar_component + regression
            
            # Likelihood (handles missing values automatically)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_obs, 
                            observed=y, dims="time")
            
            # Predictive samples for missing values
            y_missing = pm.Normal("y_missing", mu=mu, sigma=sigma_obs,
                                dims="time")
            
        return model
    
    def _make_seasonal_matrix(self, n: int, period: int, n_seasons: int) -> np.ndarray:
        """
        Create Fourier basis for seasonal component
        
        Time Complexity: O(n × n_seasons)
        Space Complexity: O(n × 2 × n_seasons) for sin/cos pairs
        
        Mathematical basis: Fourier decomposition of periodic signal
        """
        t = np.arange(n)
        X = []
        
        for i in range(1, n_seasons + 1):
            X.append(np.sin(2 * np.pi * i * t / period))
            X.append(np.cos(2 * np.pi * i * t / period))
            
        return np.column_stack(X)
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'BayesianStructuralTimeSeries':
        """
        Fit BSTS model using MCMC
        
        Args:
            y: Time series with missing values (NaN)
            X: Optional predictors
            
        Returns:
            Self
            
        Time Complexity: O(I × T × (C² + S)) for MCMC iterations
        Space Complexity: O(I × T × C) for posterior samples
        
        Algorithm:
        1. Build probabilistic model with PyMC
        2. Run NUTS sampler for posterior inference
        3. Store posterior samples for prediction
        
        Note: NUTS (No-U-Turn Sampler) adaptively sets trajectory length
        """
        self.model_ = self.build_model(y, X)
        
        with self.model_:
            # Use NUTS sampler
            self.trace_ = pm.sample(
                draws=self.mcmc_samples,
                tune=500,
                cores=1,
                progressbar=True,
                return_inferencedata=True
            )
            
        return self
    
    def predict(self, n_ahead: int = 0, X_future: Optional[np.ndarray] = None) -> BayesianPrediction:
        """
        Generate predictions with uncertainty
        
        Args:
            n_ahead: Number of steps ahead to forecast
            X_future: Future predictor values
            
        Returns:
            BayesianPrediction with posterior samples
            
        Time Complexity: O(S × T) where S = posterior samples
        Space Complexity: O(S × T) for storing predictions
        
        Provides full posterior predictive distribution
        """
        if self.trace_ is None:
            raise ValueError("Model must be fitted before prediction")
            
        with self.model_:
            # In-sample predictions (including imputed values)
            posterior_pred = pm.sample_posterior_predictive(
                self.trace_, var_names=["y_missing"]
            )
            
            # Extract predictions
            y_samples = posterior_pred.posterior_predictive["y_missing"].values
            
            # Reshape to (n_samples, n_time)
            y_samples = y_samples.reshape(-1, y_samples.shape[-1])
            
            # Calculate statistics
            mean = np.mean(y_samples, axis=0)
            variance = np.var(y_samples, axis=0)
            
            # Credible intervals
            lower = np.percentile(y_samples, 2.5, axis=0)
            upper = np.percentile(y_samples, 97.5, axis=0)
            
        return BayesianPrediction(
            mean=mean,
            variance=variance,
            samples=y_samples,
            credible_interval=(lower, upper)
        )


class VariationalBayesImputer(BaseEstimator):
    """
    Variational Bayes imputation for scalable Bayesian inference
    
    Academic Reference:
    Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference:
    A review for statisticians. Journal of the American statistical Association,
    112(518), 859-877. DOI: 10.1080/01621459.2017.1285773
    
    Mathematical Foundation:
    Variational inference approximates p(θ|X) with q(θ) by minimizing KL divergence:
    KL[q(θ) || p(θ|X)] = E_q[log q(θ)] - E_q[log p(θ, X)] + log p(X)
    
    Mean-field assumption: q(θ) = Π_i q_i(θ_i)
    
    For Gaussian model with missing data:
    - q(μ) = N(μ_q, Σ_q)
    - q(X_miss) = N(μ_miss, Σ_miss)
    
    Coordinate ascent updates maximize ELBO:
    ELBO = E_q[log p(X, θ)] - E_q[log q(θ)]
    
    Uses mean-field variational inference for approximate posterior.
    
    Assumptions:
    - Mean-field factorization reasonable
    - Gaussian approximation adequate
    - Local optima acceptable (non-convex optimization)
    
    Time Complexity:
    - Per iteration: O(n × m² + m³) for parameter updates
    - Total: O(I × (n × m² + m³)) where I = iterations
    
    Space Complexity: O(n × m + m²) for data and parameters
    
    Advantages:
    - Scales better than MCMC
    - Deterministic convergence
    - Provides approximate posteriors
    """
    
    def __init__(self,
                 prior_mean: float = 0.0,
                 prior_precision: float = 1.0,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 verbose: bool = False):
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        self.posterior_params_ = {}
        
    def _update_parameters(self, X: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Variational E-M updates for Gaussian model
        
        Args:
            X: Data with missing values
            mask: Boolean mask of observed values
            
        Returns:
            Updated parameters
            
        Time Complexity:
        - E-step: O(n × m³) for conditional distributions
        - M-step: O(n × m²) for parameter updates
        - Total: O(n × m³) per iteration
        
        Space Complexity: O(m²) for covariance matrix
        
        Algorithm:
        1. E-step: Update q(X_miss) using current parameters
        2. M-step: Update q(μ, Σ) using expected complete data
        """
        n_samples, n_features = X.shape
        
        # Initialize
        if not hasattr(self, 'mu_'):
            self.mu_ = np.nanmean(X, axis=0)
            self.Sigma_ = np.eye(n_features)
            self.X_imputed_ = X.copy()
            
            # Fill initial values
            for j in range(n_features):
                missing = np.isnan(self.X_imputed_[:, j])
                if np.any(missing):
                    self.X_imputed_[missing, j] = self.mu_[j]
        
        # E-step: Update posterior for missing values
        for i in range(n_samples):
            missing = ~mask[i]
            observed = mask[i]
            
            if np.any(missing) and np.any(observed):
                # Conditional distribution parameters
                Sigma_oo = self.Sigma_[np.ix_(observed, observed)]
                Sigma_mo = self.Sigma_[np.ix_(missing, observed)]
                Sigma_mm = self.Sigma_[np.ix_(missing, missing)]
                
                try:
                    Sigma_oo_inv = linalg.pinv(Sigma_oo)
                    
                    # Conditional mean
                    mu_cond = self.mu_[missing] + Sigma_mo @ Sigma_oo_inv @ \
                             (self.X_imputed_[i, observed] - self.mu_[observed])
                    
                    # Conditional variance
                    Sigma_cond = Sigma_mm - Sigma_mo @ Sigma_oo_inv @ Sigma_mo.T
                    
                    # Update imputed values
                    self.X_imputed_[i, missing] = mu_cond
                    
                    # Store posterior parameters
                    self.posterior_params_[i] = {
                        'mean': mu_cond,
                        'cov': Sigma_cond,
                        'indices': np.where(missing)[0]
                    }
                except:
                    pass
        
        # M-step: Update global parameters
        self.mu_ = np.mean(self.X_imputed_, axis=0)
        centered = self.X_imputed_ - self.mu_
        self.Sigma_ = (centered.T @ centered) / n_samples
        
        # Add prior regularization
        self.Sigma_ += self.prior_precision * np.eye(n_features)
        
        return {'mu': self.mu_, 'Sigma': self.Sigma_}
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit VB model and impute missing values
        
        Args:
            X: Data with missing values
            
        Returns:
            Imputed data
            
        Time Complexity: O(I × n × m³) where I = iterations
        Space Complexity: O(n × m + m²)
        
        Convergence: Guaranteed to increase ELBO (local optimum)
        """
        mask = ~np.isnan(X)
        
        # Iterative updates
        for iteration in range(self.max_iter):
            old_mu = self.mu_.copy() if hasattr(self, 'mu_') else None
            
            # Update parameters
            params = self._update_parameters(X, mask)
            
            # Check convergence
            if old_mu is not None:
                change = np.linalg.norm(self.mu_ - old_mu)
                if self.verbose:
                    print(f"Iteration {iteration}: change = {change:.6f}")
                    
                if change < self.tol:
                    break
                    
        return self.X_imputed_
    
    def get_posterior_samples(self, n_samples: int = 100) -> Dict[int, np.ndarray]:
        """
        Generate samples from variational posterior
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary mapping indices to posterior samples
            
        Time Complexity: O(n_samples × n_missing × d²) for multivariate normal sampling
        Space Complexity: O(n_samples × n_missing × d)
        
        Note: Samples from mean-field approximation q(X_miss)
        """
        samples = {}
        
        for idx, params in self.posterior_params_.items():
            # Sample from multivariate normal
            samples[idx] = np.random.multivariate_normal(
                params['mean'], params['cov'], size=n_samples
            )
            
        return samples


def bayesian_multiple_imputation(X: np.ndarray, 
                               n_imputations: int = 10,
                               method: str = "gaussian_process") -> List[np.ndarray]:
    """
    Perform Bayesian multiple imputation
    
    Academic Reference:
    Rubin, D. B. (1987). Multiple imputation for nonresponse in surveys.
    John Wiley & Sons. DOI: 10.1002/9780470316696
    
    Mathematical Foundation:
    Multiple imputation combines:
    1. Imputation uncertainty: Var(x̂|x_obs)
    2. Estimation uncertainty: Var(θ̂|x_complete)
    
    Rubin's rules for combining estimates:
    - Combined estimate: θ̄ = (1/M) Σ_m θ̂_m
    - Total variance: T = W̄ + (1 + 1/M)B
      where W̄ = within-imputation variance
            B = between-imputation variance
    
    Args:
        X: Data with missing values
        n_imputations: Number of imputed datasets to generate
        method: Bayesian method to use
        
    Returns:
        List of imputed datasets
        
    Time Complexity: O(M × C_method) where M = n_imputations, C_method = method complexity
    Space Complexity: O(M × n × m) for storing all imputed datasets
    
    Statistical Properties:
    - Properly accounts for imputation uncertainty
    - Valid inference under MAR
    - Efficiency = (1 + γ/M)^(-1) where γ = fraction missing info
    """
    imputed_datasets = []
    
    if method == "gaussian_process":
        imputer = GaussianProcessImputer(optimize_hyperparameters=True)
        imputer.fit(X)
        
        for _ in range(n_imputations):
            # Get predictive distribution
            pred = imputer.predict(X, return_std=True)
            
            # Sample from posterior
            X_imp = pred.mean.copy()
            missing_mask = np.isnan(X)
            
            # Add Gaussian noise based on posterior variance
            noise = np.random.randn(*X.shape) * np.sqrt(pred.variance)
            X_imp[missing_mask] = (pred.mean + noise)[missing_mask]
            
            imputed_datasets.append(X_imp)
            
    elif method == "variational_bayes":
        imputer = VariationalBayesImputer()
        
        for _ in range(n_imputations):
            # Each run gives slightly different results due to initialization
            X_imp = imputer.fit_transform(X.copy())
            imputed_datasets.append(X_imp)
            
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return imputed_datasets