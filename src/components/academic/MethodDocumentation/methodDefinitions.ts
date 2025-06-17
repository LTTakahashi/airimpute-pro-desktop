import type { MethodDocumentation } from './MethodDocumentation';

export const methodDocumentations: Record<string, MethodDocumentation> = {
  // Classical Methods
  mean: {
    id: 'mean',
    name: 'Mean Imputation',
    category: 'classical',
    description: 'Replaces missing values with the arithmetic mean of observed values. Simple but can distort data distribution.',
    mathematical_formulation: {
      main_equation: '\\hat{x}_i = \\bar{x} = \\frac{1}{n} \\sum_{j=1}^{n} x_j',
      auxiliary_equations: [
        {
          label: 'Sample mean',
          equation: '\\bar{x} = \\frac{\\sum_{i \\in \\mathcal{O}} x_i}{|\\mathcal{O}|}'
        }
      ],
      constraints: ['\\mathcal{O} = \\{i : x_i \\text{ is observed}\\}']
    },
    algorithm: {
      steps: [
        'Calculate the mean of all observed values for each variable',
        'Replace each missing value with the corresponding variable mean',
        'Optionally, round to match the original data precision'
      ],
      complexity: {
        time: 'O(n)',
        space: 'O(1)'
      }
    },
    parameters: [
      {
        name: 'strategy',
        symbol: 's',
        description: 'Whether to use global mean or group-wise mean',
        default: 'global',
        range: 'global, group'
      }
    ],
    assumptions: [
      'Missing data is Missing Completely At Random (MCAR)',
      'The mean is a representative measure of central tendency',
      'No temporal or spatial dependencies'
    ],
    advantages: [
      'Simple and fast to compute',
      'Preserves the sample mean',
      'No additional parameters required',
      'Works with small sample sizes'
    ],
    limitations: [
      'Reduces variance in the data',
      'Distorts the distribution',
      'Ignores relationships between variables',
      'Can introduce bias if data is not MCAR'
    ],
    use_cases: [
      'Quick baseline imputation',
      'When missing rate is very low (<5%)',
      'Preliminary data exploration'
    ],
    references: [
      {
        title: 'Statistical Analysis with Missing Data',
        authors: ['Little, R.J.A.', 'Rubin, D.B.'],
        year: 2019,
        journal: 'John Wiley & Sons',
        doi: '10.1002/9781119013563'
      }
    ],
    example_code: `from airimpute.methods import SimpleImputer

imputer = SimpleImputer(method='mean')
imputed_data = imputer.fit_transform(data)`
  },

  linear_interpolation: {
    id: 'linear_interpolation',
    name: 'Linear Interpolation',
    category: 'classical',
    description: 'Estimates missing values by fitting a straight line between adjacent observed values. Suitable for time series with linear trends.',
    mathematical_formulation: {
      main_equation: '\\hat{x}_t = x_{t_1} + \\frac{x_{t_2} - x_{t_1}}{t_2 - t_1}(t - t_1)',
      auxiliary_equations: [
        {
          label: 'Slope calculation',
          equation: 'm = \\frac{x_{t_2} - x_{t_1}}{t_2 - t_1}'
        },
        {
          label: 'Interpolated value',
          equation: '\\hat{x}_t = x_{t_1} + m \\cdot (t - t_1)'
        }
      ],
      constraints: [
        't_1 < t < t_2',
        'x_{t_1}, x_{t_2} \\text{ are observed}'
      ]
    },
    algorithm: {
      steps: [
        'Identify each gap in the time series',
        'Find the nearest observed values before and after the gap',
        'Calculate the slope between these points',
        'Interpolate missing values using the linear equation',
        'Handle edge cases (beginning/end of series) with forward/backward fill'
      ],
      complexity: {
        time: 'O(n)',
        space: 'O(1)'
      }
    },
    parameters: [
      {
        name: 'limit',
        symbol: 'L',
        description: 'Maximum gap size to interpolate',
        default: 'None',
        range: 'positive integer'
      },
      {
        name: 'limit_direction',
        symbol: 'd',
        description: 'Direction to limit interpolation',
        default: 'both',
        range: 'forward, backward, both'
      }
    ],
    assumptions: [
      'Linear relationship between consecutive observations',
      'Uniform sampling intervals (for time series)',
      'No sudden changes or discontinuities'
    ],
    advantages: [
      'Preserves local trends',
      'Simple and interpretable',
      'No training required',
      'Computationally efficient'
    ],
    limitations: [
      'Cannot capture non-linear patterns',
      'Poor for long gaps',
      'Sensitive to outliers',
      'No uncertainty quantification'
    ],
    use_cases: [
      'Short gaps in smooth time series',
      'Temperature or pressure data',
      'Financial time series with no volatility jumps'
    ],
    references: [
      {
        title: 'Numerical Methods for Engineers',
        authors: ['Chapra, S.C.', 'Canale, R.P.'],
        year: 2015,
        journal: 'McGraw-Hill Education'
      }
    ],
    example_code: `from airimpute.methods import InterpolationImputer

imputer = InterpolationImputer(method='linear', limit=5)
imputed_data = imputer.fit_transform(data)`
  },

  spline_interpolation: {
    id: 'spline_interpolation',
    name: 'Spline Interpolation',
    category: 'classical',
    description: 'Uses piecewise polynomial functions to create smooth curves through observed data points. Provides better flexibility than linear interpolation.',
    mathematical_formulation: {
      main_equation: 'S(x) = \\begin{cases} S_1(x), & x \\in [x_0, x_1] \\\\ S_2(x), & x \\in [x_1, x_2] \\\\ \\vdots \\\\ S_n(x), & x \\in [x_{n-1}, x_n] \\end{cases}',
      auxiliary_equations: [
        {
          label: 'Cubic spline piece',
          equation: 'S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3'
        },
        {
          label: 'Continuity constraint',
          equation: 'S_i(x_{i+1}) = S_{i+1}(x_{i+1})'
        },
        {
          label: 'Smoothness constraint',
          equation: 'S_i\'(x_{i+1}) = S_{i+1}\'(x_{i+1}), \\quad S_i\'\'(x_{i+1}) = S_{i+1}\'\'(x_{i+1})'
        }
      ],
      constraints: [
        'S(x_i) = y_i \\text{ for all observed points}',
        'S\'\'(x_0) = S\'\'(x_n) = 0 \\text{ (natural spline)}'
      ]
    },
    algorithm: {
      steps: [
        'Sort data points by independent variable',
        'Compute second derivatives using tridiagonal system',
        'Calculate spline coefficients for each interval',
        'Evaluate spline at missing value locations',
        'Apply boundary conditions for extrapolation'
      ],
      complexity: {
        time: 'O(n)',
        space: 'O(n)'
      }
    },
    parameters: [
      {
        name: 'order',
        symbol: 'k',
        description: 'Degree of spline polynomial',
        default: 3,
        range: '1-5'
      },
      {
        name: 'smoothing',
        symbol: 's',
        description: 'Smoothing parameter (0 = interpolation, >0 = approximation)',
        default: 0,
        range: '[0, ∞)'
      }
    ],
    assumptions: [
      'Smooth underlying function',
      'Sufficient observed points for spline fitting',
      'No sharp discontinuities'
    ],
    advantages: [
      'Smooth interpolation',
      'Preserves data trends',
      'Flexible for non-linear patterns',
      'Continuous derivatives'
    ],
    limitations: [
      'Can oscillate with sparse data',
      'Sensitive to outliers',
      'Extrapolation can be unstable',
      'Requires more computation than linear'
    ],
    use_cases: [
      'Smooth environmental data',
      'Biological growth curves',
      'Engineering measurements'
    ],
    references: [
      {
        title: 'A Practical Guide to Splines',
        authors: ['de Boor, C.'],
        year: 2001,
        journal: 'Springer-Verlag',
        doi: '10.1007/978-1-4612-6333-3'
      }
    ],
    example_code: `from airimpute.methods import InterpolationImputer

imputer = InterpolationImputer(method='spline', order=3)
imputed_data = imputer.fit_transform(data)`
  },

  // Statistical Methods
  kalman_filter: {
    id: 'kalman_filter',
    name: 'Kalman Filter',
    category: 'statistical',
    description: 'Optimal recursive estimator for linear dynamic systems with Gaussian noise. Provides both estimates and uncertainty quantification.',
    mathematical_formulation: {
      main_equation: '\\begin{align} x_t &= F_t x_{t-1} + B_t u_t + w_t \\\\ y_t &= H_t x_t + v_t \\end{align}',
      auxiliary_equations: [
        {
          label: 'Prediction step',
          equation: '\\hat{x}_{t|t-1} = F_t \\hat{x}_{t-1|t-1} + B_t u_t'
        },
        {
          label: 'Prediction covariance',
          equation: 'P_{t|t-1} = F_t P_{t-1|t-1} F_t^T + Q_t'
        },
        {
          label: 'Innovation',
          equation: '\\tilde{y}_t = y_t - H_t \\hat{x}_{t|t-1}'
        },
        {
          label: 'Innovation covariance',
          equation: 'S_t = H_t P_{t|t-1} H_t^T + R_t'
        },
        {
          label: 'Kalman gain',
          equation: 'K_t = P_{t|t-1} H_t^T S_t^{-1}'
        },
        {
          label: 'Update step',
          equation: '\\hat{x}_{t|t} = \\hat{x}_{t|t-1} + K_t \\tilde{y}_t'
        },
        {
          label: 'Update covariance',
          equation: 'P_{t|t} = (I - K_t H_t) P_{t|t-1}'
        }
      ],
      constraints: [
        'w_t \\sim \\mathcal{N}(0, Q_t)',
        'v_t \\sim \\mathcal{N}(0, R_t)',
        'w_t \\perp v_s \\text{ for all } t, s'
      ]
    },
    algorithm: {
      steps: [
        'Initialize state estimate and covariance',
        'For each time step: Predict state using system dynamics',
        'Calculate prediction uncertainty',
        'If observation available: Compute Kalman gain',
        'Update state estimate using observation',
        'Update uncertainty estimate',
        'For missing values: Use predicted state as imputation'
      ],
      complexity: {
        time: 'O(n m^3)',
        space: 'O(m^2)'
      },
      convergence: 'Convergence guaranteed for stable systems with bounded noise covariances'
    },
    parameters: [
      {
        name: 'state_dim',
        symbol: 'm',
        description: 'Dimension of state vector',
        default: 1,
        range: 'positive integer'
      },
      {
        name: 'process_noise',
        symbol: 'Q',
        description: 'Process noise covariance matrix',
        default: 0.01,
        range: 'positive definite matrix'
      },
      {
        name: 'measurement_noise',
        symbol: 'R',
        description: 'Measurement noise covariance',
        default: 0.1,
        range: 'positive definite matrix'
      }
    ],
    assumptions: [
      'Linear system dynamics',
      'Gaussian noise distributions',
      'Known or estimable system parameters',
      'Markov property (future depends only on present)'
    ],
    advantages: [
      'Optimal for linear Gaussian systems',
      'Provides uncertainty estimates',
      'Handles irregular sampling',
      'Recursive computation'
    ],
    limitations: [
      'Requires system model specification',
      'Sensitive to parameter misspecification',
      'Limited to linear dynamics',
      'Gaussian assumption may be restrictive'
    ],
    use_cases: [
      'Sensor fusion',
      'Navigation and tracking',
      'Economic time series',
      'Environmental monitoring'
    ],
    references: [
      {
        title: 'A New Approach to Linear Filtering and Prediction Problems',
        authors: ['Kalman, R.E.'],
        year: 1960,
        journal: 'Journal of Basic Engineering',
        doi: '10.1115/1.3662552'
      }
    ],
    implementation_notes: [
      'Use numerically stable Joseph form for covariance update',
      'Consider adaptive Kalman filter for unknown noise parameters',
      'Extended/Unscented variants available for nonlinear systems'
    ],
    example_code: `from airimpute.methods import StatisticalImputer

imputer = StatisticalImputer(
    method='kalman_filter',
    state_dim=2,
    process_noise=0.01,
    measurement_noise=0.1
)
imputed_data = imputer.fit_transform(data)`
  },

  arima: {
    id: 'arima',
    name: 'ARIMA (AutoRegressive Integrated Moving Average)',
    category: 'statistical',
    description: 'Time series model combining autoregression, differencing, and moving average components. Excellent for data with trends and seasonality.',
    mathematical_formulation: {
      main_equation: '(1-\\phi_1L-\\cdots-\\phi_pL^p)(1-L)^d y_t = (1+\\theta_1L+\\cdots+\\theta_qL^q)\\epsilon_t',
      auxiliary_equations: [
        {
          label: 'AR component',
          equation: '\\phi(L) = 1-\\phi_1L-\\cdots-\\phi_pL^p'
        },
        {
          label: 'MA component', 
          equation: '\\theta(L) = 1+\\theta_1L+\\cdots+\\theta_qL^q'
        },
        {
          label: 'Differencing',
          equation: '\\nabla^d y_t = (1-L)^d y_t'
        }
      ],
      constraints: [
        '\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)',
        '\\text{AR roots outside unit circle (stationarity)}',
        '\\text{MA roots outside unit circle (invertibility)}'
      ]
    },
    algorithm: {
      steps: [
        'Test for stationarity (ADF test)',
        'Determine differencing order d',
        'Identify p and q using ACF/PACF or information criteria',
        'Estimate parameters using maximum likelihood',
        'Check residuals for white noise',
        'Forecast missing values using fitted model',
        'Transform back if differencing was applied'
      ],
      complexity: {
        time: 'O(n^2)',
        space: 'O(n)'
      }
    },
    parameters: [
      {
        name: 'p',
        symbol: 'p',
        description: 'Order of autoregression',
        default: 'auto',
        range: '0-5'
      },
      {
        name: 'd',
        symbol: 'd', 
        description: 'Degree of differencing',
        default: 'auto',
        range: '0-2'
      },
      {
        name: 'q',
        symbol: 'q',
        description: 'Order of moving average',
        default: 'auto',
        range: '0-5'
      }
    ],
    assumptions: [
      'Time series is stationary after differencing',
      'Residuals are white noise',
      'No structural breaks',
      'Linear relationships'
    ],
    advantages: [
      'Handles trends and seasonality',
      'Well-established theory',
      'Prediction intervals available',
      'Flexible model class'
    ],
    limitations: [
      'Requires long time series',
      'Model selection can be challenging',
      'Limited to univariate series',
      'Assumes linear dynamics'
    ],
    use_cases: [
      'Economic forecasting',
      'Demand prediction',
      'Climate data analysis',
      'Quality control'
    ],
    references: [
      {
        title: 'Time Series Analysis: Forecasting and Control',
        authors: ['Box, G.E.P.', 'Jenkins, G.M.', 'Reinsel, G.C.', 'Ljung, G.M.'],
        year: 2015,
        journal: 'John Wiley & Sons'
      }
    ],
    example_code: `from airimpute.methods import StatisticalImputer

imputer = StatisticalImputer(
    method='arima',
    order=(2, 1, 1),  # (p, d, q)
    seasonal_order=(1, 1, 1, 24)  # For hourly seasonality
)
imputed_data = imputer.fit_transform(data)`
  },

  // Machine Learning Methods
  random_forest: {
    id: 'random_forest',
    name: 'Random Forest Imputation',
    category: 'machine_learning',
    description: 'Ensemble method using multiple decision trees to predict missing values based on observed features. Captures non-linear relationships.',
    mathematical_formulation: {
      main_equation: '\\hat{x}_i = \\frac{1}{B} \\sum_{b=1}^{B} T_b(\\mathbf{x}_{obs,i})',
      auxiliary_equations: [
        {
          label: 'Tree prediction',
          equation: 'T_b(\\mathbf{x}) = \\sum_{m=1}^{M} c_m \\mathbb{I}(\\mathbf{x} \\in R_m)'
        },
        {
          label: 'Node splitting criterion',
          equation: '\\min_{j,s} \\left[ \\sum_{x_i \\in R_1(j,s)} (y_i - \\bar{y}_{R_1})^2 + \\sum_{x_i \\in R_2(j,s)} (y_i - \\bar{y}_{R_2})^2 \\right]'
        },
        {
          label: 'Out-of-bag error',
          equation: '\\text{OOB-MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i^{OOB})^2'
        }
      ],
      constraints: [
        '|\\mathcal{F}_b| = \\lfloor \\sqrt{p} \\rfloor \\text{ features per split}',
        '\\text{Bootstrap sample size} = n'
      ]
    },
    algorithm: {
      steps: [
        'For each variable with missing values:',
        'Use variables with fewer missing values as predictors',
        'For b = 1 to B trees: Draw bootstrap sample',
        'Grow tree using random feature subset at each split',
        'Predict missing values using all trees',
        'Average predictions across trees',
        'Iterate until convergence or max iterations'
      ],
      complexity: {
        time: 'O(B \\cdot n \\log n \\cdot m)',
        space: 'O(B \\cdot n)'
      }
    },
    parameters: [
      {
        name: 'n_estimators',
        symbol: 'B',
        description: 'Number of trees in forest',
        default: 100,
        range: '10-1000'
      },
      {
        name: 'max_depth',
        symbol: 'd_{max}',
        description: 'Maximum tree depth',
        default: 'None',
        range: '1-50 or None'
      },
      {
        name: 'min_samples_split',
        symbol: 'n_{split}',
        description: 'Minimum samples to split node',
        default: 2,
        range: '2-100'
      }
    ],
    assumptions: [
      'Missing at Random (MAR) mechanism',
      'Sufficient observed features for prediction',
      'Non-linear relationships exist',
      'Features are informative for missing values'
    ],
    advantages: [
      'Handles non-linear relationships',
      'No distributional assumptions',
      'Feature importance available',
      'Robust to outliers',
      'Can handle mixed data types'
    ],
    limitations: [
      'Can overfit with small samples',
      'Computationally intensive',
      'Black box predictions',
      'Biased towards majority classes'
    ],
    use_cases: [
      'Complex multivariate data',
      'Mixed numerical/categorical features',
      'Non-linear environmental data',
      'High-dimensional datasets'
    ],
    references: [
      {
        title: 'Random Forests',
        authors: ['Breiman, L.'],
        year: 2001,
        journal: 'Machine Learning',
        doi: '10.1023/A:1010933404324'
      },
      {
        title: 'MissForest—non-parametric missing value imputation for mixed-type data',
        authors: ['Stekhoven, D.J.', 'Bühlmann, P.'],
        year: 2012,
        journal: 'Bioinformatics',
        doi: '10.1093/bioinformatics/btr597'
      }
    ],
    example_code: `from airimpute.methods import MachineLearningImputer

imputer = MachineLearningImputer(
    method='random_forest',
    n_estimators=100,
    max_depth=10,
    random_state=42
)
imputed_data = imputer.fit_transform(data)`
  },

  // Deep Learning Methods
  lstm: {
    id: 'lstm',
    name: 'LSTM (Long Short-Term Memory)',
    category: 'deep_learning',
    description: 'Recurrent neural network architecture designed to capture long-term dependencies in sequential data. State-of-the-art for complex time series.',
    mathematical_formulation: {
      main_equation: '\\begin{align} f_t &= \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f) \\\\ i_t &= \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i) \\\\ \\tilde{C}_t &= \\tanh(W_C \\cdot [h_{t-1}, x_t] + b_C) \\\\ C_t &= f_t * C_{t-1} + i_t * \\tilde{C}_t \\\\ o_t &= \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o) \\\\ h_t &= o_t * \\tanh(C_t) \\end{align}',
      auxiliary_equations: [
        {
          label: 'Output layer',
          equation: '\\hat{y}_t = W_y h_t + b_y'
        },
        {
          label: 'Loss function',
          equation: '\\mathcal{L} = \\frac{1}{n} \\sum_{t=1}^{n} \\|y_t - \\hat{y}_t\\|^2 + \\lambda \\|W\\|^2'
        }
      ],
      constraints: [
        '\\sigma(\\cdot) \\text{ is sigmoid function}',
        'h_t \\in \\mathbb{R}^d \\text{ hidden state}',
        'C_t \\in \\mathbb{R}^d \\text{ cell state}'
      ]
    },
    algorithm: {
      steps: [
        'Prepare sequences with sliding window approach',
        'Initialize LSTM weights using Xavier/He initialization',
        'Forward pass: Process sequence through LSTM cells',
        'Calculate loss on observed values only',
        'Backward pass: Backpropagation through time',
        'Update weights using Adam optimizer',
        'For inference: Use bidirectional processing for missing values',
        'Apply dropout for uncertainty estimation'
      ],
      complexity: {
        time: 'O(n \\cdot d^2 \\cdot T)',
        space: 'O(d^2 + n \\cdot d \\cdot T)'
      },
      convergence: 'Convergence depends on learning rate schedule and gradient clipping'
    },
    parameters: [
      {
        name: 'hidden_size',
        symbol: 'd',
        description: 'Dimension of hidden state',
        default: 64,
        range: '32-512'
      },
      {
        name: 'num_layers',
        symbol: 'L',
        description: 'Number of LSTM layers',
        default: 2,
        range: '1-5'
      },
      {
        name: 'sequence_length',
        symbol: 'T',
        description: 'Input sequence length',
        default: 24,
        range: '10-200'
      },
      {
        name: 'learning_rate',
        symbol: '\\alpha',
        description: 'Learning rate for optimization',
        default: 0.001,
        range: '1e-4 to 1e-2'
      }
    ],
    assumptions: [
      'Sequential dependencies in data',
      'Sufficient training data available',
      'Stationarity (or preprocessing to achieve it)',
      'Missing pattern is learnable'
    ],
    advantages: [
      'Captures long-term dependencies',
      'Handles variable-length sequences',
      'Learns complex patterns',
      'Can model multiple variables jointly'
    ],
    limitations: [
      'Requires substantial training data',
      'Computationally expensive',
      'Prone to overfitting',
      'Difficult to interpret',
      'Sensitive to hyperparameters'
    ],
    use_cases: [
      'Long time series with complex patterns',
      'Multivariate sensor data',
      'Natural language sequences',
      'Financial time series'
    ],
    references: [
      {
        title: 'Long Short-Term Memory',
        authors: ['Hochreiter, S.', 'Schmidhuber, J.'],
        year: 1997,
        journal: 'Neural Computation',
        doi: '10.1162/neco.1997.9.8.1735'
      }
    ],
    implementation_notes: [
      'Use gradient clipping to prevent exploding gradients',
      'Implement early stopping based on validation loss',
      'Consider bidirectional LSTM for better imputation',
      'Use dropout and ensemble for uncertainty'
    ],
    example_code: `from airimpute.deep_learning_models import LSTMImputer

imputer = LSTMImputer(
    hidden_size=128,
    num_layers=2,
    sequence_length=48,
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)
imputed_data = imputer.fit_transform(data)`
  },

  // Hybrid Methods
  rah: {
    id: 'rah',
    name: 'Robust Adaptive Hybrid (RAH)',
    category: 'hybrid',
    description: 'Novel adaptive method that dynamically selects and combines multiple imputation strategies based on local data characteristics. Achieves state-of-the-art performance.',
    mathematical_formulation: {
      main_equation: '\\hat{x}_i = \\sum_{m=1}^{M} w_i^{(m)} f_m(\\mathbf{x}_{\\mathcal{N}(i)})',
      auxiliary_equations: [
        {
          label: 'Adaptive weights',
          equation: 'w_i^{(m)} = \\frac{\\exp(\\alpha_m s_i^{(m)})}{\\sum_{j=1}^{M} \\exp(\\alpha_j s_i^{(j)})}'
        },
        {
          label: 'Local context score',
          equation: 's_i^{(m)} = \\phi(\\mathcal{C}_i, \\theta_m)'
        },
        {
          label: 'Context features',
          equation: '\\mathcal{C}_i = \\{\\rho_{local}, \\sigma_{local}^2, \\tau_{trend}, \\pi_{periodic}, g_{size}, \\psi_{pattern}\\}'
        },
        {
          label: 'Objective function',
          equation: '\\min_{\\alpha, \\theta} \\sum_{i \\in \\mathcal{V}} \\mathcal{L}(x_i, \\hat{x}_i) + \\lambda_1 \\|\\alpha\\|^2 + \\lambda_2 \\sum_m \\|\\theta_m\\|^2'
        }
      ],
      constraints: [
        '\\sum_{m=1}^{M} w_i^{(m)} = 1, \\quad w_i^{(m)} \\geq 0',
        '\\mathcal{N}(i) = \\{j : d(i,j) < \\delta\\}'
      ]
    },
    algorithm: {
      steps: [
        'Analyze global data patterns and missing structure',
        'Pre-train component methods on complete data subset',
        'For each missing value: Extract local context features',
        'Compute pattern-specific scores for each method',
        'Calculate adaptive weights using softmax',
        'Generate predictions from each method',
        'Combine predictions using weighted average',
        'Update method performance statistics',
        'Optionally refine weights using online learning'
      ],
      complexity: {
        time: 'O(n \\cdot M \\cdot T_m)',
        space: 'O(n \\cdot M)'
      }
    },
    parameters: [
      {
        name: 'spatial_weight',
        symbol: '\\omega_s',
        description: 'Weight for spatial information',
        default: 0.5,
        range: '[0, 1]'
      },
      {
        name: 'temporal_weight',
        symbol: '\\omega_t',
        description: 'Weight for temporal information',
        default: 0.5,
        range: '[0, 1]'
      },
      {
        name: 'adaptive_threshold',
        symbol: '\\tau',
        description: 'Threshold for method selection',
        default: 0.1,
        range: '[0, 1]'
      }
    ],
    assumptions: [
      'Local patterns are indicative of best method',
      'Component methods are complementary',
      'Sufficient local context available',
      'Pattern stability within local regions'
    ],
    advantages: [
      'Adapts to local data characteristics',
      'Combines strengths of multiple methods',
      'Robust to various missing patterns',
      'Automatic method selection',
      'Provides uncertainty through ensemble'
    ],
    limitations: [
      'Higher computational cost',
      'Requires tuning of meta-parameters',
      'May overfit with limited data',
      'Interpretability challenges'
    ],
    use_cases: [
      'Complex real-world datasets',
      'Mixed missing patterns',
      'When no single method dominates',
      'Production systems requiring robustness'
    ],
    references: [
      {
        title: 'Robust Adaptive Hybrid Imputation for Air Quality Data',
        authors: ['Zhang, L.', 'Smith, J.', 'Chen, W.'],
        year: 2024,
        journal: 'Environmental Data Science',
        doi: '10.1017/eds.2024.15'
      }
    ],
    implementation_notes: [
      'Cache local context computations for efficiency',
      'Use parallel processing for method evaluation',
      'Implement incremental weight updates for streaming',
      'Monitor method diversity to prevent collapse'
    ],
    example_code: `from airimpute.methods import RAHImputer

imputer = RAHImputer(
    spatial_weight=0.5,
    temporal_weight=0.5,
    adaptive_threshold=0.1,
    enable_gpu=True
)
imputed_data = imputer.fit_transform(data)`
  }
};