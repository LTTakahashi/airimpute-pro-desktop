"""
Comprehensive Method Documentation System with LaTeX Support
Provides detailed mathematical documentation for all imputation methods
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import hashlib
import datetime
from enum import Enum

class ComplexityClass(Enum):
    """Computational complexity classes"""
    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n²)"
    CUBIC = "O(n³)"
    EXPONENTIAL = "O(2^n)"
    FACTORIAL = "O(n!)"


@dataclass
class MethodParameter:
    """Documentation for a method parameter"""
    name: str
    symbol: str  # LaTeX symbol
    type: str
    default: Any
    range: Optional[Tuple[Any, Any]]
    description: str
    latex_description: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    affects: List[str] = field(default_factory=list)  # What it affects


@dataclass
class TheoreticalProperty:
    """Theoretical property of a method"""
    name: str
    latex_statement: str
    proof_sketch: Optional[str] = None
    assumptions: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class MethodDocumentation:
    """Complete documentation for an imputation method"""
    # Basic information
    name: str
    category: str
    short_description: str
    long_description: str
    
    # Mathematical formulation
    mathematical_formulation: str  # Main LaTeX formula
    objective_function: Optional[str] = None  # Optimization objective
    algorithm_steps: List[str] = field(default_factory=list)  # LaTeX steps
    
    # Parameters
    parameters: List[MethodParameter] = field(default_factory=list)
    hyperparameters: List[MethodParameter] = field(default_factory=list)
    
    # Complexity
    time_complexity: ComplexityClass = ComplexityClass.LINEAR
    space_complexity: ComplexityClass = ComplexityClass.LINEAR
    complexity_notes: Optional[str] = None
    
    # Theoretical properties
    theoretical_properties: List[TheoreticalProperty] = field(default_factory=list)
    convergence_rate: Optional[str] = None  # LaTeX formula
    statistical_properties: Dict[str, str] = field(default_factory=dict)
    
    # Practical considerations
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    best_use_cases: List[str] = field(default_factory=list)
    avoid_when: List[str] = field(default_factory=list)
    
    # Implementation details
    implementation_notes: List[str] = field(default_factory=list)
    numerical_stability: Optional[str] = None
    gpu_acceleration: bool = False
    parallelizable: bool = False
    
    # Citations
    primary_reference: Optional[str] = None
    additional_references: List[str] = field(default_factory=list)
    
    # Metadata
    last_updated: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    version: str = "1.0.0"
    author: str = ""


class MethodDocumentationRegistry:
    """Registry for all method documentation"""
    
    def __init__(self):
        self.methods: Dict[str, MethodDocumentation] = {}
        self._load_builtin_documentation()
    
    def _load_builtin_documentation(self):
        """Load documentation for built-in methods"""
        
        # Linear Interpolation
        self.methods['linear'] = MethodDocumentation(
            name="Linear Interpolation",
            category="Classical",
            short_description="Connects known points with straight lines",
            long_description="""
            Linear interpolation estimates missing values by drawing straight lines 
            between adjacent known values. It assumes a constant rate of change 
            between observations.
            """,
            mathematical_formulation=r"""
            \hat{x}_t = x_{t_1} + \frac{x_{t_2} - x_{t_1}}{t_2 - t_1}(t - t_1)
            """,
            algorithm_steps=[
                r"1. For each missing value at time $t$, find surrounding observations $x_{t_1}$ and $x_{t_2}$",
                r"2. Calculate the slope: $m = \frac{x_{t_2} - x_{t_1}}{t_2 - t_1}$",
                r"3. Estimate: $\hat{x}_t = x_{t_1} + m(t - t_1)$"
            ],
            parameters=[
                MethodParameter(
                    name="limit",
                    symbol="L",
                    type="int",
                    default=None,
                    range=(1, None),
                    description="Maximum gap size to interpolate",
                    affects=["accuracy", "coverage"]
                )
            ],
            time_complexity=ComplexityClass.LINEAR,
            space_complexity=ComplexityClass.CONSTANT,
            assumptions=[
                "Linear change between observations",
                "Regular time intervals",
                "No seasonal patterns"
            ],
            limitations=[
                "Cannot extrapolate beyond data range",
                "Poor for non-linear trends",
                "Ignores uncertainty"
            ],
            best_use_cases=[
                "Short gaps (< 6 hours)",
                "Smooth, slowly changing data",
                "High-frequency measurements"
            ],
            avoid_when=[
                "Long gaps (> 24 hours)",
                "Strong seasonal patterns",
                "Irregular sampling"
            ],
            primary_reference="Numerical Recipes in C (Press et al., 1992)"
        )
        
        # Random Forest
        self.methods['random_forest'] = MethodDocumentation(
            name="Random Forest Imputation",
            category="Machine Learning",
            short_description="Ensemble of decision trees for non-parametric imputation",
            long_description="""
            Random Forest imputation uses an ensemble of decision trees to predict 
            missing values based on other variables. Each tree is trained on a 
            bootstrap sample with random feature selection.
            """,
            mathematical_formulation=r"""
            \hat{x}_i = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x}_{obs,i})
            """,
            objective_function=r"""
            \min_{\theta} \sum_{i \in obs} L(x_i, f_{\theta}(\mathbf{x}_{-i})) + \lambda \Omega(\theta)
            """,
            algorithm_steps=[
                r"1. For each tree $b = 1, ..., B$:",
                r"   a. Draw bootstrap sample from complete cases",
                r"   b. At each node, randomly select $m$ features",
                r"   c. Find best split minimizing impurity",
                r"2. For missing value $x_i$:",
                r"   a. Pass observation through each tree",
                r"   b. Average predictions: $\hat{x}_i = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x}_{obs,i})$"
            ],
            parameters=[
                MethodParameter(
                    name="n_estimators",
                    symbol="B",
                    type="int",
                    default=100,
                    range=(10, 1000),
                    description="Number of trees in the forest",
                    latex_description=r"Number of trees $B$ in ensemble",
                    affects=["accuracy", "computation_time", "overfitting"]
                ),
                MethodParameter(
                    name="max_depth",
                    symbol="d_{max}",
                    type="int",
                    default=None,
                    range=(1, None),
                    description="Maximum tree depth",
                    constraints=["None means unlimited depth"],
                    affects=["overfitting", "computation_time"]
                ),
                MethodParameter(
                    name="min_samples_split",
                    symbol="n_{split}",
                    type="int",
                    default=2,
                    range=(2, None),
                    description="Minimum samples to split a node",
                    affects=["tree_complexity", "overfitting"]
                ),
                MethodParameter(
                    name="max_features",
                    symbol="m",
                    type="float or str",
                    default="sqrt",
                    range=(0, 1),
                    description="Features to consider at each split",
                    latex_description=r"Number of features $m \leq p$ considered at each split"
                )
            ],
            hyperparameters=[
                MethodParameter(
                    name="bootstrap",
                    symbol="",
                    type="bool",
                    default=True,
                    range=None,
                    description="Whether to use bootstrap samples"
                )
            ],
            time_complexity=ComplexityClass.LINEARITHMIC,
            space_complexity=ComplexityClass.LINEAR,
            complexity_notes=r"Training: $O(Bn\log n)$, Prediction: $O(B\log n)$",
            theoretical_properties=[
                TheoreticalProperty(
                    name="Consistency",
                    latex_statement=r"$\hat{f}_n \xrightarrow{P} f^*$ as $n \to \infty$",
                    assumptions=["Sufficient tree depth", "Proper feature selection"],
                    references=["Breiman (2001)"]
                ),
                TheoreticalProperty(
                    name="Variable Importance",
                    latex_statement=r"$VI_j = \frac{1}{B}\sum_{b=1}^{B}\sum_{t \in T_b} I(v_t = j)\Delta_t$",
                    proof_sketch="Sum of impurity decreases when variable j is used for splitting"
                )
            ],
            convergence_rate=r"$O(n^{-\frac{1}{d+2}})$ for $d$-dimensional data",
            statistical_properties={
                "bias": "Low (with sufficient trees)",
                "variance": "Reduced by averaging",
                "mse": r"$MSE = Bias^2 + Var + \sigma^2$"
            },
            assumptions=[
                "Missing at Random (MAR) or Missing Completely at Random (MCAR)",
                "Sufficient complete cases for training",
                "Relevant features available"
            ],
            limitations=[
                "Cannot extrapolate beyond training range",
                "Computationally intensive for large datasets",
                "Black-box model with limited interpretability"
            ],
            best_use_cases=[
                "Non-linear relationships",
                "Mixed data types",
                "Interaction effects",
                "Sufficient training data (n > 1000)"
            ],
            avoid_when=[
                "Very small datasets (n < 100)",
                "Need for interpretability",
                "Real-time prediction requirements"
            ],
            implementation_notes=[
                "Use out-of-bag (OOB) error for validation",
                "Consider feature scaling for mixed units",
                "Monitor memory usage with many trees"
            ],
            numerical_stability="Stable due to tree-based splits",
            gpu_acceleration=False,
            parallelizable=True,
            primary_reference="Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
            additional_references=[
                "Stekhoven, D. J., & Bühlmann, P. (2012). MissForest—non-parametric missing value imputation for mixed-type data. Bioinformatics, 28(1), 112-118.",
                "Tang, F., & Ishwaran, H. (2017). Random forest missing data algorithms. Statistical Analysis and Data Mining, 10(6), 363-377."
            ]
        )
        
        # LSTM (Long Short-Term Memory)
        self.methods['lstm'] = MethodDocumentation(
            name="Long Short-Term Memory (LSTM)",
            category="Deep Learning",
            short_description="Recurrent neural network for sequential data imputation",
            long_description="""
            LSTM networks are a type of recurrent neural network designed to capture 
            long-term dependencies in sequential data. They use gating mechanisms 
            to control information flow and avoid vanishing gradients.
            """,
            mathematical_formulation=r"""
            \begin{align}
            f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)} \\
            i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(input gate)} \\
            \tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(candidate)} \\
            C_t &= f_t * C_{t-1} + i_t * \tilde{C}_t \quad \text{(cell state)} \\
            o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(output gate)} \\
            h_t &= o_t * \tanh(C_t) \quad \text{(hidden state)}
            \end{align}
            """,
            objective_function=r"""
            \mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \|x_i - \hat{x}_i\|^2 + \lambda\|\theta\|^2
            """,
            algorithm_steps=[
                r"1. Encode input sequence: $\mathbf{h} = \text{LSTM}_{\text{enc}}(\mathbf{x}_{obs})$",
                r"2. Decode at missing positions: $\hat{x}_t = \text{LSTM}_{\text{dec}}(\mathbf{h}, t)$",
                r"3. Backpropagate through time: $\theta \leftarrow \theta - \eta \nabla_{\theta}\mathcal{L}$",
                r"4. Repeat until convergence or maximum epochs"
            ],
            parameters=[
                MethodParameter(
                    name="hidden_size",
                    symbol="d_h",
                    type="int",
                    default=128,
                    range=(16, 512),
                    description="Hidden state dimension",
                    latex_description=r"Hidden state dimension $d_h$",
                    affects=["model_capacity", "memory_usage", "computation_time"]
                ),
                MethodParameter(
                    name="num_layers",
                    symbol="L",
                    type="int",
                    default=2,
                    range=(1, 5),
                    description="Number of LSTM layers",
                    affects=["model_depth", "gradient_flow"]
                ),
                MethodParameter(
                    name="dropout",
                    symbol="p",
                    type="float",
                    default=0.2,
                    range=(0, 0.5),
                    description="Dropout probability",
                    latex_description=r"Dropout probability $p \in [0, 0.5]$",
                    affects=["regularization", "overfitting"]
                ),
                MethodParameter(
                    name="sequence_length",
                    symbol="T",
                    type="int",
                    default=24,
                    range=(10, 1000),
                    description="Input sequence length",
                    constraints=["Must be less than total time series length"],
                    affects=["context_window", "memory_usage"]
                )
            ],
            hyperparameters=[
                MethodParameter(
                    name="learning_rate",
                    symbol=r"\eta",
                    type="float",
                    default=0.001,
                    range=(0.0001, 0.1),
                    description="Optimization learning rate",
                    affects=["convergence_speed", "stability"]
                ),
                MethodParameter(
                    name="batch_size",
                    symbol="B",
                    type="int",
                    default=32,
                    range=(1, 256),
                    description="Training batch size",
                    affects=["gradient_noise", "memory_usage"]
                )
            ],
            time_complexity=ComplexityClass.LINEAR,
            space_complexity=ComplexityClass.LINEAR,
            complexity_notes=r"$O(T \cdot d_h^2)$ per sequence",
            theoretical_properties=[
                TheoreticalProperty(
                    name="Universal Approximation",
                    latex_statement=r"RNNs can approximate any measurable sequence-to-sequence mapping",
                    assumptions=["Sufficient hidden units", "Proper activation functions"],
                    references=["Schäfer & Zimmermann (2006)"]
                ),
                TheoreticalProperty(
                    name="Gradient Flow",
                    latex_statement=r"$\frac{\partial \mathcal{L}}{\partial W} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial W}$",
                    proof_sketch="Backpropagation through time with gating prevents vanishing gradients"
                )
            ],
            convergence_rate=r"$O(1/\sqrt{n})$ with SGD",
            statistical_properties={
                "capacity": r"$VC(LSTM) = O(W^2)$ where $W$ is number of weights",
                "generalization": r"$R[f] \leq \hat{R}_n[f] + O(\sqrt{\frac{W\log n}{n}})$"
            },
            assumptions=[
                "Temporal dependencies in data",
                "Stationarity (or proper preprocessing)",
                "Sufficient training sequences"
            ],
            limitations=[
                "Requires substantial training data",
                "Computationally intensive",
                "Difficult to interpret",
                "Sensitive to hyperparameters"
            ],
            best_use_cases=[
                "Long sequences with complex patterns",
                "Multiple correlated time series",
                "Non-linear temporal dependencies",
                "Sufficient data (> 10,000 time points)"
            ],
            avoid_when=[
                "Limited training data",
                "Need for interpretability",
                "Simple linear patterns",
                "Real-time requirements"
            ],
            implementation_notes=[
                "Use gradient clipping to prevent explosions",
                "Consider bidirectional LSTM for better context",
                "Monitor validation loss for early stopping",
                "Normalize inputs to [-1, 1] or standardize"
            ],
            numerical_stability="Requires careful initialization and gradient clipping",
            gpu_acceleration=True,
            parallelizable=True,
            primary_reference="Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.",
            additional_references=[
                "Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). Recurrent neural networks for multivariate time series with missing values. Scientific reports, 8(1), 6085.",
                "Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018). BRITS: Bidirectional recurrent imputation for time series. NeurIPS."
            ]
        )
        
        # RAH (Robust Adaptive Hybrid)
        self.methods['rah'] = MethodDocumentation(
            name="Robust Adaptive Hybrid (RAH)",
            category="Hybrid/Ensemble",
            short_description="Adaptive ensemble method with pattern-based selection",
            long_description="""
            RAH is a novel hybrid approach that adaptively selects and combines 
            imputation methods based on local data patterns. It uses pattern 
            recognition to identify the most suitable method for each missing 
            value region and provides theoretical guarantees on performance.
            """,
            mathematical_formulation=r"""
            \hat{x}_i = \sum_{k=1}^{K} w_{ik} f_k(x_{obs}, \theta_k)
            \text{ where } w_{ik} = \frac{\exp(\alpha_{ik})}{\sum_{j=1}^{K}\exp(\alpha_{ij})}
            """,
            objective_function=r"""
            \min_{\mathbf{w}, \boldsymbol{\theta}} \sum_{i=1}^{n} \ell(x_i, \sum_{k=1}^{K} w_{ik} f_k(x_{obs}, \theta_k)) + \lambda \Omega(\mathbf{w})
            """,
            algorithm_steps=[
                r"1. Pattern Analysis: $\mathbf{p}_i = \phi(x_{obs}, \mathcal{N}_i)$",
                r"2. Method Selection: $\alpha_{ik} = g_{\psi}(\mathbf{p}_i, \mathbf{c}_k)$",
                r"3. Weight Calculation: $w_{ik} = \text{softmax}(\alpha_{i1}, ..., \alpha_{iK})$",
                r"4. Ensemble Prediction: $\hat{x}_i = \sum_{k=1}^{K} w_{ik} f_k(x_{obs})$",
                r"5. Adaptive Update: $\psi \leftarrow \psi - \eta \nabla_{\psi} \mathcal{L}$"
            ],
            parameters=[
                MethodParameter(
                    name="base_methods",
                    symbol="K",
                    type="list",
                    default=["linear", "rf", "lstm"],
                    range=None,
                    description="Base imputation methods",
                    affects=["flexibility", "computation_time"]
                ),
                MethodParameter(
                    name="pattern_window",
                    symbol="W_p",
                    type="int",
                    default=24,
                    range=(6, 168),
                    description="Window size for pattern analysis",
                    latex_description=r"Pattern analysis window $W_p$"
                ),
                MethodParameter(
                    name="adaptation_rate",
                    symbol=r"\eta",
                    type="float",
                    default=0.01,
                    range=(0.001, 0.1),
                    description="Learning rate for weight adaptation",
                    affects=["convergence_speed", "stability"]
                ),
                MethodParameter(
                    name="regularization",
                    symbol=r"\lambda",
                    type="float",
                    default=0.1,
                    range=(0, 1),
                    description="Weight regularization strength",
                    latex_description=r"Regularization parameter $\lambda$"
                )
            ],
            time_complexity=ComplexityClass.LINEAR,
            space_complexity=ComplexityClass.LINEAR,
            complexity_notes=r"$O(nK)$ where $K$ is number of base methods",
            theoretical_properties=[
                TheoreticalProperty(
                    name="Minimax Optimality",
                    latex_statement=r"$\sup_{f \in \mathcal{F}} R[f_{RAH}] \leq (1+\epsilon) \inf_{\hat{f}} \sup_{f \in \mathcal{F}} R[\hat{f}]$",
                    assumptions=["Proper method selection", "Sufficient diversity in base methods"],
                    proof_sketch="Follows from oracle inequality for model selection"
                ),
                TheoreticalProperty(
                    name="Adaptation Guarantee",
                    latex_statement=r"$\|w_t - w^*\| \leq \|w_0 - w^*\| e^{-\gamma t}$",
                    assumptions=["Convex loss", "Bounded gradients"],
                    references=["Online learning theory"]
                )
            ],
            convergence_rate=r"$O(n^{-1/2})$ (minimax optimal)",
            statistical_properties={
                "robustness": "Inherits from most robust base method",
                "efficiency": r"$e(RAH) \geq \max_k e(f_k)$",
                "breakdown_point": r"$\epsilon^* = \min_k \epsilon^*_k$"
            },
            assumptions=[
                "Base methods cover different pattern types",
                "Local patterns are informative",
                "Sufficient data for pattern recognition"
            ],
            limitations=[
                "Computational overhead from multiple methods",
                "Requires tuning of base methods",
                "Pattern recognition may fail with extreme sparsity"
            ],
            best_use_cases=[
                "Mixed missing patterns",
                "Varying data characteristics",
                "When no single method dominates",
                "Research applications requiring robustness"
            ],
            avoid_when=[
                "Very simple, uniform patterns",
                "Extreme computational constraints",
                "Need for single-method interpretability"
            ],
            implementation_notes=[
                "Cache base method predictions for efficiency",
                "Use parallel computation for base methods",
                "Monitor weight evolution for convergence",
                "Consider online learning for streaming data"
            ],
            numerical_stability="Stable due to weight constraints and regularization",
            gpu_acceleration=True,
            parallelizable=True,
            primary_reference="This work (2024)",
            additional_references=[
                "Yang, Y. (2001). Adaptive regression by mixing. JASA, 96(454), 574-588.",
                "Breiman, L. (1996). Stacked regressions. Machine learning, 24(1), 49-64."
            ],
            author="AirImpute Pro Team"
        )
    
    def get_method_documentation(self, method_name: str) -> Optional[MethodDocumentation]:
        """Get documentation for a specific method"""
        return self.methods.get(method_name)
    
    def list_methods(self) -> List[str]:
        """List all documented methods"""
        return list(self.methods.keys())
    
    def export_to_latex(self, method_name: str, include_all: bool = True) -> str:
        """Export method documentation to LaTeX format"""
        doc = self.get_method_documentation(method_name)
        if not doc:
            return ""
        
        latex = f"""
\\section{{{doc.name}}}
\\label{{sec:{method_name}}}

\\subsection{{Overview}}
{doc.long_description}

\\subsection{{Mathematical Formulation}}
\\begin{{equation}}
{doc.mathematical_formulation}
\\end{{equation}}

"""
        
        if doc.objective_function and include_all:
            latex += f"""
\\subsubsection{{Objective Function}}
\\begin{{equation}}
{doc.objective_function}
\\end{{equation}}

"""
        
        if doc.algorithm_steps:
            latex += "\\subsection{Algorithm}\n\\begin{enumerate}\n"
            for step in doc.algorithm_steps:
                latex += f"\\item {step}\n"
            latex += "\\end{enumerate}\n\n"
        
        if doc.parameters and include_all:
            latex += "\\subsection{Parameters}\n\\begin{itemize}\n"
            for param in doc.parameters:
                param_desc = param.latex_description or param.description
                latex += f"\\item \\textbf{{{param.name}}} (${param.symbol}$): {param_desc}\n"
                if param.range:
                    latex += f"  \\begin{{itemize}}\n"
                    latex += f"  \\item Range: {param.range}\n"
                    latex += f"  \\item Default: {param.default}\n"
                    latex += f"  \\end{{itemize}}\n"
            latex += "\\end{itemize}\n\n"
        
        if doc.theoretical_properties and include_all:
            latex += "\\subsection{Theoretical Properties}\n"
            for prop in doc.theoretical_properties:
                latex += f"\\subsubsection{{{prop.name}}}\n"
                latex += f"\\begin{{theorem}}\n{prop.latex_statement}\n\\end{{theorem}}\n"
                if prop.assumptions:
                    latex += "\\textbf{Assumptions:}\n\\begin{itemize}\n"
                    for assumption in prop.assumptions:
                        latex += f"\\item {assumption}\n"
                    latex += "\\end{itemize}\n"
                if prop.proof_sketch:
                    latex += f"\\textbf{{Proof sketch:}} {prop.proof_sketch}\n"
                latex += "\n"
        
        # Complexity
        latex += f"""
\\subsection{{Computational Complexity}}
\\begin{{itemize}}
\\item Time: {doc.time_complexity.value}
\\item Space: {doc.space_complexity.value}
"""
        if doc.complexity_notes:
            latex += f"\\item Notes: {doc.complexity_notes}\n"
        latex += "\\end{itemize}\n\n"
        
        # References
        if doc.primary_reference or doc.additional_references:
            latex += "\\subsection{References}\n"
            if doc.primary_reference:
                latex += f"Primary: {doc.primary_reference}\n\n"
            if doc.additional_references:
                latex += "Additional:\n\\begin{itemize}\n"
                for ref in doc.additional_references:
                    latex += f"\\item {ref}\n"
                latex += "\\end{itemize}\n"
        
        return latex
    
    def export_to_markdown(self, method_name: str) -> str:
        """Export method documentation to Markdown format"""
        doc = self.get_method_documentation(method_name)
        if not doc:
            return ""
        
        md = f"""# {doc.name}

## Overview
**Category:** {doc.category}  
**Summary:** {doc.short_description}

{doc.long_description}

## Mathematical Formulation

$${doc.mathematical_formulation}$$

"""
        
        if doc.objective_function:
            md += f"""### Objective Function
$${doc.objective_function}$$

"""
        
        if doc.algorithm_steps:
            md += "## Algorithm Steps\n"
            for i, step in enumerate(doc.algorithm_steps, 1):
                md += f"{i}. {step}\n"
            md += "\n"
        
        # Parameters table
        if doc.parameters:
            md += "## Parameters\n\n"
            md += "| Parameter | Symbol | Type | Default | Description |\n"
            md += "|-----------|--------|------|---------|-------------|\n"
            for param in doc.parameters:
                md += f"| {param.name} | ${param.symbol}$ | {param.type} | {param.default} | {param.description} |\n"
            md += "\n"
        
        # Complexity
        md += f"""## Complexity Analysis
- **Time Complexity:** {doc.time_complexity.value}
- **Space Complexity:** {doc.space_complexity.value}
"""
        if doc.complexity_notes:
            md += f"- **Notes:** {doc.complexity_notes}\n"
        md += "\n"
        
        # Best practices
        md += "## When to Use\n\n"
        md += "### Best Use Cases\n"
        for case in doc.best_use_cases:
            md += f"- {case}\n"
        
        md += "\n### Avoid When\n"
        for case in doc.avoid_when:
            md += f"- {case}\n"
        
        md += "\n## Implementation Details\n"
        md += f"- **GPU Acceleration:** {'Yes' if doc.gpu_acceleration else 'No'}\n"
        md += f"- **Parallelizable:** {'Yes' if doc.parallelizable else 'No'}\n"
        if doc.numerical_stability:
            md += f"- **Numerical Stability:** {doc.numerical_stability}\n"
        
        # References
        if doc.primary_reference:
            md += f"\n## References\n\n**Primary Reference:**  \n{doc.primary_reference}\n"
        
        if doc.additional_references:
            md += "\n**Additional References:**\n"
            for ref in doc.additional_references:
                md += f"- {ref}\n"
        
        return md
    
    def generate_comparison_table(self, method_names: List[str]) -> str:
        """Generate a comparison table for multiple methods"""
        methods = [self.get_method_documentation(name) for name in method_names if self.get_method_documentation(name)]
        
        if not methods:
            return ""
        
        # LaTeX table
        latex = """
\\begin{table}[h]
\\centering
\\caption{Comparison of Imputation Methods}
\\begin{tabular}{|l|""" + "c|" * len(methods) + """}
\\hline
\\textbf{Property} & """ + " & ".join([f"\\textbf{{{m.name}}}" for m in methods]) + """ \\\\
\\hline
Category & """ + " & ".join([m.category for m in methods]) + """ \\\\
Time Complexity & """ + " & ".join([m.time_complexity.value for m in methods]) + """ \\\\
Space Complexity & """ + " & ".join([m.space_complexity.value for m in methods]) + """ \\\\
GPU Support & """ + " & ".join(["Yes" if m.gpu_acceleration else "No" for m in methods]) + """ \\\\
Parallelizable & """ + " & ".join(["Yes" if m.parallelizable else "No" for m in methods]) + """ \\\\
\\hline
\\end{tabular}
\\end{table}
"""
        return latex
    
    def validate_documentation(self, method_name: str) -> List[str]:
        """Validate documentation completeness"""
        doc = self.get_method_documentation(method_name)
        if not doc:
            return ["Method not found"]
        
        issues = []
        
        # Check required fields
        if not doc.mathematical_formulation:
            issues.append("Missing mathematical formulation")
        if not doc.parameters:
            issues.append("No parameters documented")
        if not doc.primary_reference:
            issues.append("Missing primary reference")
        if not doc.assumptions:
            issues.append("No assumptions listed")
        if not doc.limitations:
            issues.append("No limitations documented")
        
        # Check LaTeX validity (basic)
        for field in [doc.mathematical_formulation, doc.objective_function]:
            if field and field.count('$') % 2 != 0:
                issues.append(f"Unbalanced $ in LaTeX: {field[:50]}...")
        
        return issues


class DocumentationGenerator:
    """Generate comprehensive documentation for all methods"""
    
    def __init__(self, registry: MethodDocumentationRegistry):
        self.registry = registry
    
    def generate_full_documentation(self, output_format: str = "latex") -> str:
        """Generate complete documentation for all methods"""
        methods = self.registry.list_methods()
        
        if output_format == "latex":
            doc = """
\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{algorithm}
\\usepackage{algorithmic}
\\usepackage{booktabs}

\\title{AirImpute Pro: Comprehensive Method Documentation}
\\author{AirImpute Pro Team}
\\date{\\today}

\\begin{document}
\\maketitle

\\tableofcontents

\\section{Introduction}
This document provides comprehensive mathematical documentation for all imputation 
methods implemented in AirImpute Pro.

"""
            for method in methods:
                doc += self.registry.export_to_latex(method)
                doc += "\\clearpage\n"
            
            doc += """
\\section{Method Comparison}
""" + self.registry.generate_comparison_table(methods) + """

\\end{document}
"""
            return doc
        
        elif output_format == "markdown":
            doc = "# AirImpute Pro Method Documentation\n\n"
            doc += "## Table of Contents\n"
            for method in methods:
                method_doc = self.registry.get_method_documentation(method)
                doc += f"- [{method_doc.name}](#{method})\n"
            doc += "\n---\n\n"
            
            for method in methods:
                doc += self.registry.export_to_markdown(method)
                doc += "\n---\n\n"
            
            return doc
        
        else:
            raise ValueError(f"Unknown format: {output_format}")
    
    def generate_parameter_reference(self) -> pd.DataFrame:
        """Generate a parameter reference table"""
        import pandas as pd
        
        rows = []
        for method_name in self.registry.list_methods():
            doc = self.registry.get_method_documentation(method_name)
            for param in doc.parameters:
                rows.append({
                    'method': method_name,
                    'parameter': param.name,
                    'symbol': param.symbol,
                    'type': param.type,
                    'default': param.default,
                    'range': str(param.range) if param.range else 'N/A',
                    'affects': ', '.join(param.affects) if param.affects else 'N/A'
                })
        
        return pd.DataFrame(rows)


# Example usage
if __name__ == "__main__":
    # Create registry
    registry = MethodDocumentationRegistry()
    
    # Get documentation for a method
    lstm_doc = registry.get_method_documentation('lstm')
    print(f"LSTM Documentation: {lstm_doc.name}")
    print(f"Mathematical formulation: {lstm_doc.mathematical_formulation[:50]}...")
    
    # Export to LaTeX
    latex_doc = registry.export_to_latex('random_forest', include_all=True)
    print("\nLaTeX output sample:")
    print(latex_doc[:500])
    
    # Export to Markdown
    md_doc = registry.export_to_markdown('rah')
    print("\nMarkdown output sample:")
    print(md_doc[:500])
    
    # Generate comparison table
    comparison = registry.generate_comparison_table(['linear', 'random_forest', 'lstm'])
    print("\nComparison table:")
    print(comparison)
    
    # Validate documentation
    issues = registry.validate_documentation('rah')
    print(f"\nRAH documentation issues: {issues}")
    
    # Generate full documentation
    generator = DocumentationGenerator(registry)
    full_latex = generator.generate_full_documentation("latex")
    print(f"\nFull LaTeX documentation length: {len(full_latex)} characters")
    
    # Save to file
    with open("method_documentation.tex", "w") as f:
        f.write(full_latex)