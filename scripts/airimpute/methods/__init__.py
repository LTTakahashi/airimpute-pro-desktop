"""
Imputation methods for AirImpute
"""

from .base import BaseImputer
from .simple import MeanImputation, ForwardFill, BackwardFill
from .interpolation import LinearInterpolation, SplineInterpolation
from .statistical import KalmanFilter
from .machine_learning import RandomForest, KNNImputation, MatrixFactorization, DeepLearningImputation
from .rah import RAHMethod

__all__ = [
    "BaseImputer",
    "MeanImputation",
    "ForwardFill",
    "BackwardFill",
    "LinearInterpolation",
    "SplineInterpolation",
    "KalmanFilter",
    "RandomForest",
    "KNNImputation",
    "MatrixFactorization",
    "DeepLearningImputation",
    "RAHMethod",
    "get_available_methods"
]

def get_available_methods():
    """Get list of all available imputation methods"""
    return [
        {
            "id": "mean",
            "name": "Mean Imputation",
            "description": "Replace missing values with column mean",
            "category": "statistical"
        },
        {
            "id": "linear",
            "name": "Linear Interpolation",
            "description": "Linear interpolation between known values",
            "category": "statistical"
        },
        {
            "id": "spline",
            "name": "Spline Interpolation",
            "description": "Smooth spline interpolation",
            "category": "statistical"
        },
        {
            "id": "forward_fill",
            "name": "Forward Fill",
            "description": "Propagate last valid observation forward",
            "category": "statistical"
        },
        {
            "id": "backward_fill",
            "name": "Backward Fill",
            "description": "Propagate next valid observation backward",
            "category": "statistical"
        },
        {
            "id": "kalman",
            "name": "Kalman Filter",
            "description": "State-space model based imputation",
            "category": "statistical"
        },
        {
            "id": "random_forest",
            "name": "Random Forest",
            "description": "Machine learning based imputation",
            "category": "machine_learning"
        },
        {
            "id": "knn",
            "name": "K-Nearest Neighbors",
            "description": "KNN-based imputation using similar samples",
            "category": "machine_learning"
        },
        {
            "id": "matrix_factorization",
            "name": "Matrix Factorization",
            "description": "Low-rank matrix factorization for multivariate imputation",
            "category": "machine_learning"
        },
        {
            "id": "deep_learning",
            "name": "Deep Learning",
            "description": "Neural network-based imputation for complex patterns",
            "category": "deep_learning"
        },
        {
            "id": "rah",
            "name": "Robust Adaptive Hybrid (RAH)",
            "description": "Advanced hybrid method with 42.1% improvement",
            "category": "hybrid"
        }
    ]