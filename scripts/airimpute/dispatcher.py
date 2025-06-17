"""
Secure dispatcher for Python bridge commands
This replaces dangerous eval/exec with a data-oriented approach
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Global storage for dataframes (in production, use a proper store)
dataframes: Dict[str, pd.DataFrame] = {}


def handle_multiply_constant(params: Dict[str, Any]) -> Dict[str, Any]:
    """Multiply a column by a constant value"""
    try:
        df_name = params['target_df']
        result_col = params['result_col']
        source_col = params['source_col']
        value = params['value']
        
        # Validate parameters
        if not isinstance(value, (int, float)):
            raise TypeError("Multiplier must be a number")
        if not isinstance(result_col, str) or not isinstance(source_col, str):
            raise TypeError("Column names must be strings")
        
        # Get dataframe
        if df_name not in dataframes:
            raise KeyError(f"DataFrame '{df_name}' not found")
        
        df = dataframes[df_name]
        
        # Validate column exists
        if source_col not in df.columns:
            raise KeyError(f"Column '{source_col}' not found in dataframe")
        
        # Perform operation
        df[result_col] = df[source_col] * value
        
        return {
            "status": "success",
            "data": {
                "rows_affected": len(df),
                "result_column": result_col
            }
        }
    except Exception as e:
        logger.error(f"Error in multiply_constant: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


def handle_select_columns(params: Dict[str, Any]) -> Dict[str, Any]:
    """Select specific columns from dataframe"""
    try:
        df_name = params['target_df']
        columns = params['columns']
        
        if not isinstance(columns, list):
            raise TypeError("Columns must be a list")
        
        if df_name not in dataframes:
            raise KeyError(f"DataFrame '{df_name}' not found")
        
        df = dataframes[df_name]
        
        # Validate all columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found: {missing_cols}")
        
        # Create new dataframe with selected columns
        result_df = df[columns].copy()
        result_name = f"{df_name}_selected"
        dataframes[result_name] = result_df
        
        return {
            "status": "success",
            "data": {
                "result_df": result_name,
                "shape": list(result_df.shape),
                "columns": list(result_df.columns)
            }
        }
    except Exception as e:
        logger.error(f"Error in select_columns: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


def handle_calculate_stats(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate statistics for dataframe"""
    try:
        df_name = params.get('target_df', 'default')
        columns = params.get('columns', None)
        
        if df_name not in dataframes:
            raise KeyError(f"DataFrame '{df_name}' not found")
        
        df = dataframes[df_name]
        
        if columns:
            df = df[columns]
        
        # Calculate statistics
        stats = {
            "mean": df.mean().to_dict(),
            "std": df.std().to_dict(),
            "min": df.min().to_dict(),
            "max": df.max().to_dict(),
            "count": df.count().to_dict(),
            "missing": df.isnull().sum().to_dict()
        }
        
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error in calculate_stats: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


def handle_impute_advanced(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle advanced imputation with our secure methods"""
    try:
        df_name = params.get('target_df', 'default')
        method = params['method']
        method_params = params.get('params', {})
        
        if df_name not in dataframes:
            raise KeyError(f"DataFrame '{df_name}' not found")
        
        df = dataframes[df_name]
        
        # Import our imputation module
        from airimpute.core import ImputationEngine
        
        engine = ImputationEngine()
        
        # Map method names to secure function calls
        if method == 'rah':
            result = engine.impute_rah(df, **method_params)
        elif method == 'mean':
            result = engine.impute_mean(df, **method_params)
        elif method == 'interpolate':
            result = engine.impute_interpolate(df, **method_params)
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        # Store result
        result_name = f"{df_name}_imputed"
        dataframes[result_name] = result['data']
        
        return {
            "status": "success",
            "data": {
                "result_df": result_name,
                "metrics": result.get('metrics', {}),
                "gaps_filled": result.get('gaps_filled', 0)
            }
        }
    except Exception as e:
        logger.error(f"Error in impute_advanced: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# Dispatcher mapping - this is our security boundary
OPERATION_HANDLERS = {
    "multiply_constant": handle_multiply_constant,
    "select_columns": handle_select_columns,
    "calculate_stats": handle_calculate_stats,
    "impute_advanced": handle_impute_advanced,
    # Add more handlers as needed
}


def dispatch(command_json: str) -> str:
    """
    Main entry point for all bridge commands
    This is the ONLY function that should be called from Rust
    """
    try:
        # Parse command
        command = json.loads(command_json)
        operation = command.get('operation')
        params = command.get('params', {})
        
        # Validate operation exists
        if operation not in OPERATION_HANDLERS:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Call the appropriate handler
        handler = OPERATION_HANDLERS[operation]
        result = handler(params)
        
        # Return JSON response
        return json.dumps(result)
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Invalid JSON: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Dispatcher error: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


def load_dataframe(name: str, data: Any) -> bool:
    """Utility function to load data into the dataframe store"""
    try:
        if isinstance(data, pd.DataFrame):
            dataframes[name] = data
        elif isinstance(data, dict):
            dataframes[name] = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            dataframes[name] = pd.DataFrame(data)
        else:
            return False
        return True
    except Exception as e:
        logger.error(f"Error loading dataframe: {e}")
        return False