#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

console.log('🐍 Setting up Python environment for AirImpute Pro...\n');

// Check if Python is installed
try {
  const pythonVersion = execSync('python3 --version', { encoding: 'utf8' });
  console.log(`✅ Python found: ${pythonVersion.trim()}`);
} catch (error) {
  console.error('❌ Python 3 is not installed or not in PATH');
  console.error('Please install Python 3.8 or higher from https://www.python.org/');
  process.exit(1);
}

// Create virtual environment directory
const venvPath = path.join(__dirname, '..', 'python-env');
if (!fs.existsSync(venvPath)) {
  console.log('📦 Creating Python virtual environment...');
  try {
    execSync(`python3 -m venv ${venvPath}`, { stdio: 'inherit' });
    console.log('✅ Virtual environment created');
  } catch (error) {
    console.error('❌ Failed to create virtual environment');
    process.exit(1);
  }
} else {
  console.log('✅ Virtual environment already exists');
}

// Determine pip command based on OS
const isWindows = os.platform() === 'win32';
const pipCmd = isWindows 
  ? path.join(venvPath, 'Scripts', 'pip.exe')
  : path.join(venvPath, 'bin', 'pip');

// Create requirements.txt
const requirements = `# Core scientific computing
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
scipy>=1.10.0,<2.0.0
scikit-learn>=1.3.0,<2.0.0
statsmodels>=0.14.0,<1.0.0

# Machine learning
xgboost>=1.7.0,<2.0.0
lightgbm>=4.0.0,<5.0.0

# Data handling
pyarrow>=14.0.0
openpyxl>=3.1.0
h5py>=3.9.0
netCDF4>=1.6.0

# Utilities
joblib>=1.3.0
tqdm>=4.66.0
click>=8.1.0

# For development
black>=23.0.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
`;

const requirementsPath = path.join(__dirname, '..', 'requirements.txt');
fs.writeFileSync(requirementsPath, requirements);
console.log('📝 Created requirements.txt');

// Install Python packages
console.log('\n📦 Installing Python packages (this may take a few minutes)...');
try {
  execSync(`${pipCmd} install --upgrade pip`, { stdio: 'inherit' });
  execSync(`${pipCmd} install -r ${requirementsPath}`, { stdio: 'inherit' });
  console.log('\n✅ Python packages installed successfully');
} catch (error) {
  console.error('\n❌ Failed to install Python packages');
  console.error('You may need to install them manually');
}

// Create Python wrapper module
const pythonWrapper = `"""
AirImpute Pro Python Wrapper
Provides interface between Rust and Python imputation algorithms
"""

import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import imputation algorithms from the main project
sys.path.append('../../../source_code')
try:
    from enhanced_imputation_methods import EnhancedImputationMethods
    from enhanced_evaluation_framework import EnhancedEvaluationFramework
except ImportError:
    logger.warning("Could not import enhanced methods, using basic implementation")
    # Fallback implementation would go here

class ImputationBridge:
    """Bridge between Rust and Python imputation methods"""
    
    def __init__(self):
        self.methods = EnhancedImputationMethods() if 'EnhancedImputationMethods' in globals() else None
        self.evaluator = EnhancedEvaluationFramework() if 'EnhancedEvaluationFramework' in globals() else None
    
    def impute(self, data: Dict[str, Any], method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform imputation on the provided data
        
        Args:
            data: Dictionary containing the dataset
            method: Imputation method to use
            params: Method-specific parameters
            
        Returns:
            Dictionary containing imputed data and metrics
        """
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data['values'], 
                            index=pd.to_datetime(data['index']),
                            columns=data['columns'])
            
            # Apply imputation
            if self.methods and hasattr(self.methods, method):
                result = getattr(self.methods, method)(df, **params)
            else:
                # Fallback to simple mean imputation
                result = df.fillna(df.mean())
            
            # Calculate metrics
            metrics = self._calculate_metrics(df, result)
            
            return {
                'success': True,
                'imputed_data': {
                    'values': result.values.tolist(),
                    'index': result.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'columns': result.columns.tolist()
                },
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Imputation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_metrics(self, original: pd.DataFrame, imputed: pd.DataFrame) -> Dict[str, float]:
        """Calculate imputation quality metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['missing_before'] = original.isna().sum().sum()
        metrics['missing_after'] = imputed.isna().sum().sum()
        metrics['recovery_rate'] = 1 - (metrics['missing_after'] / metrics['missing_before']) if metrics['missing_before'] > 0 else 0
        
        # Statistical preservation
        for col in original.columns:
            orig_mean = original[col].mean()
            imp_mean = imputed[col].mean()
            metrics[f'{col}_mean_diff'] = abs(imp_mean - orig_mean) if not pd.isna(orig_mean) else 0
            
            orig_std = original[col].std()
            imp_std = imputed[col].std()
            metrics[f'{col}_std_ratio'] = imp_std / orig_std if orig_std > 0 else 1
        
        return metrics

# Export the bridge instance
bridge = ImputationBridge()
`;

const wrapperPath = path.join(__dirname, '..', 'src-tauri', 'python', 'imputation_bridge.py');
const pythonDir = path.dirname(wrapperPath);
if (!fs.existsSync(pythonDir)) {
  fs.mkdirSync(pythonDir, { recursive: true });
}
fs.writeFileSync(wrapperPath, pythonWrapper);
console.log('🐍 Created Python wrapper module');

console.log('\n✅ Python setup complete!');
console.log('\nNext steps:');
console.log('1. Run "npm install" to install Node.js dependencies');
console.log('2. Run "npm run dev" to start the development server');