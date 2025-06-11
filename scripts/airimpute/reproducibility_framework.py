"""
Comprehensive Reproducibility Framework with Workflow Tracking
Implements complete provenance tracking for academic reproducibility standards
"""

import os
import sys
import json
import yaml
import hashlib
import platform
import subprocess
import datetime
import uuid
import pickle
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
import git
import pkg_resources
import psutil
import GPUtil
import logging
from functools import wraps
import inspect
import ast
import dis

logger = logging.getLogger(__name__)


@dataclass
class SystemEnvironment:
    """Complete system environment information"""
    # Operating System
    os_name: str
    os_version: str
    os_release: str
    architecture: str
    
    # Hardware
    cpu_model: str
    cpu_count: int
    cpu_frequency: float
    total_memory: int
    available_memory: int
    
    # GPU Information
    gpu_count: int
    gpu_models: List[str]
    gpu_memory: List[int]
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    
    # Python Environment
    python_version: str
    python_implementation: str
    python_compiler: str
    virtual_env: Optional[str]
    conda_env: Optional[str]
    
    # Package Versions
    packages: Dict[str, str]
    pip_freeze: str
    
    # System Libraries
    system_libraries: Dict[str, str]
    
    # Environment Variables
    env_vars: Dict[str, str]
    
    # Timestamp
    captured_at: str
    
    @classmethod
    def capture(cls) -> 'SystemEnvironment':
        """Capture current system environment"""
        # OS Information
        os_info = platform.uname()
        
        # CPU Information
        cpu_freq = psutil.cpu_freq()
        
        # GPU Information
        gpu_info = cls._capture_gpu_info()
        
        # Python Information
        python_info = {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'compiler': platform.python_compiler(),
        }
        
        # Virtual environment detection
        virtual_env = os.environ.get('VIRTUAL_ENV')
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        
        # Package versions
        packages = cls._capture_package_versions()
        pip_freeze = cls._capture_pip_freeze()
        
        # System libraries
        system_libs = cls._capture_system_libraries()
        
        # Environment variables (filtered for security)
        env_vars = cls._capture_environment_variables()
        
        return cls(
            os_name=os_info.system,
            os_version=os_info.version,
            os_release=os_info.release,
            architecture=os_info.machine,
            cpu_model=platform.processor(),
            cpu_count=psutil.cpu_count(logical=True),
            cpu_frequency=cpu_freq.max if cpu_freq else 0,
            total_memory=psutil.virtual_memory().total,
            available_memory=psutil.virtual_memory().available,
            gpu_count=gpu_info['count'],
            gpu_models=gpu_info['models'],
            gpu_memory=gpu_info['memory'],
            cuda_version=gpu_info['cuda_version'],
            cudnn_version=gpu_info['cudnn_version'],
            python_version=python_info['version'],
            python_implementation=python_info['implementation'],
            python_compiler=python_info['compiler'],
            virtual_env=virtual_env,
            conda_env=conda_env,
            packages=packages,
            pip_freeze=pip_freeze,
            system_libraries=system_libs,
            env_vars=env_vars,
            captured_at=datetime.datetime.now().isoformat()
        )
    
    @staticmethod
    def _capture_gpu_info() -> Dict[str, Any]:
        """Capture GPU information"""
        info = {
            'count': 0,
            'models': [],
            'memory': [],
            'cuda_version': None,
            'cudnn_version': None
        }
        
        try:
            gpus = GPUtil.getGPUs()
            info['count'] = len(gpus)
            info['models'] = [gpu.name for gpu in gpus]
            info['memory'] = [gpu.memoryTotal for gpu in gpus]
            
            # CUDA version
            try:
                import torch
                if torch.cuda.is_available():
                    info['cuda_version'] = torch.version.cuda
                    # CuDNN version from torch
                    info['cudnn_version'] = str(torch.backends.cudnn.version())
            except ImportError:
                pass
                
        except Exception as e:
            logger.warning(f"Failed to capture GPU info: {e}")
            
        return info
    
    @staticmethod
    def _capture_package_versions() -> Dict[str, str]:
        """Capture installed package versions"""
        packages = {}
        
        # Critical packages for reproducibility
        critical_packages = [
            'numpy', 'scipy', 'pandas', 'scikit-learn',
            'torch', 'tensorflow', 'matplotlib', 'seaborn',
            'statsmodels', 'xgboost', 'lightgbm', 'airimpute'
        ]
        
        for package in critical_packages:
            try:
                version = pkg_resources.get_distribution(package).version
                packages[package] = version
            except Exception:
                packages[package] = 'not installed'
                
        return packages
    
    @staticmethod
    def _capture_pip_freeze() -> str:
        """Capture complete pip freeze output"""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'freeze'],
                capture_output=True,
                text=True
            )
            return result.stdout
        except Exception as e:
            logger.warning(f"Failed to capture pip freeze: {e}")
            return ""
    
    @staticmethod
    def _capture_system_libraries() -> Dict[str, str]:
        """Capture system library versions"""
        libraries = {}
        
        # Check common scientific libraries
        lib_commands = {
            'libblas': ['ldconfig', '-p', '|', 'grep', 'blas'],
            'liblapack': ['ldconfig', '-p', '|', 'grep', 'lapack'],
            'libfftw': ['ldconfig', '-p', '|', 'grep', 'fftw'],
        }
        
        for lib, cmd in lib_commands.items():
            try:
                result = subprocess.run(
                    ' '.join(cmd),
                    shell=True,
                    capture_output=True,
                    text=True
                )
                libraries[lib] = result.stdout.strip() if result.stdout else 'not found'
            except Exception:
                libraries[lib] = 'check failed'
                
        return libraries
    
    @staticmethod
    def _capture_environment_variables() -> Dict[str, str]:
        """Capture relevant environment variables"""
        # Only capture non-sensitive environment variables
        safe_prefixes = [
            'PYTHON', 'CONDA', 'VIRTUAL', 'CUDA', 'MKL',
            'OMP', 'OPENBLAS', 'NUMEXPR', 'PATH'
        ]
        
        env_vars = {}
        for key, value in os.environ.items():
            if any(key.startswith(prefix) for prefix in safe_prefixes):
                # Truncate long values
                if len(value) > 200:
                    value = value[:200] + '...'
                env_vars[key] = value
                
        return env_vars
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def compute_hash(self) -> str:
        """Compute hash of environment for comparison"""
        # Exclude timestamp for hash computation
        env_dict = self.to_dict()
        env_dict.pop('captured_at', None)
        
        # Sort for consistent ordering
        env_str = json.dumps(env_dict, sort_keys=True)
        return hashlib.sha256(env_str.encode()).hexdigest()


@dataclass
class DataProvenance:
    """Track data lineage and transformations"""
    dataset_id: str
    original_path: Optional[str]
    original_hash: str
    original_shape: Tuple[int, ...]
    original_columns: List[str]
    
    # Transformations applied
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Missing data information
    original_missing_count: int = 0
    original_missing_pattern: Optional[str] = None
    
    # Temporal information
    temporal_range: Optional[Tuple[str, str]] = None
    sampling_frequency: Optional[str] = None
    
    # Quality metrics
    quality_checks: Dict[str, Any] = field(default_factory=dict)
    
    def add_transformation(self, name: str, params: Dict[str, Any], 
                         result_hash: str):
        """Add a transformation to the lineage"""
        self.transformations.append({
            'name': name,
            'params': params,
            'result_hash': result_hash,
            'timestamp': datetime.datetime.now().isoformat()
        })


@dataclass
class MethodProvenance:
    """Track method execution details"""
    method_name: str
    method_version: str
    method_hash: str  # Hash of method source code
    
    # Parameters
    parameters: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    
    # Execution details
    random_seed: Optional[int]
    execution_time: float
    convergence_info: Optional[Dict[str, Any]]
    
    # Resource usage
    peak_memory: int
    cpu_time: float
    gpu_time: Optional[float]
    
    # Validation
    validation_metrics: Dict[str, float]
    cross_validation_info: Optional[Dict[str, Any]]


@dataclass
class WorkflowStep:
    """Single step in a workflow"""
    step_id: str
    step_type: str  # 'data_load', 'preprocessing', 'imputation', 'validation', 'export'
    step_name: str
    
    # Inputs and outputs
    input_hashes: List[str]
    output_hash: str
    
    # Execution details
    start_time: str
    end_time: str
    duration: float
    success: bool
    error_message: Optional[str]
    
    # Provenance
    data_provenance: Optional[DataProvenance]
    method_provenance: Optional[MethodProvenance]
    
    # Artifacts
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path


@dataclass
class Workflow:
    """Complete workflow with all steps"""
    workflow_id: str
    name: str
    description: str
    
    # Metadata
    created_at: str
    created_by: str
    project: Optional[str]
    
    # Environment
    environment: SystemEnvironment
    
    # Git information
    git_commit: Optional[str]
    git_branch: Optional[str]
    git_dirty: bool
    
    # Steps
    steps: List[WorkflowStep] = field(default_factory=list)
    
    # Global random seed
    random_seed: Optional[int]
    
    # Status
    status: str = "in_progress"  # in_progress, completed, failed
    
    def add_step(self, step: WorkflowStep):
        """Add a step to the workflow"""
        self.steps.append(step)
        
    def complete(self, status: str = "completed"):
        """Mark workflow as complete"""
        self.status = status


class ReproducibilityTracker:
    """Main class for tracking reproducibility"""
    
    def __init__(self, 
                 storage_dir: Union[str, Path] = "~/.airimpute/reproducibility",
                 auto_capture_git: bool = True):
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_capture_git = auto_capture_git
        self.current_workflow: Optional[Workflow] = None
        
        # Method source code cache
        self._method_cache: Dict[str, str] = {}
        
    def start_workflow(self, 
                      name: str,
                      description: str = "",
                      project: Optional[str] = None,
                      random_seed: Optional[int] = None) -> Workflow:
        """Start tracking a new workflow"""
        # Capture environment
        environment = SystemEnvironment.capture()
        
        # Git information
        git_info = self._capture_git_info() if self.auto_capture_git else {
            'commit': None, 'branch': None, 'dirty': False
        }
        
        # Create workflow
        workflow = Workflow(
            workflow_id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_at=datetime.datetime.now().isoformat(),
            created_by=os.environ.get('USER', 'unknown'),
            project=project,
            environment=environment,
            git_commit=git_info['commit'],
            git_branch=git_info['branch'],
            git_dirty=git_info['dirty'],
            random_seed=random_seed
        )
        
        self.current_workflow = workflow
        
        # Set random seed if provided
        if random_seed is not None:
            self._set_random_seed(random_seed)
            
        logger.info(f"Started workflow: {workflow.workflow_id}")
        return workflow
    
    def track_data_loading(self, 
                          data: Union[pd.DataFrame, np.ndarray],
                          source_path: Optional[str] = None,
                          name: str = "data") -> DataProvenance:
        """Track data loading step"""
        # Compute data hash
        data_hash = self._compute_data_hash(data)
        
        # Extract metadata
        if isinstance(data, pd.DataFrame):
            shape = data.shape
            columns = list(data.columns)
            missing_count = data.isnull().sum().sum()
            
            # Temporal information
            if 'timestamp' in data.columns or 'date' in data.columns:
                time_col = 'timestamp' if 'timestamp' in data.columns else 'date'
                temporal_range = (
                    str(data[time_col].min()),
                    str(data[time_col].max())
                )
            else:
                temporal_range = None
        else:
            shape = data.shape
            columns = [f"col_{i}" for i in range(shape[1])] if len(shape) > 1 else ["value"]
            missing_count = np.isnan(data).sum()
            temporal_range = None
        
        # Create provenance
        provenance = DataProvenance(
            dataset_id=str(uuid.uuid4()),
            original_path=source_path,
            original_hash=data_hash,
            original_shape=shape,
            original_columns=columns,
            original_missing_count=missing_count,
            temporal_range=temporal_range
        )
        
        # Add as workflow step if active
        if self.current_workflow:
            step = WorkflowStep(
                step_id=str(uuid.uuid4()),
                step_type="data_load",
                step_name=f"Load {name}",
                input_hashes=[],
                output_hash=data_hash,
                start_time=datetime.datetime.now().isoformat(),
                end_time=datetime.datetime.now().isoformat(),
                duration=0,
                success=True,
                error_message=None,
                data_provenance=provenance,
                method_provenance=None
            )
            self.current_workflow.add_step(step)
            
        return provenance
    
    def track_method_execution(self,
                             method_name: str,
                             method_func: callable,
                             parameters: Dict[str, Any],
                             input_data: Any,
                             output_data: Any,
                             execution_time: float,
                             validation_metrics: Optional[Dict[str, float]] = None) -> MethodProvenance:
        """Track method execution"""
        # Get method source code and hash
        method_source = self._get_method_source(method_func)
        method_hash = hashlib.sha256(method_source.encode()).hexdigest()
        
        # Get method version
        method_version = self._get_method_version(method_func)
        
        # Compute data hashes
        input_hash = self._compute_data_hash(input_data)
        output_hash = self._compute_data_hash(output_data)
        
        # Resource usage
        memory_info = psutil.Process().memory_info()
        
        # Create method provenance
        provenance = MethodProvenance(
            method_name=method_name,
            method_version=method_version,
            method_hash=method_hash,
            parameters=parameters,
            hyperparameters={},  # Extract if available
            random_seed=self.current_workflow.random_seed if self.current_workflow else None,
            execution_time=execution_time,
            convergence_info=None,  # Extract if available
            peak_memory=memory_info.rss,
            cpu_time=execution_time,  # Approximate
            gpu_time=None,  # Track if GPU used
            validation_metrics=validation_metrics or {},
            cross_validation_info=None
        )
        
        # Add as workflow step
        if self.current_workflow:
            step = WorkflowStep(
                step_id=str(uuid.uuid4()),
                step_type="imputation",
                step_name=f"Execute {method_name}",
                input_hashes=[input_hash],
                output_hash=output_hash,
                start_time=datetime.datetime.now().isoformat(),
                end_time=datetime.datetime.now().isoformat(),
                duration=execution_time,
                success=True,
                error_message=None,
                data_provenance=None,
                method_provenance=provenance
            )
            self.current_workflow.add_step(step)
            
        return provenance
    
    def save_workflow(self, workflow: Optional[Workflow] = None) -> Path:
        """Save workflow to disk"""
        workflow = workflow or self.current_workflow
        if not workflow:
            raise ValueError("No workflow to save")
            
        # Create workflow directory
        workflow_dir = self.storage_dir / workflow.workflow_id
        workflow_dir.mkdir(exist_ok=True)
        
        # Save workflow metadata
        workflow_path = workflow_dir / "workflow.json"
        with open(workflow_path, 'w') as f:
            json.dump(self._workflow_to_dict(workflow), f, indent=2)
            
        # Save additional files
        self._save_workflow_artifacts(workflow, workflow_dir)
        
        # Generate reproducibility certificate
        cert_path = self.generate_certificate(workflow, workflow_dir)
        
        logger.info(f"Saved workflow to: {workflow_dir}")
        return workflow_dir
    
    def load_workflow(self, workflow_id: str) -> Workflow:
        """Load workflow from disk"""
        workflow_dir = self.storage_dir / workflow_id
        workflow_path = workflow_dir / "workflow.json"
        
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow not found: {workflow_id}")
            
        with open(workflow_path, 'r') as f:
            workflow_dict = json.load(f)
            
        # Convert back to Workflow object
        workflow = self._dict_to_workflow(workflow_dict)
        
        return workflow
    
    def generate_certificate(self, 
                           workflow: Optional[Workflow] = None,
                           output_dir: Optional[Path] = None) -> Path:
        """Generate reproducibility certificate"""
        workflow = workflow or self.current_workflow
        if not workflow:
            raise ValueError("No workflow to certify")
            
        output_dir = output_dir or self.storage_dir / workflow.workflow_id
        output_dir.mkdir(exist_ok=True)
        
        # Generate certificate content
        certificate = {
            'certificate_version': '1.0',
            'generated_at': datetime.datetime.now().isoformat(),
            'workflow': {
                'id': workflow.workflow_id,
                'name': workflow.name,
                'created_at': workflow.created_at,
                'status': workflow.status
            },
            'environment_hash': workflow.environment.compute_hash(),
            'steps': len(workflow.steps),
            'data_hashes': [],
            'method_hashes': [],
            'validation': {}
        }
        
        # Collect all data and method hashes
        for step in workflow.steps:
            if step.data_provenance:
                certificate['data_hashes'].append({
                    'step': step.step_name,
                    'hash': step.output_hash
                })
            if step.method_provenance:
                certificate['method_hashes'].append({
                    'method': step.method_provenance.method_name,
                    'hash': step.method_provenance.method_hash
                })
                
        # Compute overall workflow hash
        workflow_str = json.dumps(certificate, sort_keys=True)
        certificate['workflow_hash'] = hashlib.sha256(workflow_str.encode()).hexdigest()
        
        # Save certificate
        cert_path = output_dir / "reproducibility_certificate.json"
        with open(cert_path, 'w') as f:
            json.dump(certificate, f, indent=2)
            
        # Also generate human-readable report
        report_path = self._generate_report(workflow, certificate, output_dir)
        
        return cert_path
    
    def verify_reproducibility(self, 
                             workflow_id: str,
                             rerun_workflow: Workflow) -> Dict[str, Any]:
        """Verify if a rerun matches the original workflow"""
        # Load original workflow
        original = self.load_workflow(workflow_id)
        
        # Compare environments
        env_match = (original.environment.compute_hash() == 
                    rerun_workflow.environment.compute_hash())
        
        # Compare steps
        step_matches = []
        for orig_step, rerun_step in zip(original.steps, rerun_workflow.steps):
            match = {
                'step_name': orig_step.step_name,
                'input_match': orig_step.input_hashes == rerun_step.input_hashes,
                'output_match': orig_step.output_hash == rerun_step.output_hash,
                'method_match': True  # Check if methods used
            }
            
            if orig_step.method_provenance and rerun_step.method_provenance:
                match['method_match'] = (
                    orig_step.method_provenance.method_hash == 
                    rerun_step.method_provenance.method_hash
                )
                
            step_matches.append(match)
            
        # Overall reproducibility
        fully_reproducible = (
            env_match and 
            all(s['output_match'] for s in step_matches)
        )
        
        return {
            'fully_reproducible': fully_reproducible,
            'environment_match': env_match,
            'step_matches': step_matches,
            'original_workflow': workflow_id,
            'rerun_workflow': rerun_workflow.workflow_id
        }
    
    def _capture_git_info(self) -> Dict[str, Any]:
        """Capture git repository information"""
        try:
            repo = git.Repo(search_parent_directories=True)
            return {
                'commit': repo.head.commit.hexsha,
                'branch': repo.active_branch.name,
                'dirty': repo.is_dirty()
            }
        except Exception:
            return {'commit': None, 'branch': None, 'dirty': False}
    
    def _set_random_seed(self, seed: int):
        """Set random seed for all libraries"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
            
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
    
    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash of data object"""
        if isinstance(data, pd.DataFrame):
            # Use pandas hash
            return hashlib.sha256(
                pd.util.hash_pandas_object(data, index=True).values
            ).hexdigest()
        elif isinstance(data, np.ndarray):
            # Use numpy hash
            return hashlib.sha256(data.tobytes()).hexdigest()
        else:
            # Fallback to pickle
            return hashlib.sha256(pickle.dumps(data)).hexdigest()
    
    def _get_method_source(self, func: callable) -> str:
        """Get source code of a method"""
        try:
            return inspect.getsource(func)
        except Exception:
            # Fallback to bytecode
            return str(dis.dis(func))
    
    def _get_method_version(self, func: callable) -> str:
        """Get version of method"""
        # Check if method has version attribute
        if hasattr(func, '__version__'):
            return func.__version__
            
        # Check module version
        module = inspect.getmodule(func)
        if module and hasattr(module, '__version__'):
            return module.__version__
            
        return "unknown"
    
    def _workflow_to_dict(self, workflow: Workflow) -> Dict[str, Any]:
        """Convert workflow to dictionary for serialization"""
        return {
            'workflow_id': workflow.workflow_id,
            'name': workflow.name,
            'description': workflow.description,
            'created_at': workflow.created_at,
            'created_by': workflow.created_by,
            'project': workflow.project,
            'environment': workflow.environment.to_dict(),
            'git_commit': workflow.git_commit,
            'git_branch': workflow.git_branch,
            'git_dirty': workflow.git_dirty,
            'random_seed': workflow.random_seed,
            'status': workflow.status,
            'steps': [self._step_to_dict(step) for step in workflow.steps]
        }
    
    def _step_to_dict(self, step: WorkflowStep) -> Dict[str, Any]:
        """Convert workflow step to dictionary"""
        return {
            'step_id': step.step_id,
            'step_type': step.step_type,
            'step_name': step.step_name,
            'input_hashes': step.input_hashes,
            'output_hash': step.output_hash,
            'start_time': step.start_time,
            'end_time': step.end_time,
            'duration': step.duration,
            'success': step.success,
            'error_message': step.error_message,
            'data_provenance': asdict(step.data_provenance) if step.data_provenance else None,
            'method_provenance': asdict(step.method_provenance) if step.method_provenance else None,
            'artifacts': step.artifacts
        }
    
    def _dict_to_workflow(self, workflow_dict: Dict[str, Any]) -> Workflow:
        """Convert dictionary back to Workflow object"""
        # Convert environment
        env_dict = workflow_dict['environment']
        environment = SystemEnvironment(**env_dict)
        
        # Create workflow
        workflow = Workflow(
            workflow_id=workflow_dict['workflow_id'],
            name=workflow_dict['name'],
            description=workflow_dict['description'],
            created_at=workflow_dict['created_at'],
            created_by=workflow_dict['created_by'],
            project=workflow_dict.get('project'),
            environment=environment,
            git_commit=workflow_dict.get('git_commit'),
            git_branch=workflow_dict.get('git_branch'),
            git_dirty=workflow_dict.get('git_dirty', False),
            random_seed=workflow_dict.get('random_seed'),
            status=workflow_dict.get('status', 'completed')
        )
        
        # Add steps
        for step_dict in workflow_dict['steps']:
            step = self._dict_to_step(step_dict)
            workflow.add_step(step)
            
        return workflow
    
    def _dict_to_step(self, step_dict: Dict[str, Any]) -> WorkflowStep:
        """Convert dictionary to WorkflowStep"""
        # Convert provenances if present
        data_prov = None
        if step_dict.get('data_provenance'):
            data_prov = DataProvenance(**step_dict['data_provenance'])
            
        method_prov = None
        if step_dict.get('method_provenance'):
            method_prov = MethodProvenance(**step_dict['method_provenance'])
            
        return WorkflowStep(
            step_id=step_dict['step_id'],
            step_type=step_dict['step_type'],
            step_name=step_dict['step_name'],
            input_hashes=step_dict['input_hashes'],
            output_hash=step_dict['output_hash'],
            start_time=step_dict['start_time'],
            end_time=step_dict['end_time'],
            duration=step_dict['duration'],
            success=step_dict['success'],
            error_message=step_dict.get('error_message'),
            data_provenance=data_prov,
            method_provenance=method_prov,
            artifacts=step_dict.get('artifacts', {})
        )
    
    def _save_workflow_artifacts(self, workflow: Workflow, output_dir: Path):
        """Save additional workflow artifacts"""
        # Save environment details
        env_path = output_dir / "environment.yaml"
        with open(env_path, 'w') as f:
            yaml.dump(workflow.environment.to_dict(), f)
            
        # Save pip freeze
        pip_path = output_dir / "requirements.txt"
        with open(pip_path, 'w') as f:
            f.write(workflow.environment.pip_freeze)
            
        # Save method source codes
        methods_dir = output_dir / "methods"
        methods_dir.mkdir(exist_ok=True)
        
        for step in workflow.steps:
            if step.method_provenance:
                method_path = methods_dir / f"{step.method_provenance.method_name}.py"
                # Save cached source if available
                if step.method_provenance.method_name in self._method_cache:
                    with open(method_path, 'w') as f:
                        f.write(self._method_cache[step.method_provenance.method_name])
    
    def _generate_report(self, workflow: Workflow, 
                        certificate: Dict[str, Any],
                        output_dir: Path) -> Path:
        """Generate human-readable reproducibility report"""
        report_path = output_dir / "reproducibility_report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Reproducibility Report\n\n")
            f.write(f"**Workflow:** {workflow.name}\n")
            f.write(f"**ID:** {workflow.workflow_id}\n")
            f.write(f"**Created:** {workflow.created_at}\n")
            f.write(f"**Status:** {workflow.status}\n\n")
            
            f.write(f"## Certificate\n")
            f.write(f"**Workflow Hash:** `{certificate['workflow_hash']}`\n")
            f.write(f"**Environment Hash:** `{certificate['environment_hash']}`\n\n")
            
            f.write(f"## Environment\n")
            f.write(f"- **OS:** {workflow.environment.os_name} {workflow.environment.os_version}\n")
            f.write(f"- **Python:** {workflow.environment.python_version}\n")
            f.write(f"- **CPU:** {workflow.environment.cpu_model} ({workflow.environment.cpu_count} cores)\n")
            f.write(f"- **Memory:** {workflow.environment.total_memory / 1e9:.1f} GB\n")
            
            if workflow.environment.gpu_count > 0:
                f.write(f"- **GPU:** {', '.join(workflow.environment.gpu_models)}\n")
                f.write(f"- **CUDA:** {workflow.environment.cuda_version}\n")
                
            f.write(f"\n## Workflow Steps\n")
            for i, step in enumerate(workflow.steps, 1):
                f.write(f"\n### {i}. {step.step_name}\n")
                f.write(f"- **Type:** {step.step_type}\n")
                f.write(f"- **Duration:** {step.duration:.2f}s\n")
                f.write(f"- **Output Hash:** `{step.output_hash}`\n")
                
                if step.method_provenance:
                    mp = step.method_provenance
                    f.write(f"- **Method:** {mp.method_name} (v{mp.method_version})\n")
                    f.write(f"- **Method Hash:** `{mp.method_hash}`\n")
                    
                    if mp.validation_metrics:
                        f.write(f"- **Validation Metrics:**\n")
                        for metric, value in mp.validation_metrics.items():
                            f.write(f"  - {metric}: {value:.4f}\n")
                            
            f.write(f"\n## Reproduction Instructions\n")
            f.write(f"1. Set up environment matching hash: `{certificate['environment_hash']}`\n")
            f.write(f"2. Install requirements from `requirements.txt`\n")
            f.write(f"3. Checkout git commit: `{workflow.git_commit}`\n")
            f.write(f"4. Run workflow with seed: `{workflow.random_seed}`\n")
            
        return report_path


def track_reproducibility(tracker: Optional[ReproducibilityTracker] = None):
    """Decorator to automatically track method execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create tracker
            nonlocal tracker
            if tracker is None:
                tracker = ReproducibilityTracker()
                
            # Track execution
            import time
            start_time = time.time()
            
            # Extract parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            params = dict(bound_args.arguments)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Track execution time
            execution_time = time.time() - start_time
            
            # Track method execution if data provided
            if 'data' in params and tracker.current_workflow:
                tracker.track_method_execution(
                    method_name=func.__name__,
                    method_func=func,
                    parameters=params,
                    input_data=params['data'],
                    output_data=result,
                    execution_time=execution_time
                )
                
            return result
            
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = ReproducibilityTracker()
    
    # Start workflow
    workflow = tracker.start_workflow(
        name="Air Quality Imputation Analysis",
        description="Testing multiple imputation methods on PM2.5 data",
        project="AirImpute Pro",
        random_seed=42
    )
    
    # Simulate data loading
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='h'),
        'PM25': np.random.normal(25, 10, 1000)
    })
    
    # Add some missing values
    mask = np.random.random(1000) < 0.2
    data.loc[mask, 'PM25'] = np.nan
    
    # Track data loading
    data_prov = tracker.track_data_loading(
        data, 
        source_path="simulated_data.csv",
        name="PM2.5 hourly"
    )
    
    # Simulate method execution
    @track_reproducibility(tracker)
    def impute_linear(data):
        return data.fillna(method='linear')
    
    # Run imputation
    imputed_data = impute_linear(data)
    
    # Complete workflow
    workflow.complete()
    
    # Save workflow
    workflow_dir = tracker.save_workflow()
    print(f"Workflow saved to: {workflow_dir}")
    
    # Generate certificate
    cert_path = tracker.generate_certificate()
    print(f"Certificate generated: {cert_path}")