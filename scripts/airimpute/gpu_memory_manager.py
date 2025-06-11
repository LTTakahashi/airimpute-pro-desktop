"""
GPU Memory Management System with Automatic Fallback
Implements intelligent GPU resource management for deep learning imputation methods
"""

import torch
import numpy as np
import cupy as cp
import logging
import gc
import os
import psutil
import GPUtil
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass
from contextlib import contextmanager
from functools import wraps
import warnings
import threading
import time
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Available device types"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    ROCm = "rocm"  # AMD GPUs


@dataclass
class GPUInfo:
    """GPU device information"""
    device_id: int
    name: str
    total_memory: int  # bytes
    available_memory: int  # bytes
    compute_capability: Tuple[int, int]
    temperature: float
    utilization: float
    power_draw: float
    power_limit: float


@dataclass
class MemoryStatus:
    """Current memory status"""
    device_type: DeviceType
    device_id: Optional[int]
    total_memory: int
    used_memory: int
    available_memory: int
    reserved_memory: int
    active_memory: int
    cached_memory: int
    
    @property
    def utilization(self) -> float:
        """Memory utilization percentage"""
        return (self.used_memory / self.total_memory * 100) if self.total_memory > 0 else 0


class GPUMemoryManager:
    """
    Comprehensive GPU memory management with automatic fallback
    
    Features:
    - Automatic device detection and selection
    - Memory monitoring and limits
    - Automatic fallback to CPU on OOM
    - Memory pool management
    - Multi-GPU support
    """
    
    def __init__(self, 
                 preferred_device: Optional[str] = None,
                 memory_fraction: float = 0.9,
                 enable_growth: bool = True,
                 fallback_to_cpu: bool = True,
                 monitor_interval: float = 1.0):
        """
        Initialize GPU memory manager
        
        Parameters:
        -----------
        preferred_device : str, optional
            Preferred device ('cuda', 'mps', 'cpu')
        memory_fraction : float
            Maximum fraction of GPU memory to use
        enable_growth : bool
            Allow dynamic memory growth
        fallback_to_cpu : bool
            Automatically fallback to CPU on GPU errors
        monitor_interval : float
            Interval for memory monitoring (seconds)
        """
        self.preferred_device = preferred_device
        self.memory_fraction = memory_fraction
        self.enable_growth = enable_growth
        self.fallback_to_cpu = fallback_to_cpu
        self.monitor_interval = monitor_interval
        
        # Device management
        self.available_devices = self._detect_devices()
        self.current_device = self._select_device()
        
        # Memory tracking
        self.memory_snapshots = []
        self.oom_count = 0
        self.fallback_count = 0
        
        # Monitoring thread
        self._monitor_thread = None
        self._monitoring = False
        
        # Initialize device
        self._initialize_device()
        
        logger.info(f"GPU Memory Manager initialized with device: {self.current_device}")
    
    def _detect_devices(self) -> Dict[DeviceType, List[int]]:
        """Detect available compute devices"""
        devices = {
            DeviceType.CPU: [0]  # CPU always available
        }
        
        # CUDA devices
        if torch.cuda.is_available():
            devices[DeviceType.CUDA] = list(range(torch.cuda.device_count()))
            
        # Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices[DeviceType.MPS] = [0]
            
        # AMD ROCm (if supported)
        # This would need additional detection logic
        
        return devices
    
    def _select_device(self) -> Tuple[DeviceType, Optional[int]]:
        """Select best available device"""
        if self.preferred_device:
            # Try preferred device
            device_type = DeviceType(self.preferred_device.lower())
            if device_type in self.available_devices:
                # Select device with most available memory
                if device_type == DeviceType.CUDA:
                    device_id = self._select_best_gpu()
                    return (device_type, device_id)
                else:
                    return (device_type, 0)
        
        # Auto-select best device
        if DeviceType.CUDA in self.available_devices:
            device_id = self._select_best_gpu()
            return (DeviceType.CUDA, device_id)
        elif DeviceType.MPS in self.available_devices:
            return (DeviceType.MPS, 0)
        else:
            return (DeviceType.CPU, 0)
    
    def _select_best_gpu(self) -> int:
        """Select GPU with most available memory"""
        if not torch.cuda.is_available():
            return 0
            
        best_device = 0
        max_memory = 0
        
        for device_id in self.available_devices[DeviceType.CUDA]:
            torch.cuda.set_device(device_id)
            free_memory = torch.cuda.mem_get_info(device_id)[0]
            
            if free_memory > max_memory:
                max_memory = free_memory
                best_device = device_id
        
        return best_device
    
    def _initialize_device(self):
        """Initialize selected device with memory settings"""
        device_type, device_id = self.current_device
        
        if device_type == DeviceType.CUDA:
            torch.cuda.set_device(device_id)
            
            # Set memory fraction
            if self.memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.memory_fraction, device_id
                )
            
            # Enable memory growth for CuPy
            if self.enable_growth:
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(fraction=self.memory_fraction)
            
            # Clear any existing allocations
            self._clear_gpu_memory()
    
    def get_device_info(self) -> Optional[GPUInfo]:
        """Get current GPU device information"""
        device_type, device_id = self.current_device
        
        if device_type != DeviceType.CUDA:
            return None
            
        try:
            gpus = GPUtil.getGPUs()
            if device_id < len(gpus):
                gpu = gpus[device_id]
                
                # Get compute capability
                major, minor = torch.cuda.get_device_capability(device_id)
                
                return GPUInfo(
                    device_id=device_id,
                    name=torch.cuda.get_device_name(device_id),
                    total_memory=gpu.memoryTotal * 1024 * 1024,  # Convert to bytes
                    available_memory=gpu.memoryFree * 1024 * 1024,
                    compute_capability=(major, minor),
                    temperature=gpu.temperature,
                    utilization=gpu.load * 100,
                    power_draw=gpu.powerDraw if hasattr(gpu, 'powerDraw') else 0,
                    power_limit=gpu.powerLimit if hasattr(gpu, 'powerLimit') else 0
                )
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return None
    
    def get_memory_status(self) -> MemoryStatus:
        """Get current memory status"""
        device_type, device_id = self.current_device
        
        if device_type == DeviceType.CUDA:
            torch.cuda.synchronize()
            
            # Get PyTorch memory stats
            stats = torch.cuda.memory_stats(device_id)
            
            return MemoryStatus(
                device_type=device_type,
                device_id=device_id,
                total_memory=torch.cuda.get_device_properties(device_id).total_memory,
                used_memory=torch.cuda.memory_allocated(device_id),
                available_memory=torch.cuda.mem_get_info(device_id)[0],
                reserved_memory=torch.cuda.memory_reserved(device_id),
                active_memory=stats.get("active_bytes.all.current", 0),
                cached_memory=torch.cuda.memory_cached(device_id)
            )
        else:
            # CPU memory status
            mem = psutil.virtual_memory()
            
            return MemoryStatus(
                device_type=device_type,
                device_id=None,
                total_memory=mem.total,
                used_memory=mem.used,
                available_memory=mem.available,
                reserved_memory=0,
                active_memory=mem.used,
                cached_memory=0
            )
    
    @contextmanager
    def memory_context(self, required_memory: Optional[int] = None,
                      operation_name: str = "operation"):
        """
        Context manager for GPU memory operations
        
        Parameters:
        -----------
        required_memory : int, optional
            Required memory in bytes
        operation_name : str
            Name of operation for logging
        """
        # Check if we have enough memory
        if required_memory:
            if not self._check_memory_available(required_memory):
                if self.fallback_to_cpu:
                    logger.warning(
                        f"Insufficient GPU memory for {operation_name}. "
                        f"Required: {required_memory / 1e9:.2f}GB. "
                        "Falling back to CPU."
                    )
                    self._switch_to_cpu()
                else:
                    raise torch.cuda.OutOfMemoryError(
                        f"Insufficient GPU memory for {operation_name}"
                    )
        
        # Take memory snapshot before
        before_status = self.get_memory_status()
        
        try:
            yield self
            
        except (torch.cuda.OutOfMemoryError, cp.cuda.MemoryError) as e:
            logger.error(f"GPU OOM during {operation_name}: {e}")
            self.oom_count += 1
            
            if self.fallback_to_cpu:
                logger.info("Attempting CPU fallback...")
                self._switch_to_cpu()
                self.fallback_count += 1
                yield self
            else:
                raise
                
        finally:
            # Take memory snapshot after
            after_status = self.get_memory_status()
            
            # Log memory usage
            memory_delta = after_status.used_memory - before_status.used_memory
            logger.debug(
                f"{operation_name} memory delta: {memory_delta / 1e6:.2f}MB"
            )
            
            # Store snapshot
            self.memory_snapshots.append({
                'operation': operation_name,
                'before': before_status,
                'after': after_status,
                'delta': memory_delta,
                'timestamp': time.time()
            })
            
            # Clean up if needed
            if after_status.utilization > 90:
                self._clear_gpu_memory()
    
    def _check_memory_available(self, required_bytes: int) -> bool:
        """Check if enough memory is available"""
        status = self.get_memory_status()
        
        # Add safety margin
        required_with_margin = required_bytes * 1.2
        
        return status.available_memory >= required_with_margin
    
    def _switch_to_cpu(self):
        """Switch to CPU device"""
        logger.info("Switching to CPU device")
        self.current_device = (DeviceType.CPU, 0)
        
        # Clear GPU memory before switching
        if DeviceType.CUDA in self.available_devices:
            self._clear_gpu_memory()
    
    def _clear_gpu_memory(self):
        """Clear GPU memory caches"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        if cp.cuda.runtime.getDevice() >= 0:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
        gc.collect()
    
    def allocate_tensor(self, shape: Tuple[int, ...], 
                       dtype: torch.dtype = torch.float32,
                       requires_grad: bool = False) -> torch.Tensor:
        """
        Allocate tensor with automatic device selection
        
        Parameters:
        -----------
        shape : tuple
            Tensor shape
        dtype : torch.dtype
            Data type
        requires_grad : bool
            Whether tensor requires gradients
        
        Returns:
        --------
        torch.Tensor
            Allocated tensor
        """
        # Calculate required memory
        element_size = torch.tensor([], dtype=dtype).element_size()
        required_memory = np.prod(shape) * element_size
        
        device_type, device_id = self.current_device
        
        with self.memory_context(required_memory, "tensor_allocation"):
            if device_type == DeviceType.CUDA:
                device = f"cuda:{device_id}"
            elif device_type == DeviceType.MPS:
                device = "mps"
            else:
                device = "cpu"
                
            try:
                tensor = torch.zeros(shape, dtype=dtype, device=device, 
                                   requires_grad=requires_grad)
                return tensor
            except Exception as e:
                if self.fallback_to_cpu and device != "cpu":
                    logger.warning(f"Failed to allocate on {device}, using CPU: {e}")
                    return torch.zeros(shape, dtype=dtype, device="cpu",
                                     requires_grad=requires_grad)
                raise
    
    def to_device(self, tensor: Union[torch.Tensor, np.ndarray],
                 non_blocking: bool = False) -> torch.Tensor:
        """
        Move tensor to current device
        
        Parameters:
        -----------
        tensor : torch.Tensor or np.ndarray
            Input tensor
        non_blocking : bool
            Use non-blocking transfer
        
        Returns:
        --------
        torch.Tensor
            Tensor on current device
        """
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
            
        device_type, device_id = self.current_device
        
        if device_type == DeviceType.CUDA:
            device = f"cuda:{device_id}"
        elif device_type == DeviceType.MPS:
            device = "mps"
        else:
            device = "cpu"
            
        try:
            return tensor.to(device, non_blocking=non_blocking)
        except Exception as e:
            if self.fallback_to_cpu and device != "cpu":
                logger.warning(f"Failed to move to {device}, using CPU: {e}")
                return tensor.to("cpu")
            raise
    
    def optimize_batch_size(self, model: torch.nn.Module,
                          input_shape: Tuple[int, ...],
                          initial_batch_size: int = 32,
                          max_batch_size: int = 1024) -> int:
        """
        Find optimal batch size for model
        
        Parameters:
        -----------
        model : torch.nn.Module
            Model to optimize for
        input_shape : tuple
            Shape of single input (without batch dimension)
        initial_batch_size : int
            Starting batch size
        max_batch_size : int
            Maximum batch size to try
        
        Returns:
        --------
        int
            Optimal batch size
        """
        device_type, device_id = self.current_device
        
        if device_type == DeviceType.CPU:
            # For CPU, use conservative batch size
            return min(initial_batch_size, 32)
            
        model = model.to(self.get_torch_device())
        model.eval()
        
        batch_size = initial_batch_size
        optimal_batch_size = 1
        
        while batch_size <= max_batch_size:
            try:
                # Clear memory
                self._clear_gpu_memory()
                
                # Try forward pass
                dummy_input = torch.randn(batch_size, *input_shape, 
                                         device=self.get_torch_device())
                
                with torch.no_grad():
                    _ = model(dummy_input)
                    
                # If successful, update optimal
                optimal_batch_size = batch_size
                batch_size *= 2
                
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                logger.debug(f"Batch size {batch_size} too large: {e}")
                break
                
            finally:
                # Clean up
                if 'dummy_input' in locals():
                    del dummy_input
                self._clear_gpu_memory()
        
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def get_torch_device(self) -> torch.device:
        """Get PyTorch device object"""
        device_type, device_id = self.current_device
        
        if device_type == DeviceType.CUDA:
            return torch.device(f"cuda:{device_id}")
        elif device_type == DeviceType.MPS:
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def start_monitoring(self):
        """Start memory monitoring thread"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_memory,
                daemon=True
            )
            self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring thread"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_memory(self):
        """Monitor memory usage in background"""
        while self._monitoring:
            try:
                status = self.get_memory_status()
                
                # Log if utilization is high
                if status.utilization > 85:
                    logger.warning(
                        f"High memory utilization: {status.utilization:.1f}% "
                        f"({status.used_memory / 1e9:.2f}GB / "
                        f"{status.total_memory / 1e9:.2f}GB)"
                    )
                
                # Auto-cleanup if critical
                if status.utilization > 95:
                    logger.warning("Critical memory usage, clearing caches")
                    self._clear_gpu_memory()
                    
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                
            time.sleep(self.monitor_interval)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        current_status = self.get_memory_status()
        
        summary = {
            'current_device': {
                'type': self.current_device[0].value,
                'id': self.current_device[1]
            },
            'memory_status': {
                'total_gb': current_status.total_memory / 1e9,
                'used_gb': current_status.used_memory / 1e9,
                'available_gb': current_status.available_memory / 1e9,
                'utilization_pct': current_status.utilization
            },
            'statistics': {
                'oom_count': self.oom_count,
                'fallback_count': self.fallback_count,
                'total_operations': len(self.memory_snapshots)
            }
        }
        
        # Add GPU-specific info if available
        gpu_info = self.get_device_info()
        if gpu_info:
            summary['gpu_info'] = {
                'name': gpu_info.name,
                'temperature': gpu_info.temperature,
                'power_draw': gpu_info.power_draw,
                'compute_capability': gpu_info.compute_capability
            }
        
        return summary
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_monitoring()
        self._clear_gpu_memory()


def gpu_accelerated(fallback_to_cpu: bool = True):
    """
    Decorator for GPU-accelerated functions with automatic fallback
    
    Parameters:
    -----------
    fallback_to_cpu : bool
        Whether to fallback to CPU on GPU errors
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create GPU manager
            if not hasattr(wrapper, '_gpu_manager'):
                wrapper._gpu_manager = GPUMemoryManager(
                    fallback_to_cpu=fallback_to_cpu
                )
            
            manager = wrapper._gpu_manager
            
            # Run function with memory context
            with manager.memory_context(operation_name=func.__name__):
                # Inject device into kwargs if function accepts it
                import inspect
                sig = inspect.signature(func)
                if 'device' in sig.parameters:
                    kwargs['device'] = manager.get_torch_device()
                    
                return func(*args, **kwargs)
                
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Initialize GPU manager
    gpu_manager = GPUMemoryManager(
        preferred_device="cuda",
        memory_fraction=0.8,
        fallback_to_cpu=True
    )
    
    # Start monitoring
    gpu_manager.start_monitoring()
    
    # Get device info
    print("Device Info:")
    print(gpu_manager.get_device_info())
    
    # Get memory status
    print("\nMemory Status:")
    print(gpu_manager.get_memory_status())
    
    # Allocate tensor
    with gpu_manager.memory_context(required_memory=1e9, operation_name="test"):
        tensor = gpu_manager.allocate_tensor((1000, 1000), dtype=torch.float32)
        print(f"\nAllocated tensor on: {tensor.device}")
    
    # Test batch size optimization
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    
    optimal_batch = gpu_manager.optimize_batch_size(
        model, input_shape=(100,)
    )
    print(f"\nOptimal batch size: {optimal_batch}")
    
    # Get summary
    print("\nMemory Summary:")
    print(gpu_manager.get_memory_summary())
    
    # Stop monitoring
    gpu_manager.stop_monitoring()