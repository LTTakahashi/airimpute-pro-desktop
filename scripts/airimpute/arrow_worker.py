#!/usr/bin/env python3
"""
Arrow-based Python worker for high-performance data processing.
This worker runs as a long-lived process and communicates with Rust via Arrow IPC.
"""

import asyncio
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union
import argparse
import signal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from pyarrow.cffi import ffi

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Worker-%(worker_id)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Import our imputation methods
from airimpute.methods.base import ImputationMethod
from airimpute.methods.simple import MeanImputation, MedianImputation, ModeImputation
from airimpute.methods.interpolation import LinearInterpolation, SplineInterpolation
from airimpute.methods.statistical import ForwardFillImputation, BackwardFillImputation
from airimpute.methods.machine_learning import (
    KNNImputation, RandomForestImputation, MICEImputation, XGBoostImputation
)
from airimpute.deep_learning_models import AutoencoderImputation, GAINImputation
from airimpute.spatial_kriging import KrigingImputation
from airimpute.ensemble_methods import EnsembleImputation

# For GPU support
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("PyTorch not available - GPU acceleration disabled")


class TaskStatus(Enum):
    SUCCESS = "Success"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    IN_PROGRESS = "InProgress"


@dataclass
class PythonTask:
    id: str
    action: str
    data: bytes  # Serialized Arrow IPC data
    metadata: Dict[str, Any]


@dataclass
class PythonResponse:
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None


class SecureDispatcher:
    """Secure dispatcher that maps action strings to handler functions."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.handlers = self._build_dispatch_table()
        self.active_tasks = {}
        self.cancellation_tokens = {}
        
    def _build_dispatch_table(self) -> Dict[str, callable]:
        """Build the secure dispatch table with allowed operations only."""
        return {
            # Statistical Methods
            "ImputeMean": self._impute_mean,
            "ImputeMedian": self._impute_median,
            "ImputeMode": self._impute_mode,
            "ImputeForwardFill": self._impute_forward_fill,
            "ImputeBackwardFill": self._impute_backward_fill,
            "ImputeLinearInterpolation": self._impute_linear,
            "ImputeSpline": self._impute_spline,
            
            # Machine Learning Methods
            "ImputeKNN": self._impute_knn,
            "ImputeRandomForest": self._impute_random_forest,
            "ImputeMICE": self._impute_mice,
            "ImputeXGBoost": self._impute_xgboost,
            
            # Deep Learning Methods
            "ImputeAutoencoder": self._impute_autoencoder,
            "ImputeGAIN": self._impute_gain,
            "ImputeTransformer": self._impute_transformer,
            
            # Spatial Methods
            "ImputeKriging": self._impute_kriging,
            "ImputeGNN": self._impute_gnn,
            
            # Ensemble Methods
            "ImputeEnsemble": self._impute_ensemble,
            
            # Control Operations
            "CancelJob": self._cancel_job,
            "CheckHealth": self._check_health,
        }
    
    async def dispatch(self, task: PythonTask) -> PythonResponse:
        """Dispatch a task to the appropriate handler."""
        handler = self.handlers.get(task.action)
        
        if not handler:
            logger.error(f"Invalid action received: {task.action}")
            return PythonResponse(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error={
                    "error_type": "InvalidAction",
                    "message": f"Unknown action: {task.action}",
                    "traceback": None
                }
            )
        
        # Create cancellation token for this task
        self.cancellation_tokens[task.id] = asyncio.Event()
        
        try:
            # Record active task
            self.active_tasks[task.id] = task
            
            # Execute handler
            result = await handler(task)
            
            return PythonResponse(
                task_id=task.id,
                status=TaskStatus.SUCCESS,
                result=result
            )
            
        except asyncio.CancelledError:
            logger.info(f"Task {task.id} was cancelled")
            return PythonResponse(
                task_id=task.id,
                status=TaskStatus.CANCELLED
            )
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {str(e)}")
            return PythonResponse(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error={
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            
        finally:
            # Clean up
            self.active_tasks.pop(task.id, None)
            self.cancellation_tokens.pop(task.id, None)
    
    def _reconstruct_arrow_data(self, serialized_data: bytes) -> pd.DataFrame:
        """Reconstruct Arrow data from serialized IPC format."""
        # Create a stream reader for the serialized data
        reader = ipc.open_stream(pa.BufferReader(serialized_data))
        
        # Read the record batch
        batch = reader.read_next_batch()
        
        # Convert to pandas DataFrame
        return batch.to_pandas()
    
    def _export_arrow_data(self, df: pd.DataFrame) -> bytes:
        """Export DataFrame to Arrow IPC format."""
        # Convert to Arrow RecordBatch
        batch = pa.RecordBatch.from_pandas(df)
        
        # Create a buffer to write to
        sink = pa.BufferOutputStream()
        
        # Write the batch using IPC stream format
        with ipc.new_stream(sink, batch.schema) as writer:
            writer.write_batch(batch)
        
        # Get the serialized data
        return sink.getvalue().to_pybytes()
    
    async def _check_cancellation(self, task_id: str):
        """Check if task has been cancelled."""
        token = self.cancellation_tokens.get(task_id)
        if token and token.is_set():
            raise asyncio.CancelledError()
    
    # Imputation method handlers
    async def _impute_mean(self, task: PythonTask) -> Dict[str, Any]:
        """Handle mean imputation."""
        df = self._reconstruct_arrow_data(task.data)
        
        # Check cancellation
        await self._check_cancellation(task.id)
        
        # Perform imputation
        imputer = MeanImputation()
        result = imputer.impute(df.values)
        
        # Convert back to DataFrame
        result_df = pd.DataFrame(result, columns=df.columns, index=df.index)
        
        # Export result
        serialized_data = self._export_arrow_data(result_df)
        
        return {
            "data": serialized_data,
            "metrics": {
                "imputed_values": int(df.isna().sum().sum()),
                "method": "mean"
            }
        }
    
    async def _impute_knn(self, task: PythonTask) -> Dict[str, Any]:
        """Handle KNN imputation with cancellation support."""
        df = self._reconstruct_arrow_data(task.data)
        k = task.metadata.get("k", 5)
        
        # Initialize imputer
        imputer = KNNImputation(k=k)
        
        # For large datasets, process in chunks with cancellation checks
        chunk_size = 1000
        n_rows = len(df)
        
        if n_rows > chunk_size:
            result_chunks = []
            for i in range(0, n_rows, chunk_size):
                # Check cancellation
                await self._check_cancellation(task.id)
                
                chunk = df.iloc[i:i+chunk_size]
                result_chunk = imputer.impute(chunk.values)
                result_chunks.append(result_chunk)
                
                # Report progress
                progress = min((i + chunk_size) / n_rows, 1.0)
                logger.info(f"KNN imputation progress: {progress:.1%}")
        else:
            result = imputer.impute(df.values)
        
        if n_rows > chunk_size:
            result = np.vstack(result_chunks)
        
        # Convert back to DataFrame
        result_df = pd.DataFrame(result, columns=df.columns, index=df.index)
        
        # Export result
        serialized_data = self._export_arrow_data(result_df)
        
        return {
            "data": serialized_data,
            "metrics": {
                "imputed_values": int(df.isna().sum().sum()),
                "method": "knn",
                "k": k
            }
        }
    
    async def _impute_autoencoder(self, task: PythonTask) -> Dict[str, Any]:
        """Handle Autoencoder imputation with GPU support."""
        df = self._reconstruct_arrow_data(task.data)
        
        hidden_dims = task.metadata.get("hidden_dims", [64, 32, 64])
        epochs = task.metadata.get("epochs", 100)
        use_gpu = task.metadata.get("use_gpu", True) and CUDA_AVAILABLE
        
        # Initialize imputer
        imputer = AutoencoderImputation(
            hidden_dims=hidden_dims,
            epochs=epochs,
            use_gpu=use_gpu
        )
        
        # Training callback for cancellation and progress
        async def training_callback(epoch, loss):
            await self._check_cancellation(task.id)
            if epoch % 10 == 0:
                logger.info(f"Autoencoder training - Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
        
        # Set callback
        imputer.set_callback(training_callback)
        
        # Perform imputation
        result = await imputer.impute_async(df.values)
        
        # Convert back to DataFrame
        result_df = pd.DataFrame(result, columns=df.columns, index=df.index)
        
        # Export result
        serialized_data = self._export_arrow_data(result_df)
        
        return {
            "data": serialized_data,
            "metrics": {
                "imputed_values": int(df.isna().sum().sum()),
                "method": "autoencoder",
                "device": "cuda" if use_gpu else "cpu"
            }
        }
    
    # Add more imputation handlers...
    
    async def _cancel_job(self, task: PythonTask) -> Dict[str, Any]:
        """Handle job cancellation request."""
        job_id = task.metadata.get("job_id")
        
        if job_id in self.cancellation_tokens:
            self.cancellation_tokens[job_id].set()
            logger.info(f"Cancelled job {job_id}")
            return {"cancelled": True}
        
        # Check for shutdown signal
        if task.metadata.get("shutdown") == "true":
            logger.info("Received shutdown signal")
            # Gracefully shutdown
            asyncio.get_event_loop().stop()
            return {"shutdown": True}
        
        return {"cancelled": False}
    
    async def _check_health(self, task: PythonTask) -> Dict[str, Any]:
        """Health check handler."""
        import psutil
        process = psutil.Process(os.getpid())
        
        return {
            "status": "healthy",
            "worker_id": self.worker_id,
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "cuda_available": CUDA_AVAILABLE,
            "active_tasks": len(self.active_tasks)
        }
    
    # Placeholder implementations for other methods
    async def _impute_median(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_mode(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_forward_fill(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_backward_fill(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_linear(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_spline(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_random_forest(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_mice(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_xgboost(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_gain(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_transformer(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_kriging(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_gnn(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder
    
    async def _impute_ensemble(self, task: PythonTask) -> Dict[str, Any]:
        return await self._impute_mean(task)  # Placeholder


class ArrowWorker:
    """Main worker class that manages the event loop and communication."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.dispatcher = SecureDispatcher(worker_id)
        self.running = True
        
    async def run(self):
        """Main worker loop."""
        logger.info(f"Arrow worker {self.worker_id} started")
        
        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.shutdown)
        
        try:
            while self.running:
                # Read task from stdin (JSON format)
                try:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, sys.stdin.readline
                    )
                    
                    if not line:
                        logger.warning("EOF on stdin, shutting down")
                        break
                    
                    # Parse task
                    task_data = json.loads(line.strip())
                    task = PythonTask(**task_data)
                    
                    # Process task
                    response = await self.dispatcher.dispatch(task)
                    
                    # Send response to stdout
                    response_json = json.dumps({
                        "task_id": response.task_id,
                        "status": response.status.value,
                        "result": response.result,
                        "error": response.error
                    })
                    
                    print(response_json, flush=True)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in worker loop: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Worker interrupted")
        finally:
            logger.info(f"Worker {self.worker_id} shutting down")
    
    def shutdown(self):
        """Graceful shutdown handler."""
        logger.info(f"Worker {self.worker_id} received shutdown signal")
        self.running = False
        asyncio.get_event_loop().stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Arrow-based Python worker")
    parser.add_argument("--worker-id", type=int, required=True, help="Worker ID")
    parser.add_argument("--ipc-mode", default="arrow", help="IPC mode (arrow or json)")
    args = parser.parse_args()
    
    # Update logger with worker ID
    logging.getLogger().handlers[0].setFormatter(
        logging.Formatter(
            f'%(asctime)s [Worker-{args.worker_id}] %(levelname)s: %(message)s'
        )
    )
    
    # Create and run worker
    worker = ArrowWorker(args.worker_id)
    
    # Run event loop
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()