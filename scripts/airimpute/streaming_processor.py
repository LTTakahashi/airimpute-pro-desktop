"""
Streaming Data Processor for Large Datasets
Implements memory-efficient processing for datasets that don't fit in memory
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
import zarr
import h5py
from typing import Iterator, Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
import tempfile
import shutil
import logging
import psutil
import gc
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import threading
import time
from contextlib import contextmanager
import mmap
import struct
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather
from numba import jit, prange
import warnings

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for streaming processor"""
    # Chunk settings
    chunk_size: int = 10000  # Rows per chunk
    chunk_overlap: int = 100  # Overlap for temporal methods
    
    # Memory settings
    memory_limit: int = 2 * 1024**3  # 2GB default
    memory_fraction: float = 0.8  # Use 80% of available memory
    
    # Processing settings
    n_workers: int = mp.cpu_count()
    use_gpu: bool = False
    prefetch_chunks: int = 2
    
    # Storage settings
    temp_dir: Optional[Path] = None
    compression: str = "lz4"  # Fast compression
    storage_format: str = "parquet"  # "parquet", "feather", "zarr", "hdf5"
    
    # Progress tracking
    enable_progress: bool = True
    progress_callback: Optional[Callable] = None


class ChunkIterator:
    """Iterator for data chunks with overlap and prefetching"""
    
    def __init__(self, 
                 data_source: Union[str, Path, pd.DataFrame],
                 chunk_size: int,
                 overlap: int = 0,
                 columns: Optional[List[str]] = None):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.columns = columns
        
        # Detect data format
        self._detect_format()
        
        # Initialize reader
        self._init_reader()
        
        # State
        self.current_position = 0
        self.previous_tail = None
    
    def _detect_format(self):
        """Detect data source format"""
        if isinstance(self.data_source, pd.DataFrame):
            self.format = "dataframe"
            self.total_rows = len(self.data_source)
        elif isinstance(self.data_source, (str, Path)):
            path = Path(self.data_source)
            
            if path.suffix == ".parquet":
                self.format = "parquet"
                # Get row count without loading data
                pf = pq.ParquetFile(path)
                self.total_rows = pf.metadata.num_rows
            elif path.suffix == ".feather":
                self.format = "feather"
                # Feather doesn't support easy row counting
                self.total_rows = None
            elif path.suffix == ".csv":
                self.format = "csv"
                # Count rows efficiently
                self.total_rows = sum(1 for _ in open(path)) - 1  # Subtract header
            elif path.suffix == ".h5" or path.suffix == ".hdf5":
                self.format = "hdf5"
                with h5py.File(path, 'r') as f:
                    # Assume first dataset
                    key = list(f.keys())[0]
                    self.total_rows = f[key].shape[0]
            else:
                raise ValueError(f"Unknown format: {path.suffix}")
        else:
            raise ValueError(f"Unknown data source type: {type(self.data_source)}")
    
    def _init_reader(self):
        """Initialize appropriate reader"""
        if self.format == "dataframe":
            self.reader = None  # Direct access
        elif self.format == "parquet":
            self.reader = pq.ParquetFile(self.data_source)
        elif self.format == "csv":
            # Use chunked CSV reader
            self.reader = pd.read_csv(
                self.data_source,
                chunksize=self.chunk_size,
                usecols=self.columns
            )
        elif self.format == "hdf5":
            self.reader = h5py.File(self.data_source, 'r')
            # Get dataset key
            self.dataset_key = list(self.reader.keys())[0]
    
    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over chunks"""
        self.current_position = 0
        self.previous_tail = None
        
        if self.format == "dataframe":
            # Iterate over DataFrame chunks
            for start in range(0, self.total_rows, self.chunk_size - self.overlap):
                end = min(start + self.chunk_size, self.total_rows)
                
                chunk = self.data_source.iloc[start:end]
                
                if self.columns:
                    chunk = chunk[self.columns]
                
                # Add overlap from previous chunk
                if self.previous_tail is not None and self.overlap > 0:
                    chunk = pd.concat([self.previous_tail, chunk])
                
                # Store tail for next iteration
                if end < self.total_rows and self.overlap > 0:
                    self.previous_tail = chunk.tail(self.overlap)
                
                yield chunk
        
        elif self.format == "parquet":
            # Read Parquet in chunks
            for batch in self.reader.iter_batches(
                batch_size=self.chunk_size,
                columns=self.columns
            ):
                chunk = batch.to_pandas()
                
                # Handle overlap
                if self.previous_tail is not None and self.overlap > 0:
                    chunk = pd.concat([self.previous_tail, chunk])
                
                if self.overlap > 0:
                    self.previous_tail = chunk.tail(self.overlap)
                
                yield chunk
        
        elif self.format == "csv":
            # CSV reader already chunks
            for chunk in self.reader:
                # Handle overlap
                if self.previous_tail is not None and self.overlap > 0:
                    chunk = pd.concat([self.previous_tail, chunk])
                
                if self.overlap > 0:
                    self.previous_tail = chunk.tail(self.overlap)
                
                yield chunk
        
        elif self.format == "hdf5":
            # Read HDF5 in chunks
            dataset = self.reader[self.dataset_key]
            
            for start in range(0, dataset.shape[0], self.chunk_size - self.overlap):
                end = min(start + self.chunk_size, dataset.shape[0])
                
                # Read chunk
                if self.columns:
                    # Assume columns are stored as separate datasets
                    chunk_data = {}
                    for col in self.columns:
                        if col in self.reader:
                            chunk_data[col] = self.reader[col][start:end]
                    chunk = pd.DataFrame(chunk_data)
                else:
                    chunk = pd.DataFrame(dataset[start:end])
                
                # Handle overlap
                if self.previous_tail is not None and self.overlap > 0:
                    chunk = pd.concat([self.previous_tail, chunk])
                
                if end < dataset.shape[0] and self.overlap > 0:
                    self.previous_tail = chunk.tail(self.overlap)
                
                yield chunk
    
    def __len__(self) -> int:
        """Get total number of chunks"""
        if self.total_rows:
            return int(np.ceil(self.total_rows / (self.chunk_size - self.overlap)))
        else:
            return -1  # Unknown
    
    def close(self):
        """Close any open files"""
        if hasattr(self, 'reader'):
            if self.format == "hdf5":
                self.reader.close()


class StreamingImputer:
    """Base class for streaming imputation methods"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self._setup_temp_storage()
    
    def _setup_temp_storage(self):
        """Setup temporary storage for intermediate results"""
        if self.config.temp_dir:
            self.temp_dir = Path(self.config.temp_dir)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="airimpute_stream_"))
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using temp directory: {self.temp_dir}")
    
    def fit(self, data_iterator: ChunkIterator) -> None:
        """Fit the imputer on streaming data"""
        raise NotImplementedError
    
    def transform(self, data_iterator: ChunkIterator) -> Iterator[pd.DataFrame]:
        """Transform streaming data"""
        raise NotImplementedError
    
    def fit_transform(self, data_iterator: ChunkIterator) -> Iterator[pd.DataFrame]:
        """Fit and transform in one pass"""
        self.fit(data_iterator)
        return self.transform(data_iterator)
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class StreamingMeanImputer(StreamingImputer):
    """Streaming mean imputation with running statistics"""
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self.statistics = {}
    
    def fit(self, data_iterator: ChunkIterator) -> None:
        """Calculate running mean statistics"""
        logger.info("Fitting streaming mean imputer")
        
        # Initialize running statistics
        running_sum = {}
        running_count = {}
        running_sum_sq = {}  # For variance calculation
        
        # First pass: calculate statistics
        for i, chunk in enumerate(data_iterator):
            for col in chunk.select_dtypes(include=[np.number]).columns:
                if col not in running_sum:
                    running_sum[col] = 0.0
                    running_count[col] = 0
                    running_sum_sq[col] = 0.0
                
                # Update running statistics
                valid_data = chunk[col].dropna()
                running_sum[col] += valid_data.sum()
                running_count[col] += len(valid_data)
                running_sum_sq[col] += (valid_data ** 2).sum()
            
            if i % 10 == 0:
                logger.debug(f"Processed {i} chunks")
        
        # Calculate final statistics
        for col in running_sum:
            if running_count[col] > 0:
                mean = running_sum[col] / running_count[col]
                variance = (running_sum_sq[col] / running_count[col]) - mean**2
                
                self.statistics[col] = {
                    'mean': mean,
                    'std': np.sqrt(max(0, variance)),  # Avoid negative variance
                    'count': running_count[col]
                }
            else:
                self.statistics[col] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'count': 0
                }
        
        logger.info(f"Computed statistics for {len(self.statistics)} columns")
    
    def transform(self, data_iterator: ChunkIterator) -> Iterator[pd.DataFrame]:
        """Apply mean imputation to streaming data"""
        logger.info("Transforming with streaming mean imputer")
        
        for i, chunk in enumerate(data_iterator):
            # Impute each column
            imputed_chunk = chunk.copy()
            
            for col in chunk.columns:
                if col in self.statistics and self.statistics[col]['count'] > 0:
                    imputed_chunk[col].fillna(
                        self.statistics[col]['mean'], 
                        inplace=True
                    )
            
            yield imputed_chunk


class StreamingKNNImputer(StreamingImputer):
    """Streaming KNN imputation using locality-sensitive hashing"""
    
    def __init__(self, config: StreamConfig, n_neighbors: int = 5):
        super().__init__(config)
        self.n_neighbors = n_neighbors
        self.lsh_index = None
        self.reference_points = []
    
    def fit(self, data_iterator: ChunkIterator) -> None:
        """Build LSH index for efficient neighbor search"""
        logger.info("Fitting streaming KNN imputer")
        
        # Sample reference points for LSH
        sample_size = min(10000, self.config.memory_limit // 1000)
        samples = []
        
        for i, chunk in enumerate(data_iterator):
            # Sample from chunk
            n_sample = min(len(chunk), sample_size // 10)
            if n_sample > 0:
                sample = chunk.dropna().sample(n=n_sample, replace=False)
                samples.append(sample)
            
            if len(pd.concat(samples)) >= sample_size:
                break
        
        # Build reference dataset
        self.reference_data = pd.concat(samples).reset_index(drop=True)
        self.numeric_cols = self.reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        # Normalize reference data
        self.means = self.reference_data[self.numeric_cols].mean()
        self.stds = self.reference_data[self.numeric_cols].std()
        
        normalized_ref = (self.reference_data[self.numeric_cols] - self.means) / self.stds
        
        # Build simple spatial index (for demonstration)
        # In production, use proper LSH library like datasketch
        self.reference_points = normalized_ref.values
        
        logger.info(f"Built index with {len(self.reference_points)} reference points")
    
    def transform(self, data_iterator: ChunkIterator) -> Iterator[pd.DataFrame]:
        """Apply KNN imputation to streaming data"""
        logger.info("Transforming with streaming KNN imputer")
        
        for chunk in data_iterator:
            imputed_chunk = chunk.copy()
            
            # Find rows with missing values
            missing_mask = chunk[self.numeric_cols].isnull().any(axis=1)
            missing_indices = chunk.index[missing_mask]
            
            if len(missing_indices) > 0:
                # Normalize chunk data
                normalized_chunk = (chunk[self.numeric_cols] - self.means) / self.stds
                
                # Impute each missing row
                for idx in missing_indices:
                    row = normalized_chunk.loc[idx]
                    missing_cols = row[row.isnull()].index
                    
                    if len(missing_cols) > 0:
                        # Find neighbors using available features
                        available_cols = row.dropna().index
                        
                        if len(available_cols) > 0:
                            # Simple distance calculation
                            distances = np.sum(
                                (self.reference_points[:, [self.numeric_cols.index(c) 
                                                          for c in available_cols]] - 
                                 row[available_cols].values)**2,
                                axis=1
                            )
                            
                            # Get k nearest neighbors
                            neighbor_indices = np.argpartition(
                                distances, self.n_neighbors
                            )[:self.n_neighbors]
                            
                            # Impute missing values
                            for col in missing_cols:
                                col_idx = self.numeric_cols.index(col)
                                neighbor_values = self.reference_points[
                                    neighbor_indices, col_idx
                                ]
                                
                                # Denormalize
                                imputed_value = (
                                    np.mean(neighbor_values) * self.stds[col] + 
                                    self.means[col]
                                )
                                
                                imputed_chunk.loc[idx, col] = imputed_value
            
            yield imputed_chunk


class StreamingTimeSeriesImputer(StreamingImputer):
    """Streaming time series imputation with state preservation"""
    
    def __init__(self, config: StreamConfig, method: str = "linear"):
        super().__init__(config)
        self.method = method
        self.state = {}
    
    def fit(self, data_iterator: ChunkIterator) -> None:
        """No fitting required for time series methods"""
        pass
    
    def transform(self, data_iterator: ChunkIterator) -> Iterator[pd.DataFrame]:
        """Apply time series imputation with state preservation"""
        logger.info(f"Transforming with streaming {self.method} imputer")
        
        previous_chunk_tail = None
        
        for i, chunk in enumerate(data_iterator):
            # Combine with previous tail for continuity
            if previous_chunk_tail is not None:
                combined = pd.concat([previous_chunk_tail, chunk])
            else:
                combined = chunk
            
            # Apply imputation
            if self.method == "linear":
                imputed = combined.interpolate(method='linear', limit_direction='both')
            elif self.method == "forward_fill":
                imputed = combined.fillna(method='ffill').fillna(method='bfill')
            elif self.method == "seasonal":
                # Simple seasonal decomposition
                imputed = self._seasonal_impute(combined)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Extract the current chunk portion
            if previous_chunk_tail is not None:
                # Remove the overlap portion
                imputed_chunk = imputed.iloc[len(previous_chunk_tail):]
            else:
                imputed_chunk = imputed
            
            # Store tail for next iteration
            previous_chunk_tail = chunk.tail(self.config.chunk_overlap)
            
            yield imputed_chunk.head(len(chunk) - self.config.chunk_overlap)
    
    def _seasonal_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simple seasonal imputation"""
        imputed = data.copy()
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].isnull().any():
                # Calculate seasonal pattern (daily for hourly data)
                season_length = 24
                
                # Fill using seasonal average
                for i in range(season_length):
                    seasonal_mean = data[col].iloc[i::season_length].mean()
                    mask = data[col].isnull() & (data.index % season_length == i)
                    imputed.loc[mask, col] = seasonal_mean
        
        return imputed


class StreamingDeepImputer(StreamingImputer):
    """Streaming deep learning imputation with mini-batch processing"""
    
    def __init__(self, config: StreamConfig, model_type: str = "lstm"):
        super().__init__(config)
        self.model_type = model_type
        self.model = None
        self.scaler_stats = {}
        
        # Try to import deep learning libraries
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
            self.device = torch.device(
                "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
            )
        except ImportError:
            logger.warning("PyTorch not available, using numpy fallback")
            self.torch = None
    
    def fit(self, data_iterator: ChunkIterator) -> None:
        """Train model on streaming data"""
        if self.torch is None:
            logger.warning("Skipping deep learning imputer - PyTorch not available")
            return
        
        logger.info("Fitting streaming deep imputer")
        
        # First pass: calculate normalization statistics
        self._calculate_statistics(data_iterator)
        
        # Build model
        self._build_model()
        
        # Second pass: train model
        self._train_model(data_iterator)
    
    def _calculate_statistics(self, data_iterator: ChunkIterator):
        """Calculate normalization statistics"""
        running_sum = {}
        running_sum_sq = {}
        running_count = {}
        
        for chunk in data_iterator:
            for col in chunk.select_dtypes(include=[np.number]).columns:
                if col not in running_sum:
                    running_sum[col] = 0.0
                    running_sum_sq[col] = 0.0
                    running_count[col] = 0
                
                valid_data = chunk[col].dropna()
                running_sum[col] += valid_data.sum()
                running_sum_sq[col] += (valid_data ** 2).sum()
                running_count[col] += len(valid_data)
        
        # Calculate mean and std
        for col in running_sum:
            if running_count[col] > 0:
                mean = running_sum[col] / running_count[col]
                variance = (running_sum_sq[col] / running_count[col]) - mean**2
                
                self.scaler_stats[col] = {
                    'mean': mean,
                    'std': np.sqrt(max(0, variance))
                }
    
    def _build_model(self):
        """Build neural network model"""
        if self.model_type == "lstm":
            class LSTMImputer(self.nn.Module):
                def __init__(self, input_size, hidden_size=64):
                    super().__init__()
                    self.lstm = self.nn.LSTM(
                        input_size, hidden_size, 
                        num_layers=2, batch_first=True,
                        dropout=0.2
                    )
                    self.output = self.nn.Linear(hidden_size, input_size)
                
                def forward(self, x, mask):
                    # x: [batch, seq, features]
                    # mask: [batch, seq, features] - 1 for missing
                    lstm_out, _ = self.lstm(x)
                    output = self.output(lstm_out)
                    
                    # Only predict missing values
                    return output * mask
            
            input_size = len(self.scaler_stats)
            self.model = LSTMImputer(input_size).to(self.device)
            self.optimizer = self.torch.optim.Adam(self.model.parameters())
    
    def _train_model(self, data_iterator: ChunkIterator):
        """Train model on streaming data"""
        if self.model is None:
            return
        
        self.model.train()
        sequence_length = 24  # Hours
        
        for epoch in range(3):  # Quick training
            epoch_loss = 0
            n_batches = 0
            
            for chunk in data_iterator:
                # Prepare sequences
                numeric_data = chunk.select_dtypes(include=[np.number])
                
                # Normalize
                normalized = numeric_data.copy()
                for col in numeric_data.columns:
                    if col in self.scaler_stats:
                        normalized[col] = (
                            (numeric_data[col] - self.scaler_stats[col]['mean']) / 
                            (self.scaler_stats[col]['std'] + 1e-8)
                        )
                
                # Create sequences
                for i in range(len(normalized) - sequence_length):
                    seq = normalized.iloc[i:i+sequence_length].values
                    
                    # Skip if no missing values
                    if not np.isnan(seq).any():
                        continue
                    
                    # Prepare input
                    mask = np.isnan(seq).astype(np.float32)
                    seq_filled = np.nan_to_num(seq, 0)
                    
                    # Convert to tensors
                    seq_tensor = self.torch.FloatTensor(seq_filled).unsqueeze(0).to(self.device)
                    mask_tensor = self.torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                    target = self.torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    output = self.model(seq_tensor, mask_tensor)
                    
                    # Loss only on missing values
                    loss = self.nn.functional.mse_loss(
                        output[mask_tensor == 1],
                        target[mask_tensor == 1]
                    )
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
            
            if n_batches > 0:
                logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss / n_batches:.4f}")
    
    def transform(self, data_iterator: ChunkIterator) -> Iterator[pd.DataFrame]:
        """Apply deep imputation to streaming data"""
        if self.model is None:
            # Fallback to mean imputation
            for chunk in data_iterator:
                imputed = chunk.copy()
                for col in chunk.select_dtypes(include=[np.number]).columns:
                    if col in self.scaler_stats:
                        imputed[col].fillna(self.scaler_stats[col]['mean'], inplace=True)
                yield imputed
            return
        
        self.model.eval()
        sequence_length = 24
        
        with self.torch.no_grad():
            for chunk in data_iterator:
                imputed = chunk.copy()
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                
                # Process sequences
                for i in range(0, len(chunk) - sequence_length + 1):
                    seq = chunk[numeric_cols].iloc[i:i+sequence_length]
                    
                    if seq.isnull().any().any():
                        # Normalize
                        normalized_seq = seq.copy()
                        for col in numeric_cols:
                            if col in self.scaler_stats:
                                normalized_seq[col] = (
                                    (seq[col] - self.scaler_stats[col]['mean']) / 
                                    (self.scaler_stats[col]['std'] + 1e-8)
                                )
                        
                        # Prepare input
                        seq_values = normalized_seq.values
                        mask = np.isnan(seq_values).astype(np.float32)
                        seq_filled = np.nan_to_num(seq_values, 0)
                        
                        # Predict
                        seq_tensor = self.torch.FloatTensor(seq_filled).unsqueeze(0).to(self.device)
                        mask_tensor = self.torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                        
                        output = self.model(seq_tensor, mask_tensor)
                        predictions = output.squeeze(0).cpu().numpy()
                        
                        # Denormalize and fill
                        for j, col in enumerate(numeric_cols):
                            if col in self.scaler_stats:
                                denorm_pred = (
                                    predictions[:, j] * self.scaler_stats[col]['std'] + 
                                    self.scaler_stats[col]['mean']
                                )
                                
                                # Fill missing values
                                missing_idx = seq[col].isnull()
                                imputed.loc[
                                    chunk.index[i:i+sequence_length][missing_idx], 
                                    col
                                ] = denorm_pred[missing_idx]
                
                yield imputed


class StreamingProcessor:
    """Main streaming processor for large-scale imputation"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self._monitor = MemoryMonitor(config.memory_limit)
    
    def process_file(self,
                    input_path: Union[str, Path],
                    output_path: Union[str, Path],
                    method: str = "mean",
                    columns: Optional[List[str]] = None,
                    **method_kwargs) -> Dict[str, Any]:
        """
        Process large file with streaming imputation
        
        Parameters:
        -----------
        input_path : str or Path
            Input file path
        output_path : str or Path
            Output file path
        method : str
            Imputation method
        columns : list
            Columns to process
        **method_kwargs : dict
            Method-specific parameters
        
        Returns:
        --------
        dict : Processing statistics
        """
        logger.info(f"Processing {input_path} -> {output_path}")
        
        # Create chunk iterator
        iterator = ChunkIterator(
            input_path,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            columns=columns
        )
        
        # Create imputer
        imputer = self._create_imputer(method, **method_kwargs)
        
        # Setup output writer
        writer = self._create_writer(output_path)
        
        # Processing statistics
        stats = {
            'chunks_processed': 0,
            'rows_processed': 0,
            'imputed_values': 0,
            'processing_time': 0,
            'peak_memory': 0
        }
        
        start_time = time.time()
        
        try:
            # Fit imputer (first pass)
            with self._monitor.monitor():
                imputer.fit(iterator)
            
            # Reset iterator for transform
            iterator = ChunkIterator(
                input_path,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                columns=columns
            )
            
            # Transform and write (second pass)
            with self._monitor.monitor():
                for i, imputed_chunk in enumerate(imputer.transform(iterator)):
                    # Update statistics
                    stats['chunks_processed'] += 1
                    stats['rows_processed'] += len(imputed_chunk)
                    
                    # Count imputed values
                    if columns:
                        original_chunk = next(ChunkIterator(
                            input_path,
                            chunk_size=self.config.chunk_size,
                            overlap=0,
                            columns=columns
                        ))
                        missing_before = original_chunk.isnull().sum().sum()
                        missing_after = imputed_chunk.isnull().sum().sum()
                        stats['imputed_values'] += missing_before - missing_after
                    
                    # Write chunk
                    writer.write_chunk(imputed_chunk)
                    
                    # Progress callback
                    if self.config.progress_callback:
                        progress = {
                            'chunks': stats['chunks_processed'],
                            'rows': stats['rows_processed'],
                            'memory': self._monitor.current_usage
                        }
                        self.config.progress_callback(progress)
                    
                    # Check memory
                    if self._monitor.is_limit_exceeded():
                        logger.warning("Memory limit exceeded, forcing garbage collection")
                        gc.collect()
            
            # Finalize writer
            writer.finalize()
            
            # Update final statistics
            stats['processing_time'] = time.time() - start_time
            stats['peak_memory'] = self._monitor.peak_usage
            
            logger.info(f"Processing complete: {stats}")
            
        finally:
            # Cleanup
            iterator.close()
            imputer.cleanup()
        
        return stats
    
    def _create_imputer(self, method: str, **kwargs) -> StreamingImputer:
        """Create appropriate streaming imputer"""
        if method == "mean":
            return StreamingMeanImputer(self.config)
        elif method == "knn":
            return StreamingKNNImputer(self.config, **kwargs)
        elif method in ["linear", "forward_fill", "seasonal"]:
            return StreamingTimeSeriesImputer(self.config, method=method)
        elif method in ["lstm", "gru"]:
            return StreamingDeepImputer(self.config, model_type=method)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _create_writer(self, output_path: Union[str, Path]) -> 'ChunkWriter':
        """Create appropriate output writer"""
        output_path = Path(output_path)
        
        if output_path.suffix == ".parquet" or self.config.storage_format == "parquet":
            return ParquetChunkWriter(output_path, compression=self.config.compression)
        elif output_path.suffix == ".feather" or self.config.storage_format == "feather":
            return FeatherChunkWriter(output_path)
        elif output_path.suffix == ".csv":
            return CSVChunkWriter(output_path)
        elif output_path.suffix in [".h5", ".hdf5"] or self.config.storage_format == "hdf5":
            return HDF5ChunkWriter(output_path, compression=self.config.compression)
        else:
            # Default to Parquet
            return ParquetChunkWriter(
                output_path.with_suffix(".parquet"),
                compression=self.config.compression
            )


class ChunkWriter:
    """Base class for chunk writers"""
    
    def write_chunk(self, chunk: pd.DataFrame):
        """Write a data chunk"""
        raise NotImplementedError
    
    def finalize(self):
        """Finalize writing"""
        pass


class ParquetChunkWriter(ChunkWriter):
    """Write chunks to Parquet file"""
    
    def __init__(self, output_path: Path, compression: str = "snappy"):
        self.output_path = output_path
        self.compression = compression
        self.writer = None
        self.schema = None
    
    def write_chunk(self, chunk: pd.DataFrame):
        """Write chunk to Parquet"""
        table = pa.Table.from_pandas(chunk, preserve_index=True)
        
        if self.writer is None:
            # Initialize writer with schema from first chunk
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.output_path,
                self.schema,
                compression=self.compression
            )
        
        self.writer.write_table(table)
    
    def finalize(self):
        """Close Parquet writer"""
        if self.writer:
            self.writer.close()


class FeatherChunkWriter(ChunkWriter):
    """Write chunks to Feather file"""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.chunks = []
    
    def write_chunk(self, chunk: pd.DataFrame):
        """Collect chunks (Feather doesn't support streaming)"""
        self.chunks.append(chunk)
    
    def finalize(self):
        """Write all chunks to Feather"""
        if self.chunks:
            combined = pd.concat(self.chunks, ignore_index=True)
            feather.write_feather(combined, self.output_path)


class CSVChunkWriter(ChunkWriter):
    """Write chunks to CSV file"""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.first_chunk = True
    
    def write_chunk(self, chunk: pd.DataFrame):
        """Append chunk to CSV"""
        mode = 'w' if self.first_chunk else 'a'
        header = self.first_chunk
        
        chunk.to_csv(
            self.output_path,
            mode=mode,
            header=header,
            index=True
        )
        
        self.first_chunk = False


class HDF5ChunkWriter(ChunkWriter):
    """Write chunks to HDF5 file"""
    
    def __init__(self, output_path: Path, compression: str = "gzip"):
        self.output_path = output_path
        self.compression = compression
        self.store = None
        self.key = "data"
    
    def write_chunk(self, chunk: pd.DataFrame):
        """Append chunk to HDF5"""
        if self.store is None:
            self.store = pd.HDFStore(self.output_path, mode='w')
        
        self.store.append(
            self.key,
            chunk,
            format='table',
            complib=self.compression,
            index=True
        )
    
    def finalize(self):
        """Close HDF5 store"""
        if self.store:
            self.store.close()


class MemoryMonitor:
    """Monitor memory usage during processing"""
    
    def __init__(self, limit_bytes: int):
        self.limit_bytes = limit_bytes
        self.process = psutil.Process()
        self.current_usage = 0
        self.peak_usage = 0
    
    @contextmanager
    def monitor(self):
        """Context manager for memory monitoring"""
        try:
            yield self
        finally:
            self.update()
    
    def update(self):
        """Update memory statistics"""
        self.current_usage = self.process.memory_info().rss
        self.peak_usage = max(self.peak_usage, self.current_usage)
    
    def is_limit_exceeded(self) -> bool:
        """Check if memory limit is exceeded"""
        self.update()
        return self.current_usage > self.limit_bytes
    
    def get_usage_fraction(self) -> float:
        """Get current usage as fraction of limit"""
        self.update()
        return self.current_usage / self.limit_bytes


# Optimized numerical functions using Numba
@jit(nopython=True, parallel=True)
def fast_running_stats(data: np.ndarray) -> Tuple[float, float, int]:
    """Fast calculation of running statistics"""
    total = 0.0
    total_sq = 0.0
    count = 0
    
    for i in prange(len(data)):
        if not np.isnan(data[i]):
            total += data[i]
            total_sq += data[i] * data[i]
            count += 1
    
    if count > 0:
        mean = total / count
        variance = (total_sq / count) - mean * mean
        return mean, np.sqrt(max(0, variance)), count
    else:
        return 0.0, 0.0, 0


@jit(nopython=True)
def fast_linear_interpolate(data: np.ndarray) -> np.ndarray:
    """Fast linear interpolation for 1D array"""
    result = data.copy()
    n = len(data)
    
    # Find all NaN positions
    for i in range(n):
        if np.isnan(data[i]):
            # Find previous valid value
            prev_idx = i - 1
            while prev_idx >= 0 and np.isnan(data[prev_idx]):
                prev_idx -= 1
            
            # Find next valid value
            next_idx = i + 1
            while next_idx < n and np.isnan(data[next_idx]):
                next_idx += 1
            
            # Interpolate
            if prev_idx >= 0 and next_idx < n:
                # Linear interpolation
                prev_val = data[prev_idx]
                next_val = data[next_idx]
                weight = (i - prev_idx) / (next_idx - prev_idx)
                result[i] = prev_val + weight * (next_val - prev_val)
            elif prev_idx >= 0:
                # Forward fill
                result[i] = data[prev_idx]
            elif next_idx < n:
                # Backward fill
                result[i] = data[next_idx]
    
    return result


# Example usage
if __name__ == "__main__":
    # Configuration
    config = StreamConfig(
        chunk_size=10000,
        chunk_overlap=100,
        memory_limit=1 * 1024**3,  # 1GB
        n_workers=4,
        storage_format="parquet",
        compression="snappy"
    )
    
    # Create processor
    processor = StreamingProcessor(config)
    
    # Example: Create large synthetic dataset
    n_rows = 1_000_000
    dates = pd.date_range('2020-01-01', periods=n_rows, freq='h')
    
    # Generate data in chunks to avoid memory issues
    output_file = Path("large_synthetic_data.parquet")
    writer = ParquetChunkWriter(output_file)
    
    for i in range(0, n_rows, config.chunk_size):
        end = min(i + config.chunk_size, n_rows)
        chunk_dates = dates[i:end]
        
        chunk = pd.DataFrame({
            'timestamp': chunk_dates,
            'PM25': np.sin(np.arange(i, end) * 2 * np.pi / (24 * 7)) * 20 + 50 + 
                   np.random.normal(0, 5, end - i),
            'PM10': np.sin(np.arange(i, end) * 2 * np.pi / (24 * 7)) * 30 + 80 + 
                   np.random.normal(0, 8, end - i),
            'NO2': np.sin(np.arange(i, end) * 2 * np.pi / 24) * 15 + 40 + 
                  np.random.normal(0, 3, end - i)
        })
        
        # Add missing values
        mask = np.random.random(len(chunk)) < 0.2
        chunk.loc[mask, 'PM25'] = np.nan
        
        writer.write_chunk(chunk)
    
    writer.finalize()
    
    print(f"Created synthetic dataset: {output_file}")
    
    # Process with streaming imputation
    def progress_callback(info):
        print(f"Progress: {info['chunks']} chunks, {info['rows']:,} rows, "
              f"Memory: {info['memory'] / 1024**2:.1f} MB")
    
    config.progress_callback = progress_callback
    
    # Run imputation
    stats = processor.process_file(
        input_path=output_file,
        output_path="imputed_data.parquet",
        method="mean",
        columns=['PM25', 'PM10', 'NO2']
    )
    
    print(f"\nProcessing complete:")
    print(f"  Rows processed: {stats['rows_processed']:,}")
    print(f"  Values imputed: {stats['imputed_values']:,}")
    print(f"  Time: {stats['processing_time']:.1f} seconds")
    print(f"  Peak memory: {stats['peak_memory'] / 1024**2:.1f} MB")
    
    # Cleanup
    output_file.unlink()
    Path("imputed_data.parquet").unlink()