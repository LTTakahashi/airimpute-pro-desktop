"""
Chunked Data Processor for Large Datasets
Handles streaming and memory-efficient processing
"""

import pandas as pd
import numpy as np
from typing import Iterator, Dict, Any, Optional, Callable, Tuple, List
import logging
from pathlib import Path
import json
from dataclasses import dataclass
import psutil
import gc

logger = logging.getLogger(__name__)

@dataclass
class ChunkInfo:
    """Information about a data chunk"""
    chunk_id: int
    start_row: int
    end_row: int
    n_rows: int
    n_cols: int
    memory_mb: float
    has_missing: bool
    missing_count: int


class ChunkedProcessor:
    """Processes large datasets in chunks to avoid memory issues"""
    
    def __init__(self, chunk_size: int = 10000, memory_limit_mb: Optional[float] = None):
        """
        Initialize chunked processor
        
        Args:
            chunk_size: Number of rows per chunk
            memory_limit_mb: Maximum memory to use (defaults to 50% of available)
        """
        self.chunk_size = chunk_size
        
        if memory_limit_mb is None:
            # Use 50% of available memory
            available_mb = psutil.virtual_memory().available / 1024 / 1024
            self.memory_limit_mb = available_mb * 0.5
        else:
            self.memory_limit_mb = memory_limit_mb
            
        self.current_memory_usage = 0
        self._chunk_cache = {}
        
    def estimate_chunk_size(self, file_path: str, target_memory_mb: float = 100) -> int:
        """
        Estimate optimal chunk size based on file sample
        
        Args:
            file_path: Path to CSV file
            target_memory_mb: Target memory per chunk
            
        Returns:
            Optimal chunk size
        """
        try:
            # Read small sample to estimate row size
            sample = pd.read_csv(file_path, nrows=100)
            
            # Estimate memory per row
            memory_per_row = sample.memory_usage(deep=True).sum() / len(sample) / 1024 / 1024
            
            # Calculate chunk size to fit in target memory
            optimal_chunk_size = int(target_memory_mb / memory_per_row)
            
            # Ensure reasonable bounds
            return max(1000, min(optimal_chunk_size, 100000))
            
        except Exception as e:
            logger.warning(f"Could not estimate chunk size: {e}, using default")
            return self.chunk_size
    
    def read_csv_chunks(self, file_path: str, 
                       columns: Optional[List[str]] = None,
                       progress_callback: Optional[Callable[[float], None]] = None) -> Iterator[Tuple[ChunkInfo, pd.DataFrame]]:
        """
        Read CSV file in chunks
        
        Args:
            file_path: Path to CSV file
            columns: Specific columns to read
            progress_callback: Function to call with progress (0-100)
            
        Yields:
            Tuple of (ChunkInfo, DataFrame) for each chunk
        """
        file_size = Path(file_path).stat().st_size
        bytes_read = 0
        chunk_id = 0
        
        # Auto-adjust chunk size based on file
        chunk_size = self.estimate_chunk_size(file_path)
        logger.info(f"Using chunk size: {chunk_size:,} rows")
        
        try:
            # Create chunk iterator
            chunk_iter = pd.read_csv(
                file_path,
                chunksize=chunk_size,
                usecols=columns,
                low_memory=False
            )
            
            for chunk in chunk_iter:
                # Check memory before processing
                self._check_memory()
                
                # Create chunk info
                start_row = chunk_id * chunk_size
                info = ChunkInfo(
                    chunk_id=chunk_id,
                    start_row=start_row,
                    end_row=start_row + len(chunk) - 1,
                    n_rows=len(chunk),
                    n_cols=len(chunk.columns),
                    memory_mb=chunk.memory_usage(deep=True).sum() / 1024 / 1024,
                    has_missing=chunk.isnull().any().any(),
                    missing_count=chunk.isnull().sum().sum()
                )
                
                # Update progress
                if progress_callback and file_size > 0:
                    # Estimate based on chunk number (not exact but good enough)
                    progress = min(100, (chunk_id + 1) * chunk_size / (file_size / 1000) * 100)
                    progress_callback(progress)
                
                yield info, chunk
                
                chunk_id += 1
                
                # Force garbage collection periodically
                if chunk_id % 10 == 0:
                    gc.collect()
                    
        except Exception as e:
            logger.error(f"Error reading chunks: {e}")
            raise
            
    def process_chunks(self, chunks: Iterator[Tuple[ChunkInfo, pd.DataFrame]], 
                      processor: Callable[[pd.DataFrame], pd.DataFrame],
                      combiner: Optional[Callable[[List[pd.DataFrame]], pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Process chunks with a given function and combine results
        
        Args:
            chunks: Iterator of chunks
            processor: Function to process each chunk
            combiner: Function to combine results (defaults to concat)
            
        Returns:
            Combined results
        """
        results = []
        
        for info, chunk in chunks:
            try:
                # Process chunk
                result = processor(chunk)
                results.append(result)
                
                # Check if we need to combine intermediate results
                if len(results) > 10:
                    if combiner:
                        intermediate = combiner(results)
                    else:
                        intermediate = pd.concat(results, ignore_index=True)
                    results = [intermediate]
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error processing chunk {info.chunk_id}: {e}")
                raise
                
        # Final combination
        if combiner:
            return combiner(results)
        else:
            return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def impute_chunked(self, file_path: str, method: str = 'mean', 
                      columns: Optional[List[str]] = None,
                      progress_callback: Optional[Callable[[float], None]] = None) -> Iterator[pd.DataFrame]:
        """
        Perform imputation on file in chunks
        
        Args:
            file_path: Path to CSV file
            method: Imputation method
            columns: Columns to impute
            progress_callback: Progress callback
            
        Yields:
            Imputed chunks
        """
        # For simple methods, we need to calculate statistics first
        if method in ['mean', 'median']:
            logger.info(f"Calculating {method} values...")
            stats = self._calculate_statistics(file_path, method, columns)
        else:
            stats = None
            
        # Process chunks
        chunk_count = 0
        for info, chunk in self.read_csv_chunks(file_path, columns, progress_callback):
            # Select numeric columns if not specified
            if columns is None:
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_cols = [col for col in columns if col in chunk.columns]
                
            # Apply imputation
            if method == 'mean' and stats:
                for col in numeric_cols:
                    if col in stats:
                        chunk[col].fillna(stats[col], inplace=True)
                        
            elif method == 'median' and stats:
                for col in numeric_cols:
                    if col in stats:
                        chunk[col].fillna(stats[col], inplace=True)
                        
            elif method == 'forward_fill':
                # For forward fill, we need to maintain state between chunks
                if chunk_count > 0 and hasattr(self, '_last_values'):
                    # Fill first row with last values from previous chunk
                    for col in numeric_cols:
                        if col in self._last_values and pd.isna(chunk[col].iloc[0]):
                            chunk[col].iloc[0] = self._last_values[col]
                            
                chunk[numeric_cols] = chunk[numeric_cols].fillna(method='ffill')
                
                # Store last values for next chunk
                self._last_values = {
                    col: chunk[col].iloc[-1] for col in numeric_cols
                    if not pd.isna(chunk[col].iloc[-1])
                }
                
            elif method == 'linear':
                # Linear interpolation within chunk
                chunk[numeric_cols] = chunk[numeric_cols].interpolate(
                    method='linear',
                    limit_direction='both'
                )
                
            chunk_count += 1
            yield chunk
            
    def _calculate_statistics(self, file_path: str, stat_type: str, 
                            columns: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate statistics across entire file"""
        stats = {}
        counts = {}
        
        if stat_type == 'mean':
            sums = {}
            
            for info, chunk in self.read_csv_chunks(file_path, columns):
                if columns is None:
                    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                else:
                    numeric_cols = [col for col in columns if col in chunk.columns]
                    
                for col in numeric_cols:
                    if col not in sums:
                        sums[col] = 0
                        counts[col] = 0
                        
                    valid_data = chunk[col].dropna()
                    sums[col] += valid_data.sum()
                    counts[col] += len(valid_data)
                    
            # Calculate means
            for col in sums:
                if counts[col] > 0:
                    stats[col] = sums[col] / counts[col]
                    
        elif stat_type == 'median':
            # For median, we need all values (memory intensive)
            # Use approximate median for large datasets
            all_values = {col: [] for col in columns or []}
            
            for info, chunk in self.read_csv_chunks(file_path, columns):
                if columns is None:
                    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                else:
                    numeric_cols = [col for col in columns if col in chunk.columns]
                    
                for col in numeric_cols:
                    if col not in all_values:
                        all_values[col] = []
                    all_values[col].extend(chunk[col].dropna().tolist())
                    
                    # If too many values, sample
                    if len(all_values[col]) > 1_000_000:
                        all_values[col] = np.random.choice(
                            all_values[col], 
                            size=1_000_000, 
                            replace=False
                        ).tolist()
                        
            # Calculate medians
            for col, values in all_values.items():
                if values:
                    stats[col] = np.median(values)
                    
        return stats
    
    def _check_memory(self):
        """Check memory usage and raise if exceeded"""
        current_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        if current_mb > self.memory_limit_mb:
            gc.collect()  # Try to free memory
            current_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            if current_mb > self.memory_limit_mb:
                raise MemoryError(
                    f"Memory limit exceeded: {current_mb:.1f}MB > {self.memory_limit_mb:.1f}MB"
                )
    
    def save_chunks(self, chunks: Iterator[pd.DataFrame], output_path: str,
                   format: str = 'csv') -> int:
        """
        Save processed chunks to file
        
        Args:
            chunks: Iterator of DataFrames
            output_path: Output file path
            format: Output format (csv, parquet)
            
        Returns:
            Number of rows written
        """
        total_rows = 0
        
        if format == 'csv':
            # Write header on first chunk
            first_chunk = True
            
            for chunk in chunks:
                chunk.to_csv(
                    output_path,
                    mode='w' if first_chunk else 'a',
                    header=first_chunk,
                    index=False
                )
                first_chunk = False
                total_rows += len(chunk)
                
        elif format == 'parquet':
            # For parquet, we need to collect chunks (less memory efficient)
            all_chunks = list(chunks)
            if all_chunks:
                result = pd.concat(all_chunks, ignore_index=True)
                result.to_parquet(output_path, index=False)
                total_rows = len(result)
                
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return total_rows


def process_large_file(file_path: str, output_path: str, 
                      method: str = 'mean',
                      chunk_size: int = 10000,
                      progress_callback: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
    """
    High-level function to process large file with imputation
    
    Args:
        file_path: Input CSV path
        output_path: Output CSV path
        method: Imputation method
        chunk_size: Rows per chunk
        progress_callback: Progress callback
        
    Returns:
        Processing statistics
    """
    processor = ChunkedProcessor(chunk_size=chunk_size)
    
    # Get file info
    file_stats = {
        'input_file': file_path,
        'output_file': output_path,
        'file_size_mb': Path(file_path).stat().st_size / 1024 / 1024,
        'method': method,
        'chunk_size': chunk_size
    }
    
    try:
        # Process file
        imputed_chunks = processor.impute_chunked(
            file_path, 
            method=method,
            progress_callback=progress_callback
        )
        
        # Save results
        rows_written = processor.save_chunks(imputed_chunks, output_path)
        
        file_stats.update({
            'success': True,
            'rows_processed': rows_written,
            'output_size_mb': Path(output_path).stat().st_size / 1024 / 1024,
            'memory_used_mb': psutil.Process().memory_info().rss / 1024 / 1024
        })
        
    except Exception as e:
        file_stats.update({
            'success': False,
            'error': str(e)
        })
        
    return file_stats


# Integration with desktop app
def process_file_chunked_json(request_json: str) -> str:
    """JSON interface for Rust integration"""
    try:
        request = json.loads(request_json)
        
        result = process_large_file(
            file_path=request['file_path'],
            output_path=request.get('output_path', request['file_path'].replace('.csv', '_imputed.csv')),
            method=request.get('method', 'mean'),
            chunk_size=request.get('chunk_size', 10000)
        )
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            'success': False,
            'error': str(e)
        })