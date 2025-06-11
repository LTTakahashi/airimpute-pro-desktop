"""
Local Data Processing Pipeline - Optimized for offline use
No external dependencies, all processing done locally
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Iterator
import json
import os
import gc
import psutil
from pathlib import Path
import logging
from dataclasses import dataclass
import hashlib
import pickle
import lz4.frame

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for local pipeline"""
    chunk_size: int = 10000
    max_memory_mb: float = 1000
    enable_caching: bool = True
    cache_dir: Optional[Path] = None
    compression_level: int = 6
    parallel_jobs: int = 1  # Local only, no distributed processing
    

class LocalDataPipeline:
    """Optimized pipeline for local data processing"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Set up local cache directory
        if self.config.cache_dir is None:
            self.config.cache_dir = Path.home() / ".airimpute_cache"
        self.config.cache_dir.mkdir(exist_ok=True)
        
        # Initialize memory tracker
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        
    def process_dataset(self, 
                       file_path: str,
                       operations: List[Dict[str, Any]],
                       output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process dataset through pipeline of operations
        
        Args:
            file_path: Input file path
            operations: List of operations to apply
            output_path: Optional output path
            
        Returns:
            Processing results and statistics
        """
        start_time = pd.Timestamp.now()
        stats = {
            'file_path': file_path,
            'file_size_mb': os.path.getsize(file_path) / 1024 / 1024,
            'operations': len(operations),
            'chunks_processed': 0,
            'rows_processed': 0,
            'cache_hits': 0,
            'memory_peak_mb': 0,
        }
        
        # Generate cache key for this pipeline
        pipeline_hash = self._generate_pipeline_hash(file_path, operations)
        
        # Check if result is cached
        if self.config.enable_caching:
            cached_result = self._load_from_cache(pipeline_hash)
            if cached_result is not None:
                stats['cache_hits'] = 1
                stats['from_cache'] = True
                stats['processing_time_seconds'] = 0
                return stats
        
        # Process in chunks
        results = []
        for chunk_idx, chunk in enumerate(self._read_chunks(file_path)):
            # Check memory
            current_memory = self.process.memory_info().rss / 1024 / 1024
            stats['memory_peak_mb'] = max(stats['memory_peak_mb'], 
                                         current_memory - self.initial_memory)
            
            if current_memory > self.config.max_memory_mb:
                logger.warning(f"Memory limit approaching: {current_memory:.0f}MB")
                gc.collect()
            
            # Apply operations to chunk
            processed_chunk = chunk
            for op_idx, operation in enumerate(operations):
                op_hash = self._generate_operation_hash(
                    chunk_idx, op_idx, operation, processed_chunk
                )
                
                # Check operation cache
                if self.config.enable_caching:
                    cached_op = self._load_from_cache(op_hash)
                    if cached_op is not None:
                        processed_chunk = cached_op
                        stats['cache_hits'] += 1
                        continue
                
                # Execute operation
                processed_chunk = self._execute_operation(
                    processed_chunk, operation
                )
                
                # Cache operation result
                if self.config.enable_caching:
                    self._save_to_cache(op_hash, processed_chunk)
            
            results.append(processed_chunk)
            stats['chunks_processed'] += 1
            stats['rows_processed'] += len(chunk)
            
            # Free memory periodically
            if chunk_idx % 10 == 0:
                gc.collect()
        
        # Combine results
        final_result = pd.concat(results, ignore_index=True)
        
        # Save output
        if output_path:
            self._save_output(final_result, output_path)
            stats['output_path'] = output_path
            stats['output_size_mb'] = os.path.getsize(output_path) / 1024 / 1024
        
        # Cache final result
        if self.config.enable_caching:
            self._save_to_cache(pipeline_hash, final_result)
        
        # Calculate final statistics
        end_time = pd.Timestamp.now()
        stats['processing_time_seconds'] = (end_time - start_time).total_seconds()
        stats['throughput_rows_per_second'] = stats['rows_processed'] / stats['processing_time_seconds']
        
        return stats
    
    def _read_chunks(self, file_path: str) -> Iterator[pd.DataFrame]:
        """Read file in memory-efficient chunks"""
        # Auto-detect file type
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            for chunk in pd.read_csv(file_path, chunksize=self.config.chunk_size):
                yield chunk
        elif file_ext == '.parquet':
            # Read parquet in chunks
            df = pd.read_parquet(file_path)
            for i in range(0, len(df), self.config.chunk_size):
                yield df.iloc[i:i + self.config.chunk_size]
        elif file_ext in ['.xls', '.xlsx']:
            # Excel files - read all at once but yield in chunks
            df = pd.read_excel(file_path)
            for i in range(0, len(df), self.config.chunk_size):
                yield df.iloc[i:i + self.config.chunk_size]
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def _execute_operation(self, 
                          data: pd.DataFrame, 
                          operation: Dict[str, Any]) -> pd.DataFrame:
        """Execute single operation on data"""
        op_type = operation['type']
        params = operation.get('params', {})
        
        if op_type == 'impute':
            return self._impute_data(data, **params)
        elif op_type == 'filter':
            return self._filter_data(data, **params)
        elif op_type == 'transform':
            return self._transform_data(data, **params)
        elif op_type == 'validate':
            return self._validate_data(data, **params)
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
    
    def _impute_data(self, data: pd.DataFrame, 
                    method: str = 'mean', 
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Impute missing values"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        result = data.copy()
        
        if method == 'mean':
            for col in columns:
                if col in result.columns:
                    result[col].fillna(result[col].mean(), inplace=True)
        elif method == 'median':
            for col in columns:
                if col in result.columns:
                    result[col].fillna(result[col].median(), inplace=True)
        elif method == 'forward_fill':
            result[columns] = result[columns].fillna(method='ffill')
        elif method == 'interpolate':
            result[columns] = result[columns].interpolate(method='linear')
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        return result
    
    def _filter_data(self, data: pd.DataFrame, 
                    condition: str = None,
                    remove_outliers: bool = False,
                    z_threshold: float = 3.0) -> pd.DataFrame:
        """Filter data based on conditions"""
        result = data.copy()
        
        if condition:
            # Safe evaluation of condition
            # In production, use a proper parser
            result = result.query(condition)
        
        if remove_outliers:
            # Remove outliers using z-score
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                z_scores = np.abs((result[col] - result[col].mean()) / result[col].std())
                result = result[z_scores < z_threshold]
        
        return result
    
    def _transform_data(self, data: pd.DataFrame,
                       operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply transformations to data"""
        result = data.copy()
        
        for transform in operations:
            t_type = transform['type']
            
            if t_type == 'scale':
                # Min-max scaling
                cols = transform.get('columns', result.select_dtypes(include=[np.number]).columns)
                for col in cols:
                    if col in result.columns:
                        min_val = result[col].min()
                        max_val = result[col].max()
                        if max_val > min_val:
                            result[col] = (result[col] - min_val) / (max_val - min_val)
            
            elif t_type == 'log':
                # Log transformation
                cols = transform.get('columns', [])
                for col in cols:
                    if col in result.columns:
                        result[col] = np.log1p(result[col].clip(lower=0))
            
            elif t_type == 'difference':
                # Differencing for time series
                cols = transform.get('columns', [])
                periods = transform.get('periods', 1)
                for col in cols:
                    if col in result.columns:
                        result[col] = result[col].diff(periods)
        
        return result
    
    def _validate_data(self, data: pd.DataFrame,
                      checks: List[str] = None) -> pd.DataFrame:
        """Validate data and add validation flags"""
        result = data.copy()
        
        if checks is None:
            checks = ['missing', 'range', 'type']
        
        validation_flags = []
        
        if 'missing' in checks:
            # Flag rows with any missing values
            result['has_missing'] = result.isnull().any(axis=1)
            validation_flags.append('has_missing')
        
        if 'range' in checks:
            # Check if values are within expected ranges
            # This would be configured per column
            pass
        
        if 'type' in checks:
            # Check data types are correct
            pass
        
        result['validation_flags'] = validation_flags
        
        return result
    
    def _generate_pipeline_hash(self, file_path: str, 
                               operations: List[Dict[str, Any]]) -> str:
        """Generate hash for entire pipeline"""
        hasher = hashlib.sha256()
        
        # Hash file metadata (not content for performance)
        file_stat = os.stat(file_path)
        hasher.update(f"{file_path}:{file_stat.st_size}:{file_stat.st_mtime}".encode())
        
        # Hash operations
        hasher.update(json.dumps(operations, sort_keys=True).encode())
        
        return hasher.hexdigest()
    
    def _generate_operation_hash(self, chunk_idx: int, op_idx: int,
                                operation: Dict[str, Any],
                                data_sample: pd.DataFrame) -> str:
        """Generate hash for single operation result"""
        hasher = hashlib.sha256()
        
        hasher.update(f"{chunk_idx}:{op_idx}".encode())
        hasher.update(json.dumps(operation, sort_keys=True).encode())
        
        # Sample data for hash (first few rows)
        if len(data_sample) > 0:
            sample = data_sample.head(5).to_json()
            hasher.update(sample.encode())
        
        return hasher.hexdigest()
    
    def _save_to_cache(self, key: str, data: Any) -> None:
        """Save data to local cache with compression"""
        if not self.config.enable_caching:
            return
        
        cache_path = self.config.cache_dir / f"{key}.lz4"
        
        try:
            # Serialize with pickle and compress with LZ4
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = lz4.frame.compress(serialized, 
                                          compression_level=self.config.compression_level)
            
            with open(cache_path, 'wb') as f:
                f.write(compressed)
                
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    def _load_from_cache(self, key: str) -> Optional[Any]:
        """Load data from local cache"""
        if not self.config.enable_caching:
            return None
        
        cache_path = self.config.cache_dir / f"{key}.lz4"
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                compressed = f.read()
            
            serialized = lz4.frame.decompress(compressed)
            data = pickle.loads(serialized)
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            # Remove corrupted cache file
            cache_path.unlink(missing_ok=True)
            return None
    
    def _save_output(self, data: pd.DataFrame, output_path: str) -> None:
        """Save output in appropriate format"""
        output_ext = Path(output_path).suffix.lower()
        
        if output_ext == '.csv':
            data.to_csv(output_path, index=False)
        elif output_ext == '.parquet':
            data.to_parquet(output_path, index=False, compression='snappy')
        elif output_ext in ['.xls', '.xlsx']:
            data.to_excel(output_path, index=False)
        elif output_ext == '.json':
            data.to_json(output_path, orient='records', indent=2)
        else:
            # Default to CSV
            data.to_csv(output_path, index=False)
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear local cache and return statistics"""
        stats = {
            'files_removed': 0,
            'space_freed_mb': 0
        }
        
        if self.config.cache_dir.exists():
            for cache_file in self.config.cache_dir.glob("*.lz4"):
                size = cache_file.stat().st_size
                cache_file.unlink()
                stats['files_removed'] += 1
                stats['space_freed_mb'] += size / 1024 / 1024
        
        return stats


# Optimized functions for direct use

def create_local_pipeline(config: Optional[Dict[str, Any]] = None) -> LocalDataPipeline:
    """Create pipeline with configuration"""
    if config:
        pipeline_config = PipelineConfig(**config)
    else:
        pipeline_config = PipelineConfig()
    
    return LocalDataPipeline(pipeline_config)


def process_file_locally(file_path: str, 
                        operations: List[Dict[str, Any]],
                        output_path: Optional[str] = None,
                        config: Optional[Dict[str, Any]] = None) -> str:
    """Process file through local pipeline - returns JSON stats"""
    pipeline = create_local_pipeline(config)
    stats = pipeline.process_dataset(file_path, operations, output_path)
    return json.dumps(stats, indent=2)


# Example usage
if __name__ == "__main__":
    # Example pipeline
    operations = [
        {
            'type': 'filter',
            'params': {
                'remove_outliers': True,
                'z_threshold': 3.0
            }
        },
        {
            'type': 'impute',
            'params': {
                'method': 'linear',
                'columns': None  # All numeric columns
            }
        },
        {
            'type': 'transform',
            'params': {
                'operations': [
                    {'type': 'scale', 'columns': ['PM2.5', 'PM10']}
                ]
            }
        }
    ]
    
    config = {
        'chunk_size': 5000,
        'max_memory_mb': 500,
        'enable_caching': True,
        'compression_level': 9
    }
    
    # Process file
    stats = process_file_locally(
        'data/air_quality.csv',
        operations,
        'output/processed_data.csv',
        config
    )
    
    print(stats)