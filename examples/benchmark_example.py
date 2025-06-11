#!/usr/bin/env python3
"""
Comprehensive benchmark example for AirImpute Pro.

This script demonstrates how to run a complete benchmark evaluation
with all features including GPU acceleration, statistical testing,
and reproducibility tracking.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append('scripts')

from airimpute.benchmarking import (
    BenchmarkRunner, 
    BenchmarkDatasetManager,
    ReproducibilityInfo,
    run_comprehensive_benchmark
)
from airimpute.methods.simple import MeanImputation, ForwardFillImputation
from airimpute.methods.interpolation import LinearInterpolation, SplineInterpolation
from airimpute.methods.statistical import KalmanFilterImputation, ARIMAImputation
from airimpute.methods.machine_learning import (
    RandomForestImputation, 
    XGBoostImputation,
    LightGBMImputation
)
from airimpute.methods.rah import (
    RobustAdaptiveHierarchicalImputation,
    AdaptiveRAHImputation
)


def basic_benchmark_example():
    """Basic benchmark with simple methods."""
    print("="*80)
    print("BASIC BENCHMARK EXAMPLE")
    print("="*80)
    
    # Initialize dataset manager
    manager = BenchmarkDatasetManager()
    
    # Create synthetic datasets with different characteristics
    datasets = []
    for i, (rate, pattern) in enumerate([
        (0.1, 'random'),
        (0.2, 'blocks'),
        (0.3, 'temporal')
    ]):
        name = f'synthetic_{pattern}_{int(rate*100)}'
        manager.create_synthetic_dataset(
            name=name,
            n_timesteps=1000,
            n_stations=10,
            missing_rate=rate,
            pattern=pattern,
            seed=42 + i
        )
        datasets.append(name)
    
    # Define methods to benchmark
    methods = {
        'mean': MeanImputation().impute,
        'forward_fill': ForwardFillImputation().impute,
        'linear': LinearInterpolation().impute,
        'spline': SplineInterpolation(order=3).impute,
    }
    
    # Run benchmark
    runner = BenchmarkRunner(manager, random_seed=42)
    results = runner.run_benchmark(
        methods=methods,
        datasets=datasets,
        cv_splits=3,
        parallel=True
    )
    
    # Analyze results
    comparison = runner.compare_methods(results)
    
    print("\nResults Summary:")
    print("-" * 40)
    for i, (method, score) in enumerate(comparison['ranking']):
        print(f"{i+1}. {method}: RMSE = {score:.4f}")
    
    if 'friedman_test' in comparison:
        print(f"\nStatistical Test:")
        print(f"Friedman p-value: {comparison['friedman_test']['pvalue']:.4f}")
        
    return results


def advanced_benchmark_example():
    """Advanced benchmark with ML methods and GPU acceleration."""
    print("\n" + "="*80)
    print("ADVANCED BENCHMARK EXAMPLE")
    print("="*80)
    
    # Initialize with GPU support
    manager = BenchmarkDatasetManager()
    runner = BenchmarkRunner(
        manager, 
        use_gpu=True,
        random_seed=42
    )
    
    # Create larger datasets
    datasets = []
    for i in range(3):
        name = f'large_synthetic_{i}'
        manager.create_synthetic_dataset(
            name=name,
            n_timesteps=8760,  # One year hourly
            n_stations=50,
            missing_rate=0.2 + i*0.1,
            pattern='all',
            temporal_correlation=0.8,
            spatial_correlation=0.6,
            seed=42 + i
        )
        datasets.append(name)
    
    # Advanced methods including RAH
    methods = {
        'kalman': KalmanFilterImputation().impute,
        'arima': ARIMAImputation(order=(1,1,1)).impute,
        'rf': RandomForestImputation(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1
        ).impute,
        'xgboost': XGBoostImputation(
            n_estimators=100,
            learning_rate=0.1,
            gpu_id=0 if runner.use_gpu else -1
        ).impute,
        'rah': RobustAdaptiveHierarchicalImputation(
            n_estimators=50,
            use_gpu=runner.use_gpu
        ).impute,
        'adaptive_rah': AdaptiveRAHImputation(
            base_estimators=['rf', 'xgb', 'lgb'],
            meta_learner='neural_net',
            use_gpu=runner.use_gpu
        ).impute
    }
    
    # Run comprehensive benchmark
    results = runner.run_benchmark(
        methods=methods,
        datasets=datasets,
        cv_splits=5,
        save_predictions=True,
        parallel=True
    )
    
    # Detailed analysis
    print("\nDetailed Results:")
    print("-" * 80)
    
    # Create results dataframe
    results_data = []
    for result in results:
        results_data.append({
            'Method': result.method_name,
            'Dataset': result.dataset_name,
            'RMSE': result.metrics.get('rmse', np.nan),
            'MAE': result.metrics.get('mae', np.nan),
            'R²': result.metrics.get('r2', np.nan),
            'Runtime (s)': result.runtime,
            'Memory (MB)': result.memory_usage
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Print aggregated results
    print("\nAggregated Performance (mean ± std):")
    agg_results = results_df.groupby('Method').agg({
        'RMSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'R²': ['mean', 'std'],
        'Runtime (s)': 'mean',
        'Memory (MB)': 'mean'
    }).round(4)
    print(agg_results)
    
    # Statistical comparison
    comparison = runner.compare_methods(results_df, metric='RMSE')
    
    if 'friedman_test' in comparison:
        print(f"\nFriedman Test Results:")
        print(f"χ² = {comparison['friedman_test']['statistic']:.2f}")
        print(f"p-value = {comparison['friedman_test']['pvalue']:.4f}")
        
        if comparison['friedman_test']['significant']:
            print("\nPost-hoc Analysis (Nemenyi):")
            print("Significant differences:")
            for method1, comparisons in comparison['friedman_test']['post_hoc'].items():
                for method2, is_different in comparisons.items():
                    if is_different:
                        print(f"  {method1} ≠ {method2}")
    
    # Generate publication outputs
    runner.export_results(
        results_df,
        formats=['csv', 'latex'],
        output_dir=Path('benchmark_results')
    )
    
    return results


def gpu_benchmark_example():
    """Demonstrate GPU-accelerated benchmarking."""
    print("\n" + "="*80)
    print("GPU-ACCELERATED BENCHMARK EXAMPLE")
    print("="*80)
    
    # Check GPU availability
    from airimpute.benchmarking import GPUAcceleratedMethods
    
    try:
        gpu_methods = GPUAcceleratedMethods(backend='cuda')
        print("CUDA GPU acceleration available!")
    except:
        try:
            gpu_methods = GPUAcceleratedMethods(backend='opencl')
            print("OpenCL GPU acceleration available!")
        except:
            print("No GPU acceleration available, using CPU")
            return
    
    # Create large dataset for GPU benchmark
    manager = BenchmarkDatasetManager()
    manager.create_synthetic_dataset(
        name='gpu_test_large',
        n_timesteps=100000,
        n_stations=100,
        missing_rate=0.3,
        pattern='random',
        seed=42
    )
    
    # Benchmark GPU vs CPU
    dataset = manager.get_dataset('gpu_test_large')
    data = dataset.data.values
    
    # CPU timing
    import time
    start = time.time()
    cpu_result = LinearInterpolation().impute(data)
    cpu_time = time.time() - start
    
    # GPU timing
    start = time.time()
    gpu_result = gpu_methods.gpu_linear_interpolation(data)
    gpu_time = time.time() - start
    
    print(f"\nPerformance Comparison:")
    print(f"Dataset size: {data.shape}")
    print(f"CPU time: {cpu_time:.2f} seconds")
    print(f"GPU time: {gpu_time:.2f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    
    # Verify results are similar
    diff = np.nanmean(np.abs(cpu_result - gpu_result))
    print(f"Mean absolute difference: {diff:.6f}")


def reproducibility_example():
    """Demonstrate reproducibility features."""
    print("\n" + "="*80)
    print("REPRODUCIBILITY EXAMPLE")
    print("="*80)
    
    # Create reproducibility info
    repro_info = ReproducibilityInfo()
    
    # Initialize runner with reproducibility tracking
    manager = BenchmarkDatasetManager()
    runner = BenchmarkRunner(
        manager,
        reproducibility_info=repro_info,
        random_seed=12345
    )
    
    # Create dataset
    manager.create_synthetic_dataset(
        name='repro_test',
        n_timesteps=1000,
        n_stations=10,
        missing_rate=0.2,
        seed=12345
    )
    
    # Run benchmark
    methods = {
        'rf': RandomForestImputation(random_state=12345).impute,
        'xgb': XGBoostImputation(random_state=12345).impute,
    }
    
    results1 = runner.run_benchmark(
        methods=methods,
        datasets=['repro_test'],
        cv_splits=3
    )
    
    # Print reproducibility info
    print("\nReproducibility Information:")
    print(f"Benchmark ID: {repro_info.benchmark_id}")
    print(f"Git commit: {repro_info.git_commit}")
    print(f"Platform: {repro_info.platform_info['system']} {repro_info.platform_info['release']}")
    print(f"Python version: {repro_info.python_version.split()[0]}")
    print("\nPackage versions:")
    for pkg, ver in repro_info.package_versions.items():
        print(f"  {pkg}: {ver}")
    
    # Generate certificate
    cert_hash = repro_info.generate_certificate_hash()
    print(f"\nReproducibility certificate hash: {cert_hash[:16]}...")
    
    # Verify reproducibility by running again
    runner2 = BenchmarkRunner(
        manager,
        random_seed=12345
    )
    
    results2 = runner2.run_benchmark(
        methods=methods,
        datasets=['repro_test'],
        cv_splits=3
    )
    
    # Compare results
    print("\nReproducibility verification:")
    for r1, r2 in zip(results1, results2):
        rmse_diff = abs(r1.metrics['rmse'] - r2.metrics['rmse'])
        print(f"{r1.method_name}: RMSE difference = {rmse_diff:.8f}")


def publication_example():
    """Generate publication-ready benchmark results."""
    print("\n" + "="*80)
    print("PUBLICATION-READY BENCHMARK EXAMPLE")
    print("="*80)
    
    # Run comprehensive benchmark
    methods = {
        'mean': MeanImputation().impute,
        'linear': LinearInterpolation().impute,
        'kalman': KalmanFilterImputation().impute,
        'rf': RandomForestImputation(n_estimators=100).impute,
        'xgb': XGBoostImputation(n_estimators=100).impute,
        'rah': RobustAdaptiveHierarchicalImputation().impute,
    }
    
    results = run_comprehensive_benchmark(
        methods=methods,
        n_synthetic_datasets=5,
        use_gpu=True,
        output_dir=Path('publication_results')
    )
    
    print("\nPublication outputs generated in 'publication_results/' directory:")
    print("- benchmark_results_*.csv - Raw results")
    print("- benchmark_results_*.json - JSON format")
    print("- benchmark_table_*.tex - LaTeX table")
    print("- reproducibility_info.json - Full reproducibility information")
    print("- statistical_analysis.txt - Statistical test results")
    
    return results


if __name__ == "__main__":
    # Run all examples
    print("Running AirImpute Pro Benchmark Examples")
    print("=" * 80)
    
    # 1. Basic benchmark
    basic_results = basic_benchmark_example()
    
    # 2. Advanced benchmark with ML methods
    advanced_results = advanced_benchmark_example()
    
    # 3. GPU benchmark (if available)
    gpu_benchmark_example()
    
    # 4. Reproducibility demonstration
    reproducibility_example()
    
    # 5. Publication-ready results
    publication_results = publication_example()
    
    print("\n" + "="*80)
    print("All benchmark examples completed successfully!")
    print("="*80)