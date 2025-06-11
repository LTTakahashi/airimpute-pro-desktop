# AirImpute Pro - Comprehensive Testing Guide

## Overview

This guide provides comprehensive documentation for the AirImpute Pro testing infrastructure, following IEEE/ACM standards for scientific software testing. Our testing strategy ensures reliability, accuracy, and reproducibility of air quality data imputation methods.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Testing Architecture](#testing-architecture)
3. [Running Tests](#running-tests)
4. [Writing Tests](#writing-tests)
5. [Test Categories](#test-categories)
6. [Coverage Requirements](#coverage-requirements)
7. [Performance Testing](#performance-testing)
8. [Scientific Validation](#scientific-validation)
9. [CI/CD Integration](#cicd-integration)
10. [Troubleshooting](#troubleshooting)

## Testing Philosophy

Our testing approach follows these principles:

- **Scientific Rigor**: All numerical algorithms are tested against known solutions and peer-reviewed benchmarks
- **Reproducibility**: Tests use fixed random seeds and deterministic algorithms
- **Comprehensive Coverage**: Unit, integration, E2E, and performance tests
- **Real-world Validation**: Tests include actual air quality datasets from São Paulo
- **Continuous Validation**: Automated testing on every commit

## Testing Architecture

```
├── Frontend (React/TypeScript)
│   ├── Unit Tests (Vitest + React Testing Library)
│   ├── Component Tests
│   ├── Integration Tests
│   └── E2E Tests (Playwright)
│
├── Backend (Rust)
│   ├── Unit Tests (#[test])
│   ├── Integration Tests
│   └── Benchmarks (criterion)
│
└── Scientific Computing (Python)
    ├── Unit Tests (pytest)
    ├── Numerical Tests
    ├── Statistical Validation
    └── Performance Benchmarks
```

## Running Tests

### All Tests
```bash
# Run all tests across all platforms
pnpm test:all

# With coverage
pnpm test:coverage
```

### Frontend Tests
```bash
# Unit and component tests
pnpm test:frontend

# Watch mode for development
pnpm test:frontend:watch

# E2E tests
pnpm test:e2e

# E2E with UI
pnpm test:e2e:ui
```

### Rust Backend Tests
```bash
# Run all Rust tests
cd src-tauri
cargo test

# Run with verbose output
cargo test -- --nocapture

# Run specific test
cargo test test_data_pipeline

# Run benchmarks
cargo bench
```

### Python Scientific Tests
```bash
# Run all Python tests
cd scripts
pytest

# Run with coverage
pytest --cov=airimpute --cov-report=html

# Run specific category
pytest -m "not slow"
pytest -k "test_kalman"

# Run with detailed output
pytest -vvs
```

## Writing Tests

### Frontend Component Test Example

```typescript
import { describe, it, expect } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DataImportForm } from '@/components/DataImportForm';

describe('DataImportForm', () => {
  it('validates file format before upload', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    
    render(<DataImportForm onSubmit={onSubmit} />);
    
    // Try to upload invalid file
    const file = new File(['test'], 'test.txt', { type: 'text/plain' });
    const input = screen.getByLabelText(/upload file/i);
    
    await user.upload(input, file);
    
    // Should show error
    expect(screen.getByText(/unsupported file format/i)).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
```

### Rust Unit Test Example

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    
    #[test]
    fn test_missing_data_detection() {
        let data = arr2(&[
            [1.0, 2.0, f64::NAN],
            [4.0, f64::NAN, 6.0],
        ]);
        
        let dataset = Dataset::new("test".to_string(), data, vec![], vec![]);
        
        assert_eq!(dataset.count_missing(), 2);
        assert_eq!(dataset.missing_percentage(), 33.33);
    }
}
```

### Python Scientific Test Example

```python
import pytest
import numpy as np
from airimpute.methods import KalmanFilterImputer

class TestKalmanFilter:
    def test_ar_process_recovery(self):
        """Test that Kalman filter recovers AR(1) process parameters."""
        # Generate AR(1) process
        n = 1000
        phi = 0.8
        sigma = 1.0
        
        x = np.zeros(n)
        x[0] = np.random.randn()
        for i in range(1, n):
            x[i] = phi * x[i-1] + sigma * np.random.randn()
        
        # Add 20% missing values
        missing_mask = np.random.random(n) < 0.2
        x_missing = x.copy()
        x_missing[missing_mask] = np.nan
        
        # Impute
        imputer = KalmanFilterImputer(state_dim=1)
        result = imputer.impute(x_missing.reshape(-1, 1))
        
        # Verify AR coefficient is recovered
        x_imputed = result.imputed_data[:, 0]
        phi_est = np.corrcoef(x_imputed[:-1], x_imputed[1:])[0, 1]
        
        assert abs(phi_est - phi) < 0.05
```

## Test Categories

### 1. Unit Tests
- Test individual functions and methods
- Mock external dependencies
- Fast execution (< 100ms per test)
- High coverage requirement (> 90%)

### 2. Integration Tests
- Test component interactions
- Use real implementations where possible
- Database and file system operations
- Network calls to Python bridge

### 3. E2E Tests
- Complete user workflows
- Real browser automation
- Performance measurements
- Accessibility compliance

### 4. Scientific Validation Tests
- Numerical accuracy tests
- Statistical property preservation
- Comparison with reference implementations
- Convergence tests

### 5. Performance Tests
- Benchmark critical operations
- Memory usage profiling
- Scalability tests (up to 1M data points)
- GPU acceleration validation

## Coverage Requirements

### Minimum Coverage Thresholds
- Overall: 80%
- Critical paths: 95%
- Scientific algorithms: 100%
- Error handling: 90%

### Running Coverage Reports

```bash
# Frontend coverage
pnpm test:coverage

# Rust coverage
cargo tarpaulin --out Html

# Python coverage
pytest --cov=airimpute --cov-report=html
open htmlcov/index.html
```

## Performance Testing

### Benchmarking Framework

```python
@pytest.mark.benchmark
def test_large_dataset_imputation(benchmark):
    data = generate_test_data(n_samples=100000, n_features=20)
    imputer = RandomForestImputer(n_estimators=100)
    
    result = benchmark(imputer.impute, data)
    
    assert result.execution_time_ms < 5000  # 5 second limit
```

### Load Testing

```typescript
test('handles 10k concurrent data points', async () => {
  const dataPoints = generateDataPoints(10000);
  
  const startTime = performance.now();
  await processDataPoints(dataPoints);
  const duration = performance.now() - startTime;
  
  expect(duration).toBeLessThan(1000); // 1 second limit
});
```

## Scientific Validation

### Numerical Accuracy Tests

1. **Known Solutions**: Test against analytical solutions
2. **Synthetic Data**: Generate data with known properties
3. **Recovery Tests**: Verify parameter recovery
4. **Cross-validation**: Compare with reference implementations

### Statistical Tests

```python
def test_correlation_preservation():
    """Verify that imputation preserves correlation structure."""
    # Generate correlated data
    mean = [0, 0]
    cov = [[1, 0.7], [0.7, 1]]
    data = np.random.multivariate_normal(mean, cov, 1000)
    
    # Add missing values
    data_missing = mask_values(data, 0.2)
    
    # Impute
    imputed = impute_data(data_missing)
    
    # Check correlation is preserved
    corr_original = np.corrcoef(data.T)[0, 1]
    corr_imputed = np.corrcoef(imputed.T)[0, 1]
    
    assert abs(corr_original - corr_imputed) < 0.05
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup
        uses: ./.github/actions/setup
      
      - name: Run Tests
        run: |
          pnpm test:all
          cargo test
          pytest
      
      - name: Upload Coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info,./tarpaulin-report.xml,./coverage.xml
```

### Pre-commit Hooks

```bash
# .husky/pre-commit
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

# Run quick tests
pnpm test:unit --run
cargo test --lib
pytest -m "not slow"
```

## Troubleshooting

### Common Issues

#### 1. Flaky Tests
- Use `waitFor` for async operations
- Increase timeouts for CI environments
- Mock time-dependent operations
- Use fixed random seeds

#### 2. Memory Issues
```rust
#[test]
fn test_large_data() {
    // Use streaming approach for large datasets
    let reader = CsvReader::from_path("large.csv")?;
    for batch in reader.batches(1000) {
        process_batch(batch?);
    }
}
```

#### 3. GPU Tests Failing
```python
@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")
def test_gpu_acceleration():
    # GPU-specific tests
    pass
```

### Debugging Tests

```bash
# Frontend debugging
pnpm test:debug

# Rust debugging
RUST_BACKTRACE=1 cargo test -- --nocapture

# Python debugging
pytest --pdb  # Drop into debugger on failure
```

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Arrange-Act-Assert**: Follow AAA pattern
3. **One Assertion Per Test**: Keep tests focused
4. **Mock External Services**: Don't rely on external APIs
5. **Deterministic Tests**: Avoid randomness without seeds
6. **Performance Budgets**: Set clear performance expectations
7. **Documentation**: Document why, not what

## Academic Validation

Our testing suite includes validation against published results:

1. **Kalman Filter**: Validated against Moritz et al. (2017)
2. **Random Forest**: Compared with scikit-learn reference
3. **Deep Learning**: Benchmarked against DeepAR paper
4. **RAH Method**: Validated against original implementation

Each method includes:
- Synthetic data tests
- Real-world São Paulo air quality data
- Cross-validation with multiple metrics
- Comparison with baseline methods

## Continuous Improvement

1. **Weekly Test Reviews**: Review failing and slow tests
2. **Monthly Benchmarks**: Track performance trends
3. **Quarterly Audits**: Full test suite audit
4. **Annual Validation**: Re-validate against latest research

---

For questions or contributions, please refer to our [Contributing Guide](CONTRIBUTING.md).