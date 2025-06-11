# AirImpute Pro Desktop - Practical User Guide

This guide tells you what actually works and how to use it without frustration.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Preparing Your Data](#preparing-your-data)
3. [Choosing an Imputation Method](#choosing-an-imputation-method)
4. [Running Imputation](#running-imputation)
5. [Understanding Results](#understanding-results)
6. [Common Problems & Solutions](#common-problems--solutions)
7. [Tips for Success](#tips-for-success)

## Getting Started

### First Time Setup

1. **Install the app** (see main README)
2. **Prepare your data** as CSV (Excel conversion coming later)
3. **Start small** - test with 1 month of data first
4. **Expect some hiccups** - this is research software

### What You Need

- Air quality data in CSV format
- Column for datetime (various formats work)
- Columns for pollutants (PM2.5, PM10, O3, etc.)
- At least 10% non-missing data (more is better)

## Preparing Your Data

### CSV Format That Works

```csv
datetime,PM25,PM10,O3,NO2
2024-01-01 00:00,23.5,45.2,67.8,34.1
2024-01-01 01:00,,42.1,65.3,32.5
2024-01-01 02:00,21.3,,64.9,
2024-01-01 03:00,20.8,38.7,62.1,30.2
```

### Data Checklist

- ✅ **Datetime formats supported**: ISO, US, EU, most common formats
- ✅ **Missing values**: Empty cells, 'NaN', 'NA', '-999' all work
- ✅ **Units**: Doesn't matter, but be consistent
- ❌ **Multiple stations**: Not yet - process separately
- ❌ **Irregular timestamps**: Convert to regular intervals first

### Pre-processing Tips

```python
# Quick data prep script
import pandas as pd

# Load and clean
df = pd.read_csv('your_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')
df = df.drop_duplicates(subset=['datetime'])

# Check missing percentage
print(f"Missing: {df.isnull().sum() / len(df) * 100}")

# Save clean version
df.to_csv('clean_data.csv', index=False)
```

## Choosing an Imputation Method

### Quick Decision Tree

```
Missing < 5% and gaps < 6 hours?
  → Use Linear Interpolation (fast, good enough)

Missing 5-20% with regular patterns?
  → Use Random Forest (best balance)

Missing 20-40% or complex patterns?
  → Try RAH or XGBoost (slower but better)

Missing > 40%?
  → Consider if imputation makes sense
  → Maybe try LSTM if you have years of data
```

### Method Comparison (Realistic)

| Method | Speed | Quality | When to Use | When to Avoid |
|--------|-------|---------|-------------|---------------|
| **Linear** | ⚡⚡⚡⚡⚡ | ⭐⭐ | Short regular gaps | Long gaps, trends |
| **Random Forest** | ⚡⚡⚡ | ⭐⭐⭐⭐ | **Default choice** | Real-time needs |
| **XGBoost** | ⚡⚡ | ⭐⭐⭐⭐ | Need best accuracy | Large datasets |
| **LSTM** | ⚡ | ⭐⭐⭐⭐⭐ | Complex patterns | < 1 year of data |
| **RAH** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Adaptive selection | Simple cases |

## Running Imputation

### Step-by-Step Process

1. **Import Your Data**
   - Click "Import CSV"
   - Select your cleaned file
   - Wait for preview (may take 10-30s for large files)

2. **Check Data Quality**
   - Look at the missing data summary
   - Check the gap distribution plot
   - If > 40% missing, reconsider

3. **Select Method**
   - Start with Random Forest
   - Only try complex methods if needed
   - Avoid deep learning unless you have lots of data

4. **Configure Settings** (Usually defaults are fine)
   ```
   Random Forest Settings:
   - Trees: 100 (default) - more is slightly better but slower
   - Max depth: None - let it figure out
   - Min samples: 5 - prevents overfitting
   ```

5. **Run Imputation**
   - Click "Start Imputation"
   - **Small dataset (< 10K points)**: 5-30 seconds
   - **Medium (10K-100K)**: 1-5 minutes
   - **Large (100K-1M)**: Get coffee ☕

6. **Check Results**
   - Look at before/after plots
   - Check imputed vs. original statistics
   - Verify no crazy values

### What the Progress Bar Actually Means

- 0-10%: Loading and preprocessing
- 10-20%: Analyzing patterns
- 20-80%: Actually imputing (may seem stuck)
- 80-90%: Calculating metrics
- 90-100%: Preparing visualizations

## Understanding Results

### Key Metrics Explained

**MAE (Mean Absolute Error)**: Average error in your units
- < 5 µg/m³: Excellent
- 5-10 µg/m³: Good
- 10-15 µg/m³: Acceptable
- > 15 µg/m³: Consider different method

**RMSE**: Like MAE but penalizes large errors more

**R²**: How well patterns are preserved (0-1, higher better)

### Visual Checks

1. **Time Series Plot**: Imputed values should follow trends
2. **Distribution Plot**: Shape should be similar to original
3. **Scatter Plot**: Points should cluster around diagonal

### Red Flags 🚩

- Imputed values all the same (method failed)
- Sudden spikes or drops at gap boundaries
- Negative values for concentrations
- Values way outside normal range

## Common Problems & Solutions

### Problem: "Out of Memory"
**Solution**: 
- Close other applications
- Try processing in chunks (split by year/month)
- Use simpler method (Linear/RF instead of LSTM)
- Upgrade to 16GB RAM

### Problem: "Python Error" / App Crashes
**Solution**:
- Check Python environment is activated
- Verify all packages installed correctly
- Look for specific error in console
- Restart app and try again

### Problem: Very Slow Performance
**Solution**:
- Start with subset of data
- Disable GPU if having issues
- Use Random Forest instead of deep learning
- Check CPU/RAM usage in Task Manager

### Problem: Poor Imputation Quality
**Solution**:
- Check your data quality first
- Try different methods
- Look at gap patterns - some are impossible
- Consider domain knowledge

### Problem: Can't Export Results
**Solution**:
- Check disk space
- Try different export format
- Save to different location
- Manual copy from results table

## Tips for Success

### Do's ✅

1. **Start Simple**: Test with 1 month before processing years
2. **Check Your Data**: Garbage in = garbage out
3. **Use Random Forest**: It's reliable for most cases
4. **Validate Results**: Always sanity-check imputed values
5. **Keep Originals**: Never overwrite raw data

### Don'ts ❌

1. **Don't impute > 40% missing**: Results won't be meaningful
2. **Don't trust blindly**: All methods have limitations
3. **Don't use for compliance**: This is for research only
4. **Don't expect miracles**: Can't create information from nothing
5. **Don't ignore warnings**: They're there for a reason

### Performance Tips

```python
# For large datasets, preprocess:
# 1. Remove unnecessary columns
# 2. Downsample if possible (hourly → daily)
# 3. Process in chunks by year
# 4. Use simpler methods first
```

### Validation Strategy

Always hold out some real data to test:
1. Artificially create gaps in complete sections
2. Run imputation
3. Compare imputed vs. actual
4. This gives realistic performance estimates

## Advanced Usage

### Batch Processing

```python
# Process multiple files
import os
from airimpute import ImputationEngine

engine = ImputationEngine()

for file in os.listdir('data/'):
    if file.endswith('.csv'):
        data = pd.read_csv(f'data/{file}')
        result = engine.impute(data, method='random_forest')
        result.to_csv(f'imputed/{file}')
```

### Custom Validation

```python
# Create custom validation splits
from airimpute.validation import TimeSeriesSplit

# Don't test on future data
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(data):
    # Validation code here
    pass
```

## Getting Help

1. **Check the console**: Error messages are often helpful
2. **GitHub Issues**: Include data sample if possible
3. **Community Forum**: Share what worked for you
4. **Email**: For sensitive data questions

Remember: This is research software. It's powerful but not perfect. Your expertise in interpreting results is crucial.

---

*Last updated: 2024*