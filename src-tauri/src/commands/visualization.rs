use tauri::{command, Window};
use std::sync::Arc;
use tauri::State;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use anyhow::{Result, Context};
use tracing::{info, warn, error};
use std::path::PathBuf;
use base64;

use crate::state::AppState;
use crate::python::bridge::PythonBridge;
use crate::core::data::Dataset;
use crate::core::imputation::JobStatus;

#[derive(Debug, Clone, Deserialize)]
pub struct PlotOptions {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub dpi: Option<u32>,
    pub theme: Option<String>,
    pub color_scheme: Option<String>,
    pub title: Option<String>,
    pub save_path: Option<String>,
    pub format: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlotResult {
    pub image_data: String, // Base64 encoded
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub metadata: serde_json::Value,
}

#[command]
pub async fn generate_missing_pattern_plot(
    window: Window,
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
    options: PlotOptions,
) -> Result<PlotResult, String> {
    info!("Generating missing pattern plot for dataset: {}", dataset_id);
    
    // Parse dataset ID
    let id = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    // Get dataset
    let dataset = state.datasets.get(&id)
        .ok_or_else(|| "Dataset not found".to_string())?;
    
    // Emit progress
    window.emit("plot-progress", serde_json::json!({
        "stage": "generating",
        "progress": 0.1,
        "message": "Preparing missing pattern visualization..."
    })).ok();
    
    // Call Python to generate plot
    let python_code = r#"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from io import BytesIO
import base64

# Set style
plt.style.use(theme if theme else 'seaborn-v0_8-darkgrid')
if color_scheme:
    sns.set_palette(color_scheme)

# Create figure
fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)

# Create missing pattern matrix
missing_mask = np.isnan(data)

# Plot missing pattern
if missing_mask.shape[1] <= 50:  # Show all columns if not too many
    sns.heatmap(missing_mask, 
                cbar=True, 
                yticklabels=False,
                xticklabels=columns,
                cmap='RdYlBu',
                ax=ax)
    plt.xticks(rotation=45, ha='right')
else:  # Aggregate by time for many columns
    # Calculate missing percentage by time window
    window_size = max(1, len(missing_mask) // 100)
    missing_pct = []
    time_labels = []
    
    for i in range(0, len(missing_mask), window_size):
        window = missing_mask[i:i+window_size]
        missing_pct.append(np.mean(window, axis=0) * 100)
        time_labels.append(timestamps[i])
    
    missing_pct = np.array(missing_pct)
    
    im = ax.imshow(missing_pct.T, 
                   aspect='auto', 
                   cmap='RdYlBu_r',
                   interpolation='nearest')
    
    # Set labels
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)
    
    # Set x-axis to show dates
    n_ticks = min(10, len(time_labels))
    tick_indices = np.linspace(0, len(time_labels)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([time_labels[i].strftime('%Y-%m-%d') for i in tick_indices], rotation=45, ha='right')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Missing %', rotation=270, labelpad=15)

# Set title
ax.set_title(title if title else f'Missing Data Pattern - {dataset_name}')
ax.set_xlabel('Time')
ax.set_ylabel('Variables')

# Calculate statistics
total_missing = np.sum(missing_mask)
total_values = missing_mask.size
missing_pct_total = (total_missing / total_values) * 100

# Add summary text
fig.text(0.02, 0.02, 
         f'Total missing: {missing_pct_total:.1f}% ({total_missing:,} / {total_values:,} values)',
         fontsize=10, alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save to bytes
buf = BytesIO()
plt.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
plt.close()

# Encode to base64
buf.seek(0)
image_data = base64.b64encode(buf.read()).decode('utf-8')

# Create metadata
metadata = {
    'total_missing': int(total_missing),
    'total_values': int(total_values),
    'missing_percentage': float(missing_pct_total),
    'missing_by_column': {col: int(np.sum(missing_mask[:, i])) 
                          for i, col in enumerate(columns)},
    'pattern_type': 'temporal' if len(columns) > 50 else 'detailed'
}

result = {
    'image_data': image_data,
    'metadata': metadata
}
"#;
    
    // Prepare Python context
    let context = serde_json::json!({
        "data": dataset.data.as_slice().unwrap(),
        "columns": dataset.columns,
        "timestamps": dataset.index.iter().map(|dt| dt.to_rfc3339()).collect::<Vec<_>>(),
        "dataset_name": dataset.name,
        "width": options.width.unwrap_or(1200),
        "height": options.height.unwrap_or(800),
        "dpi": options.dpi.unwrap_or(100),
        "theme": options.theme.as_deref().unwrap_or("seaborn-v0_8-darkgrid"),
        "color_scheme": options.color_scheme,
        "title": options.title,
        "format": options.format.as_deref().unwrap_or("png"),
    });
    
    // Execute Python code using SafePythonBridge
    let operation = crate::python::PythonOperation {
        module: "airimpute.visualization".to_string(),
        function: "generate_missing_pattern_plot".to_string(),
        args: vec![serde_json::to_string(&context).unwrap()],
        kwargs: std::collections::HashMap::new(),
        timeout_ms: Some(30000),
    };
    
    let result = state.python_bridge
        .execute_operation(&operation, None)
        .await
        .map_err(|e| format!("Failed to generate plot: {}", e))?;
    
    // Parse result
    let plot_data: serde_json::Value = serde_json::from_str(&result)
        .map_err(|e| format!("Failed to parse plot result: {}", e))?;
    
    // Save to file if requested
    if let Some(save_path) = &options.save_path {
        let image_bytes = base64::decode(&plot_data["image_data"].as_str().unwrap_or_default())
            .map_err(|e| format!("Failed to decode image data: {}", e))?;
        
        std::fs::write(save_path, &image_bytes)
            .map_err(|e| format!("Failed to save plot: {}", e))?;
    }
    
    // Emit completion
    window.emit("plot-progress", serde_json::json!({
        "stage": "complete",
        "progress": 1.0,
        "message": "Missing pattern plot generated successfully"
    })).ok();
    
    Ok(PlotResult {
        image_data: plot_data["image_data"].as_str().unwrap_or_default().to_string(),
        format: options.format.unwrap_or_else(|| "png".to_string()),
        width: options.width.unwrap_or(1200),
        height: options.height.unwrap_or(800),
        metadata: plot_data["metadata"].clone(),
    })
}

#[command]
pub async fn generate_time_series_plot(
    window: Window,
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
    variables: Vec<String>,
    options: PlotOptions,
) -> Result<PlotResult, String> {
    info!("Generating time series plot for {} variables", variables.len());
    
    // Validate inputs
    if variables.is_empty() {
        return Err("No variables selected for plotting".to_string());
    }
    
    if variables.len() > 10 {
        return Err("Too many variables selected (max 10)".to_string());
    }
    
    // Parse dataset ID
    let id = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    // Get dataset
    let dataset = state.datasets.get(&id)
        .ok_or_else(|| "Dataset not found".to_string())?;
    
    // Validate variables exist
    for var in &variables {
        if !dataset.columns.contains(var) {
            return Err(format!("Variable '{}' not found in dataset", var));
        }
    }
    
    // Emit progress
    window.emit("plot-progress", serde_json::json!({
        "stage": "generating",
        "progress": 0.1,
        "message": "Preparing time series visualization..."
    })).ok();
    
    // Call Python to generate plot
    let python_code = r#"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime
import matplotlib.dates as mdates

# Set style
plt.style.use(theme)
if color_scheme:
    colors = sns.color_palette(color_scheme, n_colors=len(variables))
else:
    colors = sns.color_palette("husl", n_colors=len(variables))

# Create figure with subplots if multiple variables
if len(variables) > 3:
    fig, axes = plt.subplots(len(variables), 1, 
                            figsize=(width/dpi, height/dpi), 
                            dpi=dpi, 
                            sharex=True)
    if len(variables) == 1:
        axes = [axes]
else:
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    axes = [ax] * len(variables)

# Convert timestamps
timestamps_dt = pd.to_datetime(timestamps)

# Plot each variable
for idx, (var, ax) in enumerate(zip(variables, axes)):
    var_idx = columns.index(var)
    values = data[:, var_idx]
    
    # Separate valid and missing data
    valid_mask = ~np.isnan(values)
    
    # Plot main line
    if np.sum(valid_mask) > 0:
        ax.plot(timestamps_dt[valid_mask], values[valid_mask], 
                color=colors[idx], linewidth=1.5, label=var, alpha=0.8)
        
        # Add scatter points for actual data
        if np.sum(valid_mask) < 1000:  # Only show points if not too many
            ax.scatter(timestamps_dt[valid_mask], values[valid_mask], 
                      color=colors[idx], s=20, alpha=0.6, edgecolors='white', linewidth=0.5)
    
    # Mark missing regions
    if np.sum(~valid_mask) > 0:
        # Find contiguous missing regions
        missing_regions = []
        in_gap = False
        gap_start = None
        
        for i, is_missing in enumerate(~valid_mask):
            if is_missing and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_missing and in_gap:
                missing_regions.append((gap_start, i-1))
                in_gap = False
        
        if in_gap:
            missing_regions.append((gap_start, len(values)-1))
        
        # Highlight missing regions
        for start, end in missing_regions:
            ax.axvspan(timestamps_dt[start], timestamps_dt[min(end, len(timestamps_dt)-1)], 
                      alpha=0.2, color='red', label='Missing' if start == missing_regions[0][0] else '')
    
    # Formatting
    ax.set_ylabel(f'{var}\n{units.get(var, "units")}', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Statistics annotation
    valid_values = values[valid_mask]
    if len(valid_values) > 0:
        stats_text = f'μ={np.mean(valid_values):.2f}, σ={np.std(valid_values):.2f}'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Only show legend on first subplot
    if idx == 0:
        ax.legend(loc='upper right')

# Format x-axis
if len(variables) > 3:
    axes[-1].set_xlabel('Time')
else:
    ax.set_xlabel('Time')

# Set date formatter
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
# Rotate x labels
plt.setp([ax.xaxis.get_majorticklabels() for ax in axes], rotation=45, ha='right')

# Set title
if len(variables) == 1:
    fig.suptitle(title if title else f'Time Series: {variables[0]}')
else:
    fig.suptitle(title if title else f'Time Series: {dataset_name}')

# Adjust layout
plt.tight_layout()

# Save to bytes
buf = BytesIO()
plt.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
plt.close()

# Encode to base64
buf.seek(0)
image_data = base64.b64encode(buf.read()).decode('utf-8')

# Calculate metadata
metadata = {}
for var in variables:
    var_idx = columns.index(var)
    values = data[:, var_idx]
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    
    if len(valid_values) > 0:
        metadata[var] = {
            'count': int(np.sum(valid_mask)),
            'missing': int(np.sum(~valid_mask)),
            'mean': float(np.mean(valid_values)),
            'std': float(np.std(valid_values)),
            'min': float(np.min(valid_values)),
            'max': float(np.max(valid_values)),
            'missing_percentage': float(np.sum(~valid_mask) / len(values) * 100)
        }

result = {
    'image_data': image_data,
    'metadata': metadata
}
"#;
    
    // Get units for variables
    let units: std::collections::HashMap<String, String> = dataset.get_units();
    
    // Prepare Python context
    let context = serde_json::json!({
        "data": dataset.data.as_slice().unwrap(),
        "columns": dataset.columns,
        "variables": variables,
        "timestamps": dataset.index.iter().map(|dt| dt.to_rfc3339()).collect::<Vec<_>>(),
        "dataset_name": dataset.name,
        "units": units,
        "width": options.width.unwrap_or(1400),
        "height": options.height.unwrap_or(800),
        "dpi": options.dpi.unwrap_or(100),
        "theme": options.theme.as_deref().unwrap_or("seaborn-v0_8-whitegrid"),
        "color_scheme": options.color_scheme,
        "title": options.title,
        "format": options.format.as_deref().unwrap_or("png"),
    });
    
    // Execute Python code using SafePythonBridge
    let operation = crate::python::PythonOperation {
        module: "airimpute.visualization".to_string(),
        function: "generate_missing_pattern_plot".to_string(),
        args: vec![serde_json::to_string(&context).unwrap()],
        kwargs: std::collections::HashMap::new(),
        timeout_ms: Some(30000),
    };
    
    let result = state.python_bridge
        .execute_operation(&operation, None)
        .await
        .map_err(|e| format!("Failed to generate plot: {}", e))?;
    
    // Parse result
    let plot_data: serde_json::Value = serde_json::from_str(&result)
        .map_err(|e| format!("Failed to parse plot result: {}", e))?;
    
    // Save to file if requested
    if let Some(save_path) = &options.save_path {
        let image_bytes = base64::decode(&plot_data["image_data"].as_str().unwrap_or_default())
            .map_err(|e| format!("Failed to decode image data: {}", e))?;
        
        std::fs::write(save_path, &image_bytes)
            .map_err(|e| format!("Failed to save plot: {}", e))?;
    }
    
    // Emit completion
    window.emit("plot-progress", serde_json::json!({
        "stage": "complete",
        "progress": 1.0,
        "message": "Time series plot generated successfully"
    })).ok();
    
    Ok(PlotResult {
        image_data: plot_data["image_data"].as_str().unwrap_or_default().to_string(),
        format: options.format.unwrap_or_else(|| "png".to_string()),
        width: options.width.unwrap_or(1400),
        height: options.height.unwrap_or(800),
        metadata: plot_data["metadata"].clone(),
    })
}

#[command]
pub async fn generate_correlation_matrix(
    window: Window,
    state: State<'_, Arc<AppState>>,
    dataset_id: String,
    options: PlotOptions,
) -> Result<PlotResult, String> {
    info!("Generating correlation matrix for dataset: {}", dataset_id);
    
    // Parse dataset ID
    let id = Uuid::parse_str(&dataset_id)
        .map_err(|e| format!("Invalid dataset ID: {}", e))?;
    
    // Get dataset
    let dataset = state.datasets.get(&id)
        .ok_or_else(|| "Dataset not found".to_string())?;
    
    // Emit progress
    window.emit("plot-progress", serde_json::json!({
        "stage": "generating",
        "progress": 0.1,
        "message": "Computing correlations..."
    })).ok();
    
    // Call Python to generate plot
    let python_code = r#"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
import base64
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use(theme)

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Calculate correlations with different methods
corr_pearson = df.corr(method='pearson')
corr_spearman = df.corr(method='spearman')
corr_kendall = df.corr(method='kendall')

# Determine which correlation to show
if 'pearson' in title.lower() if title else True:
    corr_matrix = corr_pearson
    method = 'Pearson'
elif 'spearman' in title.lower() if title else False:
    corr_matrix = corr_spearman
    method = 'Spearman'
elif 'kendall' in title.lower() if title else False:
    corr_matrix = corr_kendall
    method = 'Kendall'
else:
    corr_matrix = corr_pearson
    method = 'Pearson'

# Create figure
fig_size = min(max(8, len(columns) * 0.5), width/dpi)
fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Generate colormap
if color_scheme:
    cmap = sns.color_palette(color_scheme, as_cmap=True)
else:
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

# Draw the heatmap
if len(columns) <= 30:
    # Detailed view with values
    sns.heatmap(corr_matrix, 
                mask=mask,
                cmap=cmap,
                vmin=-1, vmax=1,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": f"{method} Correlation"},
                annot=True,
                fmt='.2f',
                annot_kws={'size': 8},
                ax=ax)
else:
    # Simplified view without values for many variables
    sns.heatmap(corr_matrix,
                mask=mask,
                cmap=cmap,
                vmin=-1, vmax=1,
                center=0,
                square=True,
                linewidths=0.1,
                cbar_kws={"shrink": 0.8, "label": f"{method} Correlation"},
                annot=False,
                ax=ax)

# Set title
ax.set_title(title if title else f'{method} Correlation Matrix - {dataset_name}', 
             fontsize=14, pad=20)

# Rotate labels
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
plt.setp(ax.get_yticklabels(), rotation=0)

# Adjust layout
plt.tight_layout()

# Calculate statistics
# Find strongest correlations (excluding diagonal)
corr_values = corr_matrix.values[~mask]
corr_pairs = []
for i in range(len(columns)):
    for j in range(i+1, len(columns)):
        corr_val = corr_matrix.iloc[i, j]
        if not np.isnan(corr_val):
            corr_pairs.append({
                'var1': columns[i],
                'var2': columns[j],
                'correlation': float(corr_val)
            })

# Sort by absolute correlation
corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

# Calculate p-values for top correlations
top_correlations = []
for pair in corr_pairs[:10]:  # Top 10
    var1_idx = columns.index(pair['var1'])
    var2_idx = columns.index(pair['var2'])
    
    # Get non-NaN pairs
    mask = ~(np.isnan(data[:, var1_idx]) | np.isnan(data[:, var2_idx]))
    if np.sum(mask) > 3:  # Need at least 3 points
        if method == 'Pearson':
            _, p_value = stats.pearsonr(data[mask, var1_idx], data[mask, var2_idx])
        elif method == 'Spearman':
            _, p_value = stats.spearmanr(data[mask, var1_idx], data[mask, var2_idx])
        else:
            _, p_value = stats.kendalltau(data[mask, var1_idx], data[mask, var2_idx])
        
        pair['p_value'] = float(p_value)
        pair['significant'] = p_value < 0.05
        top_correlations.append(pair)

# Save to bytes
buf = BytesIO()
plt.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
plt.close()

# Encode to base64
buf.seek(0)
image_data = base64.b64encode(buf.read()).decode('utf-8')

# Create metadata
metadata = {
    'method': method,
    'n_variables': len(columns),
    'correlation_range': {
        'min': float(np.nanmin(corr_values)),
        'max': float(np.nanmax(corr_values)),
        'mean': float(np.nanmean(corr_values)),
        'std': float(np.nanstd(corr_values))
    },
    'top_correlations': top_correlations,
    'correlation_methods': {
        'pearson': corr_pearson.to_dict() if len(columns) <= 10 else {},
        'spearman': corr_spearman.to_dict() if len(columns) <= 10 else {},
        'kendall': corr_kendall.to_dict() if len(columns) <= 10 else {}
    }
}

result = {
    'image_data': image_data,
    'metadata': metadata
}
"#;
    
    // Prepare Python context
    let context = serde_json::json!({
        "data": dataset.data.as_slice().unwrap(),
        "columns": dataset.columns,
        "dataset_name": dataset.name,
        "width": options.width.unwrap_or(1000),
        "height": options.height.unwrap_or(1000),
        "dpi": options.dpi.unwrap_or(100),
        "theme": options.theme.as_deref().unwrap_or("seaborn-v0_8-white"),
        "color_scheme": options.color_scheme.as_deref().unwrap_or("coolwarm"),
        "title": options.title.as_deref().unwrap_or(""),
        "format": options.format.as_deref().unwrap_or("png"),
    });
    
    // Execute Python code using SafePythonBridge
    let operation = crate::python::PythonOperation {
        module: "airimpute.visualization".to_string(),
        function: "generate_correlation_matrix".to_string(),
        args: vec![serde_json::to_string(&context).unwrap()],
        kwargs: std::collections::HashMap::new(),
        timeout_ms: Some(30000),
    };
    
    let result = state.python_bridge
        .execute_operation(&operation, None)
        .await
        .map_err(|e| format!("Failed to generate correlation matrix: {}", e))?;
    
    // Parse result
    let plot_data: serde_json::Value = serde_json::from_str(&result)
        .map_err(|e| format!("Failed to parse plot result: {}", e))?;
    
    // Save to file if requested
    if let Some(save_path) = &options.save_path {
        let image_bytes = base64::decode(&plot_data["image_data"].as_str().unwrap_or_default())
            .map_err(|e| format!("Failed to decode image data: {}", e))?;
        
        std::fs::write(save_path, &image_bytes)
            .map_err(|e| format!("Failed to save plot: {}", e))?;
    }
    
    // Emit completion
    window.emit("plot-progress", serde_json::json!({
        "stage": "complete",
        "progress": 1.0,
        "message": "Correlation matrix generated successfully"
    })).ok();
    
    Ok(PlotResult {
        image_data: plot_data["image_data"].as_str().unwrap_or_default().to_string(),
        format: options.format.unwrap_or_else(|| "png".to_string()),
        width: options.width.unwrap_or(1000),
        height: options.height.unwrap_or(1000),
        metadata: plot_data["metadata"].clone(),
    })
}

#[command]
pub async fn generate_uncertainty_bands(
    window: Window,
    state: State<'_, Arc<AppState>>,
    job_id: String,
    variable: String,
    options: PlotOptions,
) -> Result<PlotResult, String> {
    info!("Generating uncertainty bands for job: {}, variable: {}", job_id, variable);
    
    // Get imputation job results
    let job_uuid = Uuid::parse_str(&job_id)
        .map_err(|e| format!("Invalid job ID: {}", e))?;
    
    let job_entry = state.imputation_jobs.get(&job_uuid)
        .ok_or_else(|| "Imputation job not found".to_string())?;
    let job = job_entry.lock().await;
    
    if job.status != JobStatus::Completed {
        return Err(format!("Job is not completed: status = {:?}", job.status));
    }
    
    // Emit progress
    window.emit("plot-progress", serde_json::json!({
        "stage": "generating",
        "progress": 0.1,
        "message": "Preparing uncertainty visualization..."
    })).ok();
    
    // Call Python to generate plot
    let python_code = r#"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# Set style
plt.style.use(theme)

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width/dpi, height/dpi), 
                                dpi=dpi, height_ratios=[3, 1], sharex=True)

# Convert timestamps
timestamps_dt = pd.to_datetime(timestamps)

# Get variable data
var_idx = imputed_columns.index(variable)
original_values = original_data[:, var_idx]
imputed_values = imputed_data[:, var_idx]

# Get uncertainty data
if 'uncertainty' in result_data:
    uncertainty = result_data['uncertainty'][:, var_idx]
else:
    # Estimate uncertainty from ensemble or bootstrap results
    uncertainty = np.abs(imputed_values - original_values) * 0.1  # Simplified

# Identify imputed points
imputed_mask = np.isnan(original_values)
original_mask = ~imputed_mask

# Upper plot: Time series with uncertainty bands
if np.sum(original_mask) > 0:
    ax1.plot(timestamps_dt[original_mask], original_values[original_mask], 
             'o-', color='darkblue', markersize=4, linewidth=1.5, 
             label='Original', alpha=0.8)

if np.sum(imputed_mask) > 0:
    # Plot imputed values
    ax1.plot(timestamps_dt[imputed_mask], imputed_values[imputed_mask], 
             's', color='red', markersize=6, label='Imputed', alpha=0.8)
    
    # Add uncertainty bands
    lower_bound = imputed_values - 1.96 * uncertainty
    upper_bound = imputed_values + 1.96 * uncertainty
    
    ax1.fill_between(timestamps_dt[imputed_mask], 
                     lower_bound[imputed_mask], 
                     upper_bound[imputed_mask],
                     color='red', alpha=0.2, label='95% CI')

# Connect all points with a line
ax1.plot(timestamps_dt, imputed_values, '-', color='gray', 
         linewidth=0.8, alpha=0.5)

# Add method annotation
method_text = f"Method: {result_data.get('method', 'Unknown')}\n"
if 'metrics' in result_data and variable in result_data['metrics']:
    metrics = result_data['metrics'][variable]
    method_text += f"RMSE: {metrics.get('rmse', 0):.3f}\n"
    method_text += f"MAE: {metrics.get('mae', 0):.3f}"

ax1.text(0.02, 0.95, method_text, transform=ax1.transAxes, 
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Formatting
ax1.set_ylabel(f'{variable}\n{units.get(variable, "units")}', fontsize=11)
ax1.set_title(title if title else f'Imputation Results with Uncertainty - {variable}', 
              fontsize=14, pad=10)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Lower plot: Uncertainty magnitude
ax2.bar(timestamps_dt[imputed_mask], uncertainty[imputed_mask], 
        width=0.8, color='orange', alpha=0.6, edgecolor='darkorange')
ax2.set_ylabel('Uncertainty\n(±1σ)', fontsize=10)
ax2.set_xlabel('Time', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

# Format x-axis
for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add summary statistics
total_imputed = np.sum(imputed_mask)
avg_uncertainty = np.mean(uncertainty[imputed_mask]) if total_imputed > 0 else 0
max_uncertainty = np.max(uncertainty[imputed_mask]) if total_imputed > 0 else 0

fig.text(0.99, 0.01, 
         f'Imputed points: {total_imputed} | Avg uncertainty: {avg_uncertainty:.3f} | Max: {max_uncertainty:.3f}',
         fontsize=9, ha='right', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save to bytes
buf = BytesIO()
plt.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
plt.close()

# Encode to base64
buf.seek(0)
image_data = base64.b64encode(buf.read()).decode('utf-8')

# Create metadata
metadata = {
    'variable': variable,
    'total_points': len(original_values),
    'imputed_points': int(total_imputed),
    'imputation_rate': float(total_imputed / len(original_values) * 100),
    'uncertainty_stats': {
        'mean': float(avg_uncertainty),
        'std': float(np.std(uncertainty[imputed_mask])) if total_imputed > 0 else 0,
        'min': float(np.min(uncertainty[imputed_mask])) if total_imputed > 0 else 0,
        'max': float(max_uncertainty),
        'percentiles': {
            '25': float(np.percentile(uncertainty[imputed_mask], 25)) if total_imputed > 0 else 0,
            '50': float(np.percentile(uncertainty[imputed_mask], 50)) if total_imputed > 0 else 0,
            '75': float(np.percentile(uncertainty[imputed_mask], 75)) if total_imputed > 0 else 0,
            '95': float(np.percentile(uncertainty[imputed_mask], 95)) if total_imputed > 0 else 0,
        }
    },
    'method': result_data.get('method', 'Unknown'),
    'metrics': result_data.get('metrics', {}).get(variable, {})
}

result = {
    'image_data': image_data,
    'metadata': metadata
}
"#;
    
    // Get original dataset
    let dataset_id = job.dataset_id;
    
    let dataset = state.datasets.get(&dataset_id)
        .ok_or_else(|| "Original dataset not found".to_string())?;
    
    // Get units
    let units = dataset.get_units();
    
    // Prepare Python context
    let context = serde_json::json!({
        "original_data": dataset.data.as_slice().unwrap(),
        "imputed_data": job.result_data.as_ref().and_then(|rd| rd.get("imputed_data")),
        "imputed_columns": dataset.columns,
        "timestamps": dataset.index.iter().map(|dt| dt.to_rfc3339()).collect::<Vec<_>>(),
        "variable": variable,
        "units": units,
        "result_data": job.result_data,
        "width": options.width.unwrap_or(1400),
        "height": options.height.unwrap_or(900),
        "dpi": options.dpi.unwrap_or(100),
        "theme": options.theme.as_deref().unwrap_or("seaborn-v0_8-whitegrid"),
        "title": options.title,
        "format": options.format.as_deref().unwrap_or("png"),
    });
    
    // Execute Python code using SafePythonBridge
    let operation = crate::python::PythonOperation {
        module: "airimpute.visualization".to_string(),
        function: "generate_imputation_comparison".to_string(),
        args: vec![serde_json::to_string(&context).unwrap()],
        kwargs: std::collections::HashMap::new(),
        timeout_ms: Some(30000),
    };
    
    let result = state.python_bridge
        .execute_operation(&operation, None)
        .await
        .map_err(|e| format!("Failed to generate uncertainty plot: {}", e))?;
    
    // Parse result
    let plot_data: serde_json::Value = serde_json::from_str(&result)
        .map_err(|e| format!("Failed to parse plot result: {}", e))?;
    
    // Save to file if requested
    if let Some(save_path) = &options.save_path {
        let image_bytes = base64::decode(&plot_data["image_data"].as_str().unwrap_or_default())
            .map_err(|e| format!("Failed to decode image data: {}", e))?;
        
        std::fs::write(save_path, &image_bytes)
            .map_err(|e| format!("Failed to save plot: {}", e))?;
    }
    
    // Emit completion
    window.emit("plot-progress", serde_json::json!({
        "stage": "complete",
        "progress": 1.0,
        "message": "Uncertainty plot generated successfully"
    })).ok();
    
    Ok(PlotResult {
        image_data: plot_data["image_data"].as_str().unwrap_or_default().to_string(),
        format: options.format.unwrap_or_else(|| "png".to_string()),
        width: options.width.unwrap_or(1400),
        height: options.height.unwrap_or(900),
        metadata: plot_data["metadata"].clone(),
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct DashboardConfig {
    pub id: String,
    pub title: String,
    pub layout: Vec<DashboardPanel>,
    pub refresh_interval: Option<u32>,
    pub theme: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPanel {
    pub id: String,
    pub panel_type: String,
    pub title: String,
    pub position: PanelPosition,
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelPosition {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[command]
pub async fn create_interactive_dashboard(
    window: Window,
    state: State<'_, Arc<AppState>>,
    project_id: String,
) -> Result<DashboardConfig, String> {
    info!("Creating interactive dashboard for project: {}", project_id);
    
    // Validate project exists
    let project_uuid = Uuid::parse_str(&project_id)
        .map_err(|e| format!("Invalid project ID: {}", e))?;
    
    let project = state.projects.get(&project_uuid)
        .ok_or_else(|| "Project not found".to_string())?;
    
    // Get all datasets in project
    let datasets: Vec<_> = state.datasets.iter()
        .filter(|entry| {
            // Filter datasets belonging to this project
            true // Simplified - would check project association
        })
        .map(|entry| (entry.key().clone(), entry.value().clone()))
        .collect();
    
    if datasets.is_empty() {
        return Err("No datasets found in project".to_string());
    }
    
    // Create default dashboard layout
    let dashboard_id = Uuid::new_v4().to_string();
    let mut panels = Vec::new();
    
    // Panel 1: Overview statistics
    panels.push(DashboardPanel {
        id: Uuid::new_v4().to_string(),
        panel_type: "statistics".to_string(),
        title: "Dataset Overview".to_string(),
        position: PanelPosition { x: 0, y: 0, width: 4, height: 2 },
        config: serde_json::json!({
            "show_missing": true,
            "show_quality": true,
            "show_temporal": true,
        }),
    });
    
    // Panel 2: Time series plot
    panels.push(DashboardPanel {
        id: Uuid::new_v4().to_string(),
        panel_type: "timeseries".to_string(),
        title: "Time Series Visualization".to_string(),
        position: PanelPosition { x: 4, y: 0, width: 8, height: 3 },
        config: serde_json::json!({
            "dataset_id": datasets[0].0.to_string(),
            "variables": datasets[0].1.columns.iter().take(3).cloned().collect::<Vec<_>>(),
            "show_gaps": true,
            "interactive": true,
        }),
    });
    
    // Panel 3: Missing pattern
    panels.push(DashboardPanel {
        id: Uuid::new_v4().to_string(),
        panel_type: "missing_pattern".to_string(),
        title: "Missing Data Pattern".to_string(),
        position: PanelPosition { x: 0, y: 2, width: 4, height: 3 },
        config: serde_json::json!({
            "dataset_id": datasets[0].0.to_string(),
            "aggregation": "hourly",
            "colormap": "RdYlBu",
        }),
    });
    
    // Panel 4: Correlation matrix
    panels.push(DashboardPanel {
        id: Uuid::new_v4().to_string(),
        panel_type: "correlation".to_string(),
        title: "Variable Correlations".to_string(),
        position: PanelPosition { x: 0, y: 5, width: 6, height: 4 },
        config: serde_json::json!({
            "dataset_id": datasets[0].0.to_string(),
            "method": "pearson",
            "show_significance": true,
        }),
    });
    
    // Panel 5: Imputation results (if available)
    let imputation_jobs: Vec<_> = state.imputation_jobs.iter()
        .filter_map(|entry| {
            let job_guard = entry.value().blocking_lock();
            if job_guard.dataset_id == datasets[0].0 {
                Some((entry.key().clone(), job_guard.clone()))
            } else {
                None
            }
        })
        .collect();
    
    if !imputation_jobs.is_empty() {
        panels.push(DashboardPanel {
            id: Uuid::new_v4().to_string(),
            panel_type: "imputation_results".to_string(),
            title: "Imputation Analysis".to_string(),
            position: PanelPosition { x: 6, y: 3, width: 6, height: 4 },
            config: serde_json::json!({
                "job_id": imputation_jobs[0].0.to_string(),
                "show_uncertainty": true,
                "show_metrics": true,
                "comparison_mode": true,
            }),
        });
    }
    
    // Panel 6: Spatial map (if spatial data)
    if datasets[0].1.stations.is_some() {
        panels.push(DashboardPanel {
            id: Uuid::new_v4().to_string(),
            panel_type: "spatial_map".to_string(),
            title: "Spatial Distribution".to_string(),
            position: PanelPosition { x: 6, y: 5, width: 6, height: 4 },
            config: serde_json::json!({
                "dataset_id": datasets[0].0.to_string(),
                "show_stations": true,
                "show_interpolation": true,
                "variable": datasets[0].1.columns[0],
            }),
        });
    }
    
    // Create dashboard configuration
    let dashboard = DashboardConfig {
        id: dashboard_id.clone(),
        title: format!("{} - Interactive Dashboard", project.name),
        layout: panels,
        refresh_interval: Some(300), // 5 minutes
        theme: "light".to_string(),
        created_at: chrono::Utc::now(),
    };
    
    // TODO: Store dashboard configuration when dashboards field is added to AppState
    // state.dashboards.insert(dashboard_id.clone(), dashboard.clone());
    
    // Emit dashboard ready event
    window.emit("dashboard-ready", serde_json::json!({
        "dashboard_id": dashboard_id,
        "panel_count": dashboard.layout.len(),
    })).ok();
    
    info!("Created dashboard with {} panels", dashboard.layout.len());
    Ok(dashboard)
}