use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Offline resource manager for complete offline functionality
pub struct OfflineResourceManager {
    help_content: HashMap<String, HelpContent>,
    method_docs: HashMap<String, MethodDocumentation>,
    tutorials: HashMap<String, Tutorial>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelpContent {
    pub title: String,
    pub content: String,
    pub category: String,
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodDocumentation {
    pub name: String,
    pub description: String,
    pub mathematical_formulation: String,
    pub parameters: Vec<ParameterDoc>,
    pub complexity: String,
    pub references: Vec<String>,
    pub example_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDoc {
    pub name: String,
    pub description: String,
    pub type_info: String,
    pub default_value: String,
    pub constraints: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tutorial {
    pub title: String,
    pub description: String,
    pub steps: Vec<TutorialStep>,
    pub difficulty: String,
    pub estimated_time: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialStep {
    pub title: String,
    pub content: String,
    pub code_example: Option<String>,
    pub expected_output: Option<String>,
}

impl OfflineResourceManager {
    pub fn new() -> Self {
        let mut manager = Self {
            help_content: HashMap::new(),
            method_docs: HashMap::new(),
            tutorials: HashMap::new(),
        };
        
        // Initialize with embedded content
        manager.initialize_help_content();
        manager.initialize_method_docs();
        manager.initialize_tutorials();
        
        manager
    }
    
    fn initialize_help_content(&mut self) {
        // Getting Started
        self.help_content.insert(
            "getting-started".to_string(),
            HelpContent {
                title: "Getting Started with AirImpute Pro".to_string(),
                content: r#"
# Getting Started with AirImpute Pro

AirImpute Pro is a professional desktop application for air quality data imputation that works completely offline.

## Quick Start

1. **Import Data**: Click "Import Data" to load your air quality dataset (CSV, Excel, NetCDF, HDF5)
2. **Analyze**: View missing data patterns and statistics
3. **Choose Method**: Select an imputation method based on your data characteristics
4. **Configure**: Adjust parameters for optimal results
5. **Run**: Execute imputation and review results
6. **Export**: Save imputed data in your preferred format

## Key Features

- **100% Offline**: All processing happens locally on your machine
- **No Authentication**: No login or internet connection required
- **Academic Rigor**: All methods include proper citations and validation
- **High Performance**: GPU acceleration available for deep learning methods
- **Reproducible**: Generate certificates for reproducibility

## Data Privacy

Your data never leaves your computer. All processing is done locally with no external communication.
"#.to_string(),
                category: "basics".to_string(),
                keywords: vec!["start".to_string(), "begin".to_string(), "introduction".to_string()],
            }
        );
        
        // Data Formats
        self.help_content.insert(
            "data-formats".to_string(),
            HelpContent {
                title: "Supported Data Formats".to_string(),
                content: r#"
# Supported Data Formats

## Input Formats

### CSV (Comma-Separated Values)
- Standard tabular format
- Configurable delimiters (comma, tab, semicolon)
- Header row required
- Date/time column for temporal data

### Excel (.xlsx, .xls)
- Single or multiple sheets
- Named ranges supported
- Preserves formatting on export

### NetCDF
- Scientific data format
- Multidimensional arrays
- Metadata preservation
- CF conventions supported

### HDF5
- Hierarchical data format
- Large dataset support
- Compressed storage
- Complex data structures

## Data Structure Requirements

1. **Tabular Format**: Rows represent time points, columns represent variables
2. **Time Column**: ISO format (YYYY-MM-DD HH:MM:SS) recommended
3. **Missing Values**: Represented as NaN, NA, null, or empty cells
4. **Variable Names**: Alphanumeric, no special characters

## Export Formats

All input formats can be exported, plus:
- LaTeX tables for publications
- R/Python scripts for reproducibility
- JSON for web applications
"#.to_string(),
                category: "data".to_string(),
                keywords: vec!["csv".to_string(), "excel".to_string(), "netcdf".to_string(), "hdf5".to_string()],
            }
        );
        
        // Offline Mode
        self.help_content.insert(
            "offline-mode".to_string(),
            HelpContent {
                title: "Offline Operation".to_string(),
                content: r#"
# Offline Operation

AirImpute Pro is designed to work completely offline for maximum data security and privacy.

## What This Means

- **No Internet Required**: The application works without any network connection
- **No Updates**: Updates must be manually downloaded and installed
- **No Cloud Features**: All data storage and processing is local
- **No Telemetry**: No usage data is collected or transmitted

## Benefits

1. **Data Security**: Your sensitive data never leaves your computer
2. **Compliance**: Meet strict data governance requirements
3. **Performance**: No network latency or bandwidth limitations
4. **Reliability**: Work anywhere without connectivity concerns

## Local Resources

All documentation, examples, and references are embedded in the application:
- Press F1 for context-sensitive help
- Access tutorials from the Help menu
- View method documentation in the Method Selection screen

## Updates

To update AirImpute Pro:
1. Download the latest version from official sources
2. Close the current application
3. Install the new version
4. Your data and settings will be preserved
"#.to_string(),
                category: "features".to_string(),
                keywords: vec!["offline".to_string(), "privacy".to_string(), "security".to_string()],
            }
        );
    }
    
    fn initialize_method_docs(&mut self) {
        // Mean Imputation
        self.method_docs.insert(
            "mean".to_string(),
            MethodDocumentation {
                name: "Mean Imputation".to_string(),
                description: "Replace missing values with the arithmetic mean of observed values.".to_string(),
                mathematical_formulation: r"x̂ᵢ = (1/n) Σⱼ xⱼ where j ∈ observed".to_string(),
                parameters: vec![],
                complexity: "O(n)".to_string(),
                references: vec![
                    "Little, R.J.A. and Rubin, D.B. (2019). Statistical Analysis with Missing Data, 3rd Edition.".to_string()
                ],
                example_code: r#"
# Example usage
imputed_data = impute_mean(data)
"#.to_string(),
            }
        );
        
        // KNN Imputation
        self.method_docs.insert(
            "knn".to_string(),
            MethodDocumentation {
                name: "K-Nearest Neighbors Imputation".to_string(),
                description: "Impute missing values using weighted average of k nearest neighbors.".to_string(),
                mathematical_formulation: r"x̂ᵢ = Σⱼ wⱼxⱼ / Σⱼ wⱼ where j ∈ k-nearest neighbors".to_string(),
                parameters: vec![
                    ParameterDoc {
                        name: "k".to_string(),
                        description: "Number of neighbors to consider".to_string(),
                        type_info: "integer".to_string(),
                        default_value: "5".to_string(),
                        constraints: "1 ≤ k ≤ n-1".to_string(),
                    },
                    ParameterDoc {
                        name: "weights".to_string(),
                        description: "Weight function for neighbors".to_string(),
                        type_info: "string".to_string(),
                        default_value: "distance".to_string(),
                        constraints: "uniform, distance".to_string(),
                    }
                ],
                complexity: "O(n²)".to_string(),
                references: vec![
                    "Troyanskaya, O. et al. (2001). Missing value estimation methods for DNA microarrays. Bioinformatics, 17(6), 520-525.".to_string()
                ],
                example_code: r#"
# Example usage
imputed_data = impute_knn(data, k=5, weights='distance')
"#.to_string(),
            }
        );
        
        // Transformer Imputation
        self.method_docs.insert(
            "transformer".to_string(),
            MethodDocumentation {
                name: "Transformer Imputation (ImputeFormer)".to_string(),
                description: "State-of-the-art deep learning approach using transformer architecture with low-rank attention.".to_string(),
                mathematical_formulation: r"Attention(Q,K,V) = softmax(QK^T/√d)V with low-rank decomposition".to_string(),
                parameters: vec![
                    ParameterDoc {
                        name: "context_length".to_string(),
                        description: "Length of input sequence context".to_string(),
                        type_info: "integer".to_string(),
                        default_value: "96".to_string(),
                        constraints: "24 ≤ context_length ≤ 512".to_string(),
                    },
                    ParameterDoc {
                        name: "hidden_dims".to_string(),
                        description: "Hidden layer dimensions".to_string(),
                        type_info: "array".to_string(),
                        default_value: "[128, 256, 128]".to_string(),
                        constraints: "Each dim > 0".to_string(),
                    }
                ],
                complexity: "O(n²·d) where n is sequence length, d is embedding dimension".to_string(),
                references: vec![
                    "Nie, T. et al. (2023). ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation. arXiv:2312.01728.".to_string()
                ],
                example_code: r#"
# Example usage (GPU recommended)
imputed_data = impute_transformer(
    data, 
    context_length=96,
    epochs=100,
    use_gpu=True
)
"#.to_string(),
            }
        );
    }
    
    fn initialize_tutorials(&mut self) {
        // Basic Tutorial
        self.tutorials.insert(
            "basic-imputation".to_string(),
            Tutorial {
                title: "Basic Data Imputation".to_string(),
                description: "Learn how to perform basic imputation on air quality data".to_string(),
                difficulty: "Beginner".to_string(),
                estimated_time: "10 minutes".to_string(),
                steps: vec![
                    TutorialStep {
                        title: "Load Sample Data".to_string(),
                        content: "Start by loading the included sample air quality dataset.".to_string(),
                        code_example: Some("File -> Import Data -> Select 'samples/air_quality_sample.csv'".to_string()),
                        expected_output: Some("Dataset loaded: 1000 rows, 10 columns, 15% missing".to_string()),
                    },
                    TutorialStep {
                        title: "Analyze Missing Patterns".to_string(),
                        content: "Examine the missing data patterns to choose appropriate method.".to_string(),
                        code_example: Some("Analysis -> Missing Pattern Analysis".to_string()),
                        expected_output: Some("Pattern identified: Random missing (MCAR)".to_string()),
                    },
                    TutorialStep {
                        title: "Select Imputation Method".to_string(),
                        content: "For random missing data, KNN imputation works well.".to_string(),
                        code_example: Some("Imputation -> Select Method -> K-Nearest Neighbors".to_string()),
                        expected_output: None,
                    },
                    TutorialStep {
                        title: "Configure Parameters".to_string(),
                        content: "Set k=5 for balanced accuracy and speed.".to_string(),
                        code_example: Some("Set k = 5, weights = 'distance'".to_string()),
                        expected_output: None,
                    },
                    TutorialStep {
                        title: "Run Imputation".to_string(),
                        content: "Execute the imputation process.".to_string(),
                        code_example: Some("Click 'Run Imputation'".to_string()),
                        expected_output: Some("Imputation complete: RMSE = 0.023".to_string()),
                    },
                ],
            }
        );
    }
    
    pub fn search_help(&self, query: &str) -> Vec<&HelpContent> {
        let query_lower = query.to_lowercase();
        self.help_content
            .values()
            .filter(|content| {
                content.title.to_lowercase().contains(&query_lower) ||
                content.content.to_lowercase().contains(&query_lower) ||
                content.keywords.iter().any(|k| k.to_lowercase().contains(&query_lower))
            })
            .collect()
    }
    
    pub fn get_method_doc(&self, method_id: &str) -> Option<&MethodDocumentation> {
        self.method_docs.get(method_id)
    }
    
    pub fn get_tutorial(&self, tutorial_id: &str) -> Option<&Tutorial> {
        self.tutorials.get(tutorial_id)
    }
    
    pub fn list_tutorials(&self) -> Vec<(&String, &Tutorial)> {
        self.tutorials.iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_offline_resources() {
        let manager = OfflineResourceManager::new();
        
        // Test help search
        let results = manager.search_help("offline");
        assert!(!results.is_empty());
        
        // Test method documentation
        let knn_doc = manager.get_method_doc("knn");
        assert!(knn_doc.is_some());
        
        // Test tutorials
        let tutorials = manager.list_tutorials();
        assert!(!tutorials.is_empty());
    }
}