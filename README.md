# Universal ML Analysis Suite

A comprehensive, dataset-agnostic machine learning analysis toolkit for classification, regression, and clustering tasks. Works with any CSV dataset!

## Overview

This project performs advanced machine learning analysis on CSV datasets, including:
- **Classification**: Predicting categorical outcomes
- **Regression**: Predicting continuous values
- **Clustering**: Discovering natural groupings in data

The suite is designed to work with **any CSV dataset** - simply configure your dataset details and run the analysis.

## Key Features

✅ **Dataset Agnostic** - Works with any CSV file  
✅ **Memory Optimized** - Handles large datasets efficiently  
✅ **Hybrid Interface** - Both interactive menu and command-line  
✅ **Automatic Preprocessing** - Handles missing data and encoding  
✅ **Multiple Algorithms** - Comprehensive ML algorithm coverage  
✅ **Easy Configuration** - Simple setup for new datasets  

## Quick Start

### 1. Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Basic Usage (with diamonds dataset)

```bash
# Interactive mode
python main.py

# EDA only
python exploratory_data_analysis.py

# Command-line mode for ML tasks
python main.py --task classification
python main.py --task regression
python main.py --task clustering
python main.py --task all
```

### 3. Use Your Own Dataset

```bash
# Create configuration template
python main.py --create-config

# Edit dataset_config.py with your dataset details
# Then run analysis
python main.py
```

## Dataset Configuration

### Method 1: Configuration File (Recommended)

Create `dataset_config.py`:

```python
DATASET_CONFIG = {
    'dataset_file': 'your_dataset.csv',
    'target_column': 'your_target_column',
    'columns_to_drop': ['id', 'unnecessary_columns'],
    'categorical_features': ['category1', 'category2']
}
```

### Method 2: Default Setup

Place your CSV file as `diamonds.csv` with target column `price`.

## Example Configurations

### Iris Dataset
```python
DATASET_CONFIG = {
    'dataset_file': 'iris.csv',
    'target_column': 'species',
    'columns_to_drop': [],
    'categorical_features': []
}
```

### House Prices Dataset  
```python
DATASET_CONFIG = {
    'dataset_file': 'house_prices.csv',
    'target_column': 'SalePrice',
    'columns_to_drop': ['Id'],
    'categorical_features': ['Neighborhood', 'HouseStyle', 'SaleType']
}
```

### Titanic Dataset
```python
DATASET_CONFIG = {
    'dataset_file': 'titanic.csv',
    'target_column': 'Survived',
    'columns_to_drop': ['PassengerId', 'Name', 'Ticket'],
    'categorical_features': ['Sex', 'Embarked', 'Pclass']
}
```

## Usage

### Two-Step Workflow (Recommended)

**Step 1: Exploratory Data Analysis (EDA)**
```bash
# Generate all data visualizations and insights
python exploratory_data_analysis.py
```

**Step 2: Machine Learning Analysis**
```bash
# Run clean ML tasks without repetitive plots
python main.py --task all
```

### Interactive Mode

Launch the menu interface:
```bash
python main.py
```

Display:
```
🤖 UNIVERSAL ML ANALYSIS SUITE
==================================================
📄 Current dataset: your_dataset.csv
🎯 Target column: target_name
==================================================
1. Run Classification
2. Run Regression
3. Run Clustering
4. Run All Tasks
5. Exit
==================================================
```

### Command-Line Mode

```bash
# Run specific ML tasks (clean output, focused on results)
python main.py --task classification
python main.py --task regression
python main.py --task clustering

# Run everything
python main.py --task all

# Create config template
python main.py --create-config

# Generate data visualizations separately
python exploratory_data_analysis.py
```

## Analysis Components

### 1. Exploratory Data Analysis (`exploratory_data_analysis.py`)
- **Purpose**: Data visualization and exploration
- **Features**: Comprehensive plotting and data insights
- **Output**: Correlation plots, histograms, pair plots, box plots
- **When to Use**: Run first to understand your data
- **Command**: `python exploratory_data_analysis.py`

### 2. Classification Analysis (`classifier.py`)
- **Purpose**: Predict categorical outcomes
- **Algorithms**: Logistic Regression, KNN, Decision Tree, Random Forest, SVC, ANN
- **Output**: Clean results summary with accuracy comparison table
- **Best For**: Category prediction (species, quality grades, etc.)
- **Command**: `python main.py --task classification`

### 3. Regression Analysis (`regressor.py`) 
- **Purpose**: Predict continuous values
- **Algorithms**: Linear Regression, KNN, Decision Tree, Random Forest, SVR, ANN
- **Output**: Clean results summary with R² scores and RMSE comparison table
- **Best For**: Price prediction, score estimation, etc.
- **Command**: `python main.py --task regression`

### 4. Clustering Analysis (`clustoring.py`)
- **Purpose**: Discover natural data groupings
- **Algorithms**: K-Means, Agglomerative, MeanShift
- **Output**: Clean results summary with silhouette scores and recommendations
- **Best For**: Market segmentation, pattern discovery
- **Command**: `python main.py --task clustering`

### 5. Core Data Processing (`analyzer.py`)
- **Purpose**: Data preprocessing and cleaning
- **Features**: Missing data handling, encoding, feature preparation
- **Output**: Clean, ML-ready data (no repetitive plots)
- **Note**: Used internally by all ML tasks

## Memory Optimization Features

Advanced optimizations for large datasets:

- **Intelligent Sampling**: Uses representative samples for memory-intensive algorithms
- **Parameter Optimization**: Reduced search ranges for efficiency
- **Automatic Cleanup**: Memory management between tasks  
- **Progress Monitoring**: Real-time status updates
- **Algorithm-Specific Tuning**: Optimized for each ML method

These optimizations prevent system crashes on consumer hardware while maintaining analysis quality.

## Dataset Requirements

### Supported Formats
- **CSV files** with headers
- **Mixed data types** (numerical + categorical)
- **Any size** (memory optimization handles large datasets)

### Required Structure
```
your_data.csv
├── feature1       (numerical or categorical)
├── feature2       (numerical or categorical) 
├── ...
└── target_column  (what you want to predict)
```

### Data Types Handled
- **Numerical**: Integers, floats, continuous values
- **Categorical**: Strings, categories (auto-encoded)
- **Mixed**: Datasets with both types

## Output Files

### ML Results (Clean, Focused Output)
- **Classification**: Clean console output with algorithm comparison table
- **Regression**: Clean console output with R² scores and RMSE comparison 
- **Clustering**: Clean console output with comprehensive results summary table

### Analysis Reports (Generated Separately)
- `classification_analysis_summary.md` - Classification insights and recommendations
- `regression_analysis_summary.md` - Regression performance analysis
- `clustering_analysis_summary.md` - Market segmentation insights

### Data Visualizations (EDA Script)
- `Correlation.png` - Feature correlation heatmap
- `PairPlot.png` - Pairwise feature relationships  
- `Histograms_Numeric.png` - Distribution plots
- `Histograms_Categorical.png` - Category distributions
- `BoxPlot.png` - Outlier detection plots

### Workflow Benefits
✅ **ML Tasks**: Clean, focused output showing only algorithm performance  
✅ **EDA Script**: Comprehensive visualizations when you need them  
✅ **No Redundancy**: Plots generated once, not repeated for each ML task

## Example Workflows

### Complete Analysis (Recommended)
```bash
# 1. First understand your data
python exploratory_data_analysis.py

# 2. Run clean ML analysis
python main.py --task all
```

### New Dataset Setup
```bash
# 1. Create config template
python main.py --create-config

# 2. Edit dataset_config.py for your data
# 3. Explore the data first
python exploratory_data_analysis.py

# 4. Run ML analysis
python main.py --task all
```

### Individual Tasks
```bash
# Data exploration only
python exploratory_data_analysis.py

# Specific ML task only  
python main.py --task classification
python main.py --task regression
python main.py --task clustering
```

### Research Workflow
```bash
# 1. Visualize and understand data
python exploratory_data_analysis.py

# 2. Test specific algorithms
python main.py --task classification

# 3. Compare with regression
python main.py --task regression

# 4. Find market segments
python main.py --task clustering
```

## Troubleshooting

### Common Issues

**"Dataset file not found"**
```bash
# Check file exists and path is correct
ls your_dataset.csv

# Or create config with correct path
python main.py --create-config
```

**"Target column not found"**
- Verify target column name in your CSV
- Update `target_column` in `dataset_config.py`

**Memory issues during clustering**  
- The system automatically uses sampling for large datasets
- No action needed - optimization is automatic

**Import errors**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Performance Guidelines

### Dataset Size Recommendations
- **Small** (<1K rows): All algorithms run quickly
- **Medium** (1K-50K rows): Automatic optimizations applied
- **Large** (50K+ rows): Sampling used for memory-intensive tasks

### Expected Runtime
- **Classification**: 2-10 minutes depending on size
- **Regression**: 2-10 minutes depending on size  
- **Clustering**: 3-15 minutes (optimized with sampling)

## Advanced Usage

### Custom Preprocessing
Modify `analyzer.py` to add domain-specific preprocessing steps.

### Algorithm Selection
Each module supports different algorithms - check the source code to enable/disable specific methods.

### Output Customization
Modify result file formats and visualization styles in individual modules.

## Example Datasets to Try

### Beginner Friendly
- **Iris**: Species classification (150 rows)
- **Wine**: Quality classification (1599 rows)
- **Boston Housing**: Price regression (506 rows)

### Intermediate
- **Titanic**: Survival prediction (891 rows)
- **Diamonds**: Price prediction (53K rows) ⭐ Included
- **Heart Disease**: Medical diagnosis (303 rows)

### Advanced
- **House Prices**: Complex regression (1460+ features)
- **Customer Segmentation**: Large clustering datasets
- **E-commerce**: Transaction analysis

## Project Structure

```
universal-ml-suite/
├── main.py                        # Universal ML execution controller
├── exploratory_data_analysis.py   # Data visualization and EDA 📊
├── analyzer.py                    # Data preprocessing (clean, no plots)  
├── classifier.py                  # Classification algorithms
├── regressor.py                   # Regression algorithms
├── clustoring.py                  # Clustering algorithms (memory-optimized)
├── dataset_config.py              # Your dataset configuration
├── your_dataset.csv               # Your data file
├── README.md                      # This documentation
└── results/                       # Generated outputs
    ├── *_analysis_summary.md      # Analysis insights & recommendations
    └── *.png                      # Data visualizations
```

### Key Files Explained
- **`main.py`**: Clean ML analysis with focused results tables
- **`exploratory_data_analysis.py`**: Generates all plots and visualizations
- **`analyzer.py`**: Core data processing (no repetitive outputs)
- **ML modules**: Classification, regression, clustering algorithms
- **Analysis summaries**: Comprehensive insights and recommendations

## Contributing

We welcome contributions! Areas for improvement:
- Additional ML algorithms
- New visualization types  
- Enhanced preprocessing options
- Performance optimizations
- Dataset format support (JSON, Excel, etc.)

## License

This project is for educational and research purposes.

---

**Ready to analyze any dataset! 🚀📊**

*Transform your CSV data into actionable machine learning insights with just a few commands.*