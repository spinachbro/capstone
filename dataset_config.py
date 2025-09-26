"""
Dataset Configuration File
Customize this file for your specific dataset
"""

DATASET_CONFIG = {
    # Dataset file (CSV format)
    'dataset_file': 'diamonds.csv',
    
    # Columns to drop (optional)
    'columns_to_drop': ['Unnamed: 0'],
    
    # Categorical features to encode (optional)
    'categorical_features': ['cut', 'color', 'clarity'],
    
    # Multi-target configuration - specify different targets for different tasks
    'targets': {
        'classification': {
            'target_column': 'cut',
            'description': 'Predict diamond cut quality'
        },
        'regression': {
            'target_column': 'price',
            'description': 'Predict diamond price'
        }
    }
}

# Example configurations for common datasets:

# For diamonds dataset:
# DATASET_CONFIG = {
#     'dataset_file': 'diamonds.csv',
#     'target_column': 'price',
#     'columns_to_drop': ['Unnamed: 0'],
#     'categorical_features': ['cut', 'color', 'clarity']
# }

# For iris dataset:
# DATASET_CONFIG = {
#     'dataset_file': 'iris.csv',
#     'target_column': 'species',
#     'columns_to_drop': [],
#     'categorical_features': []
# }

# For house prices dataset:
# DATASET_CONFIG = {
#     'dataset_file': 'house_prices.csv',
#     'target_column': 'SalePrice',
#     'columns_to_drop': ['Id'],
#     'categorical_features': ['Neighborhood', 'HouseStyle', 'SaleType']
# }
