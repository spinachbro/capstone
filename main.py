#!/usr/bin/env python3
"""
Universal ML Analysis Suite - Main Execution Controller
Hybrid interface supporting both command-line arguments and interactive menu
Works with any CSV dataset for classification, regression, and clustering
"""

import argparse
import sys
import gc
import os
from datetime import datetime

# Default dataset configuration (can be overridden)
DEFAULT_CONFIG = {
    'dataset_file': 'diamonds.csv',
    'columns_to_drop': ['Unnamed: 0'],
    'categorical_features': ['cut', 'color', 'clarity'],
    
    # Multi-target configuration
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

def load_dataset_config(config_file='dataset_config.py'):
    """Load dataset configuration if available"""
    if os.path.exists(config_file):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_file)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            return config.DATASET_CONFIG
        except:
            pass
    
    # If no config file but diamonds.csv exists, use diamonds configuration
    if os.path.exists('diamonds.csv'):
        print("📄 Using built-in diamonds dataset configuration")
        return DEFAULT_CONFIG
    
    # Otherwise return template config for user to customize
    return {
        'dataset_file': 'your_dataset.csv',
        'columns_to_drop': ['id', 'index', 'unnecessary_column'],
        'categorical_features': ['category1', 'category2', 'category3'],
        'targets': {
            'classification': {
                'target_column': 'categorical_target',
                'description': 'Predict categories or classes'
            },
            'regression': {
                'target_column': 'numerical_target', 
                'description': 'Predict continuous values'
            }
        }
    }

def run_classification(dataset_config=None):
    """Execute classification analysis"""
    config = dataset_config or load_dataset_config()
    
    # Get classification target
    if 'targets' in config and 'classification' in config['targets']:
        target_info = config['targets']['classification']
        target_column = target_info['target_column']
        description = target_info.get('description', '')
    else:
        # Fallback to old format
        target_column = config.get('target_column', 'cut')
        description = ''
    
    print("🔍 Starting Classification Analysis...")
    print(f"📄 Dataset: {config['dataset_file']}")
    print(f"🎯 Target: {target_column}")
    if description:
        print(f"📝 Task: {description}")
    print("="*50)
    
    try:
        from classifier import Classifier
        from analyzer import Analyzer
        
        # Data preparation
        analyzer = Analyzer()
        analyzer.read_dataset(config['dataset_file'])
        analyzer.drop_missing_data()
        
        # Drop specified columns
        if config.get('columns_to_drop'):
            analyzer.drop_columns(config['columns_to_drop'])
        
        # Encode categorical features
        if config.get('categorical_features'):
            analyzer.encode_features(config['categorical_features'])
            
        data = analyzer.retrieve_data()
        
        # Prepare features and target  
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
            
        X = data.drop([target_column], axis=1)
        y = data[target_column]
        
        # Validate for classification
        n_unique = y.nunique()
        print(f"📊 Features: {list(X.columns)}")
        print(f"📈 Data shape: {X.shape}")
        print(f"🎯 Target classes: {n_unique}")
        
        if n_unique > 100:
            print(f"⚠️  WARNING: Target has {n_unique} unique values - this may be a regression problem!")
            print("💡 Consider using regression instead of classification for continuous targets")
            
        if n_unique > 1000:
            raise ValueError(f"Too many classes ({n_unique}) for classification. Use regression for continuous targets like price.")
        
        # Run classification with multiple estimators and collect results
        estimators = ['logistic_regression', 'knn', 'decision_tree', 'random_forest', 'svc', 'ann']
        results = []
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for estimator in estimators:
            print(f"\n🔄 Testing {estimator}:")
            start_time = datetime.now()
            
            try:
                classifier = Classifier()
                classifier.fit(X_train, y_train, estimator)
                
                # Get accuracy score
                accuracy = classifier.score(X_test, y_test)
                
                # Calculate training time
                end_time = datetime.now()
                training_time = end_time - start_time
                
                # Store results
                results.append({
                    'Algorithm': estimator.replace('_', ' ').title(),
                    'Accuracy': accuracy,
                    'Training Time': str(training_time).split('.')[0],
                    'Status': '✅ Success'
                })
                
                # Memory cleanup between estimators
                gc.collect()
                
            except Exception as e:
                end_time = datetime.now()
                training_time = end_time - start_time
                
                results.append({
                    'Algorithm': estimator.replace('_', ' ').title(),
                    'Accuracy': None,
                    'Training Time': str(training_time).split('.')[0],
                    'Status': f'❌ Failed: {str(e)[:30]}...'
                })
                
                print(f"⚠️ {estimator} failed: {e}")
                continue
        
        # Display results table
        print("\n" + "="*80)
        print("📊 CLASSIFICATION RESULTS SUMMARY")
        print("="*80)
        
        if results:
            # Create formatted table
            print(f"{'Algorithm':<18} {'Accuracy':<12} {'Training Time':<15} {'Status'}")
            print("-" * 80)
            
            for result in results:
                algo = result['Algorithm']
                acc = f"{result['Accuracy']:.4f}" if result['Accuracy'] else "N/A"
                time_str = result['Training Time']
                status = result['Status']
                
                print(f"{algo:<18} {acc:<12} {time_str:<15} {status}")
            
            # Show best performer
            successful_results = [r for r in results if r['Accuracy'] is not None]
            if successful_results:
                best_result = max(successful_results, key=lambda x: x['Accuracy'])
                print(f"\n🏆 Best Performer: {best_result['Algorithm']} ({best_result['Accuracy']:.4f} accuracy)")
        else:
            print("No results to display")
        
        print("✅ Classification completed successfully!")
        gc.collect()  # Memory cleanup
        return True
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return False

def run_regression(dataset_config=None):
    """Execute regression analysis"""
    config = dataset_config or load_dataset_config()
    
    # Get regression target
    if 'targets' in config and 'regression' in config['targets']:
        target_info = config['targets']['regression']
        target_column = target_info['target_column']
        description = target_info.get('description', '')
    else:
        # Fallback to old format
        target_column = config.get('target_column', 'price')
        description = ''
    
    print("📈 Starting Regression Analysis...")
    print(f"📄 Dataset: {config['dataset_file']}")
    print(f"🎯 Target: {target_column}")
    if description:
        print(f"📝 Task: {description}")
    print("="*50)
    
    try:
        from regressor import Regressor
        from analyzer import Analyzer
        
        # Data preparation
        analyzer = Analyzer()
        analyzer.read_dataset(config['dataset_file'])
        analyzer.drop_missing_data()
        
        # Drop specified columns
        if config.get('columns_to_drop'):
            analyzer.drop_columns(config['columns_to_drop'])
        
        # Encode categorical features
        if config.get('categorical_features'):
            analyzer.encode_features(config['categorical_features'])
            
        data = analyzer.retrieve_data()
        
        # Prepare features and target
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
            
        X = data.drop([target_column], axis=1)
        y = data[target_column]
        
        print(f"📊 Features: {list(X.columns)}")
        print(f"📈 Data shape: {X.shape}")
        print(f"🎯 Target range: {y.min():.2f} - {y.max():.2f}")
        print(f"🎯 Target mean: {y.mean():.2f}")
        
        # Run regression with multiple estimators and collect results
        estimators = ['linear_regression', 'knn', 'decision_tree', 'random_forest', 'svr', 'ann']
        results = []
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for estimator in estimators:
            print(f"\n🔄 Testing {estimator}:")
            start_time = datetime.now()
            
            try:
                regressor = Regressor()
                regressor.fit(X_train, y_train, estimator)
                
                # Get R² score and RMSE
                score_dict = regressor.score(X_test, y_test)
                r2_score = score_dict['r2'] if isinstance(score_dict, dict) else score_dict
                rmse = score_dict['rmse'] if isinstance(score_dict, dict) else None
                
                # If RMSE not available from score, calculate manually
                if rmse is None:
                    from sklearn.metrics import mean_squared_error
                    predictions = regressor.predict(X_test)
                    rmse = mean_squared_error(y_test, predictions) ** 0.5 if predictions is not None else None
                
                # Calculate training time
                end_time = datetime.now()
                training_time = end_time - start_time
                
                # Store results
                results.append({
                    'Algorithm': estimator.replace('_', ' ').title(),
                    'R2 Score': r2_score,
                    'RMSE': rmse,
                    'Training Time': str(training_time).split('.')[0],
                    'Status': '✅ Success'
                })
                
                # Memory cleanup between estimators
                gc.collect()
                
            except Exception as e:
                end_time = datetime.now()
                training_time = end_time - start_time
                
                results.append({
                    'Algorithm': estimator.replace('_', ' ').title(),
                    'R2 Score': None,
                    'RMSE': None,
                    'Training Time': str(training_time).split('.')[0],
                    'Status': f'❌ Failed: {str(e)[:30]}...'
                })
                
                print(f"⚠️ {estimator} failed: {e}")
                continue
        
        # Display results table
        print("\n" + "="*90)
        print("📊 REGRESSION RESULTS SUMMARY")
        print("="*90)
        
        if results:
            try:
                # Create formatted table
                print(f"{'Algorithm':<18} {'R² Score':<10} {'RMSE':<12} {'Training Time':<15} {'Status'}")
                print("-" * 90)
                
                for result in results:
                    algo = result['Algorithm']
                    r2 = f"{result['R2 Score']:.4f}" if result['R2 Score'] is not None else "N/A"
                    rmse = f"{result['RMSE']:.2f}" if result['RMSE'] is not None else "N/A"
                    time_str = result['Training Time']
                    status = result['Status']
                    
                    print(f"{algo:<18} {r2:<10} {rmse:<12} {time_str:<15} {status}")
                
                # Show best performer
                successful_results = [r for r in results if r['R2 Score'] is not None and r['R2 Score'] > -999]
                if successful_results:
                    best_result = max(successful_results, key=lambda x: x['R2 Score'])
                    print(f"\n🏆 Best Performer: {best_result['Algorithm']} (R² = {best_result['R2 Score']:.4f}, RMSE = {best_result['RMSE']:.2f})")
                    
            except Exception as table_error:
                print(f"❌ Error formatting results table: {table_error}")
                print("Raw results:", results)
        else:
            print("No results to display")
        
        print("✅ Regression completed successfully!")
        gc.collect()  # Memory cleanup
        return True
        
    except Exception as e:
        print(f"❌ Regression failed: {e}")
        return False


def run_clustering(dataset_config=None):
    """Execute clustering analysis"""
    config = dataset_config or load_dataset_config()
    
    print("🔗 Starting Clustering Analysis...")
    print(f"📄 Dataset: {config['dataset_file']}")
    print("="*50)
    
    try:
        from clustoring import Clustering
        from analyzer import Analyzer
        
        # Data preparation
        analyzer = Analyzer()
        analyzer.read_dataset(config['dataset_file'])
        analyzer.drop_missing_data()
        
        # Drop specified columns
        if config.get('columns_to_drop'):
            analyzer.drop_columns(config['columns_to_drop'])
        
        # Encode categorical features
        if config.get('categorical_features'):
            analyzer.encode_features(config['categorical_features'])
            
        data = analyzer.retrieve_data()
        
        # Use all features for clustering (no target needed)
        # Remove target columns if they exist for clustering
        target_columns = []
        
        # Check for multi-target format
        if 'targets' in config:
            for task_type, target_info in config['targets'].items():
                target_col = target_info.get('target_column')
                if target_col and target_col in data.columns:
                    target_columns.append(target_col)
        # Check for old format
        elif 'target_column' in config and config['target_column'] in data.columns:
            target_columns.append(config['target_column'])
        
        # Drop target columns for clustering
        if target_columns:
            X = data.drop(target_columns, axis=1).values
            print(f"📝 Excluded target columns: {target_columns}")
        else:
            X = data.values
        
        print(f"📊 Features for clustering: {X.shape}")
        
        # Test all clustering algorithms with memory efficiency and collect results
        algorithms = ['kmeans', 'agglomerative', 'meanshift']
        results = []
        
        for algorithm in algorithms:
            print(f"\n🔄 Testing {algorithm}:")
            clustering = Clustering()
            labels = clustering.fit(X, algorithm, memory_efficient=True)
            
            # Collect results for summary
            result = {
                'algorithm': algorithm,
                'n_clusters': clustering.n_clusters,
                'silhouette_score': clustering.silhouette_score,
                'optimal_param': 'N/A',
                'data_coverage': '100%',
                'notes': 'Full dataset'
            }
            
            # Calculate data coverage and get algorithm-specific metrics
            if hasattr(labels, 'shape'):
                total_points = len(labels)
                valid_points = len(labels[labels >= 0])
                coverage_pct = (valid_points / total_points) * 100
                result['data_coverage'] = f"{coverage_pct:.1f}%"
            
            # Get algorithm-specific parameters
            if algorithm == 'kmeans':
                result['optimal_param'] = f"K={clustering.optimal_k}"
            elif algorithm == 'agglomerative':
                result['optimal_param'] = f"N={clustering.optimal_n_clusters}"
                result['notes'] = 'Sampled data'
            elif algorithm == 'meanshift':
                if hasattr(clustering, 'estimated_bandwidth'):
                    result['optimal_param'] = f"BW={clustering.estimated_bandwidth:.3f}"
                else:
                    result['optimal_param'] = 'Auto BW'
                result['notes'] = 'Sampled data'
            
            results.append(result)
            
            # Memory cleanup between algorithms
            gc.collect()
        
        # Import and display comprehensive summary
        from clustoring import print_clustering_summary
        print_clustering_summary(results)
        
        print("✅ Clustering completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Clustering failed: {e}")
        return False

def run_all_tasks(dataset_config=None):
    """Execute all ML tasks in sequence"""
    config = dataset_config or load_dataset_config()
    
    print("🚀 Starting Complete ML Analysis Suite...")
    print(f"📄 Dataset: {config['dataset_file']}")
    print("="*60)
    
    tasks = [
        ("Classification", lambda: run_classification(config)),
        ("Regression", lambda: run_regression(config)), 
        ("Clustering", lambda: run_clustering(config))
    ]
    
    results = {}
    
    for task_name, task_func in tasks:
        print(f"\n{'='*20} {task_name.upper()} {'='*20}")
        start_time = datetime.now()
        
        success = task_func()
        results[task_name] = success
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"⏱️ {task_name} completed in {duration}")
        
        # Memory cleanup between tasks
        gc.collect()
    
    # Summary
    print("\n" + "="*60)
    print("📊 EXECUTION SUMMARY")
    print("="*60)
    
    for task_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{task_name:15} {status}")
    
    successful_tasks = sum(results.values())
    total_tasks = len(results)
    print(f"\nOverall: {successful_tasks}/{total_tasks} tasks completed successfully")
    
    return results

def interactive_menu():
    """Display interactive menu for task selection"""
    config = load_dataset_config()
    
    while True:
        print("\n" + "="*50)
        print("🤖 UNIVERSAL ML ANALYSIS SUITE")
        print("="*50)
        print(f"📄 Current dataset: {config['dataset_file']}")
        
        # Show target information
        if 'targets' in config:
            if 'classification' in config['targets']:
                cls_target = config['targets']['classification']['target_column']
                cls_desc = config['targets']['classification'].get('description', '')
                print(f"🎯 Classification target: {cls_target} ({cls_desc})")
            
            if 'regression' in config['targets']:
                reg_target = config['targets']['regression']['target_column']
                reg_desc = config['targets']['regression'].get('description', '')
                print(f"🎯 Regression target: {reg_target} ({reg_desc})")
        else:
            # Fallback for old format
            target = config.get('target_column', 'price')
            print(f"🎯 Target column: {target}")
            
        print("="*50)
        print("1. Run Classification")
        print("2. Run Regression")
        print("3. Run Clustering")
        print("4. Run All Tasks")
        print("5. Exit")
        print("="*50)
        
        try:
            choice = input("Choose option [1-5]: ").strip()
            
            if choice == '1':
                run_classification(config)
            elif choice == '2':
                run_regression(config)
            elif choice == '3':
                run_clustering(config)
            elif choice == '4':
                run_all_tasks(config)
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def create_example_config():
    """Create example dataset configuration file"""
    config_content = '''"""
Dataset Configuration File
Customize this file for your specific dataset
"""

DATASET_CONFIG = {
    # Dataset file (CSV format)
    'dataset_file': 'your_dataset.csv',
    
    # Columns to drop (optional)
    'columns_to_drop': ['id', 'index', 'unnecessary_column'],
    
    # Categorical features to encode (optional)
    'categorical_features': ['category1', 'category2', 'category3'],
    
    # Multi-target configuration - specify different targets for different tasks
    'targets': {
        'classification': {
            'target_column': 'categorical_target',
            'description': 'Predict categories or classes'
        },
        'regression': {
            'target_column': 'numerical_target', 
            'description': 'Predict continuous values'
        }
    }
}

# Example configurations for common datasets:

# For diamonds dataset (multi-target):
# DATASET_CONFIG = {
#     'dataset_file': 'diamonds.csv',
#     'columns_to_drop': ['Unnamed: 0'],
#     'categorical_features': ['cut', 'color', 'clarity'],
#     'targets': {
#         'classification': {
#             'target_column': 'cut',
#             'description': 'Predict diamond cut quality'
#         },
#         'regression': {
#             'target_column': 'price',
#             'description': 'Predict diamond price'
#         }
#     }
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
'''
    with open('dataset_config.py', 'w') as f:
        f.write(config_content)
    print("✅ Created dataset_config.py template")

def main():
    """Main entry point with hybrid CLI/interactive support"""
    parser = argparse.ArgumentParser(
        description="Universal ML Analysis Suite - Works with any CSV dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Interactive menu
  python main.py --task classification        # Run classification only
  python main.py --task regression            # Run regression only
  python main.py --task clustering            # Run clustering only
  python main.py --task all                  # Run all tasks
  python main.py --create-config             # Create dataset configuration template
  
Dataset Configuration:
  Create 'dataset_config.py' to specify your dataset details, or use the default diamonds configuration.
        """
    )
    
    parser.add_argument(
        '--task', 
        choices=['classification', 'regression', 'clustering', 'all'],
        help='ML task to execute'
    )
    
    parser.add_argument(
        '--create-config', 
        action='store_true',
        help='Create dataset configuration template file'
    )
    
    args = parser.parse_args()
    
    # Create config template
    if args.create_config:
        create_example_config()
        return
    
    # Load dataset configuration
    config = load_dataset_config()
    
    # Check if dataset file exists
    if not os.path.exists(config['dataset_file']):
        print(f"❌ Dataset file '{config['dataset_file']}' not found!")
        print("💡 Either:")
        print(f"   1. Add your CSV file as '{config['dataset_file']}'")
        print("   2. Create 'dataset_config.py' with your dataset details")
        print("   3. Run 'python main.py --create-config' for a template")
        
        # Show available files to help user
        print("\n📁 Files in current directory:")
        for file in sorted(os.listdir('.')):
            if file.endswith('.csv'):
                print(f"   📄 {file}")
        
        sys.exit(1)
    
    # Command-line mode
    if args.task:
        print(f"🎯 Command-line mode: {args.task}")
        print(f"📄 Using dataset: {config['dataset_file']}")
        
        if args.task == 'classification':
            success = run_classification(config)
        elif args.task == 'regression':
            success = run_regression(config)
        elif args.task == 'clustering':
            success = run_clustering(config)
        elif args.task == 'all':
            results = run_all_tasks(config)
            success = all(results.values())
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
    
    # Interactive mode
    else:
        print("🎮 Interactive mode - Use --help for command-line options")
        interactive_menu()

if __name__ == "__main__":
    main()