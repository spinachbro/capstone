import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import math

class Regressor:
    def __init__(self):
        self.estimator = None
        self.estimator_name = None
        self.X_train = None
        self.y_train = None
        self.scaler = None
        # Store optimal parameters for each algorithm
        self.optimal_params = {}

    def fit(self, X_train, y_train, estimator_name):
        """
        A function that takes the training set and trains the input data using the current estimator.
        """
        # Store training data and target
        self.X_train = X_train
        self.y_train = y_train
        self.estimator_name = estimator_name

        print(f"Training {estimator_name} regressor...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Target range: ${y_train.min():.0f} - ${y_train.max():.0f}")
        print(f"Target mean: ${y_train.mean():.0f}")

              # Initialize the estimator based on name
        if estimator_name == 'linear_regression':
            self.estimator = LinearRegression()

        elif estimator_name == 'knn':
            # ADD THE KNN CODE HERE
            print("Finding optimal K value for KNN regressor...")

            best_k = 5  # default
            best_score = -float('inf')

            print("Testing K values for regression:")

            # Test K values from 1 to 20
            for k in range(1, 21):
                knn_temp = KNeighborsRegressor(n_neighbors=k)
                cv_scores = cross_val_score(knn_temp, X_train, y_train, cv=5, scoring='r2')
                avg_score = cv_scores.mean()

                print(f"K={k}: CV R² Score = {avg_score:.4f} ({avg_score*100:.2f}%)")

                if avg_score > best_score:
                    best_score = avg_score
                    best_k = k

            print(f"\nOptimal K: {best_k}")
            print(f"Best CV R² Score: {best_score:.4f} ({best_score*100:.2f}%)")

            # Create the final KNN with optimal K
            self.estimator = KNeighborsRegressor(n_neighbors=best_k)
            self.optimal_k = best_k  # Store for reference

        elif estimator_name == 'decision_tree':
            print("Finding optimal criterion for Decision Tree regressor...")

        # Test different criterion values for regression
            criterions = ['squared_error', 'friedman_mse','absolute_error']
            best_criterion = 'squared_error'  # default
            best_score = -float('inf')

            print("Testing different criterion values:")

            for criterion in criterions:
                dt_temp = DecisionTreeRegressor(criterion=criterion,random_state=42)
                # Use cross-validation to get reliable R² score
                cv_scores = cross_val_score(dt_temp, X_train, y_train, cv=3, scoring='r2')  # Reduced CV folds for speed
                avg_score = cv_scores.mean()

                print(f"Criterion='{criterion}': CV R² Score = {avg_score:.4f} ({avg_score*100:.2f}%)")

                if avg_score > best_score:
                    best_score = avg_score
                    best_criterion = criterion

            print(f"\nOptimal Criterion: '{best_criterion}'")
            print(f"Best CV R² Score: {best_score:.4f} ({best_score*100:.2f}%)")

            # Create final Decision Tree with optimal criterion
            self.estimator = DecisionTreeRegressor(criterion=best_criterion,random_state=42)
            self.optimal_criterion = best_criterion  # Store for reference

        elif estimator_name == 'random_forest':
            print("Finding optimal parameters for Random Forest regressor...")
            # Test different criterion and n_estimators values
            criterions = ['squared_error', 'friedman_mse']  # Reduced from 3 to 2 for speed
            n_estimators_list = [50, 100]  # Reduced from [50, 100, 200] for speed

            best_criterion = 'squared_error'
            best_n_estimators = 100
            best_score = -float('inf')

            print("Testing different parameter combinations:")

            for criterion in criterions:
                for n_est in n_estimators_list:
                    rf_temp = RandomForestRegressor(
                        criterion=criterion,
                        n_estimators=n_est,
                        random_state=42,
                        n_jobs=-1  # Use all CPU cores for speed
                    )
                    # Use cross-validation for reliable R² score
                    cv_scores = cross_val_score(rf_temp, X_train, y_train, cv=3, scoring='r2')  # Reduced CV folds for speed
                    avg_score = cv_scores.mean()

                    print(f"Criterion='{criterion}', n_estimators={n_est}: CV R² Score = {avg_score:.4f} ({avg_score*100:.2f}%)")

                    if avg_score > best_score:
                        best_score = avg_score
                        best_criterion = criterion
                        best_n_estimators = n_est

            print(f"\nOptimal Parameters:")
            print(f"  Criterion: '{best_criterion}'")
            print(f"  N_estimators: {best_n_estimators}")
            print(f"  Best CV R² Score: {best_score:.4f} ({best_score*100:.2f}%)")

            # Create final Random Forest with optimal parameters
            self.estimator = RandomForestRegressor(
                criterion=best_criterion,
                n_estimators=best_n_estimators,
                random_state=42,
                n_jobs=-1
            )
            self.optimal_criterion = best_criterion
            self.optimal_n_estimators = best_n_estimators

        elif estimator_name == 'svr':
            print("Training SVR (Support Vector Regression)...")

            # Scale features for SVR (required for optimal performance)
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Create SVR with RBF kernel (good default for regression)
            self.estimator = SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                epsilon=0.1  # Add epsilon parameter for regression
            )

            # Fit the scaled data
            self.estimator.fit(X_train_scaled, y_train)

            print(f"SVR training completed!")
            print("Features scaled for optimal SVR performance")
            return  # Early return since we already fitted

        elif estimator_name == 'ann':
            print("Finding optimal parameters for ANN regressor...")

            # Scale features for ANN (required)
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Optimized parameter grid for speed
            architectures = [(100, 50)]  # Reduced from 3 to 1 architecture for speed
            activations = ['relu']  # Reduced from ['tanh', 'relu'] to 1 for speed
            solvers = ['adam']
            learning_rates = [0.001]  # Reduced from [0.001, 0.01, 0.1] to 1 for speed

            # Total: 1×1×1×1 = 1 combination (much faster!)

            best_params = {}
            best_score = -float('inf')

            print("Testing different parameter combinations:")
            total_combinations = len(architectures) * len(activations) * len(solvers) * len(learning_rates)
            print(f"Total combinations: {total_combinations}")

            combination_count = 0
            for arch in architectures:
                for activation in activations:
                    for solver in solvers:
                        for lr in learning_rates:
                            combination_count += 1
                            print(f"Testing combination {combination_count}...")

                            ann_temp = MLPRegressor(
                                hidden_layer_sizes=arch,
                                activation=activation,
                                solver=solver,
                                learning_rate_init=lr,
                                random_state=42,
                                max_iter=200,  # Reduced from 500 for speed
                                early_stopping=True,     # Add early stopping
                                validation_fraction=0.1  # Use 10% for validation
                            )

                            # Use cross-validation for R² score
                            cv_scores = cross_val_score(ann_temp, X_train_scaled, y_train, cv=3, scoring='r2')
                            avg_score = cv_scores.mean()

                            print(f"Arch={arch}, activation='{activation}', solver='{solver}', lr={lr}: CV R² Score = {avg_score:.4f}")

                            if avg_score > best_score:
                                best_score = avg_score
                                best_params = {
                                    'hidden_layer_sizes': arch,
                                    'activation': activation,
                                    'solver': solver,
                                    'learning_rate_init': lr
                                }

            print(f"\nOptimal Parameters:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            print(f"  Best CV R² Score: {best_score:.4f} ({best_score*100:.2f}%)")

            # Create final ANN with optimal parameters
            self.estimator = MLPRegressor(
                **best_params,
                random_state=42,
                max_iter=200,  # Reduced from 500 for speed
                early_stopping=True,
                validation_fraction=0.1
            )
            self.estimator.fit(X_train_scaled, y_train)
            self.optimal_ann_params = best_params

            print(f"ANN training completed!")
            return  # Early return since we already fitted

        else:
            print(f"Error: Unknown estimator '{estimator_name}'")
            print("Available estimators: 'linear_regression', 'knn', 'decision_tree', 'random_forest', 'svr', 'ann'")
            return

        # Fit the estimator (for non-scaling and non-optimization algorithms)
        if estimator_name not in ['knn', 'decision_tree',
        'random_forest', 'svr', 'ann']:
            self.estimator.fit(X_train, y_train)
        else:
            if estimator_name not in ['svr', 'ann']:  # These already fitted above
                self.estimator.fit(X_train, y_train)

        print(f"{estimator_name} training completed!")

    def predict(self, X_test):
        """
        A function that takes the data and predicts their values using the current trained estimator.
        """
        if self.estimator is None:
            print("Error: No trained estimator found. Please call fit() first.")
            return None

        print(f"Making predictions using {self.estimator_name}...")
        print(f"Test data shape: {X_test.shape}")

        # Handle scaling for SVR and ANN
        if self.estimator_name in ['svr', 'ann']:
            if self.scaler is None:
                print("Error: Scaler not found for scaled estimator.")
                return None
            X_test_scaled = self.scaler.transform(X_test)
            predictions = self.estimator.predict(X_test_scaled)
        else:
            predictions = self.estimator.predict(X_test)

        print(f"Predictions completed! Generated {len(predictions)} predictions.")
        print(f"Predicted price range: ${predictions.min():.0f} - ${predictions.max():.0f}")
        print(f"Mean predicted price: ${predictions.mean():.0f}")

        return predictions

    def score(self, X_test, y_test):
        """
        A function that takes data and its true values and returns regression metrics.
        """
        if self.estimator is None:
            print("Error: No trained estimator found. Please call fit() first.")
            return None

        print(f"Calculating regression metrics for {self.estimator_name}...")

        # Get predictions
        predictions = self.predict(X_test)
        if predictions is None:
            return None

        # Calculate the three required metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Regression Metrics:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MSE Score: ${mse:.2f}")
        print(f"  RMSE Score: ${rmse:.2f}")
        print(f"  MAE Score: ${mae:.2f}")

        # Return as dictionary for easy access
        return {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

# Test section will go here...
  # Test all regressor algorithms with diamond data
if __name__ == "__main__":
    from analyzer import Analyzer
    from sklearn.model_selection import train_test_split

    # Get diamond data
    analyzer = Analyzer()
    analyzer.read_dataset('diamonds.csv')
    analyzer.drop_missing_data()
    analyzer.drop_columns(['Unnamed: 0'])
    analyzer.encode_features(['cut', 'color', 'clarity'])
    data = analyzer.retrieve_data()

    # Prepare for price regression
    X = data.drop(['price'], axis=1)  # All features except price
    y = data['price']                 # Target: price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

    print("DIAMOND PRICE PREDICTION - REGRESSION COMPARISON")
    print("="*70)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {list(X.columns)}")
    print(f"Target: Price (${y.min():.0f} - ${y.max():.0f})")
    print("="*70)

    # Test all estimators
    estimators = ['linear_regression', 'knn','decision_tree', 'random_forest', 'svr', 'ann']
    results = []

    for estimator in estimators:
        print(f"\n{'='*50}")
        print(f"TESTING: {estimator.upper().replace('_', ' ')}")
        print(f"{'='*50}")

        try:
            # Create new regressor instance for each estimator
            regressor = Regressor()

            # Train
            regressor.fit(X_train, y_train, estimator)

            # Get scores
            scores = regressor.score(X_test, y_test)

            if scores:
                results.append((estimator, scores['r2'], scores['rmse'], scores['mae']))
                print(f"✓ {estimator} completed successfully!")
            else:
                results.append((estimator, 0.0, float('inf'), float('inf')))
                print(f"✗ {estimator} - scoring failed")

        except Exception as e:
            print(f"✗ {estimator} failed: {e}")
            results.append((estimator, 0.0, float('inf'), float('inf')))

    # Summary of all results
    print(f"\n{'='*70}")
    print("FINAL REGRESSION COMPARISON")
    print(f"{'='*70}")
    print(f"{'Algorithm':<20} {'R² Score':<12} {'RMSE':<12} {'MAE':<12}")
    print("-" * 70)

    # Sort by R² score (best first)
    results.sort(key=lambda x: x[1], reverse=True)

    for estimator, r2, rmse, mae in results:
        name = estimator.replace('_', ' ').title()
        if r2 > 0:
            print(f"{name:<20} {r2:<12.4f} ${rmse:<11.2f} ${mae:<11.2f}")
        else:
            print(f"{name:<20} {'Failed':<12} {'Failed':<12} {'Failed':<12}")

    print(f"{'='*70}")
    if results and results[0][1] > 0:
        best = results[0]
        print(f"Best Performer: {best[0].replace('_', ' ').title()} (R² = {best[1]:.4f})")
        print(f"Target Achievement: {'✓ PASSED' if best[1] >= 0.90 else '✗ NEEDS IMPROVEMENT'} (Goal: R² ≥ 90%)")