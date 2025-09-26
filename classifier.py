import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Classifier:

    def __init__(self):
        self.estimator = None
        self.estimator_name = None
        self.X_train = None
        self.y_train = None
        self.scaler = None

    def fit(self, X_train, y_train, estimator_name):
        """
        A function that takes the training set and trains the input data using the current estimator.
        """
        # Store training data and target
        self.X_train = X_train
        self.y_train = y_train
        self.estimator_name = estimator_name

        print(f"Training {estimator_name} classifier...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Number of classes: {len(set(y_train))}")

        # Initialize the estimator based on name
        if estimator_name == 'logistic_regression':
            self.estimator = LogisticRegression(random_state=42)
        
        elif estimator_name == 'knn':
            print("Finding optimal K value for KNN classifier...")

            # Test different K values to find the best one
            from sklearn.model_selection import cross_val_score

            best_k = 5  # default
            best_score = 0

            # Test K values from 1 to 20
            for k in range(1, 21):
                knn_temp = KNeighborsClassifier(n_neighbors=k)
                # Use cross-validation to get more reliable score
                cv_scores = cross_val_score(knn_temp, X_train, y_train, cv=5)
                avg_score = cv_scores.mean()

                print(f"K={k}: CV Accuracy = {avg_score:.4f} ({avg_score*100:.2f}%)")

                if avg_score > best_score:
                    best_score = avg_score
                    best_k = k

            print(f"\nOptimal K: {best_k}")
            print(f"Best CV Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")

            # Now create the final KNN with optimal K
            self.estimator = KNeighborsClassifier(n_neighbors=best_k)
            self.optimal_k = best_k  # Store for reference

        elif estimator_name == 'decision_tree':
            print("Finding optimal criterion for Decision Tree classifier...")

            # Test different criterion values
            from sklearn.model_selection import cross_val_score

            criterions = ['gini', 'entropy']
            best_criterion = 'gini'  # default
            best_score = 0

            for criterion in criterions:
                dt_temp = DecisionTreeClassifier(criterion=criterion,random_state=42)
                # Use cross-validation to get more reliable score
                cv_scores = cross_val_score(dt_temp, X_train, y_train, cv=5)
                avg_score = cv_scores.mean()

                print(f"Criterion='{criterion}': CV Accuracy = {avg_score:.4f} ({avg_score*100:.2f}%)")

                if avg_score > best_score:
                    best_score = avg_score
                    best_criterion = criterion

            print(f"\nOptimal Criterion: '{best_criterion}'")
            print(f"Best CV Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")

            # Create final Decision Tree with optimal criterion
            self.estimator = DecisionTreeClassifier(criterion=best_criterion, random_state=42)
            self.optimal_criterion = best_criterion  # Store for reference

        elif estimator_name == 'random_forest':
            print("Finding optimal parameters for Random Forest classifier...")
            # Test different criterion and n_estimators values
            from sklearn.model_selection import cross_val_score
            criterions = ['gini', 'entropy']
            n_estimators_list = [50, 100]  # Reduced from [50, 100, 200] for speed
            best_criterion = 'gini'
            best_n_estimators = 100
            best_score = 0
            print("Testing different parameter combinations:")
            for criterion in criterions:
                for n_est in n_estimators_list:
                    rf_temp = RandomForestClassifier(
                        criterion=criterion,
                        n_estimators=n_est,
                        random_state=42
                    )
                    # Use cross-validation
                    cv_scores = cross_val_score(rf_temp, X_train, y_train, cv=5)
                    avg_score = cv_scores.mean()
                    print(f"Criterion='{criterion}', n_estimators={n_est}: CV Accuracy = {avg_score:.4f} ({avg_score*100:.2f}%)")
                    if avg_score > best_score:
                        best_score = avg_score
                        best_criterion = criterion
                        best_n_estimators = n_est
            print(f"\nOptimal Parameters:")
            print(f"  Criterion: '{best_criterion}'")
            print(f"  N_estimators: {best_n_estimators}")
            print(f"  Best CV Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
            # Create final Random Forest with optimal parameters
            self.estimator = RandomForestClassifier(
                criterion=best_criterion,
                n_estimators=best_n_estimators,
                random_state=42
            )
            self.optimal_criterion = best_criterion
            self.optimal_n_estimators = best_n_estimators

        elif estimator_name == 'svc':
            # Scale features for SVC
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.estimator = SVC(kernel='rbf', random_state=42)
            self.estimator.fit(X_train_scaled, y_train)
            print(f"{estimator_name} training completed!")
            return
        
        elif estimator_name == 'ann':
            print("Finding optimal parameters for ANN classifier...")

            # Scale features for ANN
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Test different ANN parameters
            from sklearn.model_selection import cross_val_score

            architectures = [(100, 50)]
            activations = ['tanh']
            solvers = ['adam']
            learning_rates = [0.001]  # Reduced from [0.001, 0.01] for speed

            best_params = {}
            best_score = 0

            print("Testing different parameter combinations:")

            for arch in architectures:
                for activation in activations:
                    for solver in solvers:
                        for lr in learning_rates:
                            ann_temp = MLPClassifier(
                                hidden_layer_sizes=arch,
                                activation=activation,
                                solver=solver,
                                learning_rate_init=lr,
                                random_state=42,
                                max_iter=300  # Reduced from 500 for speed
                            )

                            # Use cross-validation
                            cv_scores = cross_val_score(ann_temp, X_train_scaled, y_train, cv=3)  # Reduced CV folds
                            avg_score = cv_scores.mean()

                            print(f"Arch={arch}, activation='{activation}', solver='{solver}', lr={lr}: CV Accuracy = {avg_score:.4f}")

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
            print(f"  Best CV Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")

            # Create final ANN with optimal parameters
            self.estimator = MLPClassifier(**best_params, random_state=42, max_iter=500)
            self.estimator.fit(X_train_scaled, y_train)
            self.optimal_ann_params = best_params

            print(f"{estimator_name} training completed!")
            return

        else:
            print(f"Error: Unknown estimator '{estimator_name}'")
            print("Available estimators: 'logistic_regression', 'knn', 'decision_tree', 'random_forest', 'svc', 'ann'")
            return

        # Fit the estimator (for non-scaling algorithms)
        self.estimator.fit(X_train, y_train)
        print(f"{estimator_name} training completed!")

    def predict(self, X_test):
        """
        A function that takes the data and predicts their label using the current trained estimator.
        """
        if self.estimator is None:
            print("Error: No trained estimator found. Please call fit() first.")
            return None

        if self.estimator_name is None:
            print("Error: No estimator name found. Please call fit() first.")
            return None

        print(f"Making predictions using {self.estimator_name}...")
        print(f"Test data shape: {X_test.shape}")

        # Handle scaling for SVC and ANN
        if self.estimator_name in ['svc', 'ann']:
            if self.scaler is None:
                print("Error: Scaler not found for scaled estimator.")
                return None
            X_test_scaled = self.scaler.transform(X_test)
            predictions = self.estimator.predict(X_test_scaled)
        else:
            # For non-scaling algorithms
            predictions = self.estimator.predict(X_test)

        print(f"Predictions completed! Generated {len(predictions)} predictions.")
        print(f"Unique predicted classes: {sorted(set(predictions))}")

        return predictions
    
    def score(self, X_test, y_test):
        """
        A function that takes data and its true labels and returns the accuracy score.
        """
        if self.estimator is None:
            print("Error: No trained estimator found. Please call fit() first.")
            return None

        if self.estimator_name is None:
            print("Error: No estimator name found. Please call fit() first.")
            return None

        print(f"Calculating accuracy score for {self.estimator_name}...")
        print(f"Test data shape: {X_test.shape}")
        print(f"True labels shape: {y_test.shape}")

        # Get predictions first
        predictions = self.predict(X_test)

        if predictions is None:
            print("Error: Failed to get predictions.")
            return None

        # Calculate accuracy score
        accuracy = accuracy_score(y_test, predictions)

        print(f"Accuracy score: {accuracy:.4f} ({accuracy*100:.2f}%)")

        return accuracy
    
    def plot_confusionMatrix(self, X_test, y_test):
        """
        A function that takes data predictions and the true labels, plots the confusion matrix, then returns it.
        """
        if self.estimator is None:
            print("Error: No trained estimator found. Please call fit() first.")
            return None

        if self.estimator_name is None:
            print("Error: No estimator name found. Please call fit() first.")
            return None

        print(f"Creating confusion matrix for {self.estimator_name}...")

        # Get predictions
        predictions = self.predict(X_test)

        if predictions is None:
            print("Error: Failed to get predictions.")
            return None

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Plot confusion matrix heatmap
        sns.heatmap(cm, 
                    annot=True,           # Show numbers in cells
                    fmt='d',              # Format as integers
                    cmap='Blues',         # Color scheme
                    square=True,          # Make cells square
                    cbar_kws={'shrink': 0.8})

        plt.title(f'Confusion Matrix - {self.estimator_name.replace("_", " ").title()}', fontsize=14, pad=20)
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.tight_layout()

        # Save the plot
        filename = f'ConfusionMatrix_{self.estimator_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory

        print(f"Confusion matrix plotted and saved as '{filename}'")
        print(f"Matrix shape: {cm.shape}")
        print("Confusion Matrix:")
        print(cm)

        return cm




# Test with diamond data for 'cut' classification
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

    # Prepare for cut classification
    X = data.drop(['cut'], axis=1)
    y = data['cut']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

    estimators = ['logistic_regression', 'knn', 'decision_tree', 'random_forest', 'svc', 'ann']

    print("DIAMOND CUT CLASSIFICATION - ACCURACY COMPARISON")
    print("="*60)

    for estimator in estimators:
        classifier = Classifier()
        classifier.fit(X_train, y_train, estimator)
        accuracy = classifier.score(X_test, y_test)
        print(f"{estimator.replace('_', ' ').title():<20}: {accuracy:.4f} ({accuracy*100:.2f}%)")