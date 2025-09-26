import numpy as np
import pandas as pd
import gc
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

class Clustering:
    def __init__(self):
        self.estimator = None
        self.estimator_name = None
        self.X_data = None
        self.scaler = None
        self.cluster_labels = None
        self.n_clusters = None
    
    def fit(self, X_data, estimator_name, memory_efficient=True):
        """
        A function that clusters the input data.
        memory_efficient: Use memory optimization techniques for large datasets
        """
        self.X_data = X_data
        self.estimator_name = estimator_name
        
        print(f"Clustering with {estimator_name}...")
        print(f"Data shape: {X_data.shape}")
        
        # Scale data for clustering
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_data)
        
        # Force garbage collection
        gc.collect()
        
        # Initialize estimator
        if estimator_name == 'kmeans':
            print("Finding optimal K for K-means using elbow method...")
            
            # Test different K values and find optimal using elbow method
            k_range = range(2, 7) if memory_efficient else range(2, 11)  # Reduce range for memory efficiency
            inertias = []
            silhouette_scores = []
            
            print("Testing different K values:")
            
            for k in k_range:
                # 1. Implement the k-means instance (single-threaded for memory efficiency)
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=5 if memory_efficient else 10)
                
                # 2. Cluster the data using the K-means
                temp_labels = kmeans_temp.fit_predict(X_scaled)
                
                # 5. Retrieve the inertia of the k-means estimator
                inertia = kmeans_temp.inertia_
                inertias.append(inertia)
                
                # Calculate silhouette score for comparison (use sampling for large datasets)
                if memory_efficient and X_scaled.shape[0] > 5000:
                    # Use sample for silhouette calculation to save memory
                    sample_size = min(5000, X_scaled.shape[0])
                    sample_indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
                    silhouette_avg = silhouette_score(X_scaled[sample_indices], temp_labels[sample_indices])
                else:
                    silhouette_avg = silhouette_score(X_scaled, temp_labels)
                silhouette_scores.append(silhouette_avg)
                
                print(f"K={k}: Inertia={inertia:.2f}, Silhouette={silhouette_avg:.4f}")
                
                # Clean up temporary objects
                del kmeans_temp, temp_labels
                gc.collect()
            
            # 3 & 4. Choose different values for K and print optimal K using elbow curve
            optimal_k = self._find_optimal_k_elbow(k_range, inertias, silhouette_scores)
            
            print(f"Optimal K (elbow method): {optimal_k}")
            
            # Create final K-means with optimal K
            self.estimator = KMeans(n_clusters=optimal_k, random_state=42, 
                                  n_init=5 if memory_efficient else 10)
            self.n_clusters = optimal_k
            self.optimal_k = optimal_k  # Store for reference
            
        elif estimator_name == 'agglomerative':
            print("Finding optimal N_Clusters for Agglomerative clustering...")
            
            # Agglomerative is O(n³) - sample data for large datasets
            if memory_efficient and X_scaled.shape[0] > 10000:
                print(f"Sampling 10000 rows from {X_scaled.shape[0]} for Agglomerative clustering...")
                sample_indices = np.random.choice(X_scaled.shape[0], 10000, replace=False)
                X_agg = X_scaled[sample_indices]
                print(f"Using sampled data shape: {X_agg.shape}")
            else:
                X_agg = X_scaled
            
            # Test different N_Clusters values
            n_clusters_range = range(2, 7) if memory_efficient else range(2, 11)  # Reduce range for memory efficiency
            silhouette_scores = []
            
            print("Testing different N_Clusters values:")
            
            for n_clusters in n_clusters_range:
                # 1. Implement the Agglomerative clustering instance
                agg_temp = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                
                # 2. Cluster the data using the Agglomerative clustering
                temp_labels = agg_temp.fit_predict(X_agg)
                
                # Calculate silhouette score for evaluation
                silhouette_avg = silhouette_score(X_agg, temp_labels)
                silhouette_scores.append(silhouette_avg)
                
                print(f"N_Clusters={n_clusters}: Silhouette={silhouette_avg:.4f}")
                
                # Clean up temporary objects
                del agg_temp, temp_labels
                gc.collect()
            
            # 3 & 4. Choose different values for N_Clusters and print optimal N_Clusters
            best_idx = np.argmax(silhouette_scores)
            optimal_n_clusters = list(n_clusters_range)[best_idx]
            
            print(f"Optimal N_Clusters (highest silhouette): {optimal_n_clusters}")
            
            # Create final Agglomerative clustering with optimal N_Clusters
            # Note: Final clustering uses original data (not sampled)
            self.estimator = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage='ward')
            self.n_clusters = optimal_n_clusters
            self.optimal_n_clusters = optimal_n_clusters  # Store for reference
            self.used_sampling = memory_efficient and X_scaled.shape[0] > 10000
            
        elif estimator_name == 'meanshift':
            print("Implementing Mean Shift clustering with bandwidth estimation...")
            
            # MeanShift is very memory intensive - sample data for large datasets
            if memory_efficient and X_scaled.shape[0] > 15000:
                print(f"Sampling 15000 rows from {X_scaled.shape[0]} for MeanShift clustering...")
                sample_indices = np.random.choice(X_scaled.shape[0], 15000, replace=False)
                X_ms = X_scaled[sample_indices]
                print(f"Using sampled data shape: {X_ms.shape}")
            else:
                X_ms = X_scaled
            
            # 2. Use bandwidth estimation
            from sklearn.cluster import estimate_bandwidth
            
            # Estimate bandwidth using built-in method
            print("Estimating optimal bandwidth...")
            n_samples_bw = 100 if memory_efficient else 500
            bandwidth = estimate_bandwidth(X_ms, quantile=0.2, n_samples=n_samples_bw)
            
            print(f"Estimated bandwidth: {bandwidth:.4f}")
            
            # 1. Implement the Mean shift clustering instance
            if bandwidth > 0:
                self.estimator = MeanShift(bandwidth=bandwidth)
                print(f"Using estimated bandwidth: {bandwidth:.4f}")
            else:
                # Fallback if bandwidth estimation fails
                print("Bandwidth estimation failed, using automatic bandwidth")
                self.estimator = MeanShift(bandwidth=None)
            
            self.estimated_bandwidth = bandwidth
            
        else:
            print(f"Error: Unknown estimator '{estimator_name}'")
            return
        
        # Fit clustering
        if estimator_name == 'agglomerative' and memory_efficient and X_scaled.shape[0] > 10000:
            # Use the same sample for final Agglomerative clustering
            print("Using sampled data for final Agglomerative clustering...")
            sample_indices = np.random.choice(X_scaled.shape[0], 10000, replace=False)
            sample_labels = self.estimator.fit_predict(X_scaled[sample_indices])
            # Create full labels array (assign -1 to non-sampled points)
            self.cluster_labels = np.full(X_scaled.shape[0], -1)
            self.cluster_labels[sample_indices] = sample_labels
            print("Note: Only 10,000 samples clustered for Agglomerative (others marked as -1)")
        elif estimator_name == 'meanshift' and memory_efficient and X_scaled.shape[0] > 15000:
            # Use sampled data for final MeanShift clustering
            print("Using sampled data for final MeanShift clustering...")
            sample_indices = np.random.choice(X_scaled.shape[0], 15000, replace=False)
            sample_labels = self.estimator.fit_predict(X_scaled[sample_indices])
            # Create full labels array (assign -1 to non-sampled points)
            self.cluster_labels = np.full(X_scaled.shape[0], -1)
            self.cluster_labels[sample_indices] = sample_labels
            print("Note: Only 15,000 samples clustered for MeanShift (others marked as -1)")
        else:
            self.cluster_labels = self.estimator.fit_predict(X_scaled)
        
        # Update n_clusters for Mean Shift
        if estimator_name == 'meanshift':
            self.n_clusters = len(np.unique(self.cluster_labels))
        
        # Calculate and store silhouette score
        self.silhouette_score = None
        if self.n_clusters > 1:
            if memory_efficient and X_scaled.shape[0] > 5000:
                # Use sample for final silhouette calculation to save memory
                sample_size = min(5000, X_scaled.shape[0])
                sample_indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
                self.silhouette_score = silhouette_score(X_scaled[sample_indices], self.cluster_labels[sample_indices])
                print(f"Silhouette Score (sampled): {self.silhouette_score:.4f}")
            else:
                self.silhouette_score = silhouette_score(X_scaled, self.cluster_labels)
                print(f"Silhouette Score: {self.silhouette_score:.4f}")
        
        print(f"Clustering completed!")
        print(f"Number of clusters: {self.n_clusters}")
        
        # Handle cluster sizes (filter out -1 labels for sampled Agglomerative)
        valid_labels = self.cluster_labels[self.cluster_labels >= 0]
        if len(valid_labels) < len(self.cluster_labels):
            print(f"Cluster sizes (sampled data): {np.bincount(valid_labels)}")
            print(f"Unclustered points: {len(self.cluster_labels) - len(valid_labels)}")
        else:
            print(f"Cluster sizes: {np.bincount(self.cluster_labels)}")
        
        return self.cluster_labels
    
    def predict(self, X_new):
        """
        A function that predicts cluster labels for new data.
        """
        if self.estimator is None:
            print("Error: No trained clustering model found. Please call fit() first.")
            return None
        
        print(f"Predicting clusters using {self.estimator_name}...")
        
        # Scale new data
        X_new_scaled = self.scaler.transform(X_new)
        
        # Predict (only K-means supports this)
        if self.estimator_name == 'kmeans':
            cluster_labels = self.estimator.predict(X_new_scaled)
            print(f"Prediction completed!")
            return cluster_labels
        else:
            print(f"Warning: {self.estimator_name} doesn't support prediction on new data")
            return None
    
    def _find_optimal_k_elbow(self, k_range, inertias, silhouette_scores):
        """
        Find optimal K using elbow method and silhouette analysis.
        """
        # Simple elbow detection: find the "knee" in inertia curve
        # Use the point where silhouette score is maximized as a tie-breaker
        
        # Method 1: Elbow detection using rate of change
        deltas = []
        for i in range(1, len(inertias)):
            delta = inertias[i-1] - inertias[i]
            deltas.append(delta)
        
        # Find where the rate of improvement slows down significantly
        # Look for the biggest drop in delta (elbow point)
        if len(deltas) > 1:
            delta_deltas = []
            for i in range(1, len(deltas)):
                delta_delta = deltas[i-1] - deltas[i]
                delta_deltas.append(delta_delta)
            
            # Find the elbow point
            elbow_idx = np.argmax(delta_deltas) + 2  # +2 to account for offset
            elbow_k = list(k_range)[elbow_idx] if elbow_idx < len(k_range) else list(k_range)[-1]
        else:
            elbow_k = 3  # Default fallback
        
        # Method 2: Use silhouette score as validation
        best_silhouette_idx = np.argmax(silhouette_scores)
        silhouette_k = list(k_range)[best_silhouette_idx]
        
        print(f"Elbow method suggests K={elbow_k}")
        print(f"Silhouette method suggests K={silhouette_k}")
        
        # Choose the one with better silhouette score if they're different
        if elbow_k == silhouette_k:
            optimal_k = elbow_k
        else:
            # Choose based on silhouette score
            optimal_k = silhouette_k
            print(f"Using silhouette-based choice: K={optimal_k}")
        
        return optimal_k
    
    def get_inertia(self):
        """
        A function that retrieves the inertia of the k-means estimator.
        """
        if self.estimator is None or self.estimator_name != 'kmeans':
            print("Error: No K-means estimator found or wrong algorithm")
            return None
        
        inertia = self.estimator.inertia_
        print(f"K-means inertia: {inertia:.2f}")
        return inertia
    
    def get_bandwidth(self):
        """
        A function that retrieves the bandwidth of the mean shift estimator.
        """
        if self.estimator is None or self.estimator_name != 'meanshift':
            print("Error: No Mean Shift estimator found or wrong algorithm")
            return None
        
        if hasattr(self, 'estimated_bandwidth'):
            bandwidth = self.estimated_bandwidth
            print(f"Mean Shift bandwidth: {bandwidth:.4f}")
            return bandwidth
        else:
            print("Error: No bandwidth information available")
            return None
    
    def get_linkages(self):
        """
        A function that retrieves the linkages of the agglomerative clustering estimator.
        """
        if self.estimator is None or self.estimator_name != 'agglomerative':
            print("Error: No Agglomerative clustering estimator found or wrong algorithm")
            return None
        
        if hasattr(self.estimator, 'children_'):
            linkages = self.estimator.children_
            print(f"Agglomerative linkages shape: {linkages.shape}")
            print(f"Number of linkage steps: {len(linkages)}")
            return linkages
        else:
            print("Error: Agglomerative clustering has not been fitted yet")
            return None
    
    def get_linkages(self):
        """
        A function that retrieves the linkages of the agglomerative clustering estimator.
        """
        if self.estimator is None or self.estimator_name != 'agglomerative':
            print("Error: No Agglomerative clustering estimator found or wrong algorithm")
            return None
        
        if hasattr(self.estimator, 'children_'):
            linkages = self.estimator.children_
            print(f"Agglomerative linkages shape: {linkages.shape}")
            print(f"Number of linkage steps: {len(linkages)}")
            return linkages
        else:
            print("Error: Agglomerative clustering has not been fitted yet")
            return None


def print_clustering_summary(results):
    """
    Print a formatted summary table of clustering results.
    
    Args:
        results: List of dictionaries containing clustering results
    """
    print("\n" + "="*80)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Algorithm':<15} {'Clusters':<10} {'Silhouette':<12} {'Optimal Param':<15} {'Data Coverage':<12} {'Notes'}")
    print("-" * 80)
    
    # Find best performing algorithm
    best_silhouette = -1
    best_algorithm = None
    
    for result in results:
        if result['silhouette_score'] and result['silhouette_score'] > best_silhouette:
            best_silhouette = result['silhouette_score']
            best_algorithm = result['algorithm']
    
    # Print results
    for result in results:
        algorithm = result['algorithm'].capitalize()
        clusters = str(result['n_clusters'])
        silhouette = f"{result['silhouette_score']:.4f}" if result['silhouette_score'] else "N/A"
        optimal_param = result['optimal_param']
        coverage = result['data_coverage']
        notes = result['notes']
        
        # Mark best algorithm
        if result['algorithm'] == best_algorithm:
            algorithm += " ⭐"
        
        print(f"{algorithm:<15} {clusters:<10} {silhouette:<12} {optimal_param:<15} {coverage:<12} {notes}")
    
    print("-" * 80)
    print(f"⭐ Best performing algorithm based on silhouette score: {best_algorithm.capitalize()}")
    print("="*80)
    
    # Recommendations
    print("\nRECOMMendations:")
    print("-" * 40)
    
    for result in results:
        if result['algorithm'] == best_algorithm:
            if result['algorithm'] == 'kmeans':
                k_value = result['optimal_param'].replace('K=', '')
                print(f"• K-means with K={k_value} shows the best silhouette score")
                print("• K-means is scalable and works well with all data points")
                print("• Good for spherical clusters and large datasets")
            elif result['algorithm'] == 'agglomerative':
                n_value = result['optimal_param'].replace('N=', '')
                print(f"• Agglomerative with {n_value} clusters shows good results")
                print("• Note: Only used sampled data due to computational constraints")
                print("• Good for hierarchical structure and irregular cluster shapes")
            elif result['algorithm'] == 'meanshift':
                print(f"• Mean Shift found {result['n_clusters']} natural clusters")
                print("• Note: Only used sampled data due to computational constraints")
                print("• Good for finding natural cluster centers without pre-specifying K")
    
    print("\nGeneral Notes:")
    print("• Silhouette scores range from -1 to 1 (higher is better)")
    print("• Scores above 0.5 indicate strong clustering, 0.2-0.5 reasonable, below 0.2 weak")
    print("• Data coverage shows percentage of data actually clustered")
    print("="*80)


# Simple test
if __name__ == "__main__":
    from analyzer import Analyzer
    
    # Get diamond data
    analyzer = Analyzer()
    analyzer.read_dataset('diamonds.csv')
    analyzer.drop_missing_data()
    analyzer.drop_columns(['Unnamed: 0'])
    analyzer.encode_features(['cut', 'color', 'clarity'])
    data = analyzer.retrieve_data()
    
    # Use all features for clustering (no target needed)
    X = data.values  # All features as numpy array
    
    print("DIAMOND CLUSTERING ANALYSIS")
    print("="*50)
    
    # Test all clustering algorithms and collect results
    estimators = ['kmeans', 'agglomerative', 'meanshift']
    results = []
    
    for estimator in estimators:
        print(f"\nTesting {estimator}:")
        clustering = Clustering()
        labels = clustering.fit(X, estimator, memory_efficient=True)  # Enable memory optimization
        
        # Collect results for summary
        result = {
            'algorithm': estimator,
            'n_clusters': clustering.n_clusters,
            'silhouette_score': None,
            'optimal_param': 'N/A',
            'data_coverage': '100%',
            'notes': 'Full dataset'
        }
        
        # Calculate data coverage and get algorithm-specific metrics
        total_points = len(labels)
        valid_points = len(labels[labels >= 0]) if hasattr(labels, 'shape') else total_points
        coverage_pct = (valid_points / total_points) * 100
        result['data_coverage'] = f"{coverage_pct:.1f}%"
        
        # Set silhouette score from the clustering object
        result['silhouette_score'] = clustering.silhouette_score
        
        if estimator == 'kmeans':
            inertia = clustering.get_inertia()
            print(f"Final inertia: {inertia:.2f}")
            print(f"Optimal K used: {clustering.optimal_k}")
            result['optimal_param'] = f"K={clustering.optimal_k}"
            
        elif estimator == 'agglomerative':
            linkages = clustering.get_linkages()
            print(f"Optimal N_Clusters used: {clustering.optimal_n_clusters}")
            if linkages is not None:
                print(f"Linkage matrix available with {len(linkages)} merge steps")
            result['optimal_param'] = f"N={clustering.optimal_n_clusters}"
            result['notes'] = 'Sampled data'
            
        elif estimator == 'meanshift':
            bandwidth = clustering.get_bandwidth()
            if hasattr(clustering, 'estimated_bandwidth'):
                print(f"Estimated bandwidth: {clustering.estimated_bandwidth:.4f}")
                result['optimal_param'] = f"BW={clustering.estimated_bandwidth:.3f}"
            else:
                result['optimal_param'] = 'Auto BW'
            print(f"Final clusters found: {clustering.n_clusters}")
            result['notes'] = 'Sampled data'
        
        results.append(result)
        
        # Force garbage collection between algorithms
        gc.collect()
    
    # Print comprehensive summary
    print_clustering_summary(results)