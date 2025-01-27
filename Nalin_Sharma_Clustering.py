import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class CustomerSegmentation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_n_clusters = None
        self.best_db_score = float('inf')
        self.customers_df = None
        self.transactions_df = None
        
    def load_data(self):
        """Load the datasets"""
        self.customers_df = pd.read_csv('Customers.csv')
        self.transactions_df = pd.read_csv('Transactions.csv')
        return self.customers_df, self.transactions_df
        
    def create_features(self):
        """Create customer features for clustering"""
        customer_metrics = self.transactions_df.groupby('CustomerID').agg({
            'TotalValue': ['sum', 'mean', 'count'],
            'Quantity': ['sum', 'mean'],
            'ProductID': 'nunique'
        }).reset_index()
        
        customer_metrics.columns = ['CustomerID', 'total_spend', 'avg_transaction_value', 
                                  'transaction_count', 'total_quantity', 'avg_quantity',
                                  'unique_products']
        
        customer_features = self.customers_df.merge(customer_metrics, on='CustomerID')
        region_dummies = pd.get_dummies(customer_features['Region'], prefix='region')
        customer_features = pd.concat([customer_features, region_dummies], axis=1)
        
        return customer_features
        
    def find_optimal_clusters(self, features, min_clusters=2, max_clusters=10):
        """Find optimal number of clusters"""
        feature_cols = ['total_spend', 'avg_transaction_value', 'transaction_count',
                       'total_quantity', 'avg_quantity', 'unique_products'] + \
                      [col for col in features.columns if col.startswith('region_')]
        
        X = self.scaler.fit_transform(features[feature_cols])
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            db_score = davies_bouldin_score(X, cluster_labels)
            
            if db_score < self.best_db_score:
                self.best_db_score = db_score
                self.best_n_clusters = n_clusters
                self.best_model = kmeans
        
        return X
        
    def visualize_clusters(self, X, features):
        """Visualize clusters using PCA"""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=self.best_model.labels_, cmap='viridis')
        plt.title(f'Customer Segments (n_clusters={self.best_n_clusters})')
        plt.colorbar(scatter)
        plt.savefig('cluster_visualization.png')
        plt.close()
        
        features['Cluster'] = self.best_model.labels_
        return features

def main():
    # Initialize segmentation
    segmentation = CustomerSegmentation()
    
    # Load data
    segmentation.load_data()
    
    # Create features and find optimal clusters
    customer_features = segmentation.create_features()
    X = segmentation.find_optimal_clusters(customer_features)
    
    # Visualize clusters
    clustered_features = segmentation.visualize_clusters(X, customer_features)
    
    # Save clustering results
    results = {
        'n_clusters': segmentation.best_n_clusters,
        'db_index': segmentation.best_db_score,
        'cluster_sizes': clustered_features['Cluster'].value_counts().to_dict()
    }
    
    # Save results to PDF
    with open('Nalin_Sharma_Clustering.pdf', 'w') as f:
        f.write(f"Number of clusters: {results['n_clusters']}\n")
        f.write(f"Davies-Bouldin Index: {results['db_index']:.4f}\n")
        f.write("\nCluster sizes:\n")
        for cluster, size in results['cluster_sizes'].items():
            f.write(f"Cluster {cluster}: {size} customers\n")

if __name__ == "__main__":
    main() 