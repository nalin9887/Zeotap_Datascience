import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class LookalikeModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_matrix = None
        self.customer_ids = None
        self.customers_df = None
        self.products_df = None
        self.transactions_df = None
        
    def load_data(self):
        """Load the datasets"""
        self.customers_df = pd.read_csv('Customers.csv')
        self.products_df = pd.read_csv('Products.csv')
        self.transactions_df = pd.read_csv('Transactions.csv')
        return self.customers_df, self.products_df, self.transactions_df
        
    def create_customer_features(self):
        """Create customer features for similarity calculation"""
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
        
    def fit(self, customer_features):
        """Fit the model with customer features"""
        feature_cols = ['total_spend', 'avg_transaction_value', 'transaction_count',
                       'total_quantity', 'avg_quantity', 'unique_products'] + \
                      [col for col in customer_features.columns if col.startswith('region_')]
        
        self.feature_matrix = self.scaler.fit_transform(customer_features[feature_cols])
        self.customer_ids = customer_features['CustomerID'].values
        
    def get_lookalikes(self, customer_id, n_recommendations=3):
        """Get top similar customers"""
        customer_index = np.where(self.customer_ids == customer_id)[0][0]
        similarity_scores = cosine_similarity([self.feature_matrix[customer_index]], self.feature_matrix)[0]
        similar_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations+1]
        
        return [(self.customer_ids[idx], similarity_scores[idx]) for idx in similar_indices]

def main():
    # Initialize model
    model = LookalikeModel()
    
    # Load data
    model.load_data()
    
    # Create features and fit model
    customer_features = model.create_customer_features()
    model.fit(customer_features)
    
    # Generate recommendations for first 20 customers
    results = []
    for i in range(1, 21):
        customer_id = f'C{i:04d}'
        lookalikes = model.get_lookalikes(customer_id)
        results.append({
            'customer_id': customer_id,
            'lookalikes': lookalikes
        })
    
    # Save results to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv('Nalin_Sharma_Lookalike.csv', index=False)

if __name__ == "__main__":
    main() 