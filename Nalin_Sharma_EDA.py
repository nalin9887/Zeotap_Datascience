import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class EDAAnalysis:
    def __init__(self):
        self.customers_df = None
        self.products_df = None
        self.transactions_df = None
        self.merged_df = None
        
    def load_data(self):
        """Load the datasets"""
        self.customers_df = pd.read_csv('Customers.csv')
        self.products_df = pd.read_csv('Products.csv')
        self.transactions_df = pd.read_csv('Transactions.csv')
        
    def preprocess_data(self):
        """Preprocess and merge the datasets"""
        # Convert date columns
        self.customers_df['SignupDate'] = pd.to_datetime(self.customers_df['SignupDate'])
        self.transactions_df['TransactionDate'] = pd.to_datetime(self.transactions_df['TransactionDate'])
        
        # Merge datasets
        self.merged_df = self.transactions_df.merge(self.customers_df, on='CustomerID')
        self.merged_df = self.merged_df.merge(self.products_df, on='ProductID')
        
    def analyze_sales_by_region(self):
        """Analyze sales performance by region"""
        region_sales = self.merged_df.groupby('Region').agg({
            'TotalValue': ['sum', 'mean', 'count']
        }).round(2)
        
        plt.figure(figsize=(12, 6))
        region_sales['TotalValue']['sum'].plot(kind='bar')
        plt.title('Total Sales by Region')
        plt.xlabel('Region')
        plt.ylabel('Total Sales (USD)')
        plt.tight_layout()
        plt.savefig('region_sales.png')
        plt.close()
        
        return region_sales
        
    def analyze_product_categories(self):
        """Analyze product category performance"""
        category_analysis = self.merged_df.groupby('Category').agg({
            'TotalValue': ['sum', 'mean'],
            'Quantity': 'sum',
            'TransactionID': 'count'
        }).round(2)
        
        plt.figure(figsize=(12, 6))
        category_analysis['TotalValue']['sum'].plot(kind='bar')
        plt.title('Sales by Product Category')
        plt.xlabel('Category')
        plt.ylabel('Total Sales (USD)')
        plt.tight_layout()
        plt.savefig('category_sales.png')
        plt.close()
        
        return category_analysis
        
    def generate_insights(self):
        """Generate business insights"""
        insights = []
        
        # Regional performance
        region_sales = self.analyze_sales_by_region()
        top_region = region_sales['TotalValue']['sum'].idxmax()
        insights.append(f"1. Regional Performance: {top_region} leads in sales with ${region_sales['TotalValue']['sum'].max():,.2f}")
        
        # Category performance
        category_analysis = self.analyze_product_categories()
        top_category = category_analysis['TotalValue']['sum'].idxmax()
        insights.append(f"2. Product Categories: {top_category} is the best-performing category")
        
        # Customer behavior
        avg_transaction = self.merged_df['TotalValue'].mean()
        insights.append(f"3. Customer Behavior: Average transaction value is ${avg_transaction:,.2f}")
        
        # Seasonal trends
        monthly_sales = self.merged_df.groupby(self.merged_df['TransactionDate'].dt.month)['TotalValue'].sum()
        peak_month = monthly_sales.idxmax()
        insights.append(f"4. Seasonal Trends: Peak sales observed in month {peak_month}")
        
        # Product diversity
        category_count = len(category_analysis)
        insights.append(f"5. Product Mix: {category_count} distinct categories contributing to revenue")
        
        return insights

def main():
    # Initialize analysis
    eda = EDAAnalysis()
    
    # Load and process data
    eda.load_data()
    eda.preprocess_data()
    
    # Generate insights
    insights = eda.generate_insights()
    
    # Save insights to PDF
    with open('Nalin_Sharma_EDA.pdf', 'w') as f:
        f.write("Business Insights from EDA\n\n")
        for insight in insights:
            f.write(f"{insight}\n\n")

if __name__ == "__main__":
    main() 