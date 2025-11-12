"""
Data Loader for Enterprise Datasets
Handles loading and preprocessing of enterprise data from various sources
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

class EnterpriseDataLoader:
    """
    Loads enterprise datasets from various sources (Kaggle, local files, etc.)
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()
        self.logger = logging.getLogger(__name__)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def load_kaggle_dataset(self, dataset_name: str, file_pattern: str = None) -> pd.DataFrame:
        """
        Load dataset from Kaggle
        """
        try:
            # Download dataset
            self.kaggle_api.dataset_download_files(
                dataset_name, 
                path=self.data_dir, 
                unzip=True
            )
            
            # Find the downloaded files
            files = os.listdir(self.data_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            
            if file_pattern:
                csv_files = [f for f in csv_files if file_pattern in f]
            
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
            
            # Load the first CSV file
            file_path = os.path.join(self.data_dir, csv_files[0])
            df = pd.read_csv(file_path)
            
            self.logger.info(f"Loaded dataset {dataset_name} with shape {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading Kaggle dataset {dataset_name}: {e}")
            raise
    
    def load_sample_hr_data(self) -> pd.DataFrame:
        """
        Load sample HR data for testing
        """
        return pd.DataFrame({
            'employee_id': [101, 102, 103, 104, 105],
            'emp_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
            'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Sales'],
            'salary_amount': [75000, 65000, 80000, 55000, 70000],
            'hire_date': ['2020-01-15', '2019-03-20', '2021-06-10', '2018-11-05', '2020-02-14'],
            'performance_rating': [4, 3, 5, 3, 4],
            'attendance_days': [240, 235, 245, 230, 242]
        })
    
    def load_sample_sales_data(self) -> pd.DataFrame:
        """
        Load sample sales data for testing
        """
        return pd.DataFrame({
            'transaction_id': [1001, 1002, 1003, 1004, 1005],
            'customer_name': ['ABC Corp', 'XYZ Ltd', 'Global Inc', 'Tech Solutions', 'Innovate Co'],
            'product_category': ['Software', 'Hardware', 'Services', 'Software', 'Hardware'],
            'sales_amount': [15000, 25000, 18000, 22000, 19000],
            'transaction_date': ['2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19'],
            'region': ['North', 'South', 'East', 'West', 'North'],
            'sales_rep': ['Sarah Chen', 'Mike Johnson', 'Lisa Wang', 'David Kim', 'Sarah Chen']
        })
    
    def load_sample_finance_data(self) -> pd.DataFrame:
        """
        Load sample finance data for testing
        """
        return pd.DataFrame({
            'invoice_number': ['INV001', 'INV002', 'INV003', 'INV004', 'INV005'],
            'vendor_name': ['Office Supplies Inc', 'Tech Equipment Co', 'Software Solutions', 'Consulting Services', 'Marketing Agency'],
            'invoice_amount': [1500.50, 7500.00, 12000.75, 8500.25, 3200.00],
            'due_date': ['2023-02-15', '2023-02-20', '2023-02-25', '2023-03-01', '2023-03-05'],
            'payment_status': ['Paid', 'Pending', 'Overdue', 'Paid', 'Pending'],
            'cost_center': ['IT', 'Operations', 'R&D', 'Consulting', 'Marketing']
        })
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataframe for semantic analysis
        """
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Convert date columns to datetime
        date_columns = [col for col in processed_df.columns 
                       if any(keyword in col.lower() for keyword in ['date', 'time'])]
        
        for col in date_columns:
            try:
                processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
            except:
                pass  # Keep as string if conversion fails
        
        # Clean column names
        processed_df.columns = [col.strip().replace(' ', '_').lower() for col in processed_df.columns]
        
        return processed_df
    
    def get_enterprise_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Get multiple enterprise datasets for semantic analysis
        """
        datasets = {}
        
        # Load sample datasets
        datasets['hr_data'] = self.preprocess_data(self.load_sample_hr_data())
        datasets['sales_data'] = self.preprocess_data(self.load_sample_sales_data())
        datasets['finance_data'] = self.preprocess_data(self.load_sample_finance_data())
        
        # Try to load real datasets from Kaggle if available
        kaggle_datasets = [
            ('pavansubhasht/ibm-hr-analytics-attrition-dataset', 'hr_analytics'),
            # ('rohanrao/sales-data-for-forecasting', 'sales_forecasting'),
            # ('nelgiriyewithana/global-superstore-dataset', 'superstore_sales')
        ]
        
        for dataset_name, dataset_key in kaggle_datasets:
            try:
                df = self.load_kaggle_dataset(dataset_name)
                datasets[dataset_key] = self.preprocess_data(df)
                self.logger.info(f"Successfully loaded Kaggle dataset: {dataset_name}")
            except Exception as e:
                self.logger.warning(f"Could not load Kaggle dataset {dataset_name}: {e}")
        
        return datasets

# Example usage
if __name__ == "__main__":
    loader = EnterpriseDataLoader()
    
    # Load sample datasets
    datasets = loader.get_enterprise_datasets()
    
    print("Available datasets:")
    for name, df in datasets.items():
        print(f"  {name}: {df.shape}")
        print(f"    Columns: {list(df.columns)}")
