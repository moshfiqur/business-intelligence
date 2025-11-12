"""
Data Loader for Enterprise Datasets
Handles loading enterprise datasets from Kaggle
"""

import pandas as pd
import os
from typing import Dict, List, Optional
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

class EnterpriseDataLoader:
    """
    Loads enterprise datasets from Kaggle
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()
        self.logger = logging.getLogger(__name__)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def load_kaggle_dataset(self, kaggle_ref: str, dataset_name: str) -> pd.DataFrame:
        """
        Download a Kaggle dataset into a dataset-specific folder under data/ without renaming files.
        """
        try:
            # dataset-specific directory inside data/
            dataset_path = os.path.join(self.data_dir, kaggle_ref)
            os.makedirs(dataset_path, exist_ok=True)

            # Check if any CSV already exists in the dataset folder; if not, download
            existing_csvs = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
            if not existing_csvs:
                self.kaggle_api.dataset_download_files(
                    kaggle_ref,
                    path=dataset_path,
                    unzip=True
                )
            
            # Find CSV files within the dataset folder
            files = os.listdir(dataset_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {dataset_path}")

            # Fallback to the first CSV if main_csv not specified
            file_path = os.path.join(dataset_path, csv_files[0])

            df = pd.read_csv(file_path)

            self.logger.info(f"Loaded dataset {kaggle_ref} (folder: {dataset_name}) with shape {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading Kaggle dataset {kaggle_ref}: {e}")
            raise
    
    def get_enterprise_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Get multiple enterprise datasets from Kaggle
        Downloads datasets and saves with custom filenames
        """
        datasets = {}
        
        # Kaggle datasets to download into dataset-specific folders
        kaggle_datasets = [
            # (kaggle_ref, dataset_generic_name)
            ('pavansubhasht/ibm-hr-analytics-attrition-dataset', 'hr_analytics'),
            ('aslanahmedov/walmart-sales-forecast', 'sales_forecasting'),  # this dataset contains 4 files: features.csv, stores.csv, test.csv, train.csv
            ('rohitsahoo/sales-forecasting', 'superstore_sales')
        ]
        
        for kaggle_ref, dataset_name in kaggle_datasets:
            try:
                df = self.load_kaggle_dataset(kaggle_ref, dataset_name)
                datasets[dataset_name] = df
                self.logger.info(
                    f"Successfully loaded Kaggle dataset: {kaggle_ref} into folder '{dataset_name}'"
                )
            except Exception as e:
                self.logger.warning(f"Could not load Kaggle dataset {kaggle_ref}: {e}")
        
        return datasets

# Example usage
if __name__ == "__main__":
    loader = EnterpriseDataLoader()
    
    # Load datasets
    datasets = loader.get_enterprise_datasets()
    
    print("Available datasets:")
    for name, df in datasets.items():
        print(f"  {name}: {df.shape}")
        print(f"    Columns: {list(df.columns)}")
