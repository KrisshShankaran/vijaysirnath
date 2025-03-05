import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.label_encoders = {}
        
    def load_data(self, file):
        """Load data from CSV file"""
        try:
            self.raw_data = pd.read_csv(file)
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self, data):
        """Preprocess the input data"""
        try:
            # Create a copy of input data
            self.processed_data = data.copy()
            
            # Handle missing values
            self.processed_data.fillna(method='ffill', inplace=True)
            self.processed_data.fillna(method='bfill', inplace=True)
            
            # Convert categorical columns to string type
            categorical_cols = self.processed_data.select_dtypes(include=['object']).columns
            self.processed_data[categorical_cols] = self.processed_data[categorical_cols].astype(str)
            
            # Encode categorical variables
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                self.processed_data[col] = self.label_encoders[col].fit_transform(self.processed_data[col])
            
            # Remove duplicates
            self.processed_data.drop_duplicates(inplace=True)
            
            return self.processed_data
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            return None
    
    def get_data_summary(self):
        """Get summary statistics of the processed data"""
        if self.processed_data is None:
            return None
        
        try:
            # Basic Information
            basic_info = pd.DataFrame({
                'Metric': ['Total Rows', 'Total Columns', 'Memory Usage'],
                'Value': [
                    len(self.processed_data),
                    len(self.processed_data.columns),
                    f"{self.processed_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                ]
            })
            
            # Missing Values
            missing_values = pd.DataFrame({
                'Column': self.processed_data.columns,
                'Missing Values': self.processed_data.isnull().sum(),
                'Missing %': (self.processed_data.isnull().sum() / len(self.processed_data) * 100).round(2)
            })
            
            # Column Types
            column_types = pd.DataFrame({
                'Column': self.processed_data.columns,
                'Data Type': self.processed_data.dtypes.astype(str)
            })
            
            # Numerical Summary
            numerical_summary = self.processed_data.describe()
            
            return {
                'basic_info': basic_info,
                'missing_values': missing_values,
                'column_types': column_types,
                'numerical_summary': numerical_summary
            }
        except Exception as e:
            print(f"Error getting data summary: {str(e)}")
            return None 