import pandas as pd
import numpy as np
from ctgan import CTGAN
from scipy.stats import ks_2samp
from sdv.metrics.tabular import CSTest
import streamlit as st
import plotly.express as px

class CTGANModule:
    def __init__(self, epochs=300, batch_size=500):
        self.model = CTGAN(epochs=epochs, batch_size=batch_size)
        self.fitted = False
        
    def fit(self, data):
        """Fit the CTGAN model to the data"""
        try:
            # Check for null values in the data
            if data.isnull().any().any():
                null_cols = data.columns[data.isnull().any()].tolist()
                st.error(f"CTGAN does not support null values. Please preprocess your data first. Columns with null values: {null_cols}")
                return False
                
            # Check data types
            for col in data.columns:
                if data[col].dtype not in ['int64', 'float64', 'object']:
                    st.warning(f"Column {col} has unsupported data type: {data[col].dtype}. Converting to string.")
                    data[col] = data[col].astype(str)
            
            # Fit the model
            self.model.fit(data)
            self.fitted = True
            return True
        except Exception as e:
            st.error(f"Error fitting CTGAN model: {str(e)}")
            return False
    
    def generate_samples(self, n_samples):
        """Generate synthetic samples"""
        if not self.fitted:
            st.error("Model not fitted yet. Please fit the model first.")
            return None
        
        try:
            synthetic_data = self.model.sample(n_samples)
            return synthetic_data
        except Exception as e:
            st.error(f"Error generating synthetic samples: {str(e)}")
            return None
    
    def evaluate_synthetic_data(self, real_data, synthetic_data):
        """Evaluate the quality of synthetic data"""
        try:
            similarity_scores = {}
            
            # KS Test for numerical columns
            numerical_cols = real_data.select_dtypes(include=['int64', 'float64']).columns
            for col in numerical_cols:
                ks_stat, ks_pval = ks_2samp(real_data[col], synthetic_data[col])
                similarity_scores[col] = {'KS_Stat': ks_stat, 'P-Value': ks_pval}
            
            # Categorical Similarity Test
            categorical_cols = real_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                similarity_scores[col] = CSTest.compute(real_data[col], synthetic_data[col])
            
            return similarity_scores
        except Exception as e:
            st.error(f"Error evaluating synthetic data: {str(e)}")
            return None
    
    def visualize_similarity(self):
        """Visualize the similarity between real and synthetic data"""
        try:
            if not hasattr(self, 'similarity_scores'):
                st.error("No similarity scores available. Please run evaluate_synthetic_data first.")
                return
            
            # Create a DataFrame for visualization
            similarity_df = pd.DataFrame(self.similarity_scores).T
            
            # Create a heatmap
            fig = px.imshow(similarity_df,
                          title="Similarity Scores Heatmap",
                          color_continuous_scale="RdBu")
            
            st.plotly_chart(fig)
            
            # Display summary statistics
            st.subheader("Similarity Summary")
            st.dataframe(similarity_df.describe())
            
        except Exception as e:
            st.error(f"Error visualizing similarity: {str(e)}")
            return
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.fitted:
            st.error("No trained model to save.")
            return False
            
        try:
            self.model.save(filepath)
            return True
        except Exception as e:
            st.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            self.model = CTGAN.load(filepath)
            self.fitted = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False 