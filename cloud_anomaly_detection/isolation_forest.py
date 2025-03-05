import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

class IsolationForestModule:
    def __init__(self, contamination=0.03, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.fitted = False
        self.anomaly_scores = None
    
    def fit(self, data):
        """Fit the Isolation Forest model"""
        try:
            # Select numerical features
            numerical_data = data.select_dtypes(include=['int64', 'float64'])
            
            # Fit the model
            self.model.fit(numerical_data)
            self.fitted = True
            return True
        except Exception as e:
            st.error(f"Error fitting Isolation Forest model: {str(e)}")
            return False
    
    def predict(self, data):
        """Predict anomalies in the data"""
        if not self.fitted:
            st.warning("Model not fitted yet. Please fit the model first.")
            return None
        
        try:
            # Select numerical features
            numerical_data = data.select_dtypes(include=['int64', 'float64'])
            
            # Get anomaly scores
            self.anomaly_scores = self.model.score_samples(numerical_data)
            
            # Get predictions (-1 for anomalies, 1 for normal)
            predictions = self.model.predict(numerical_data)
            
            return predictions
        except Exception as e:
            st.error(f"Error predicting anomalies: {str(e)}")
            return None
    
    def visualize_anomalies(self, data, timestamp_col=None):
        """Visualize anomalies in the data"""
        if self.anomaly_scores is None:
            st.warning("No anomaly scores available. Please run predict first.")
            return
        
        try:
            # Create DataFrame for visualization
            viz_data = data.copy()
            viz_data['anomaly_score'] = self.anomaly_scores
            viz_data['is_anomaly'] = self.model.predict(data.select_dtypes(include=['int64', 'float64']))
            
            # Display model performance metrics
            st.subheader("Model Performance Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Anomaly Score", f"{np.mean(self.anomaly_scores):.4f}")
            with col2:
                st.metric("Anomaly Ratio", f"{(viz_data['is_anomaly'] == -1).mean():.4f}")
            
            # Time series visualization if timestamp is available
            if timestamp_col and timestamp_col in viz_data.columns:
                st.subheader("Anomaly Scores Over Time")
                fig = px.scatter(
                    viz_data,
                    x=timestamp_col,
                    y='anomaly_score',
                    color='is_anomaly',
                    title="Anomaly Scores Over Time"
                )
                st.plotly_chart(fig)
            
            # Distribution of anomaly scores
            st.subheader("Distribution of Anomaly Scores")
            fig = px.histogram(
                viz_data,
                x='anomaly_score',
                color='is_anomaly',
                title="Distribution of Anomaly Scores"
            )
            st.plotly_chart(fig)
            
            # t-SNE visualization
            st.subheader("t-SNE Visualization")
            try:
                tsne = TSNE(n_components=2, random_state=42)
                numerical_data = data.select_dtypes(include=['int64', 'float64'])
                tsne_results = tsne.fit_transform(numerical_data)
                
                fig = px.scatter(
                    x=tsne_results[:, 0],
                    y=tsne_results[:, 1],
                    color=viz_data['is_anomaly'],
                    title="t-SNE Visualization of Anomalies"
                )
                st.plotly_chart(fig)
            except Exception as e:
                st.warning("Unable to perform t-SNE visualization. Displaying alternative visualization.")
                try:
                    img = Image.open(r"C:\Users\kriss\Downloads\code_tsne.png")
                    st.image(img, caption="t-SNE Visualization Example", use_column_width=True)
                except Exception as img_error:
                    st.error(f"Error displaying alternative visualization: {str(img_error)}")
            
        except Exception as e:
            st.error(f"Error visualizing anomalies: {str(e)}")
            return
    
    def get_model_info(self):
        """Get information about the model"""
        if not self.fitted:
            return None
        
        try:
            info = {
                'contamination': self.model.contamination,
                'n_estimators': self.model.n_estimators,
                'max_samples': self.model.max_samples,
                'random_state': self.model.random_state,
                'offset_': self.model.offset_,
                'n_features_in_': self.model.n_features_in_,
                'feature_names_in_': self.model.feature_names_in_.tolist() if hasattr(self.model, 'feature_names_in_') else None
            }
            return info
        except Exception as e:
            st.error(f"Error getting model info: {str(e)}")
            return None