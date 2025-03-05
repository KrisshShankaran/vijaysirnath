import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import shap
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class RCAModule:
    def __init__(self):
        self.fault_tree = None
        self.correlation_matrix = None
        self.mutual_info_scores = None
        self.shap_values = None
    
    def initialize_fault_tree(self):
        """Initialize the fault tree structure"""
        try:
            self.fault_tree = nx.DiGraph()
            
            # Add edges to create the fault tree structure
            edges = [
                ("Root Failure", "Network Issue"),
                ("Root Failure", "Server Overload"),
                ("Root Failure", "Unauthorized Access"),
                ("Root Failure", "Configuration Issues"),
                ("Root Failure", "Service Failures"),
                ("Root Failure", "Malicious Activity"),
                
                ("Network Issue", "Network Latency"),
                ("Network Issue", "Packet Loss"),
                ("Network Issue", "IP Blacklisting"),
                ("Network Latency", "DNS Failure"),
                ("Network Latency", "High Traffic Load"),
                ("Packet Loss", "Infrastructure Congestion"),
                
                ("Server Overload", "Resource Exhaustion"),
                ("Server Overload", "Memory Leak"),
                ("Resource Exhaustion", "High CPU Usage"),
                ("Resource Exhaustion", "Insufficient Scaling"),
                
                ("Unauthorized Access", "Brute Force Attempts"),
                ("Unauthorized Access", "Invalid API Calls"),
                ("Brute Force Attempts", "Repeated Login Failures"),
                ("Invalid API Calls", "Stolen Access Keys"),
                
                ("Configuration Issues", "Misconfigured IAM Roles"),
                ("Configuration Issues", "Incorrect Permissions"),
                ("Misconfigured IAM Roles", "Excessive Privileges"),
                ("Incorrect Permissions", "Lack of Role Separation"),
                
                ("Service Failures", "Database Failure"),
                ("Service Failures", "API Timeout"),
                ("Database Failure", "Slow Query Performance"),
                ("API Timeout", "Load Balancer Failure"),
                
                ("Malicious Activity", "Suspicious API Calls"),
                ("Malicious Activity", "Abnormal User Behavior"),
                ("Suspicious API Calls", "Unusual Traffic Patterns"),
                ("Abnormal User Behavior", "Access from New Locations")
            ]
            
            self.fault_tree.add_edges_from(edges)
            return True
        except Exception as e:
            print(f"Error initializing fault tree: {str(e)}")
            return False
    
    def analyze_correlations(self, data, target_col):
        """Analyze correlations between features and target"""
        try:
            # Calculate correlation matrix
            self.correlation_matrix = data.corr()
            
            # Get correlations with target
            target_correlations = self.correlation_matrix[target_col].abs()
            
            return target_correlations
        except Exception as e:
            print(f"Error analyzing correlations: {str(e)}")
            return None
    
    def calculate_mutual_info(self, data, target_col):
        """Calculate mutual information scores"""
        try:
            # Calculate mutual information scores
            mi_scores = mutual_info_regression(data, data[target_col])
            
            # Create dictionary of feature scores
            self.mutual_info_scores = dict(zip(data.columns, mi_scores))
            
            return self.mutual_info_scores
        except Exception as e:
            print(f"Error calculating mutual information: {str(e)}")
            return None
    
    def apply_shap_analysis(self, model, data):
        """Apply SHAP analysis to identify feature importance"""
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            self.shap_values = explainer.shap_values(data)
            
            return self.shap_values
        except Exception as e:
            print(f"Error applying SHAP analysis: {str(e)}")
            return None
    
    def identify_root_causes(self, data, target_col, threshold=0.1):
        """Identify potential root causes based on analysis"""
        try:
            # Get correlation scores
            correlations = self.analyze_correlations(data, target_col)
            
            # Get mutual information scores
            mi_scores = self.calculate_mutual_info(data, target_col)
            
            if correlations is None or mi_scores is None:
                return None
            
            # Combine scores
            root_causes = {}
            for feature in data.columns:
                if feature != target_col:
                    combined_score = (
                        correlations[feature] * 0.5 +
                        mi_scores[feature] * 0.5
                    )
                    
                    if combined_score > threshold:
                        root_causes[feature] = {
                            'correlation': correlations[feature],
                            'mutual_info': mi_scores[feature],
                            'combined_score': combined_score
                        }
            
            return root_causes
        except Exception as e:
            print(f"Error identifying root causes: {str(e)}")
            return None
    
    def visualize_fault_tree(self):
        """Visualize the fault tree"""
        if self.fault_tree is None:
            print("Fault tree not initialized. Please call initialize_fault_tree first.")
            return
        
        try:
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Set up the layout
            pos = nx.spring_layout(self.fault_tree, k=1, iterations=50)
            
            # Draw the graph
            nx.draw(
                self.fault_tree,
                pos,
                with_labels=True,
                node_color='lightblue',
                node_size=2000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                edge_color='gray'
            )
            
            # Add title
            plt.title("Fault Tree Analysis")
            
            # Display in Streamlit
            st.pyplot(plt)
            
        except Exception as e:
            print(f"Error visualizing fault tree: {str(e)}")
            return
    
    def visualize_root_causes(self):
        """Visualize root cause analysis results"""
        if not hasattr(self, 'root_causes') or self.root_causes is None:
            print("No root causes available. Please run identify_root_causes first.")
            return
        
        try:
            # Create DataFrame for visualization
            causes_df = pd.DataFrame([
                {
                    'Feature': feature,
                    'Combined Score': scores['combined_score'],
                    'Correlation': scores['correlation'],
                    'Mutual Information': scores['mutual_info']
                }
                for feature, scores in self.root_causes.items()
            ])
            
            # Sort by combined score
            causes_df = causes_df.sort_values('Combined Score', ascending=False)
            
            # Create bar plot
            fig = px.bar(
                causes_df,
                x='Feature',
                y=['Combined Score', 'Correlation', 'Mutual Information'],
                title="Root Cause Analysis Results",
                barmode='group'
            )
            
            st.plotly_chart(fig)
            
        except Exception as e:
            print(f"Error visualizing root causes: {str(e)}")
            return
    
    def get_analysis_summary(self):
        """Get a summary of the analysis"""
        try:
            summary = {
                'correlation_matrix': self.correlation_matrix.to_dict() if self.correlation_matrix is not None else None,
                'mutual_info_scores': self.mutual_info_scores if self.mutual_info_scores is not None else None,
                'root_causes': getattr(self, 'root_causes', None),
                'fault_tree_initialized': self.fault_tree is not None
            }
            return summary
        except Exception as e:
            print(f"Error getting analysis summary: {str(e)}")
            return None 