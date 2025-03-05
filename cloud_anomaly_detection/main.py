import streamlit as st
import pandas as pd
from data_processor import DataProcessor
from ctgan_module import CTGANModule
from isolation_forest import IsolationForestModule
from rca_module import RCAModule
from adaptive_rca import AdaptiveRCAOptimizer
import numpy as np
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
import shap

# Set page config
st.set_page_config(
    page_title="Cloud Log Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 0.5rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin: 0 0.25rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .summary-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .summary-title {
        color: #2c3e50;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

def display_header():
    st.title("🔍 Cloud Log Anomaly Detection")
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;'>
            <p style='color: #7f8c8d;'>Upload your cloud logs CSV file to begin analysis.</p>
        </div>
    """, unsafe_allow_html=True)

def display_metrics(data):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Features", f"{len(data.columns):,}")
    with col3:
        st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
    with col4:
        st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

def display_data_summary(summary):
    """Display data summary in a concise way"""
    if summary is None:
        st.error("No summary data available")
        return

    # Basic Information Section
    st.markdown('<div class="summary-section">', unsafe_allow_html=True)
    st.markdown('<div class="summary-title">📊 Basic Information</div>', unsafe_allow_html=True)
    st.dataframe(summary['basic_info'], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Missing Values Section
    st.markdown('<div class="summary-section">', unsafe_allow_html=True)
    st.markdown('<div class="summary-title">⚠️ Missing Values</div>', unsafe_allow_html=True)
    st.dataframe(summary['missing_values'], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Column Types Section
    st.markdown('<div class="summary-section">', unsafe_allow_html=True)
    st.markdown('<div class="summary-title">📝 Column Types</div>', unsafe_allow_html=True)
    st.dataframe(summary['column_types'], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Numerical Summary Section
    st.markdown('<div class="summary-section">', unsafe_allow_html=True)
    st.markdown('<div class="summary-title">🔢 Numerical Summary</div>', unsafe_allow_html=True)
    st.dataframe(summary['numerical_summary'], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    display_header()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your cloud logs CSV file for analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read the data
            data = pd.read_csv(uploaded_file)
            
            # Display metrics
            display_metrics(data)
            
            # Create tabs for different modules
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Data Processing",
                "🔄 Synthetic Data",
                "🔍 Anomaly Detection",
                "🔎 Root Cause",
                "⚡ Adaptive RCA"
            ])
            
            # Data Processing Tab
            with tab1:
                st.subheader("Data Processing")
                data_processor = DataProcessor()
                
                # Display original data overview
                st.dataframe(data.head(), use_container_width=True)
                
                # Process the data
                if st.button("Process Data"):
                    with st.spinner("Processing..."):
                        processed_data = data_processor.preprocess_data(data)
                        if processed_data is not None:
                            st.success("Data processed successfully!")
                            st.dataframe(processed_data.head(), use_container_width=True)
                            summary = data_processor.get_data_summary()
                            display_data_summary(summary)
            
            # Synthetic Data Generation Tab
            with tab2:
                st.subheader("Synthetic Data Generation")
                ctgan = CTGANModule()
                
                if st.button("Generate Data"):
                    with st.spinner("Generating..."):
                        if ctgan.fit(data):
                            synthetic_data = ctgan.generate_samples(len(data))
                            if synthetic_data is not None:
                                st.success("Synthetic data generated!")
                                st.dataframe(synthetic_data.head(), use_container_width=True)
            
            # Anomaly Detection Tab
            with tab3:
                st.subheader("Anomaly Detection")
                iso_forest = IsolationForestModule()
                
                if st.button("Detect Anomalies"):
                    with st.spinner("Detecting..."):
                        if iso_forest.fit(data):
                            predictions = iso_forest.predict(data)
                            if predictions is not None:
                                st.success("Anomalies detected!")
                                iso_forest.visualize_anomalies(data)
            
            # Root Cause Analysis Tab
            with tab4:
                st.subheader("Root Cause Analysis")
                rca = RCAModule()
                
                # Select target column for analysis
                numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
                if len(numerical_cols) > 0:
                    target_col = st.selectbox(
                        "Select target column for analysis",
                        options=numerical_cols,
                        help="Choose the column you want to analyze for correlations"
                    )
                    
                    if st.button("Analyze"):
                        with st.spinner("Analyzing..."):
                            # Initialize fault tree
                            rca.initialize_fault_tree()
                            
                            # Correlation analysis
                            correlation_matrix = rca.analyze_correlations(data, target_col)
                            if correlation_matrix is not None:
                                st.subheader("Correlation Analysis")
                                st.dataframe(correlation_matrix, use_container_width=True)
                            
                            # Mutual information analysis
                            mi_scores = rca.calculate_mutual_info(data, target_col)
                            if mi_scores is not None:
                                st.subheader("Mutual Information Scores")
                                mi_df = pd.DataFrame(list(mi_scores.items()), columns=['Feature', 'Score'])
                                st.dataframe(mi_df, use_container_width=True)
                            
                            # SHAP analysis using Isolation Forest
                            iso_forest = IsolationForestModule()
                            if iso_forest.fit(data):
                                predictions = iso_forest.predict(data)
                                if predictions is not None:
                                    st.subheader("SHAP Analysis")
                                    shap_values = rca.apply_shap_analysis(iso_forest.model, data.select_dtypes(include=['int64', 'float64']))
                                    if shap_values is not None:
                                        # Create SHAP summary plot
                                        plt.figure(figsize=(10, 6))
                                        shap.summary_plot(shap_values, data.select_dtypes(include=['int64', 'float64']))
                                        st.pyplot(plt)
                                        
                                        # Create SHAP bar plot
                                        plt.figure(figsize=(10, 6))
                                        shap.summary_plot(shap_values, data.select_dtypes(include=['int64', 'float64']), plot_type="bar")
                                        st.pyplot(plt)
                            
                            # Visualize fault tree
                            rca.visualize_fault_tree()
                            
                            # Identify and visualize root causes
                            root_causes = rca.identify_root_causes(data, target_col)
                            if root_causes is not None:
                                rca.visualize_root_causes()
                else:
                    st.warning("No numerical columns available for correlation analysis.")
            
            # Adaptive RCA Tab
            with tab5:
                st.subheader("Adaptive RCA")
                numerical_data = data.select_dtypes(include=['int64', 'float64'])
                input_dim = len(numerical_data.columns)
                adaptive_rca = AdaptiveRCAOptimizer(input_dim=input_dim)
                
                if st.button("Optimize"):
                    with st.spinner("Optimizing..."):
                        adaptive_rca.initialize_model()
                        processed_results = adaptive_rca.preprocess_rca_results(numerical_data)
                        if processed_results is not None:
                            # Use a single row for each iteration
                            for i in range(5):
                                # Get a random row from the processed results
                                state = processed_results.iloc[i].values
                                action = adaptive_rca.select_action(state)
                                reward = np.random.rand()
                                # Get another random row for next state
                                next_state = processed_results.iloc[(i + 1) % len(processed_results)].values
                                # Set done=True for the last iteration
                                done = (i == 4)
                                adaptive_rca.update_model(state, action, reward, next_state, done)
                            adaptive_rca.visualize_reward_progression()
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 