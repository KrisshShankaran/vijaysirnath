# Cloud Log Anomaly Detection

A comprehensive Streamlit application for detecting anomalies in cloud logs using various machine learning techniques.

## Features

- Data Processing and Analysis
- Synthetic Data Generation using CTGAN
- Anomaly Detection using Isolation Forest
- Root Cause Analysis with SHAP
- Adaptive RCA Optimization using Reinforcement Learning

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cloud-anomaly-detection.git
cd cloud-anomaly-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app locally:
```bash
streamlit run main.py
```

2. Access the application at http://localhost:8502

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app"
4. Select your repository
5. Set the main file path to `main.py`
6. Click "Deploy"

## Project Structure

```
cloud-anomaly_detection/
├── main.py                 # Main Streamlit application
├── data_processor.py       # Data preprocessing module
├── isolation_forest.py     # Anomaly detection module
├── rca_module.py          # Root cause analysis module
├── adaptive_rca.py        # Adaptive RCA optimization
├── ctgan_module.py        # Synthetic data generation
└── requirements.txt       # Project dependencies
```

## License

MIT License 