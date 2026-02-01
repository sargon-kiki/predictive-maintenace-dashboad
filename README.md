# Predictive Maintenance Dashboard

A machine learning-powered predictive maintenance system that predicts equipment failures, assesses machine health, and estimates remaining useful life (RUL) based on sensor data.

## Features

- **Failure Type Prediction**: Multi-class classification predicting 6 failure types
- **Health Assessment**: Binary health classification with 1-10 health score
- **RUL Estimation**: Remaining useful life prediction in hours
- **Interactive Dashboard**: Streamlit-based web interface
- **Batch Processing**: CSV upload for bulk predictions

## Installation

### Prerequisites

- Python 3.9+
- Conda (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd predictive-maintenace-dashboad
```

2. Create and activate conda environment:
```bash
conda create -n kens-project python=3.11
conda activate kens-project
```

3. Install dependencies:
```bash
pip install pandas numpy scikit-learn streamlit matplotlib joblib imbalanced-learn
```

## Usage

### Running the Dashboard

```bash
conda activate kens-project
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### Training Models

Train individual models:

```bash
# Failure Type Prediction (Multi-class)
python train/train_failure_prediction.py

# Health Classification
python train/train_classification.py

# RUL Estimation
python train/train_rul_model.py
```

## Project Structure

```
predictive-maintenace-dashboad/
├── app.py                      # Streamlit dashboard
├── data/
│   ├── ai4i_2020.csv          # Training dataset
│   └── example-test.csv       # Test data with failure scenarios
├── models/
│   ├── failure_prediction_model.pkl
│   ├── failure_config.pkl
│   ├── health_classification_model.pkl
│   ├── health_config.pkl
│   ├── rul_model.pkl
│   └── rul_config.pkl
├── train/
│   ├── train_failure_prediction.py
│   ├── train_classification.py
│   ├── train_rul_model.py
│   └── inspect_data.py
├── notebooks/
│   └── train.ipynb
├── report.md                   # Academic report
└── README.md
```

## Models

### 1. Failure Type Prediction

| Attribute | Value |
|-----------|-------|
| Algorithm | Random Forest Classifier |
| Classes | No Failure, TWF, HDF, PWF, OSF, RNF |
| Accuracy | 95.70% |
| Balancing | SMOTE oversampling |

### 2. Health Classification

| Attribute | Value |
|-----------|-------|
| Algorithm | Gradient Boosting Classifier |
| Output | Healthy/Unhealthy + Score (1-10) |
| Accuracy | 99.90% |

### 3. RUL Estimation

| Attribute | Value |
|-----------|-------|
| Algorithm | Gradient Boosting Regressor |
| Output | Hours remaining |
| R² Score | 0.9999 |
| MAE | 0.28 hours |

## Input Features

| Feature | Description | Unit |
|---------|-------------|------|
| Air temperature | Ambient temperature | Kelvin |
| Process temperature | Operating temperature | Kelvin |
| Rotational speed | Spindle speed | RPM |
| Torque | Applied torque | Nm |
| Tool wear | Cumulative usage | Minutes |
| Product Type | Quality category | H, M, L |

## Failure Types

| Code | Failure Type | Condition |
|------|--------------|-----------|
| TWF | Tool Wear Failure | Tool wear ≥ 200 min |
| HDF | Heat Dissipation Failure | temp_diff < 8.6K AND RPM < 1380 |
| PWF | Power Failure | Power < 3500W OR > 9000W |
| OSF | Overstrain Failure | Strain > 11000 |
| RNF | Random Failure | Random occurrence |

## Testing

Use the provided test file with failure scenarios:

```bash
# Upload data/example-test.csv in the dashboard
```

Test cases included:
1. Heat Dissipation Failure
2. Power Failure (Low)
3. Power Failure (High)
4. Overstrain + Tool Wear
5. Tool Wear Failure
6. Multiple Failures (Critical)
7. Healthy Machine (Control)

## Dataset

AI4I 2020 Predictive Maintenance Dataset
- 10,000 samples
- 14 features
- 5 failure types
- Source: UCI Machine Learning Repository

## License

MIT License

## Author

Kowalski Devops
