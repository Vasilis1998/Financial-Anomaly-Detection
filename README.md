# Financial Transaction Anomaly Detection

An end-to-end Machine Learning project for detecting unusual financial transaction volumes using **Databricks, PySpark, Delta Lake, Scikit-learn, and MLflow**.

## Architecture

```mermaid
flowchart TD
    A["Synthetic Financial Data<br/>Python · NumPy · Pandas"]
    B["Data Processing & Feature Engineering<br/>Databricks · PySpark"]
    C[("Delta Lake<br/>Raw & Feature Tables")]
    D["Isolation Forest<br/>Training · Tuning · Evaluation"]
    E["MLflow<br/>Experiment Tracking"]
    F["Unity Catalog<br/>Registered Model · Versioning"]
    G["New Batch Transactions"]
    H["Batch Feature Engineering<br/>PySpark"]
    I["Anomaly Scoring<br/>Calibrated Threshold"]
    J[("Delta Lake<br/>Predictions Table")]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J

## Project Overview

This project explores how machine learning can identify anomalous patterns in financial transaction data.

The solution uses synthetic time-series transaction data, applies feature engineering techniques, trains an Isolation Forest model, evaluates its performance, registers the model using MLflow and Unity Catalog, and implements a batch prediction pipeline that stores its results in Delta Lake.

## Technologies Used

- **Python** – Data processing and machine learning
- **PySpark** – Distributed data processing and feature engineering
- **Databricks** – Development and execution environment
- **Delta Lake** – Storage and management of data tables
- **Pandas & NumPy** – Data manipulation and synthetic data generation
- **Scikit-learn** – Isolation Forest anomaly detection
- **MLflow** – Experiment tracking and model logging
- **Unity Catalog** – Model registration and version management
- **Matplotlib** – Model evaluation visualizations

## Dataset

The project uses synthetically generated financial transaction time-series data.

The initial dataset contains 2,000 records with timestamps at 15-minute intervals. Transaction volumes are generated using a normal distribution, and 20 artificial anomalies are injected by multiplying selected transaction values by random factors.

The dataset includes:

- `timestamp` – Transaction timestamp
- `transaction_volume` – Transaction volume
- `true_label` – Ground-truth anomaly label (0: normal, 1: anomaly)

The synthetic data allows controlled evaluation of anomaly detection performance.

## Project Workflow

### 1. Data Generation and Storage
- Generate synthetic financial transaction data.
- Inject artificial anomalies.
- Convert Pandas DataFrames into Spark DataFrames.
- Store the data in Delta Lake.

### 2. Exploratory Data Processing
- Extract time-based features.
- Examine transaction volume distributions.
- Create ground-truth labels for evaluation.

### 3. Feature Engineering
Create additional features using PySpark:

- `hour`
- `day_of_week`
- `previous_volume`
- `rolling_mean`
- `rolling_std`
- `volume_deviation`

### 4. Model Development
- Train an Isolation Forest model.
- Experiment with different feature combinations.
- Tune model parameters, including contamination and number of estimators.
- Compare model performance using precision, recall, and F1-score.

### 5. Model Evaluation
Use a chronological training/test split to evaluate the selected model on unseen data.

The final model uses the following features:

- `transaction_volume`
- `hour`
- `day_of_week`

Selected configuration:

- Algorithm: Isolation Forest
- Contamination: 0.005
- Number of estimators: 500
- Random state: 42

### 6. MLflow and Model Registry
- Track model parameters and evaluation metrics.
- Log the trained model with MLflow.
- Register the final model in Databricks Unity Catalog.
- Create and register model version 1.

Registered model:

`workspace.default.financial_anomaly_detection_model`

### 7. Batch Prediction Pipeline
- Load the registered model.
- Generate a new synthetic batch of 100 transactions.
- Apply feature engineering.
- Calculate anomaly scores.
- Apply a calibrated anomaly-score threshold.
- Save the final predictions as a Delta Lake table.

Output table:

`financial_anomaly_predictions_delta`

## Model Evaluation Results

The final model was evaluated on a separate chronological test set containing 400 records, including 4 injected anomalies.

| Metric | Result |
|---|---:|
| Accuracy | 99.75% |
| Precision | 100% |
| Recall | 75% |
| F1-score | 85.71% |

### Confusion Matrix

| | Predicted Normal | Predicted Anomaly |
|---|---:|---:|
| Actual Normal | 396 | 0 |
| Actual Anomaly | 1 | 3 |

The model detected 3 out of 4 anomalies without producing false positives on the test set.

## Batch Prediction Results

A separate synthetic batch of 100 transactions was used to demonstrate the batch prediction pipeline.

- Total transactions: 100
- Normal transactions: 95
- Injected anomalies: 5
- Detected anomalies: 5

The calibrated threshold identified all five injected anomalies in this demonstration batch.

**Important:** The threshold was selected after examining the known labels of this same synthetic batch. Therefore, the batch results are illustrative and should not be interpreted as an independent estimate of real-world model performance.

## Limitations and Future Improvements

- The dataset is synthetic and does not represent real financial transaction behavior.
- The test set contains only four anomalies, so evaluation metrics are sensitive to individual predictions.
- Threshold calibration should be validated on a separate labeled calibration dataset.
- Further evaluation should use larger datasets and more realistic anomaly patterns.
- Future improvements could include additional behavioral features, alternative anomaly detection algorithms, and a production-oriented real-time inference pipeline.

## Repository Contents

- `Financial_Anomaly_Detection.ipynb` – Complete Databricks/Jupyter Notebook containing the project workflow.
- `README.md` – Project documentation.

## Author

Data Science portfolio project focused on Machine Learning, anomaly detection, and data engineering technologies.
