# IoT23 Anomaly Detection

Cybersecurity anomaly detection pipeline using a subset of the IoT-23 dataset with Snowflake, dbt, Databricks, MLflow, Groq and Slack.

## Project Description

This project is about building a machine learning pipeline for detecting malicious network traffic using a subset of the IoT-23 dataset.

The selected data is based on Zeek network log data from the IoT-23 dataset and was converted to CSV before being loaded into Snowflake. The data is then transformed using dbt and accessed through Unity Catalog in Databricks. Databricks is used for data analysis, preprocessing, machine learning and experiment tracking.

The project includes:

- Data storage and transformation using Snowflake and dbt
- Data preprocessing and feature engineering
- Random Forest classification
- Isolation Forest anomaly detection
- A/B testing of Random Forest models with 100 and 200 trees
- MLflow experiment tracking
- Model performance comparison
- Feature importance analysis
- AI-assisted analysis of the model results using Groq
- Automated reporting to Slack
- An interactive Databricks dashboard

## Project Structure

The repository contains the following main components:

- `models/` – dbt models and trained machine learning models
- `output/` – the CSV dataset used in the project
- `screenshots/` – dashboard and Slack report screenshots
- `scripts/` – Python scripts for data preparation
- `IoT23_ML_Analysis_and_Experiment_Tracking.py` – main machine learning workflow
- `dbt_project.yml` – dbt project configuration
- `IoT23_ML_Analysis_and_Experiment_Tracking.ipynb` - main machine learning workflow (Jupyter Notebook)

## Machine Learning

The main supervised model used in the project is Random Forest with 100 decision trees.

An A/B test was performed to compare Random Forest models with 100 and 200 trees. The purpose was to see if increasing the number of trees would improve the model performance.

Isolation Forest was also tested as an unsupervised anomaly detection model.

The dataset is highly imbalanced. Approximately 99.44% of the traffic is malicious and 0.56% is benign. Because of this, accuracy alone is not enough to evaluate the models, so precision, recall and F1-score are also considered.

## Results

The Random Forest models achieved very similar results. Increasing the number of trees from 100 to 200 did not result in any measurable improvement.

Random Forest performed substantially better than Isolation Forest on the selected dataset.

The feature importance analysis showed that `ORIG_IP_BYTES`, `DURATION` and `TOTAL_PACKETS` were the three most important features for the Random Forest model.

The very high Random Forest results should be interpreted with some caution because the dataset is highly imbalanced. The model performance should be further validated using a proper held-out or cross-validated dataset before using the model in a real-world environment.

## MLflow

MLflow is used to track the machine learning experiments.

The following are logged:

- Model parameters
- Evaluation metrics
- Trained models
- Experiment runs

Three experiments are tracked:

1. Random Forest – 100 trees
2. Random Forest – 200 trees
3. Isolation Forest

## Running the Project

The project is designed to use Snowflake, dbt and Databricks.

The dbt project contains the data transformation models used to prepare the data for machine learning. The machine learning workflow is executed in Databricks.

When using the notebook, replace `<your-email>` in the MLflow experiment path with the email address associated with your Databricks account.

The Slack webhook URL and Groq API key are entered when the notebook is run. They are not stored directly in the source code.

## Dashboard

The Databricks dashboard contains visualizations for:

- Benign and malicious traffic
- Network connections by hour
- Traffic by protocol
- Total connections
- Model performance comparison
- Random Forest A/B-test results

## Technologies

- Python
- Pandas
- Scikit-learn
- Snowflake
- dbt
- Databricks
- MLflow
- Groq
- Slack

## Dataset

This project uses a subset of the IoT-23 dataset created by the Stratosphere Laboratory.

The data used in this project was derived from Zeek network log data from the IoT-23 dataset.

**Citation:**

Garcia, S., Parmisano, A., & Erquiaga, M. J. (2020). *IoT-23: A labeled dataset with malicious and benign IoT network traffic* (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4743746

Dataset source: https://www.stratosphereips.org/datasets-iot23
