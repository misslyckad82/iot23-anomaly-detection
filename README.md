# IoT23 Anomaly Detection

End-to-end cybersecurity anomaly detection pipeline using Snowflake, dbt, Databricks and MLflow on a subset of the IoT-23 dataset.

## Project Description

This project develops a machine learning pipeline for detecting malicious network traffic using a subset of the IoT-23 dataset.

The data is stored and transformed using Snowflake and dbt, while Databricks is used for data analysis, machine learning and experiment tracking.

The project includes:

- Data storage and transformation using Snowflake and dbt
- Data preprocessing and feature engineering
- Random Forest classification
- Isolation Forest anomaly detection
- A/B testing of Random Forest models with 100 and 200 trees
- MLflow experiment tracking
- Model performance comparison
- Feature importance analysis
- Groq-based analysis of model results
- Automated reporting to Slack
- An interactive Databricks dashboard

## Machine Learning

The main supervised model is Random Forest with 100 decision trees.

An A/B test compares Random Forest models using 100 and 200 trees to determine whether increasing the number of trees improves performance.

Isolation Forest is also evaluated as an unsupervised anomaly detection approach.

The dataset is highly imbalanced, with approximately 99.44% malicious traffic and 0.56% benign traffic. Therefore, accuracy is interpreted together with precision, recall and F1-score.

## Results

The Random Forest models achieved very similar evaluation results on the selected dataset. Increasing the number of trees from 100 to 200 did not provide a measurable improvement.

The Random Forest models substantially outperformed the Isolation Forest model on the selected dataset.

Feature importance analysis identified `ORIG_IP_BYTES`, `DURATION` and `TOTAL_PACKETS` as the three most influential features for the Random Forest model.

Because the dataset is highly imbalanced and the Random Forest results were extremely high, the results should be interpreted carefully and validated on an appropriate held-out or cross-validated dataset before deployment.

## MLflow

MLflow is used to track:

- Model parameters
- Evaluation metrics
- Trained models
- Experiment runs

Three experiments are tracked:

1. Random Forest – 100 trees
2. Random Forest – 200 trees
3. Isolation Forest

## Running the Project

The project is designed to run in Databricks.

The user should provide their own Databricks MLflow experiment path and credentials for external services when prompted.

The Slack webhook URL and Groq API key are entered at runtime and are not stored directly in the source code.

## Dashboard

The Databricks dashboard provides visualizations of:

- Benign and malicious traffic
- Network connections over time
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
