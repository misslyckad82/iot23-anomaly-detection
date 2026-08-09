# Databricks notebook source
# MAGIC %md
# MAGIC # IoT-23 Machine Learning Pipeline
# MAGIC
# MAGIC This notebook demonstrates a machine learning pipeline for detecting malicious network traffic using a subset of a Zeek network log from the IoT-23 dataset.
# MAGIC
# MAGIC The selected dataset was converted to CSV and loaded into Snowflake. It was then transformed with dbt, accessed through Unity Catalog, and prepared for machine learning. The processed dataset is then used to train and compare a Random Forest model and an Isolation Forest model.
# MAGIC
# MAGIC The notebook also logs the experiments with MLflow, generates an AI-based analysis using Groq, and sends an automated report to Slack.

# COMMAND ----------

# Import the required libraries.
# Pandas is used for data manipulation.
# NumPy is used for numerical operations.
# Matplotlib is used for visualization.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import scikit-learn modules.
# These libraries are used to split the dataset,
# train the Random Forest model and evaluate
# its classification performance.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Import MLflow for experiment tracking.

import mlflow
import mlflow.sklearn

# Import pickle.
# Pickle is used to save the trained model
# as a .pkl file.

import pickle

print("Libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data from Snowflake
# MAGIC
# MAGIC Load the prepared feature dataset created from a subset of a Zeek network log from the IoT-23 dataset. The data is stored in Snowflake and accessed through Unity Catalog before being converted to a Pandas DataFrame for machine learning with scikit-learn.

# COMMAND ----------

# Load the transformed fact table from Snowflake into a Spark DataFrame.
# The spark.table() method reads a table registered in Unity Catalog
# and returns it as a Spark DataFrame.
# The DataFrame is stored in the variable 'df' and will be used
# throughout the notebook.

df = spark.table("iot23_snowflake_catalog.dbt_iot23.fct_iot23")

# Display the contents of the DataFrame.
# The display() function renders the data as an interactive table,
# making it easier to inspect the dataset.

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert the dataset to a Pandas DataFrame
# MAGIC
# MAGIC Convert the Spark DataFrame to a Pandas DataFrame to enable machine learning with scikit-learn.

# COMMAND ----------

# Convert the Spark DataFrame to a Pandas DataFrame.
# The scikit-learn models used in this notebook
# are trained using a Pandas DataFrame.

df = df.toPandas()

# Display the first five rows of the dataset.
# This provides a quick overview of the imported data.

display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset overview
# MAGIC
# MAGIC Explore the structure of the dataset by displaying the total number of rows, columns and data types. This provides a quick overview of the dataset before data preprocessing and machine learning.

# COMMAND ----------

# Display the number of rows and columns.
# This provides an overview of the dataset size.

print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Display the data types of each column.
# This helps identify numerical and categorical features.

print("\nData types:")
display(df.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Class Distribution
# MAGIC
# MAGIC The chosen dataset is highly imbalanced, with malicious traffic representing approximately 99.44% of the observations and benign traffic approximately 0.56%.
# MAGIC
# MAGIC This class imbalance is important when interpreting the machine learning results, since accuracy alone may not fully reflect model performance.
# MAGIC

# COMMAND ----------

# Count the number of observations in each class.
# 1 represents malicious traffic and 0 represents benign traffic.
print(df["IS_MALICIOUS"].value_counts())

print()

# Calculate the proportion of observations in each class.
# normalize=True converts the counts into relative proportions.
print(df["IS_MALICIOUS"].value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select the input features
# MAGIC
# MAGIC Select the numerical network traffic features that will be used to train the Random Forest model. These features describe the network connections and are used to distinguish malicious from benign traffic.

# COMMAND ----------

# Select the input features used for machine learning.
# These features describe network traffic characteristics.

features = [
    "DURATION",
    "ORIG_BYTES",
    "RESP_BYTES",
    "ORIG_PKTS",
    "RESP_PKTS",
    "ORIG_IP_BYTES",
    "RESP_IP_BYTES",
    "MISSED_BYTES",
    "TOTAL_BYTES",
    "TOTAL_PACKETS",
    "BYTES_PER_PACKET",
    "HAS_MISSED_BYTES",
    "IS_LOCAL_TRAFFIC"
]

# Create a new DataFrame containing only the selected features.

X = df[features].copy()

# Create the target variable.

y = df["IS_MALICIOUS"]

# Display the first rows.

display(X.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preprocessing
# MAGIC
# MAGIC Convert all selected features to numerical values. Machine learning algorithms in scikit-learn require numerical input, so any non-numeric values are converted before training the model.

# COMMAND ----------

# Convert all selected feature columns to numeric values.
# Invalid values are replaced with NaN.

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Replace missing values with 0.
# This ensures that the dataset contains only numerical values.

X = X.fillna(0)

# Display the data types after conversion.

display(X.dtypes)

# COMMAND ----------

# Convert the target variable to numeric values.
# Machine learning models require numerical class labels.

y = pd.to_numeric(
    y,
    errors="coerce"
)

# Replace missing values.

y = y.fillna(0)

# Convert the labels to integers.

y = y.astype(int)

# Display the data type.

print(y.dtype)
print(y.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataset into training and test sets
# MAGIC
# MAGIC Divide the dataset into training and test sets. The training set is used to train the Random Forest model, while the test set is used to evaluate the model on previously unseen data.

# COMMAND ----------

# Split the dataset into training and test sets.
# 80% of the data is used for training and
# 20% is reserved for testing.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the size of each dataset.

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the Random Forest model
# MAGIC
# MAGIC Train a Random Forest model using the training dataset. The model learns patterns from labelled network traffic and will later be used to classify previously unseen network connections.

# COMMAND ----------

# Create the Random Forest model.
# The model consists of 100 decision trees.
# A fixed random_state ensures reproducible results.

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model using the training dataset.

model.fit(X_train, y_train)

print("Random Forest model trained successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make predictions
# MAGIC
# MAGIC Use the trained Random Forest model to predict whether the network traffic in the test dataset is malicious or benign.

# COMMAND ----------

# Predict the class labels for the test dataset.
# The trained model classifies each network connection
# as either malicious (1) or benign (0).

y_pred = model.predict(X_test)

# Display the first predictions.

print("First 10 predictions:")
print(y_pred[:10])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the Random Forest model
# MAGIC
# MAGIC Evaluate the Random Forest model using the test dataset. The model performance is measured using Accuracy, Precision, Recall and F1-score.

# COMMAND ----------

# Calculate the evaluation metrics.
# These metrics measure how well the Random Forest model
# distinguishes malicious and benign network traffic.

rf_accuracy = accuracy_score(y_test, y_pred)
rf_precision = precision_score(y_test, y_pred)
rf_recall = recall_score(y_test, y_pred)
rf_f1 = f1_score(y_test, y_pred)

# Display the evaluation metrics.

print(f"Accuracy : {rf_accuracy:.6f}")
print(f"Precision: {rf_precision:.6f}")
print(f"Recall   : {rf_recall:.6f}")
print(f"F1-score : {rf_f1:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion Matrix
# MAGIC
# MAGIC Display the confusion matrix to compare the predicted class labels with the actual labels. This provides a detailed overview of the model's classification performance.

# COMMAND ----------

# Create the confusion matrix.
# The confusion matrix compares the predicted labels
# with the actual labels.

cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix.

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malicious"]
)

disp.plot()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC
# MAGIC Display the importance of each feature used by the Random Forest model. Feature importance indicates how much each variable contributed to the model's predictions.

# COMMAND ----------

# Create a DataFrame containing the feature importance values.

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

# Sort the features by importance.

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Display the results.

display(feature_importance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance visualization
# MAGIC
# MAGIC Visualize the importance of each feature used by the Random Forest model. The chart highlights which network traffic features contributed the most to the model's predictions.

# COMMAND ----------

# Plot the feature importance values.
# The most important features are displayed at the top.

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=True
)

feature_importance.plot(
    kind="barh",
    x="Feature",
    y="Importance",
    legend=False,
    figsize=(8, 6)
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the trained model
# MAGIC
# MAGIC Save the trained Random Forest model as a pickle (.pkl) file. The saved model can later be loaded without retraining, making it suitable for deployment and future predictions.

# COMMAND ----------

# Import the os module.
# The os module is used to create folders if they do not exist.

import os

# Create a directory for storing trained models.

os.makedirs("models", exist_ok=True)

# Save the trained Random Forest model as a pickle file.

with open("models/random_forest_100_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Random Forest model saved successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest Conclusion
# MAGIC
# MAGIC The Random Forest model achieved excellent classification performance on the selected subset of the IoT-23 Zeek network log.
# MAGIC
# MAGIC The evaluation metrics showed very high Accuracy, Precision, Recall and F1-score, indicating that the model was able to distinguish malicious and benign network traffic with a high degree of accuracy.
# MAGIC
# MAGIC Feature importance analysis revealed that **ORIG_IP_BYTES**, **DURATION**, and **TOTAL_PACKETS** were the most influential features for the model's predictions, while several other features contributed very little.
# MAGIC
# MAGIC  Overall, the results show that Random Forest performed very well on the selected subset of the IoT-23 dataset. However, the perfect evaluation metrics should be interpreted with caution because the dataset is highly imbalanced and the possibility of data leakage or overfitting should be considered. Further validation on an independent dataset or using cross-validation would be needed before drawing conclusions about real-world performance.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # A/B Test – Random Forest (100 vs 200 Trees)
# MAGIC
# MAGIC This section compares two versions of the Random Forest model using different numbers of decision trees.
# MAGIC
# MAGIC Model A:
# MAGIC - Random Forest (100 trees)
# MAGIC
# MAGIC Model B:
# MAGIC - Random Forest (200 trees)
# MAGIC
# MAGIC Both models are trained and evaluated using the same training and test datasets.
# MAGIC
# MAGIC The purpose of this A/B test is to evaluate whether increasing the number of trees improves the model's performance.

# COMMAND ----------

# Create a second Random Forest model.
# This model uses 200 decision trees for the A/B test.

rf_200_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("Random Forest (200 trees) model created successfully!")

# COMMAND ----------

# Train the Random Forest model with 200 trees.
# The model is trained using the same training dataset.

rf_200_model.fit(X_train, y_train)

print("Random Forest (200 trees) model trained successfully!")

# COMMAND ----------

# Save the trained Random Forest (200 trees) model.

import pickle

with open("models/random_forest_200_model.pkl", "wb") as file:
    pickle.dump(rf_200_model, file)

print("Random Forest (200 trees) model saved successfully!")

# COMMAND ----------

# Predict the class labels using the Random Forest model with 200 trees.
# The trained model predicts whether each network connection
# is malicious (1) or benign (0).

rf_200_pred = rf_200_model.predict(X_test)

print("First 10 predictions:")
print(rf_200_pred[:10])

# COMMAND ----------

# Calculate the evaluation metrics for the Random Forest model with 200 trees.

rf_200_accuracy = accuracy_score(y_test, rf_200_pred)
rf_200_precision = precision_score(y_test, rf_200_pred)
rf_200_recall = recall_score(y_test, rf_200_pred)
rf_200_f1 = f1_score(y_test, rf_200_pred)

# Display the evaluation metrics.

print(f"Accuracy : {rf_200_accuracy:.6f}")
print(f"Precision: {rf_200_precision:.6f}")
print(f"Recall   : {rf_200_recall:.6f}")
print(f"F1-score : {rf_200_f1:.6f}")

# COMMAND ----------

ab_test_df = pd.DataFrame({

    "Model": [
        "Random Forest (100 trees)",
        "Random Forest (200 trees)"
    ],

    "Trees": [
        100,
        200
    ],

    "Accuracy": [
        rf_accuracy,
        rf_200_accuracy
    ],

    "Precision": [
        rf_precision,
        rf_200_precision
    ],

    "Recall": [
        rf_recall,
        rf_200_recall
    ],

    "F1-score": [
        rf_f1,
        rf_200_f1
    ]

})

display(ab_test_df.round(6))

# COMMAND ----------

# MAGIC %md
# MAGIC ## A/B Test Conclusion
# MAGIC
# MAGIC The A/B test compared two Random Forest models with different numbers of decision trees.
# MAGIC
# MAGIC Model A used 100 trees, while Model B used 200 trees.
# MAGIC
# MAGIC Both models achieved identical evaluation metrics on the selected dataset. This indicates that increasing the number of trees from 100 to 200 did not improve the model's performance for this particular dataset.
# MAGIC
# MAGIC The results suggest that 100 trees were sufficient to achieve stable classification performance on the selected subset of the IoT-23 dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC # Isolation Forest
# MAGIC
# MAGIC This section demonstrates how an Isolation Forest model can be used to detect anomalies on the selected subset of the IoT-23 Zeek network log.
# MAGIC
# MAGIC Unlike Random Forest, Isolation Forest is an unsupervised machine learning algorithm. It identifies anomalies by isolating observations that differ from the majority of the data.
# MAGIC
# MAGIC The same preprocessed dataset is used to enable a direct comparison between the two machine learning models.

# COMMAND ----------

# Import the Isolation Forest model.
# Isolation Forest is an unsupervised machine learning
# algorithm used for anomaly detection.

from sklearn.ensemble import IsolationForest

print("Isolation Forest imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the Isolation Forest model
# MAGIC
# MAGIC Train an Isolation Forest model using the preprocessed network traffic data.
# MAGIC
# MAGIC Unlike Random Forest, Isolation Forest is an unsupervised learning algorithm. It identifies anomalies by isolating observations that differ from the majority of the data rather than learning from labelled examples.

# COMMAND ----------

# Create the Isolation Forest model.
# The contamination parameter specifies the expected
# proportion of anomalies in the training data.
# A fixed random_state ensures reproducible results.

isolation_model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

# Train the model using only the training dataset.
# Isolation Forest does not require class labels.

isolation_model.fit(X_train)

print("Isolation Forest model trained successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Detect anomalies
# MAGIC
# MAGIC Use the trained Isolation Forest model to classify each network connection.
# MAGIC
# MAGIC The model returns:
# MAGIC - **1** for normal observations (inliers)
# MAGIC - **-1** for anomalies (outliers)
# MAGIC
# MAGIC The predictions are converted to the same format as the Random Forest model to simplify the comparison between the two models.

# COMMAND ----------

# Predict anomalies for the test dataset.
# The model returns:
#   1  = normal observation
#  -1  = anomaly

isolation_predictions = isolation_model.predict(X_test)

# Convert the predictions to binary labels.
# 0 = benign
# 1 = malicious

isolation_predictions = np.where(
    isolation_predictions == -1,
    1,
    0
)

# Display the first predictions.

print("First 10 predictions:")
print(isolation_predictions[:10])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the Isolation Forest model
# MAGIC
# MAGIC Evaluate the Isolation Forest model by comparing its anomaly predictions with the ground truth labels.
# MAGIC
# MAGIC Although Isolation Forest is an unsupervised learning algorithm, the labelled dataset is used here only for performance evaluation.

# COMMAND ----------

# Calculate the evaluation metrics.
# Compare the predicted anomaly labels
# with the true labels from the test dataset.

if_accuracy = accuracy_score(y_test, isolation_predictions)
if_precision = precision_score(y_test, isolation_predictions)
if_recall = recall_score(y_test, isolation_predictions)
if_f1 = f1_score(y_test, isolation_predictions)

# Display the evaluation metrics.

print(f"Accuracy : {if_accuracy:.6f}")
print(f"Precision: {if_precision:.6f}")
print(f"Recall   : {if_recall:.6f}")
print(f"F1-score : {if_f1:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion Matrix
# MAGIC
# MAGIC Display the confusion matrix for the Isolation Forest model. The confusion matrix compares the predicted anomaly labels with the true class labels and provides a detailed overview of the model's performance.

# COMMAND ----------

# Create the confusion matrix.
cm = confusion_matrix(y_test, isolation_predictions)

# Display the confusion matrix.

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malicious"]
)

disp.plot()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the Isolation Forest model
# MAGIC
# MAGIC Save the trained Isolation Forest model as a pickle file.
# MAGIC
# MAGIC The saved model can later be loaded into MLflow for experiment tracking without retraining the model.

# COMMAND ----------

# Create a directory for storing trained models.
# The directory will be created if it does not already exist.

import os

os.makedirs("models", exist_ok=True)

# Save the trained Isolation Forest model as a pickle file.
# The saved model can later be loaded into MLflow
# without retraining the model.

with open("models/isolation_forest_model.pkl", "wb") as file:
    pickle.dump(isolation_model, file)

print("Isolation Forest model saved successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Isolation Forest Conclusion
# MAGIC
# MAGIC The Isolation Forest model was evaluated as an unsupervised anomaly detection algorithm on the selected dataset.
# MAGIC
# MAGIC Unlike the Random Forest model, Isolation Forest does not use labelled training data. Instead, it identifies observations that differ from the majority of the dataset.
# MAGIC
# MAGIC The evaluation showed substantially lower Accuracy, Precision, Recall and F1-score compared to the Random Forest model. This result indicates that the anomalies detected by Isolation Forest did not correspond well to the malicious labels in the dataset.
# MAGIC
# MAGIC A likely explanation is that the selected dataset is highly imbalanced, with approximately 99.44% of observations labelled as malicious. In addition, the Isolation Forest model was configured with a contamination value of 0.05, meaning that it assumes approximately 5% of the training observations are anomalies. This assumption does not match the class distribution of the selected dataset, where malicious traffic is the dominant class. As a result, the anomalies detected by Isolation Forest did not correspond well to the provided malicious labels.
# MAGIC
# MAGIC Overall, the results suggest that Isolation Forest is less suitable than Random Forest for supervised intrusion detection on the selected dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Comparison of Machine Learning Models
# MAGIC
# MAGIC This section compares the performance of two machine learning models using the same selected dataset.
# MAGIC
# MAGIC - Random Forest (Supervised Learning)
# MAGIC - Isolation Forest (Unsupervised Learning)
# MAGIC
# MAGIC Both models were trained using the same feature set and evaluated using the same test dataset.
# MAGIC
# MAGIC The performance of each model is compared using the following evaluation metrics:
# MAGIC
# MAGIC - Accuracy
# MAGIC - Precision
# MAGIC - Recall
# MAGIC - F1-score
# MAGIC
# MAGIC The purpose of this comparison is to evaluate how the two different machine learning approaches perform on the selected dataset and to assess their suitability for the given intrusion-detection task.

# COMMAND ----------

# Create a comparison table for both machine learning models.

comparison_df = pd.DataFrame({
    "Model": [
        "Random Forest (100 trees)",
        "Isolation Forest"
    ],
    "Learning Method": [
        "Supervised",
        "Unsupervised"
    ],
    "Accuracy": [
        rf_accuracy,
        if_accuracy
    ],
    "Precision": [
        rf_precision,
        if_precision
    ],
    "Recall": [
        rf_recall,
        if_recall
    ],
    "F1-score": [
        rf_f1,
        if_f1
    ]
})
comparison_df = comparison_df.round(6)

display(comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpretation of the comparison
# MAGIC
# MAGIC The Random Forest model significantly outperformed the Isolation Forest model across all evaluation metrics.
# MAGIC
# MAGIC Random Forest is a supervised learning algorithm that learns from labelled training data, allowing it to accurately distinguish malicious and benign network traffic.
# MAGIC
# MAGIC In contrast, Isolation Forest is an unsupervised anomaly detection algorithm. It identifies statistically unusual observations rather than learning malicious behaviour from class labels. Since the selected dataset is highly imbalanced and dominated by malicious traffic, the detected anomalies did not correspond well to the provided labels.
# MAGIC
# MAGIC These results indicate that Random Forest is the more suitable algorithm for supervised intrusion detection on the selected dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow Experiment Tracking
# MAGIC
# MAGIC This notebook demonstrates how MLflow can be used to track machine learning experiments.
# MAGIC
# MAGIC MLflow records model parameters, evaluation metrics and trained machine learning models. By storing this information, different experiments can easily be compared and reproduced.
# MAGIC
# MAGIC The notebook logs three machine learning experiments: two Random Forest models (100 and 200 trees) and one Isolation Forest model. MLflow is used to store model parameters, evaluation metrics and trained models for experiment tracking and comparison.

# COMMAND ----------

# Import the required libraries.
# MLflow is used for experiment tracking.
# The sklearn module in MLflow is used to
# log trained scikit-learn models.

import mlflow
import mlflow.sklearn

# Display the installed MLflow version.
# This confirms that MLflow has been imported successfully.

print(f"MLflow version: {mlflow.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an MLflow experiment
# MAGIC
# MAGIC Create an MLflow experiment to store all machine learning runs.
# MAGIC
# MAGIC The experiment acts as a container for model parameters, evaluation metrics and trained models. Each execution of the notebook creates a new run within the experiment, making it possible to compare multiple experiments over time.

# COMMAND ----------

# Import the MlflowClient class.
# The client is used to create and manage
# MLflow experiments programmatically.

from mlflow.tracking import MlflowClient

# Create an MLflow client.

client = MlflowClient()

# MAGIC %md
# MAGIC ## MLflow Experiment Setup
# MAGIC
# MAGIC Before running the notebook, replace `<your-email>` in the MLflow experiment path with the email address associated with your Databricks account.
# MAGIC
# MAGIC For example:
# MAGIC
# MAGIC `/Users/<your-email>/iot23_mlflow`
# MAGIC
# MAGIC The experiment path is used to create and track the MLflow experiments for the models in this project.

experiment_name = "/Users/<your-email>/iot23_mlflow"

print("Experiment name:")
print(experiment_name)

# COMMAND ----------

# Create or retrieve the MLflow experiment.
# If the experiment already exists, it will be reused.
# Otherwise, a new experiment will be created.

try:
    experiment_id = client.create_experiment(experiment_name)
    print("New experiment created.")
except Exception:
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    print("Existing experiment found.")

# Display the experiment ID.

print(f"Experiment ID: {experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the trained machine learning models
# MAGIC
# MAGIC Load the previously trained Random Forest and Isolation Forest models.
# MAGIC
# MAGIC The models were saved as pickle files earlier in this notebook. Loading them here makes it possible to track, compare and register the models with MLflow without retraining them.

# COMMAND ----------

# Import the pickle module.
# Pickle is used to load previously saved machine learning models.

import pickle

# Load the trained Random Forest (100 trees) model.

with open("models/random_forest_100_model.pkl", "rb") as file:
    random_forest_model = pickle.load(file)

# Load the trained Isolation Forest model.

with open("models/isolation_forest_model.pkl", "rb") as file:
    isolation_forest_model = pickle.load(file)

# Confirm that both models were loaded successfully.

print("Random Forest model loaded successfully!")
print("Isolation Forest model loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## MLflow Experiment Overview
# MAGIC The notebook logs three machine learning experiments: two Random Forest models (100 and 200 trees) and one Isolation Forest model. MLflow is used to store model parameters, evaluation metrics and trained models for experiment tracking and comparison.

# COMMAND ----------

mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the Random Forest experiment
# MAGIC
# MAGIC Log the complete Random Forest experiment to MLflow.
# MAGIC
# MAGIC The experiment includes the model parameters, evaluation metrics and the trained machine learning model. Storing all information in a single MLflow run makes it easy to compare experiments and reproduce the results.

# COMMAND ----------

# Start a new MLflow run for the Random Forest experiment.
# The run stores model parameters, evaluation metrics
# and the trained machine learning model.

with mlflow.start_run(run_name="Random Forest (100 trees)"):

    # Log the model parameters.

    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Log the evaluation metrics.

    mlflow.log_metric("accuracy", rf_accuracy)
    mlflow.log_metric("precision", rf_precision)
    mlflow.log_metric("recall", rf_recall)
    mlflow.log_metric("f1_score", rf_f1)

    # Log the trained model.

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model"
    )

    print("Random Forest (100 trees) experiment logged successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the A/B-testing experiment

# COMMAND ----------

# Start a new MLflow run for the Random Forest (200 trees) experiment.
# The run stores model parameters, evaluation metrics
# and the trained machine learning model.

with mlflow.start_run(run_name="Random Forest (200 trees)"):

    # Log the model parameters.

    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("random_state", 42)

    # Log the evaluation metrics.

    mlflow.log_metric("accuracy", rf_200_accuracy)
    mlflow.log_metric("precision", rf_200_precision)
    mlflow.log_metric("recall", rf_200_recall)
    mlflow.log_metric("f1_score", rf_200_f1)

    # Log the trained model.

    mlflow.sklearn.log_model(
    sk_model=rf_200_model,
    artifact_path="random_forest_200_model"
)


    print("Random Forest (200 trees) experiment logged successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the Isolation Forest experiment
# MAGIC
# MAGIC Log the complete Isolation Forest experiment to MLflow.
# MAGIC
# MAGIC The experiment stores the model parameters, evaluation metrics and the trained machine learning model in a single MLflow run. This makes it possible to compare the Isolation Forest model with the Random Forest model.

# COMMAND ----------

# Start a new MLflow run for the Isolation Forest experiment.
# The run stores the model parameters,
# evaluation metrics and the trained model.

with mlflow.start_run(run_name="Isolation Forest"):

    # Log the model parameters.

    mlflow.log_param("model_type", "Isolation Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("contamination", 0.05)
    mlflow.log_param("random_state", 42)

    # Log the evaluation metrics.

    mlflow.log_metric("accuracy", if_accuracy)
    mlflow.log_metric("precision", if_precision)
    mlflow.log_metric("recall", if_recall)
    mlflow.log_metric("f1_score", if_f1)

    # Log the trained model.

    mlflow.sklearn.log_model(
        sk_model=isolation_model,
        artifact_path="isolation_forest_model"
    )

    print("Isolation Forest experiment logged successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## View the logged MLflow experiments
# MAGIC
# MAGIC Display information about the logged MLflow experiments.
# MAGIC
# MAGIC The MLflow interface can now be used to compare all logged experiments, including the Random Forest A/B test and the Isolation Forest model.

# COMMAND ----------

# Display the current MLflow experiment.

experiment = mlflow.get_experiment_by_name(experiment_name)

print("Experiment name:")
print(experiment.name)

print("\nExperiment ID:")
print(experiment.experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display the logged MLflow runs
# MAGIC
# MAGIC Display the MLflow runs created during the notebook execution. This makes it possible to verify that all machine learning experiments were logged successfully.

# COMMAND ----------

# Display the latest MLflow runs.

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id]
)

display(runs)

# COMMAND ----------

# Display the most important information
display(
    runs[
        [
            "tags.mlflow.runName",
            "status"
        ]
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Experiment analysis
# MAGIC
# MAGIC MLflow was used to track and compare three machine learning experiments: two Random Forest models with different numbers of decision trees (100 and 200) and one Isolation Forest model.
# MAGIC
# MAGIC The logged parameters and evaluation metrics showed that increasing the number of trees from 100 to 200 did not improve the performance of the Random Forest model on the selected dataset.
# MAGIC
# MAGIC The experiment tracking functionality made it possible to compare the different model configurations, reproduce the results, and evaluate both the A/B test and the comparison between the Random Forest and Isolation Forest models.

# COMMAND ----------

# MAGIC %md
# MAGIC # Slack and Groq Automation
# MAGIC
# MAGIC This section demonstrates how the machine learning results can be automatically analysed and distributed using external services.
# MAGIC
# MAGIC Groq is used to generate a short interpretation of the model evaluation results. The analysis includes the Random Forest models with 100 and 200 trees, the Isolation Forest model, and the impact of the class imbalance on the results.
# MAGIC
# MAGIC The generated analysis is then sent to a Slack channel using a Slack Incoming Webhook. This provides an automated way to communicate the machine learning results without manually copying the output from the notebook.

# COMMAND ----------

# -------------------------------------------------
# API Configuration
# -------------------------------------------------

%pip install groq
import requests

from getpass import getpass
from groq import Groq

# Slack Incoming Webhook
# Used to send the automated report to Slack.
SLACK_WEBHOOK_URL = getpass("Enter your Slack Webhook URL: ")

# Groq API Key
# Used to authenticate requests to the Groq API.
GROQ_API_KEY = getpass("Enter your Groq API key: ")


# COMMAND ----------

# Create a Groq client using the API key.
client = Groq(api_key=GROQ_API_KEY)

# COMMAND ----------

# Test the connection to the Groq API.
# Verify that the API key is working correctly.
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Groq connection successful."
        }
    ]
)

print(response.choices[0].message.content)

# COMMAND ----------

prompt = f"""
Analyze the following machine learning model evaluation results.

Random Forest (100 trees):
- Accuracy: {rf_accuracy:.6f}
- Precision: {rf_precision:.6f}
- Recall: {rf_recall:.6f}
- F1-score: {rf_f1:.6f}

Random Forest (200 trees):
- Accuracy: {rf_200_accuracy:.6f}
- Precision: {rf_200_precision:.6f}
- Recall: {rf_200_recall:.6f}
- F1-score: {rf_200_f1:.6f}

Isolation Forest:
- Accuracy: {if_accuracy:.6f}
- Precision: {if_precision:.6f}
- Recall: {if_recall:.6f}
- F1-score: {if_f1:.6f}

The dataset is highly imbalanced, with approximately 99.44% malicious traffic and 0.56% benign traffic.

Write a short comparison (6–9 sentences).

Explain:
- Which model performed best.
- Whether increasing Random Forest from 100 to 200 trees improved performance.
- What the evaluation metrics indicate.
- How the class imbalance affects the interpretation of the results.
- Whether the models appear suitable for IoT network traffic anomaly detection.
"""

# COMMAND ----------

# Send the prompt to Groq.
# Generate an AI-based comparison of the machine learning experiments.
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# Store the AI-generated analysis.
analysis = response.choices[0].message.content

print(analysis)

# COMMAND ----------

# Create the Slack report.
# Include the evaluation metrics and the AI-generated analysis.
message = {
    "text": f"""
🤖 *IoT23 Machine Learning Evaluation Report* 🤖 

_Generated automatically from Databricks_

*Random Forest (100 trees)*
• Accuracy: {rf_accuracy:.6f}
• Precision: {rf_precision:.6f}
• Recall: {rf_recall:.6f}
• F1-score: {rf_f1:.6f}

*Random Forest (200 trees)*
• Accuracy: {rf_200_accuracy:.6f}
• Precision: {rf_200_precision:.6f}
• Recall: {rf_200_recall:.6f}
• F1-score: {rf_200_f1:.6f}

*Isolation Forest*
• Accuracy: {if_accuracy:.6f}
• Precision: {if_precision:.6f}
• Recall: {if_recall:.6f}
• F1-score: {if_f1:.6f}

*Groq Analysis*
{analysis}
"""
}

# Send the report to Slack.
# The Incoming Webhook posts the message to the selected channel.
response = requests.post(
    SLACK_WEBHOOK_URL,
    json=message
)

# Display the response returned by Slack.
print(f"Status code: {response.status_code}")
print(response.text)

# MAGIC %md
# MAGIC ## Check Saved Models
# MAGIC
# MAGIC This cell checks the current working directory and lists the saved `.pkl` model files.
# MAGIC It is used to verify that the trained machine learning models have been saved correctly.

# COMMAND ----------

# MAGIC %sh
# MAGIC pwd
# MAGIC find . -name "*.pkl" 2>/dev/null
