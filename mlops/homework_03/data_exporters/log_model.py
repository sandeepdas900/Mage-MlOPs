import click
import mlflow
import os
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from typing import Tuple

# Define the data preparation and model training function
def prepare_and_train(raw_data_path: str) -> Tuple[DictVectorizer, LinearRegression, pd.DataFrame]:
    df_train = read_dataframe(os.path.join(raw_data_path, "green_tripdata_2023-01.parquet"))
    y_train = df_train.pop('duration').values
    # Initialize DictVectorizer
    dv = DictVectorizer()
    # Prepare datasets to train the model
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    # Train a linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return dv, lr, df_train


# Define the data exporter function with decorator
# Adjust the decorator name and 'mage_ai.data_preparation.decorators' import if needed
if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_model_and_vectorizer(dv: DictVectorizer, lr: LinearRegression, df: pd.DataFrame, dest_path: str):
    # MLflow setup
    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.set_experiment('taxi_trip_duration')

    # Start a new MLflow run for logging the model and artifacts
    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        print(f"Model intercept: {lr.intercept_}")

        # Save the dict vectorizer using pickle
        dv_path = os.path.join(dest_path, "dv.pkl")
        with open(dv_path, 'wb') as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(dv_path, artifact_path="dict_vectorizer")

        # Optional: log the size of the model as a metric
        os.makedirs(dest_path, exist_ok=True)
        local_model_path = os.path.join(dest_path, "model.pkl")
        mlflow.sklearn.save_model(lr, local_model_path)
        model_size_bytes = os.path.getsize(local_model_path)
        mlflow.log_metric('model_size_bytes', model_size_bytes)

        print(f"The size of the model is: {model_size_bytes} bytes")


# Click command for orchestrating the data preparation and export
@click.command()
@click.option("--raw_data_path", default=".", help="Location where the raw NYC taxi trip data was saved")
@click.option("--dest_path", default=".", help="Location where the resulting files will be saved")
def orchestrate_pipeline(raw_data_path: str, dest_path: str):
    # Run data prep and model training
    dv, lr, df_train = prepare_and_train(raw_data_path)

    # Run the export to MLflow
    export_model_and_vectorizer(dv, lr, df_train, dest_path)


if __name__ == '__main__':
    orchestrate_pipeline()