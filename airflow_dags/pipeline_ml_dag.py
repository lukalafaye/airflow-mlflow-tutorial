import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
import os 
import sys

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(SRC_PATH)

from project_paths import run as run_project_setup
from data_ingestion import run as run_ingestion
from preprocessing import run as run_preprocessing
from train_rf import run as run_train_rf
from train_lr import run as run_train_lr
from validate import run as run_validation


with DAG(
    dag_id="pipeline_ml",
    start_date=datetime.datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "example"]
):
    setup_project_task = PythonOperator(
        task_id="setup_projet",
        python_callable=run_project_setup
    )

    ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=run_ingestion
    )

    preprocessing_task = PythonOperator(
        task_id="preprocessing",
        python_callable=run_preprocessing
    )

    train_rf_task = PythonOperator(
        task_id='train_random_forest',
        python_callable=run_train_rf
    )

    train_lr_task = PythonOperator(
        task_id='train_logistic_regression',
        python_callable=run_train_lr
    )

    validate_task = PythonOperator(
        task_id='validate_models',
        python_callable=run_validation
    )

    setup_project_task >> ingestion_task
    ingestion_task >> preprocessing_task
    preprocessing_task >> [train_rf_task, train_lr_task]
    [train_rf_task, train_lr_task] >> validate_task