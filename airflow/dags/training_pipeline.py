from __future__ import annotations
from textwrap import dedent
from airflow import DAG
from airflow.operators.bash import BashOperator
from src.gemstonePrediction.pipeline.training_pipeline import TrainingPipeline
import pendulum
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

training_pipeline=TrainingPipeline()
with DAG(
  "gemstone_training_pipeline",
  default_args={"retries": 2},
  description="Training pipeline for gemstone prediction",
  schedule=timedelta(minutes=120),
  start_date=pendulum.datetime(2024, 12, 31, tz=ZoneInfo("Asia/Jakarta")),
  catchup=False,
  tags=["machine_learning ","regression", "gemstone"],
) as dag:
  dag.doc_md = __doc__
  
  data_ingestion_task = BashOperator(
    task_id="data_ingestion",
    bash_command="cd /app && dvc repro -s data_ingestion >> /app/logs/data_ingestion.log 2>&1",
  )
  data_ingestion_task.doc_md = dedent(
    """
    #### Ingestion Task
    This task is responsible for ingesting the data.
    
    """
  )

  data_transform_task = BashOperator(
    task_id="data_transformation",
    bash_command="cd /app && dvc repro -s data_transformation >> /app/logs/data_transformation.log 2>&1",
  )
  data_transform_task.doc_md = dedent(
    """
    #### Transformation Task
    This task is responsible for transforming the data.
    
    """
  )

  model_trainer_task = BashOperator(
    task_id="model_trainer",
    bash_command="cd /app && dvc repro -s model_training >> /app/logs/model_training.log 2>&1",
  )
  model_trainer_task.doc_md = dedent(
    """
    #### Model Training Task
    This task is responsible for training the model.
    
    """
  )

  push_to_s3_task = BashOperator(
    task_id="push_to_s3",
    bash_command="cd /app && dvc push >> /app/logs/dvc_push.log 2>&1",
  )
data_ingestion_task >> data_transform_task >> model_trainer_task >> push_to_s3_task