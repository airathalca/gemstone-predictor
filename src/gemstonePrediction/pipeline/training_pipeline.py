import os
import sys
import pandas as pd

from src.gemstonePrediction.components.data_ingestion import DataIngestion
from src.gemstonePrediction.components.data_transformation import DataTransformation
from src.gemstonePrediction.components.model_training import ModelTrainer
from src.gemstonePrediction.components.model_evaluation import ModelEvaluation
from src.gemstonePrediction.logger import logging
from src.gemstonePrediction.exception import CustomException

data_ingestion = DataIngestion()
train_data_path, test_data_path = data_ingestion.read_data('notebooks/data/train.csv', 'notebooks/data/cubic_zirconia.csv')

data_transformation=DataTransformation()
train_arr, test_arr =data_transformation.transform_data(train_data_path,test_data_path)

model_trainer=ModelTrainer()
model_trainer.train_model(train_arr)

model_eval = ModelEvaluation()
model_eval.evaluate_model(test_arr)