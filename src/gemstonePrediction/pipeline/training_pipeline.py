import os
import sys
import pandas as pd

from src.gemstonePrediction.components.data_ingestion import DataIngestion
from src.gemstonePrediction.components.data_transformation import DataTransformation
from src.gemstonePrediction.components.model_training import ModelTrainer
from src.gemstonePrediction.components.model_evaluation import ModelEvaluation
from src.gemstonePrediction.logger import logging
from src.gemstonePrediction.exception import CustomException

class TrainingPipeline:
  def start_data_ingestion(self, original_path, generated_path):
    try:
      data_ingestion = DataIngestion()
      train_path, test_path = data_ingestion.read_data(generated_path, original_path)
      return train_path, test_path
    except Exception as e:
      raise CustomException(e,sys)
      
  def start_data_transformation(self, train_path, test_path): 
    try:
      data_transformation = DataTransformation()
      train_arr, test_arr = data_transformation.transform_data(train_path,test_path)
      return train_arr, test_arr
    except Exception as e:
      raise CustomException(e,sys)

  def start_model_training(self, train_arr):
    try:
      model_trainer = ModelTrainer()
      model_trainer.train_model(train_arr)
    except Exception as e:
      raise CustomException(e,sys)
              
  def start_trainig(self):
    try:
      train_path, test_path = self.start_data_ingestion('notebooks/data/train.csv', 'notebooks/data/cubic_zirconia.csv')
      train_arr, test_arr = self.start_data_transformation(train_path, test_path)
      self.start_model_training(train_arr)
    except Exception as e:
      raise CustomException(e,sys)