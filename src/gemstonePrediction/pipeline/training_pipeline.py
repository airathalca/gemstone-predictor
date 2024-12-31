import argparse
import os
import sys
import pandas as pd

from src.gemstonePrediction.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.gemstonePrediction.components.data_transformation import DataTransformation, DataTransformationConfig
from src.gemstonePrediction.components.model_training import ModelTrainer, ModelTrainerConfig
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
              
  def start_training(self):
    try:
      train_path, test_path = self.start_data_ingestion('notebooks/data/train.csv', 'notebooks/data/cubic_zirconia.csv')
      train_arr_path, test_arr_path = self.start_data_transformation(train_path, test_path)
      self.start_model_training(train_arr_path)
    except Exception as e:
      raise CustomException(e,sys)

def main():
  parser = argparse.ArgumentParser(description="Training pipeline")
  parser.add_argument('stage', choices=['data_ingestion', 'data_transformation', 'model_training', 'full_pipeline'],
                      help="Specify the stage of the pipeline to run.")
  args = parser.parse_args()

  pipeline = TrainingPipeline()

  if args.stage == 'data_ingestion':
    pipeline.start_data_ingestion('notebooks/data/train.csv', 'notebooks/data/cubic_zirconia.csv')

  elif args.stage == 'data_transformation':
    config = DataIngestionConfig()
    pipeline.start_data_transformation(config.train_data_path, config.test_data_path)

  elif args.stage == 'model_training':
    config = DataTransformationConfig()
    pipeline.start_model_training(config.train_array_path)

  elif args.stage == 'full_pipeline':
    pipeline.start_training_pipeline()

if __name__ == '__main__':
    main()