import os
import sys
import pandas as pd

from src.gemstonePrediction.exception import CustomException
from src.gemstonePrediction.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
  train_data_path: str = os.path.join('artifacts', 'train.csv')
  test_data_path: str = os.path.join('artifacts', 'test.csv')
  raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
  def __init__(self):
    self.config = DataIngestionConfig()

  def read_data(self, data_path_generated, data_path_original):
    logging.info("data ingestion started")
    try:
      data_generated = pd.read_csv(data_path_generated, index_col=0)
      data_original = pd.read_csv(data_path_original, index_col=0)
      data = pd.concat([data_generated, data_original], ignore_index=True)
      logging.info('CSV file read successfully')
    
      os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
      data.to_csv(self.config.raw_data_path, index=False)
      logging.info('Raw data saved successfully')

      train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
      train_set.to_csv(self.config.train_data_path, index=False)
      test_set.to_csv(self.config.test_data_path, index=False)
      logging.info('Train and Test data saved successfully')

      return [
        self.config.train_data_path,
        self.config.test_data_path
      ]

    except Exception as e:
      logging.info("data ingestion failed")
      raise CustomException(e, sys)
    
if __name__ == "__main__":
  data_ingestion = DataIngestion()
  data_ingestion.read_data('notebooks/data/train.csv', 'notebooks/data/cubic_zirconia.csv')