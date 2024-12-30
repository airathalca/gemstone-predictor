import os
import sys
import pandas as pd
import numpy as np

from src.gemstonePrediction.exception import CustomException
from src.gemstonePrediction.logger import logging
from src.gemstonePrediction.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
  preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
  def __init__(self):
    self.config = DataTransformationConfig()

  def create_preprocessor(self):
    pass

  def transform_data(self, train_path, test_path):
    try:
      logging.info("data transformation started")
      
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)
      logging.info('Train and Test data read successfully')

      preprocessor = self.create_preprocessor()
      target = 'price'
      logging.info('Preprocessor created successfully')

      feature_train_df, target_train_df = train_df.drop(target, axis=1), train_df[target]
      feature_test_df, target_test_df = test_df.drop(target, axis=1), test_df[target]

      feature_train_df = preprocessor.fit_transform(feature_train_df)
      feature_test_df = preprocessor.tranform(feature_test_df)
      logging.info('Data transformed successfully')

      save_object(self.config.preprocessor_obj_path, preprocessor)
      logging.info('Preprocessor saved successfully')
    except Exception as e:
      logging.info("data transformation failed")
      raise CustomException(e, sys)