import os
import sys
import numpy as np
import pickle
from urllib.parse import urlparse

from sklearn.metrics import mean_squared_error, r2_score
from src.gemstonePrediction.components.data_transformation import DataTransformation
from src.gemstonePrediction.utils import load_object
from src.gemstonePrediction.logger import logging

import mlflow
import mlflow.sklearn

from dataclasses import dataclass

@dataclass
class ModelEvaluationConfig:
  trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelEvaluation:
  def __init__(self):
    logging.info('Loading model and preprocessor from artifacts folder')
    self.config = ModelEvaluationConfig()
    self.model = load_object(self.config.trained_model_file_path)
    logging.info('Preprocessor and model loaded successfully')

  
  def eval_metrics(self,actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    logging.info(f'Evaluation metrics: RMSE: {rmse}, R2: {r2}')
    return rmse, r2

  def evaluate_model(self, test_array):
    try:
      X_test,y_test=(test_array[:,:-1], test_array[:,-1])
      logging.info('Model evaluation started')

      tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme

      with mlflow.start_run():
        predicted_qualities = self.model.predict(X_test)
        (rmse, r2) = self.eval_metrics(y_test, predicted_qualities)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # this condition is for the dagshub
        # Model registry does not work with file store
        if tracking_url_type != "file":
          mlflow.sklearn.log_model(self.model, "model", registered_model_name="ml_model")
        # it is for the local 
        else:
          mlflow.sklearn.log_model(self.model, "model")
    except Exception as e:
      raise e
    
if __name__ == '__main__':
  data_transformation = DataTransformation()
  train, test = data_transformation.transform_data('artifacts/train.csv', 'artifacts/test.csv')
  model_evaluation = ModelEvaluation()
  model_evaluation.evaluate_model(test)