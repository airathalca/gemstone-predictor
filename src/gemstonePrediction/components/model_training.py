import os
import sys
from urllib.parse import urlparse
import dagshub
import numpy as np 
import pandas as pd

from src.gemstonePrediction.components.data_transformation import DataTransformation
from src.gemstonePrediction.exception import CustomException
from src.gemstonePrediction.logger import logging
from src.gemstonePrediction.utils import save_object

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
import optuna

import mlflow.sklearn
import mlflow.xgboost
import mlflow

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
  def __init__(self):
    self.config = ModelTrainerConfig()
    self.models = {
      'XGBRegressor': (XGBRegressor, self.objective_xgb),
      'Ridge': (Ridge, self.objective_ridge),
      'Lasso': (Lasso, self.objective_lasso),
    }

  def train_model(self, train_arr_path):
    try:
      logging.info('Loading and preparing training data')
      train_arr = np.load(train_arr_path)
      train_set = pd.DataFrame(train_arr)
      X_train, y_train = train_set.iloc[:, :-1], train_set.iloc[:, -1]
      best_models = {}
      logging.info('Training data loaded successfully')
      
      env = os.getenv('MLFLOW_ENV', 'local')
      if env == 'local':
        dagshub.init(repo_owner='airathalca', repo_name='mlops-project', mlflow=True)

      logging.info('Training model with Optuna as hyperparameter tuner started')
      for model_name, (model_class, objective_func) in self.models.items():
        logging.info(f'Training {model_name} with Optuna optimization')
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective_func(trial, X_train, y_train),
            n_trials=10,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_trial.params
        logging.info(f'Best parameters for {model_name}: {best_params}')

        with mlflow.start_run(run_name=model_name):
          mlflow.log_params(best_params)

          model = model_class(**best_params)
          model.fit(X_train, y_train)

          if model_name == 'XGBRegressor':
            mlflow.xgboost.log_model(model, model_name, registered_model_name=model_name)
          else:
            mlflow.sklearn.log_model(model, model_name, registered_model_name=model_name)
        
          y_pred = model.predict(X_train)
          rmse = np.sqrt(mean_squared_error(y_train, y_pred))
          r2 = r2_score(y_train, y_pred)
          logging.info(f'RMSE for {model_name}: {rmse}')
          best_models[model_name] = {
            'model': model,
            'rmse': rmse,
            'r2': r2,
            'best_params': best_params
          }

          mlflow.log_metric(f'RMSE', rmse)
          mlflow.log_metric(f'R2', r2)

      best_model_name = min(best_models, key=lambda x: best_models[x]['rmse'])
      best_model = best_models[best_model_name]['model']
      best_rmse = best_models[best_model_name]['rmse']
      logging.info(f'Best model: {best_model_name} with RMSE: {best_rmse}')

      logging.info('Saving best model')
      save_object(self.config.trained_model_file_path, best_model)
      logging.info('Best model saved successfully')
      return best_model

    except Exception as e:
      logging.error('Error occurred while training models')
      raise CustomException(e, sys)

  def objective_xgb(self, trial, X_train, y_train):
    params = {
            'objective': 'reg:squarederror',
            'tree_method': trial.suggest_categorical('tree_method', ['hist']),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1e2, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0, step=0.1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
            'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1e0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=10),
            'max_depth': trial.suggest_int('max_depth', 4, 10, step=2),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        }
    return self._get_cross_val_rmse(XGBRegressor, params, X_train, y_train)

  def objective_ridge(self, trial, X_train, y_train):
    params = {
            'alpha': trial.suggest_float('alpha', 1e-3, 1e2, log=True),
            'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
            'max_iter': 10000,
            'random_state': 42
        }
    return self._get_cross_val_rmse(Ridge, params, X_train, y_train)

  def objective_lasso(self, trial, X_train, y_train):
    params = {
            'alpha': trial.suggest_float('alpha', 1e-3, 1e2, log=True),
            'max_iter': 10000,
            'random_state': 42
        }
    return self._get_cross_val_rmse(Lasso, params, X_train, y_train)

  def _get_cross_val_rmse(self, model_class, params, X_train, y_train):
    kf = KFold(n_splits=4, random_state=42, shuffle=True)
    val_split_rmse = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_split, X_val_split = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_split, y_val_split = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = model_class(**params)
        model.fit(X_train_split, y_train_split)
        y_pred_val = model.predict(X_val_split)
        rmse = np.sqrt(mean_squared_error(y_val_split, y_pred_val))
        val_split_rmse.append(rmse)

    val_rmse = np.mean(val_split_rmse)
    return val_rmse
  
if __name__ == '__main__':
  data_transformation = DataTransformation()
  train, test = data_transformation.transform_data('artifacts/train.csv', 'artifacts/test.csv')
  model_trainer = ModelTrainer()
  best_model = model_trainer.train_model(train)
  print(f"Best Model: {type(best_model).__name__}")