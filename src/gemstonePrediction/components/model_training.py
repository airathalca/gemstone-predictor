import os
import sys
import numpy as np 
import pandas as pd

from src.gemstonePrediction.components.data_transformation import DataTransformation
from src.gemstonePrediction.exception import CustomException
from src.gemstonePrediction.logger import logging
from src.gemstonePrediction.utils import save_object

from sklearn.model_selection import KFold, RepeatedKFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import optuna

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
  def __init__(self):
    self.config = ModelTrainerConfig()

  def train_model(self, train_arr: np.ndarray):
    try:
      logging.info('Training model with XGBoost and Optuna as hyperparameter tuner started')
      study = optuna.create_study(direction='minimize')
      train_set = pd.DataFrame(train_arr)
      X_train, y_train = train_set.iloc[:, :-1], train_set.iloc[:, -1]

      study.optimize(lambda trial: self.objective_xgb(trial, X_train, y_train), n_trials=20, show_progress_bar=True)
      logging.info(f'Best trial: {study.best_trial}')
      best_params = study.best_trial.params

      logging.info('Training model with best hyperparameters')
      estimator = XGBRegressor(**best_params)
      estimator.fit(X_train, y_train)

      y_pred = estimator.predict(X_train)
      train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
      logging.info(f'Training RMSE: {train_rmse}')

      logging.info('Saving trained model')
      save_object(self.config.trained_model_file_path, estimator)
      logging.info('Model training completed')

      return estimator

    except Exception as e:
      logging.error(f'Error occurred while training model')
      raise CustomException(e, sys)
  
  def objective_xgb(self, trial, X_train, y_train):
    params = {
      'objective': 'reg:squarederror',
      'tree_method': trial.suggest_categorical(
        'tree_method', ['hist']
      ),
      'reg_lambda': trial.suggest_float(
        'reg_lambda', 1e-3, 1e2, log=True
      ),
      'colsample_bytree': trial.suggest_float(
        'colsample_bytree', 0.5, 1.0, step=0.1
      ),
      'colsample_bylevel': trial.suggest_float(
        'colsample_bylevel', 0.5, 1.0, step=0.1
      ),
      'subsample': trial.suggest_float(
        'subsample', 0.5, 1.0, step=0.1
      ),
      'learning_rate': trial.suggest_float(
        'learning_rate', 1e-2, 1e0, log=True
      ),
      'n_estimators': trial.suggest_int(
        'n_estimators', 50, 200, step=10
      ),
      'max_depth': trial.suggest_int(
        'max_depth', 4, 10, step=2
      ),
      'grow_policy': trial.suggest_categorical(
        'grow_policy', ['depthwise', 'lossguide']
      )
    }
    kf = KFold(n_splits=4, random_state=42, shuffle=True)
    val_split_rmse = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_split, X_val_split = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_split, y_val_split = y_train.iloc[train_idx], y_train.iloc[val_idx]
        estimator = XGBRegressor(**params, early_stopping_rounds=10, eval_metric='rmse')
        estimator.fit(X_train_split, y_train_split, eval_set=[(X_val_split, y_val_split)], verbose=False)
        Y_pred_val = pd.Series(estimator.predict(X_val_split), index=X_val_split.index)
        rmse = np.sqrt(mean_squared_error(y_val_split, Y_pred_val))
        val_split_rmse.append(rmse)
        
    val_rmse = np.mean(val_split_rmse)
    return val_rmse
  
if __name__ == '__main__':
  data_transformation = DataTransformation()
  train, test = data_transformation.transform_data('artifacts/train.csv', 'artifacts/test.csv')
  model_trainer = ModelTrainer()
  print(model_trainer.train_model(train))