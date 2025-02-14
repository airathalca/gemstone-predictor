import os
import sys
import pandas as pd
import numpy as np

from src.gemstonePrediction.exception import CustomException
from src.gemstonePrediction.logger import logging
from src.gemstonePrediction.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from dataclasses import dataclass

class CustomOrdinalEncoder:
  def __init__(self, mapping):
    self.mapping = mapping

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X_copy = X.copy()
    for col in X_copy.columns:
      X_copy[col] = X_copy[col].map(self.mapping[col])
    return X_copy

class FeatureGenerator(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X_copy = X.copy()
    # Create new features
    X_copy['volume'] = X_copy['x'] * X_copy['y'] * X_copy['z']
    X_copy['density'] = X_copy['carat'] / X_copy['volume']
    X_copy['depth_per_density'] = X_copy['depth'] / X_copy['density']
    X_copy['depth_per_volume'] = X_copy['depth'] / X_copy['volume']
    X_copy['depth_per_table'] = X_copy['depth'] / X_copy['table']
    X_copy = X_copy.drop(['x', 'y', 'z'], axis=1)
    return X_copy

@dataclass
class DataTransformationConfig:
  preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')
  train_array_path = os.path.join('artifacts', 'train_array.npy')
  test_array_path = os.path.join('artifacts', 'test_array.npy')

class DataTransformation:
  def __init__(self):
    self.config = DataTransformationConfig()

  def create_preprocessor(self):
    try:
      logging.info('Preprocessor creation started')
      
      categorical_cols = ['cut', 'color', 'clarity']
      numerical_cols = ['carat', 'depth', 'table', 'volume', 'density', 'depth_per_density', 'depth_per_volume', 'depth_per_table']
      
      category_orders = {
        'cut': {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5},
        'color': {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7},
        'clarity': {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5,
                    'VVS2': 6, 'VVS1': 7, 'IF': 8}
      }
      
      logging.info('Pipeline Initiated')

      num_pipeline = Pipeline([
        ('scaler', RobustScaler())
      ])

      cat_pipeline = Pipeline([
        ('ordinal_encoder', CustomOrdinalEncoder(category_orders)),
        ('std_scaler', StandardScaler())
      ])

      preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
      ])

      full_preprocessor = Pipeline([
        ('feature_generator', FeatureGenerator()),
        ('preprocessor', preprocessor)
      ])
      logging.info('Preprocessor created successfully')

      return full_preprocessor
    except Exception as e:
      logging.error(f"Error creating preprocessor: {e}")
      raise e

  def transform_data(self, train_path, test_path):
    try:
      logging.info("data transformation started")
      
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)
      logging.info('Train and Test data read successfully')

      preprocessor = self.create_preprocessor()
      target = 'price'

      train_df = train_df.dropna(axis=0)
      test_df = test_df.dropna(axis=0)
      logging.info('Null values removed successfully')

      numerical_col = train_df.select_dtypes(include=[np.number]).columns
      for col in numerical_col:
        train_df = train_df[train_df[col] > 0.0]
        test_df = test_df[test_df[col] > 0.0]
      logging.info('Invalid values removed successfully')

      feature_train_df, target_train_df = train_df.drop(target, axis=1), train_df[target]
      feature_test_df, target_test_df = test_df.drop(target, axis=1), test_df[target]

      logging.info('Applying transformation to data')
      feature_train_df = preprocessor.fit_transform(feature_train_df)
      feature_test_df = preprocessor.transform(feature_test_df)
      logging.info('Data transformed successfully')

      train_arr = np.concatenate((feature_train_df, target_train_df.values.reshape(-1, 1)), axis=1)
      test_arr = np.concatenate((feature_test_df, target_test_df.values.reshape(-1, 1)), axis=1)

      np.save(self.config.train_array_path, train_arr)
      np.save(self.config.test_array_path, test_arr)
      save_object(self.config.preprocessor_obj_path, preprocessor)
      logging.info('Preprocessor saved successfully')

      return (
        self.config.train_array_path,
        self.config.test_array_path
      )
    except Exception as e:
      logging.info("data transformation failed")
      raise CustomException(e, sys)
    
if __name__ == '__main__':
  data_transformation = DataTransformation()
  data_transformation.transform_data('artifacts/train.csv', 'artifacts/test.csv')