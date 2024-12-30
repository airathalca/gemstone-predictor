import os
import sys
import pandas as pd
import numpy as np

from src.gemstonePrediction.exception import CustomException
from src.gemstonePrediction.logger import logging
from src.gemstonePrediction.utils import load_object
from src.gemstonePrediction.enum import EnumCut, EnumColor, EnumClarity

class PredictionPipeline:
  def __init__(self):
    logging.info('Loading model and preprocessor from artifacts folder')
    self.preprocessor = load_object(os.path.join('artifacts', 'preprocessor.pkl'))
    self.model = load_object(os.path.join('artifacts', 'model.pkl'))
    logging.info('Preprocessor and model loaded successfully')

  def predict(self, features):
    try:
      features_scaled = self.preprocessor.transform(features)
      preds = self.model.predict(features_scaled)
      return preds
    except Exception as e:
      raise CustomException(e, sys)
    
class CustomData:
  def __init__(self, cut: EnumCut, color: EnumColor, clarity: EnumClarity, 
               carat: float, depth: float, table: float, x: float, y: float, z: float):
    self.cut = cut
    self.color = color
    self.clarity = clarity
    self.carat = carat
    self.depth = depth
    self.table = table
    self.x = x
    self.y = y
    self.z = z

  def get_data_as_df(self):
    try:
      custom_data = {
        'cut': [self.cut],
        'color': [self.color],
        'clarity': [self.clarity],
        'carat': [self.carat],
        'depth': [self.depth],
        'table': [self.table],
        'x': [self.x],
        'y': [self.y],
        'z': [self.z]
      }
      return pd.DataFrame(custom_data)
    except Exception as e:
      return CustomException(e, sys)