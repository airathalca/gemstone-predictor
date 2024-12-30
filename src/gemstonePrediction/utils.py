import os
import sys
import pickle

from src.gemstonePrediction.logger import logging
from src.gemstonePrediction.exception import CustomException

def save_object(file_path: str, obj):
  try:
    logging.info(f'Saving object to {file_path}')
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, 'wb') as f:
      pickle.dump(obj, f)
    logging.info('Object saved successfully')
  except Exception as e:
    raise CustomException(e, sys)