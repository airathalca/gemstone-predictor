import os
import sys
import pandas as pd

from src.gemstonePrediction.exception import CustomException
from src.gemstonePrediction.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass