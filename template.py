import os
from pathlib import Path

package = 'gemstone-prediction'

list_of_files = [
  ".github/workflows/.gitkeep",
  f"src/{package}/__init__.py",
  f"src/{package}/logger.py",
  f"src/{package}/utils.py",
  f"src/{package}/exception.py",
  f"src/{package}/config.py",
  f"src/{package}/components/__init__.py",
  f"src/{package}/components/data_ingestion.py",
  f"src/{package}/components/data_transformation.py",
  f"src/{package}/components/model_training.py",
  f"src/{package}/components/model_evaluation.py",
  f"src/{package}/pipeline/__init__.py",
  f"src/{package}/pipeline/data_pipeline.py",
  f"src/{package}/pipeline/training_pipeline.py",
  f"src/{package}/pipeline/prediction_pipeline.py",
  "notebooks/experimentation.ipynb",
  "notebooks/data/.gitkeep",
  "requirements.txt",
  "setup.py",
  "init_setup.sh",
]

for filename in list_of_files:
  filepath = Path(filename)
  filedir, filename = os.path.split(filepath)
  if filedir:
    os.makedirs(filedir, exist_ok=True)
  # create the file
  if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
    with open(filepath, 'w') as f:
      pass
  else:
    print(f"File {filepath} already exists and is not empty")
