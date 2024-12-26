from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
  with open(file_path) as f:
    req = f.readlines()
    req = [r.replace('\n', '') for r in req]

    if HYPEN_E_DOT in req:
      req.remove(HYPEN_E_DOT)
  
  return req

setup(
  name='GemstonePricePredictor',
  version='0.0.1',
  author='airathalca',
  author_email='airathalca@gmail.com',
  packages=find_packages(),
  install_requires=get_requirements('requirements.txt'),
)
