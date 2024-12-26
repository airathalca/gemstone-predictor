echo [$(date)]: "START"

echo [$(date)]: "creating env with python 3.10 version" 

conda create -p venv python=3.11 -y

echo [$(date)]: "activating the environment" 

conda activate venv/

echo [$(date)]: "installing the requirements" 

pip install -r requirements.txt

echo [$(date)]: "END" 