# Gemstone Price Predictor

## How to run?
1. Create Virtual Environment with conda or other tools.
```bash
conda create -p venv python=3.11 -y
```
2. Activate the environment
```bash
conda activate venv/
```
3. Install the requirements
```bash
pip install -r requirements.txt
```
4. Create a .env file and add the following variables
```bash
AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
```
5. Run the Apache Airflow
```bash
nohup airflow webserver --port 8080 & airflow scheduler &
```
6. Run the flask app
```bash
python app.py
```

## Docker
If you want to run the application in docker, you can follow the following steps:
1. Build both the images (Apache Airflow and Flask)
```bash
docker-compose build
```
2. Run the docker-compose
```bash
docker-compose up
```
3. To stop the docker-compose
```bash
docker-compose down
```


# AWS-CICD-Deployment-with-Github-Actions
## 1. Login to AWS console.
## 2. Create new user for deployment with following permissions:
- AmazonEC2FullAccess
- AmazonEC2ContainerRegistryFullAccess
- AmazonS3FullAccess
## 3. Create Access Key for the user and save it.
## 4. Create ECR Repository to store docker images.
## 5. Create S3 bucket to store dvc files.
## 6. Create EC2 Ubuntu instance and Install Docker on it.
- sudo apt-get update -y
- sudo apt-get upgrade
- curl -fsSL https://get.docker.com -o get-docker.sh
- sudo sh get-docker.sh
- sudo usermod -aG docker ubuntu
- newgrp docker
## 7. Create self-hosted GitHub runner and configure it on EC2 instance by following the steps on GitHub.
## 8. Create following secrets in GitHub repository:
- AWS_ACCESS_KEY_ID: Access key of the user created in step 2.
- AWS_SECRET_ACCESS_KEY: Secret key of the user created in step 2.
- AWS_REGION: Region of the AWS services.
- ECR_REPOSITORY_AIRFLOW: ECR repository created in step 4.
- ECR_REPOSITORY_FLASK: ECR repository created in step 4.
- AWS_ECR_LOGIN_URI: URI from AWS ECR login.
## 9. Run the GitHub Actions.