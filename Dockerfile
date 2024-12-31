FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN apt update -y
RUN if [ ! -e /usr/local/bin/python ]; then ln -s /usr/local/bin/python3 /usr/local/bin/python; fi
RUN pip install -r requirements.txt
RUN if ! dvc remote list | grep -q 'datastore'; then \
  dvc remote add -d datastore s3://aira-dvc-gemstone; \
  fi
CMD ["python3", "app.py"]