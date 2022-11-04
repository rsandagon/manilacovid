FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# NOTE: Run download_data prior to run

COPY ./app /app
COPY ./requirements.txt .
# COPY ./app/download_model.py /app/download_model.py

RUN pip3 install -r requirements.txt --no-cache-dir
# RUN python3 /app/download_model.py --model scibert-nli