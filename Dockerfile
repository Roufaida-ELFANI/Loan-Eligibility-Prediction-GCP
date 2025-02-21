FROM python:3.12.7

WORKDIR /

COPY app/GBM_Model_version.pkl ./GBM_Model_version.pkl

COPY app/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY app/templates ./templates

COPY app/app.py ./app.py

ENV PORT=8080
CMD uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 0
