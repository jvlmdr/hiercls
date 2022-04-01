FROM pytorch/pytorch:latest

RUN apt-get update && apt-get -y install git && git config --global http.sslverify "false"

COPY requirements.txt /app/
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt
