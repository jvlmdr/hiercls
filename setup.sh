#!/bin/bash

apt-get update
apt-get install -y git tmux curl htop vim

curl https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-384.0.1-linux-x86_64.tar.gz | tar -xz
./google-cloud-sdk/install.sh
gcloud init  # in new shell

mkdir dl
gsutil cp gs://aiml-valmadre-research-data/dl/inaturalist2021/* dl/

mkdir data
cd data
ls ../dl/*.tar.gz | xargs -n 1 -P 2 tar -xvzf

git clone https://gitlab.aiml.team/jack.valmadre/hier-class.git
cd hier-class
pip install --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt

# ...

gsutil -m rsync -r experiments/ gs://aiml-valmadre-research-data/experiments/
