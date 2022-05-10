#!/bin/bash

apt-get update
apt-get install -y git
apt-get install -y tmux

git clone https://gitlab.aiml.team/jack.valmadre/hier-class.git

pip install --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt
