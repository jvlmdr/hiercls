FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Not sure of best way to provide ssh key (or https token) to image to clone private repo.
# Just copy the current directory - docker caches each stage anyway.
# RUN apt-get update
# RUN apt-get -y install git
# RUN git config --global http.sslverify "false"

# Copy just requirements.txt in order to do pip install.
# Hope that this enables caching of environment without code.
# Note: Use pipreqs to generate minimal requirements.txt.
# Note: Check builtin numpy version: `docker run IMAGE python -c 'import numpy; print(numpy.__version__)'
COPY requirements.txt /app/
# New version of setuptools (setuptools-62.1.0) gave error about distutils.version.
# RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --extra-index-url https://download.pytorch.org/whl/cu113 -r /app/requirements.txt

COPY . /app/
