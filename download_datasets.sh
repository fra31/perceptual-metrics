#!/bin/bash

# Adpted from https://github.com/ssundaram21/dreamsim/blob/main/dataset/download_dataset.sh.
mkdir -p /tmlscratch/fcroce/datasets
cd /tmlscratch/fcroce/datasets

# Download NIGHTS dataset
wget -O nights.zip https://data.csail.mit.edu/nights/nights.zip

unzip nights.zip
rm nights.zip