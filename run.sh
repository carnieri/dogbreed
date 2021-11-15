#!/bin/bash

pip install -r requirements.txt

python download_dataset.py

python cleanup_dataset.py

# convert notebook to python script so user can reproduce steps
jupyter-nbconvert --to python --no-prompt train_and_test.ipynb

python train_and_test.py