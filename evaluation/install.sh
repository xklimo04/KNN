#!/bin/bash

python3 -m venv venv

./venv/bin/python3 -m pip install --upgrade pip
./venv/bin/python3 -m pip install -r requirements.txt

source venv/bin/activate

python3 load_model.py

deactivate