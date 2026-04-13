#!/bin/bash

if [ -e "valid.csv" ]; then
    unzip valid.zip -d valid
    mv valid.csv valid/valid/dataset/target
    rm -rf valid.zip
fi

source venv/bin/activate

python3 evaluation.py --model_path "$1" --proc_path "$1"  --output_file "$2" --target_path valid/valid/dataset/target --eval "$3"

deactivate