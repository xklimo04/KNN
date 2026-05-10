import pandas as pd
import numpy as np
from latex_normalizer import normalize_latex_tokens
import argparse


def calculate_norm_acc(csv_path):
    df = pd.read_csv(csv_path)
    matches = 0
    
    print(f"Processing {len(df)} rows...")

    for _, row in df.iterrows():
        try:
            gt_norm = normalize_latex_tokens(str(row['ground truth']))
            pred_norm = normalize_latex_tokens(str(row['text']))
        except Exception:
            pass
        
        if gt_norm == pred_norm:
            matches += 1

    return matches / len(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True)

    args = parser.parse_args()
    
    try:
        acc = calculate_norm_acc(args.input_file)
        print("-" * 30)
        print(f"Accuracy:    {(acc)*100:.2f}%")
    except Exception as e:
        print(f"Error: {e}")

