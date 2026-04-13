import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import auc

def compute_curve(df, performance_measure):
    # Sort by selected confidence measure column
    df_sorted = df.sort_values(by=performance_measure, ascending=False).reset_index(drop=True)

    cer_values = df_sorted["CER"].values

    cum_cer = np.cumsum(cer_values)
    counts = np.arange(1, len(cer_values) + 1)

    # Y axis points
    cum_avg_cer = cum_cer / counts

    # X axis points
    fractions = counts / len(cer_values)

    return fractions, cum_avg_cer


def main(args):
    df = pd.read_csv(args.input_file, header=0)

    curves = {}
    aucs = {}

    # For all performance measures
    for col in df.columns.tolist()[3:]:
        fractions, cum_avg_cer = compute_curve(df, col)
        curves[col] = fractions, cum_avg_cer
        aucs[col] = auc(fractions, cum_avg_cer)

    # Optimal performance measure
    fractions, cum_avg_cer = compute_curve(df, "CER")
    curves["optimal"] = fractions, cum_avg_cer
    aucs["optimal"] = auc(fractions, cum_avg_cer)

    # Random selection performance measure
    df_shuffled = df.sample(frac=1, random_state=42)
    fractions, cum_avg_cer = compute_curve(df_shuffled, "CER")
    curves["random"] = fractions, cum_avg_cer
    aucs["random"] = auc(fractions, cum_avg_cer)

    plt.figure()

    for name, (fractions, cum_avg_cer) in curves.items():
        plt.plot(fractions, cum_avg_cer, label=f"{name} (AUC={aucs[name]:.3f})")

    plt.xlabel("Fraction of selected data")
    plt.ylabel("Cumulative CER")
    plt.title("Predictive Power of Confidence Measures")
    plt.legend()
    plt.grid()

    plt.savefig(args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", default="performance_measures.png")

    args = parser.parse_args()

    main(args)