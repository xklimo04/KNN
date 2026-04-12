import json
import argparse
import matplotlib.pyplot as plt


def main(args):
    with open(args.log_file, "r") as f:
        data = json.load(f)

    train = [
        x for x in data
        if x.get("type") == "train"
        and "loss" in x
        and "epoch" in x
    ]

    epoch_train = [x["epoch"] for x in train]
    loss = [x["loss"] for x in train]

    eval_data = [
        x for x in data
        if x.get("type") == "eval"
        and "eval_cer" in x
        and "epoch" in x
    ]

    epoch_val = [x["epoch"] for x in eval_data]
    cer = [x["eval_cer"] for x in eval_data]

    _, ax1 = plt.subplots()

    l1 = ax1.plot(epoch_train, loss, label="Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    l2 = ax2.plot(epoch_val, cer, "-r", label="CER")
    ax2.set_ylabel("CER")

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)

    plt.title(args.title)
    plt.savefig(args.title + ".png")
    plt.close()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_file", required=True)
    parser.add_argument("--title", required=True)

    args = parser.parse_args()

    main(args)
