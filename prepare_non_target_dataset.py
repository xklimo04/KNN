from datasets import load_dataset, concatenate_datasets
from latex_normalizer import normalize_latex_tokens
import json
import os

def process_sample(sample):
    try:
        sample["formula"] = normalize_latex_tokens(sample["formula"])
        return sample
    except Exception:
        return None  # drop invalid

def get_vocabulary(dataset):
    vocabulary = set()
    for split in dataset.keys():
        for formula in dataset[split]["formula"]:
            if formula:
                tokens = formula.split()
                vocabulary.update(tokens)
    return sorted(list(vocabulary))

# Create dataset directory if it does not already exist
dataset_dir = "./datasets/im2latex-100k-norm"
os.makedirs(dataset_dir, exist_ok=True)
vocab_path = os.path.join(dataset_dir, "vocab.json")


# Load dataset and normalize
dataset = load_dataset("yuntian-deng/im2latex-100k")
dataset = dataset.map(process_sample, num_proc=4, load_from_cache_file=False)
dataset = dataset.filter(lambda x: x is not None and x["formula"] is not None)

# Test will not be used, so it is added to train split
dataset["train"] = concatenate_datasets([
    dataset["train"],
    dataset["test"]
])
del dataset["test"]

# Get dataset alphabet
alphabet = get_vocabulary(dataset)
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(alphabet, f, ensure_ascii=False, indent=4)

dataset.save_to_disk("./datasets/im2latex-100k-norm")