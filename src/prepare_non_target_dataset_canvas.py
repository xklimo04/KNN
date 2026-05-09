from datasets import load_dataset, concatenate_datasets
from src.latex_normalizer import normalize_latex_tokens
import json
import os
from PIL import Image, ImageChops

def crop_whitespace(img, background_color=(255, 255, 255)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    bg = Image.new(img.mode, img.size, background_color)
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    
    if bbox:
        return img.crop(bbox)
    return img

# Var 1: Crop whitespace + resize to max 64x384 + paste on canvas 384x384
def process_sample(sample, target_height=64, max_width=384, canvas_size=384):
    try:
        sample["formula"] = normalize_latex_tokens(sample["formula"])
        if not sample["formula"]:
            return None

        img = sample["image"].convert("RGB")
        img_cropped = crop_whitespace(img)
        
        w, h = img_cropped.size
        if w == 0 or h == 0:
            return None

        scale = min(max_width / w, target_height / h)
        new_width = int(w * scale)
        new_height = int(h * scale)

        img_resized = img_cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
        paste_x = (canvas_size - new_width) // 2
        paste_y = (canvas_size - new_height) // 2
        canvas.paste(img_resized, (paste_x, paste_y))

        sample["image"] = canvas
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
dataset_dir = "./datasets/im2latex-100k-norm-1"
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

dataset.save_to_disk(dataset_dir)