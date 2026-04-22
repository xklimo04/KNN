import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from PIL import Image
import os

class Im2LatexDataset(Dataset):
    def __init__(self, hf_dataset, processor, max_target_length=128, df=False, path_to_images=""):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_target_length = max_target_length
        self.df = df
        self.path_to_images = path_to_images

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.df:
            img_path = os.path.join(self.path_to_images, self.dataset["image"].iloc[idx])
            
            if not os.path.exists(img_path):
                return None  
            
            image = Image.open(img_path).convert('RGB')
        else:
            image = self.dataset["image"].iloc[idx].convert('RGB')

        w, h = image.size
        aspect_ratio = w / h
        target_width = int(64 * aspect_ratio)
        image = image.resize((target_width, 64), Image.Resampling.LANCZOS)
        # canvas = Image.new("RGB", (384, 384), (255, 255, 255))
        # canvas.paste(image, (0, (384 - image.size[1]) // 2))

        # image=canvas
        
        text = self.dataset['formula'].iloc[idx]

        pixel_values = self.processor(image, return_tensors='pt').pixel_values

        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length,
            truncation=True
        ).input_ids

        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values.squeeze(), 
            "labels": torch.tensor(labels)
        }

        

def load_datasets(dataset_path, processor):
    full_dataset = load_from_disk(dataset_path)
    
    train_ds = full_dataset["train"]
    valid_ds = full_dataset["val"]

    train_dataset = Im2LatexDataset(train_ds, processor)
    valid_dataset = Im2LatexDataset(valid_ds, processor)

    return train_dataset, valid_dataset

