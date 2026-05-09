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
        target_ratio = 320 / 64  # 5.0
        current_ratio = w / h

        # Pad na poměr stran 320:64 (5:1)
        if current_ratio < target_ratio:
            # obrázek je moc úzký → přidáme padding do šířky
            new_w = int(h * target_ratio)
            new_h = h
        else:
            # obrázek je moc široký → přidáme padding do výšky
            new_w = w
            new_h = int(w / target_ratio)

        # bílý canvas
        canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))

        # zarovnání vpravo nahoru
        paste_x = new_w - w   # úplně doprava
        paste_y = 0           # úplně nahoru

        canvas.paste(image, (paste_x, paste_y))

        # finální resize na 320x64
        image = canvas.resize((320, 64), Image.Resampling.LANCZOS)
        
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

class Im2LatexCanvasDataset(Dataset):
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
            img_path = os.path.join(
                self.path_to_images,
                self.dataset["image"].iloc[idx]
            )

            if not os.path.exists(img_path):
                return None

            image = Image.open(img_path).convert("RGB")
        else:
            image = self.dataset["image"].iloc[idx].convert("RGB")

        # původní rozměry
        w, h = image.size

        canvas_size = 384

        # zachování poměru stran
        scale = min(canvas_size / w, canvas_size / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # resize bez deformace
        resized = image.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS
        )

        # bílý canvas 384x384
        canvas = Image.new(
            "RGB",
            (canvas_size, canvas_size),
            (255, 255, 255)
        )

        # vystředění
        paste_x = (canvas_size - new_w) // 2
        paste_y = (canvas_size - new_h) // 2

        canvas.paste(resized, (paste_x, paste_y))

        text = self.dataset["formula"].iloc[idx]

        # důležité:
        # processor by měl mít do_resize=False
        pixel_values = self.processor(
            canvas,
            return_tensors="pt"
        ).pixel_values

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids

        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }

