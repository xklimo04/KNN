import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from PIL import Image
import os

class TargetDataset(Dataset):
    def __init__(self, hf_dataset, processor, max_target_length=128, df=False, path_to_images="", target_height=64, max_width=384, canvas_size=384):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_target_length = max_target_length
        self.df = df
        self.path_to_images = path_to_images
        self.target_height = target_height
        self.max_width = max_width
        self.canvas_size = canvas_size

    def __len__(self):
        return len(self.dataset)
    
    def preprocess_image(self, image):
        image = image.convert("RGB")


        w, h = image.size
        if w == 0 or h == 0:
            return None

        scale = min(
            self.max_width / w,
            self.target_height / h
        )

        new_width = int(w * scale)
        new_height = int(h * scale)

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            "RGB",
            (self.canvas_size, self.canvas_size),
            (255, 255, 255)
        )

        paste_x = (self.canvas_size - new_width) // 2
        paste_y = (self.canvas_size - new_height) // 2

        canvas.paste(image, (paste_x, paste_y))

        return canvas

    def __getitem__(self, idx):
        if self.df:
            img_path = os.path.join(self.path_to_images, self.dataset["image"].iloc[idx])
            
            if not os.path.exists(img_path):
                return None  
            
            image = Image.open(img_path).convert('RGB')
        else:
            image = self.dataset["image"].iloc[idx].convert('RGB')

        image = self.preprocess_image(image)
        if image is None:
            return None
        
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