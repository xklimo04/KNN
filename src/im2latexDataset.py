import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# Augmentation pipeline
aug_pipeline = A.Compose([
    A.SafeRotate(
        limit=(-2, 2),
        border_mode=cv2.BORDER_CONSTANT,
        fill=(255, 255, 255),
        p=0.7
    ),
    A.Affine(
        shear=(-1, 1),
        scale=(0.98, 1.02),
        border_mode=cv2.BORDER_CONSTANT,
        fill=(255, 255, 255),
        p=0.7
    ),
    A.OneOf([
        A.Morphological(p=1.0, scale=(2, 3), operation="dilation"),
        A.Morphological(p=1.0, scale=(2, 3), operation="erosion"),
        ], p=0.2
    ),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.3
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.3
    ),

    # Simulate yellowesh paper
    A.RGBShift(
        r_shift_limit=(5, 15),
        g_shift_limit=(3, 10),
        b_shift_limit=(-10, 0),
        p=0.4
    ),

    # Simulate scan artefacts
    A.MultiplicativeNoise(
        multiplier=(0.85, 1.15),
        elementwise=True,
        p=0.2
    ),
])

class Im2LatexDataset(Dataset): 
    def __init__(self, hf_dataset, augment=False): 
        self.dataset = hf_dataset 
        self.augment = augment 
    
    def __len__(self): 
        return len(self.dataset) 
    
    def __getitem__(self, idx): 
        item = self.dataset[idx] 
        image = np.array(item["image"]) 
        formula = item["formula"] 

        if self.augment: 
            image = aug_pipeline(image=image)["image"] 

        image = Image.fromarray(image)
        return { 
            "image": image, 
            "formula": formula
        }

