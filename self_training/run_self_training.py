import os
import argparse
from trainer import Trainer
import zipfile
import pandas as pd
import random
from datasets import Dataset, DatasetDict
import glob
from self_training import SelfTraining

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--train_data_path_images", type=str, required=True)
    parser.add_argument("--zip_path", type=str, required=True)
    parser.add_argument("--iterations", type=int, required=True)

    args = parser.parse_args()

    trainermodel = Trainer(args)


    zip_path = args.zip_path
    target_count = 10000

    unlabeled_df = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = z.namelist()
        
        image_files = [f for f in all_files if f.lower().endswith('.jpg') and not f.startswith('__MACOSX')]
        
        print(f"Celkem nalezeno {len(image_files)} obrázků v ZIPu.")
        
        sample_size = min(target_count, len(image_files))
        selected_files = random.sample(image_files, sample_size)
        
        unlabeled_df = pd.DataFrame({
            "image": selected_files,
            "formula": [""] * len(selected_files)
    })

    df = pd.read_csv(args.train_data_path)
    train_end = int(len(df) * 0.8)
    val_end = int(len(df) * 0.9)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    #test_df = df.iloc[val_end:]

    def filter_data(df, path):
        exists = df['image'].apply(lambda x: os.path.exists(os.path.join(path, x)))
        print(f"Filtrace: Ponecháno {exists.sum()} z {len(df)} vzorků.")
        return df[exists].reset_index(drop=True)

    image_path = args.train_data_path_images

    train_df_clean = filter_data(train_df, image_path)
    val_df_clean = filter_data(val_df, image_path)


    self_training_logic = SelfTraining(
        iterations=args.iterations,
        train=train_df_clean,
        val=val_df_clean,
        unlabeled_df=unlabeled_df,
        basemodel=trainermodel,
        conf_threshold=0.01
    )

    print(f"Started self training. Found {len(unlabeled_df)} unlabeled samples.")
    self_training_logic.run_self_training()
