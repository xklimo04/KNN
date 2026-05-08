import pandas as pd
import numpy as np

class SelfTraining:
    def __init__(self, iterations, train, val, unlabeled_df, basemodel, conf_threshold):
        self.iterations = iterations
        self.model = basemodel
        self.conf_threshold = conf_threshold
        self.train_dataset = train.copy() 
        self.val_dataset = val.copy()
        self.unlabeled_df = unlabeled_df.copy()

    def run_self_training(self):
        iter_idx = 0
        
        while len(self.unlabeled_df) > 0 and iter_idx < self.iterations:
            print(f"\n--- Iteration: {iter_idx} ---")
            print(f"Current labeled pool size: {len(self.train_dataset)}")
            print(f"Unlabeled pool size: {len(self.unlabeled_df)}")

            # 1. Predikce na neoznačených datech
            pseudo_labels, conf = self.model.predict(self.unlabeled_df)
            
            temp_df = self.unlabeled_df.copy()
            temp_df["formula"] = pseudo_labels
            temp_df["confidence"] = conf

            # 2. Výběr horních X procent nejjistějších vzorků
            percent_to_take = self.conf_threshold 
            n_to_take = int(len(temp_df) * percent_to_take)
            

            n_to_take = max(1, n_to_take) 


            temp_df = temp_df.sort_values(by="confidence", ascending=False)


            newly_labeled = temp_df.head(n_to_take).copy()

            if len(newly_labeled) == 0:
                print(f"No samples passed threshold ({self.conf_threshold}). Stopping.")
                break

            # 3. Přidání pseudolabelů
            train_addition = newly_labeled.drop(columns=["confidence"])
            self.train_dataset = pd.concat([self.train_dataset, train_addition], ignore_index=True)

            # 4. Odstranění použitých vzorků z unlabeled poolu
            self.unlabeled_df = self.unlabeled_df.drop(newly_labeled.index).reset_index(drop=True)

            # 4. Trénování s nově rozšířeným datasetem
            print(f"Starting model fine-tuning (Iteration {iter_idx})...")
            self.model.fit(self.train_dataset, self.val_dataset, iter_idx)

            iter_idx += 1
            
        print("\nSelf-training finished!")
        return self.train_dataset