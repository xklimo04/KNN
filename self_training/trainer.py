import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    GenerationConfig
)
from tqdm import tqdm
import os
import pandas as pd
from target_dataset import TargetDataset
from logger import LoggerCallback
from compute_metrics import get_compute_metrics

class Trainer():
    def __init__(self, args):
        self.args = args
        
        processor_dir = os.path.join(args.project_dir, args.model_dir)
        model_dir = os.path.join(args.project_dir, args.model_dir)
        
        self.processor = TrOCRProcessor.from_pretrained(processor_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)

        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id

    def collate_fn_train(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return {
                "pixel_values": torch.zeros((1, 3, 384, 384)),
                "labels": torch.full((1, self.args.max_length), -100)
            }



        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])
        # if pixel_values.shape[-1] != 384 or pixel_values.shape[-2] != 384:
        #   pixel_values = torch.nn.functional.interpolate(pixel_values, size=(384, 384))

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

    def fit(self, train_df, val_df, it):
        print("--- Starting training phase (fit) ---")
        
        # Synchronizace konfigurace generování (prevence ValueError)
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id

        self.model.generation_config = GenerationConfig(
            max_length=self.args.max_length,
            num_beams=self.args.num_beams,
            early_stopping=self.args.early_stopping,
            decoder_start_token_id=self.processor.tokenizer.cls_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.sep_token_id
        )

        train_set = TargetDataset(
            train_df, 
            self.processor, 
            max_target_length=self.args.max_length, 
            df=True, 
            path_to_images=self.args.train_data_path_images
        )
        val_set = TargetDataset(
            val_df, 
            self.processor, 
            max_target_length=self.args.max_length, 
            df=True, 
            path_to_images=self.args.train_data_path_images
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.args.output_dir,
            learning_rate=self.args.lr,
            warmup_steps=50,
            per_device_train_batch_size=self.args.batch_size,
            num_train_epochs=self.args.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            fp16=True,
            predict_with_generate=True,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            data_collator=self.collate_fn_train,
            train_dataset=train_set,
            eval_dataset=val_set,
            processing_class=self.processor,
            compute_metrics=get_compute_metrics(self.processor),
            callbacks=[LoggerCallback(os.path.join(self.args.output_dir, "log.json"))]
        )

        trainer.train()

        trainer.save_model(f"self-training-second-stage-{it}")

    def predict(self, dataset_df):
      predict_dataset = TargetDataset(
          dataset_df, 
          self.processor, 
          max_target_length=self.args.max_length, 
          df=True, 
          path_to_images=self.args.target_path
      )

      dataloader = DataLoader(
          predict_dataset, 
          batch_size=self.args.batch_size, 
          collate_fn=self.collate_fn_train, 
          num_workers=0 
      )
    
      preds = []
      conf = []
      self.model.to("cuda")
      self.model.eval()

      with torch.no_grad():
          for batch in tqdm(dataloader, desc="Inference"):
              if batch is None: continue
              
              inputs = batch["pixel_values"].to("cuda")
              
              outputs = self.model.generate(
                  inputs,
                  max_length=self.args.max_length,
                  num_beams=self.args.num_beams,
                  return_dict_in_generate=True,
                  output_scores=True
              )

              predictions = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
              confidences = torch.exp(outputs.sequences_scores).cpu().tolist()

              preds.extend(predictions)
              conf.extend(confidences)

      return preds, conf
