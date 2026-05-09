import argparse
import os
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig
)
from src.target_dataset_canvas import TargetDataset
from src.evaluation.compute_metrics import get_compute_metrics
from src.logger import LoggerCallback
import pandas as pd
import torch


def collate_fn(batch):

    # remove invalid samples
    batch = [x for x in batch if x is not None]

    if len(batch) == 0:
        return None

    pixel_values = torch.stack([
        x["pixel_values"] for x in batch
    ])

    labels = torch.stack([
        x["labels"] for x in batch
    ])

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

def main(args):

    # Define paths
    if args.model_dir is None:
        processor_dir = os.path.join(args.project_dir, "models/trocr/trocr_processor")
        model_dir = os.path.join(args.project_dir, "models/trocr/trocr_model")
    else:
        processor_dir = os.path.join(args.project_dir, args.model_dir)
        model_dir = os.path.join(args.project_dir, args.model_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained model from disk
    processor = TrOCRProcessor.from_pretrained(processor_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    # Set special tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    model.generation_config = GenerationConfig(
        max_length=args.max_length, 
        num_beams=args.num_beams, 
        early_stopping=args.early_stopping
    )

    for param in model.encoder.parameters():
        if args.freeze_encoder:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Load dataset from disk
    df = pd.read_csv(os.path.join(args.target_path, "valid.csv"))
    train_end = int(len(df) * 0.8)
    val_end = int(len(df) * 0.9)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    # test_df = df.iloc[val_end:]
    train_dataset = TargetDataset(train_df, processor, max_target_length=args.max_length, df=True, path_to_images=args.target_path + "/images/")
    val_dataset = TargetDataset(val_df, processor, max_target_length=args.max_length, df=True, path_to_images=args.target_path + "/images/")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        warmup_steps=50,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        dataloader_num_workers=4,
        logging_steps=100,
        logging_strategy="steps",
        gradient_accumulation_steps=2,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model, 
        args=training_args, 
        data_collator=collate_fn,
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        processing_class=processor,
        compute_metrics=get_compute_metrics(processor), 
        callbacks=[ 
            LoggerCallback(os.path.join(args.output_dir, "log.json"))
        ],
    ) 
    
    trainer.train() 

    trainer.save_model(os.path.join(args.output_dir, "best_model"))

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

    args = parser.parse_args()

    main(args)