import argparse
import os
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig
)
from datasets import load_from_disk
from src.im2latexDataset import Im2LatexDataset
from src.evaluation.evaluate import get_compute_metrics
from src.logger import LoggerCallback


def collate_fn(batch, processor):

    images = [x["image"] for x in batch]
    formulas = [x["formula"] for x in batch]

    pixel_values = processor(images=images, return_tensors="pt").pixel_values

    labels = processor.tokenizer(
        formulas,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).input_ids

    # Ignore pad tokens
    labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

def main(args):

    # Define paths
    if args.model_dir is None:
        processor_dir = os.path.join(args.project_dir, "models/trocr_processor")
        model_dir = os.path.join(args.project_dir, "models/trocr_model")
    else:
        processor_dir = os.path.join(args.project_dir, args.model_dir)
        model_dir = os.path.join(args.project_dir, args.model_dir)
    dataset_dir = os.path.join(args.project_dir, "datasets/im2latex-100k-norm")
    os.makedirs(args.output_dir, exist_ok=True)
    print (processor_dir, model_dir, args.output_dir, dataset_dir)

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
    dataset = load_from_disk(dataset_dir)
    train_dataset = Im2LatexDataset(dataset['train'], augment=args.augment)
    val_dataset = Im2LatexDataset(dataset['val'], augment=False)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        warmup_steps=300,
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
        data_collator=lambda b: collate_fn(b, processor),
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

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--augment", action="store_true")

    args = parser.parse_args()

    main(args)