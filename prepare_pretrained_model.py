from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import json
import os

os.makedirs("./models", exist_ok=True)

# Load model
model_name = "microsoft/trocr-small-printed"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# Add new tokens to tokenizer
with open("./datasets/im2latex-100k-norm/vocab.json", "r") as f:
    new_tokens = json.load(f)
added_tokens = processor.tokenizer.add_tokens(new_tokens)
print(f"Added {added_tokens} new tokens to the tokenizer.")
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# Set special tokens
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

# Save model
processor.save_pretrained("./models/trocr_processor")
model.save_pretrained("./models/trocr_model")