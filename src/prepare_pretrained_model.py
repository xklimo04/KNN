from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import json
import os

os.makedirs("./models", exist_ok=True)

# Load model
model_name = "microsoft/trocr-small-printed"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Add new tokens to tokenizer
with open("./datasets/im2latex-100k-norm/vocab.json", "r") as f:
    new_tokens = json.load(f)
added_tokens = processor.tokenizer.add_tokens(new_tokens)
print(f"Added {added_tokens} new tokens to the tokenizer.")
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# Save model
processor.save_pretrained("./models/trocr_processor")
model.save_pretrained("./models/trocr_model")