from PIL import Image
import torch
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pandas as pd
import argparse
from computecer import compute_cer
from load_dataset import Im2LatexDataset
import torch.nn.functional as F
import gc


parser = argparse.ArgumentParser(description="Validation on target dataset")

parser.add_argument("--model_path", type=str, help="Model path")
parser.add_argument("--proc_path", type=str, help="Processor path")
parser.add_argument("--output_file", type=str, default="results.csv", help="Output csv file")
parser.add_argument("--target_path", type=str, help="Path to target validation dataset")
parser.add_argument("--eval", type=str, help="Evaluation dataset", choices=['test', 'valid'])

args = parser.parse_args()


processor = TrOCRProcessor.from_pretrained("xklimo04/MER", subfolder=args.proc_path)
model = VisionEncoderDecoderModel.from_pretrained("xklimo04/MER", subfolder=args.proc_path)
model.to('cuda')
model.eval()

data_df = pd.read_csv(args.target_path + "/valid.csv")
if args.eval == 'valid':
    data_df = data_df.iloc[:1100]
else:
    data_df = data_df.iloc[1100:]
    
valid_data = Im2LatexDataset(data_df, processor, df=True, path_to_images=args.target_path + "/images/")

results = []
valid_cer = 0.0


with torch.no_grad():
    print("Started evaluation...\n")
    for batch in valid_data:
        if batch is None:
            continue
        pixel_values = batch["pixel_values"].to('cuda').unsqueeze(0)

        outputs = model.generate(
            pixel_values,
            max_length=256,
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=4,            
            early_stopping=True
        )

        label_ids = batch["labels"].to('cuda').unsqueeze(0)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        generated_text = processor.batch_decode(outputs.sequences[0], skip_special_tokens=True)
        label_str = label_str[0]

        cer = compute_cer(pred_ids = outputs.sequences, label_ids=label_ids, processor=processor)
        valid_cer += cer

        confidences = torch.exp(outputs.sequences_scores).tolist()

        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
        )
        probs = torch.exp(transition_scores)

        token_entropies = []
        for token_logits in outputs.scores:
            logits = token_logits[0]
            tokens_probs = F.softmax(logits, dim=-1)

            token_entropy = -torch.sum(tokens_probs * torch.log(tokens_probs + 1e-12))

            token_entropies.append(token_entropy.item())

        mean_prob = probs[0].mean().item()
        min_prob = probs[0].min().item()
        mean_sequence_entropy = sum(token_entropies) / len(token_entropies)

        for text, conf in zip(generated_text, confidences):
            results.append({
                "text": text,
                "ground truth": label_str,
                "CER": cer,
                "posterior probability": conf,
                "mean sequence probability": mean_prob,
                "min sequence probability": min_prob,
                "mean sequence entropy": mean_sequence_entropy
            })
        
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
    
total_cer = valid_cer / len(valid_data)
# print(f"Total CER: {total_cer}")
# results.append({
#     "text" : "TOTAL CER",
#     "ground truth": "",
#     "CER": total_cer,
#     "posterior probability": "",
#     "mean sequence probability": "",
#     "min sequence probability": "",
#     "mean sequence entropy": ""
# })
df = pd.DataFrame(results)
df.to_csv(args.output_file, index=False)

print(f"Results saved into {args.output_file}")
