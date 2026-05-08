import numpy as np
import editdistance

def get_compute_metrics(processor):

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        pad_id = processor.tokenizer.pad_token_id
        bos_id = processor.tokenizer.cls_token_id
        eos_id = processor.tokenizer.sep_token_id

        total_distance = 0
        total_tokens = 0
        exact_matches = 0

        for pred, label in zip(preds, labels):

            label = [
                int(x) for x in label
                if x not in (-100, pad_id, bos_id, eos_id)
            ]

            pred = [
                int(x) for x in pred
                if x not in (pad_id, bos_id, eos_id)
            ]

            distance = editdistance.eval(pred, label)

            total_distance += distance
            total_tokens += len(label)

            if pred == label:
                exact_matches += 1

        cer = total_distance / max(total_tokens, 1)
        acc = exact_matches / len(preds)

        return {
            "cer": cer,
            "accuracy": acc
        }

    return compute_metrics