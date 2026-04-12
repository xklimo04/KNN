import numpy as np
import editdistance

def get_compute_metrics(processor):
    def normalize(s):
        return s.replace(" ", "")
    
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        labels = np.where(labels == -100, processor.tokenizer.pad_token_id, labels)

        pred_str = processor.batch_decode(preds, skip_special_tokens=True)
        label_str = processor.batch_decode(labels, skip_special_tokens=True)

        pred_str = [normalize(p) for p in pred_str]
        label_str = [normalize(g) for g in label_str]

        cer = sum(
            editdistance.eval(p, g) / max(len(g), 1)
            for p, g in zip(pred_str, label_str)
        ) / len(pred_str)

        acc = sum(p == g for p, g in zip(pred_str, label_str)) / len(pred_str)

        return {
            "cer": cer,
            "accuracy": acc
        }
    return compute_metrics