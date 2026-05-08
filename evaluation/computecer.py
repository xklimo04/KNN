import editdistance
import torch

def normalize(s):
    return s.strip()

def compute_cer(pred_ids, label_ids, processor):
    """
    Compute the Character Error Rate (CER) between predicted and label sequences.

    Args:
        pred_ids (List[List[int]]): List of predicted token IDs.
        label_ids (List[List[int]]): List of label token IDs.
        processor: The processor object used for decoding.

    Returns:
        float: The computed Character Error Rate (CER).
    """
    total_distance = 0
    total_tokens = 0

    pad_id = processor.tokenizer.pad_token_id
    bos_id = processor.tokenizer.cls_token_id
    eos_id = processor.tokenizer.sep_token_id

    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.tolist()

    if isinstance(label_ids, torch.Tensor):
        label_ids = label_ids.tolist()

    for pred, label in zip(pred_ids, label_ids):

        label = [
            x for x in label
            if x not in (-100, pad_id, bos_id, eos_id)
        ]

        pred = [
            x for x in pred
            if x not in (pad_id, bos_id, eos_id)
        ]

        distance = editdistance.eval(pred, label)

        total_distance += distance
        total_tokens += len(label)

    if total_tokens == 0:
        return 0.0
    return total_distance / total_tokens