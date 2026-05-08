from transformers import TrainerCallback
import json
import os

class LoggerCallback(TrainerCallback):
    def __init__(self, output_path):
        self.output_path = output_path
        self.logs = []

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def save(self):
        with open(self.output_path, "w") as f:
            json.dump(self.logs, f, indent=4)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        entry = {
            "type": "train",
            "step": state.global_step,
            "epoch": state.epoch,
            **logs
        }

        self.logs.append(entry)
        self.save()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        entry = {
            "type": "eval",
            "step": state.global_step,
            "epoch": state.epoch,
            **metrics
        }

        self.logs.append(entry)
        self.save()