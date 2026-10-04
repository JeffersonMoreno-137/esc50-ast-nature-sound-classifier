"""Training engine with Cost-Sensitive Loss and Hugging Face Trainer orchestration."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, ASTForAudioClassification
from datasets import Dataset
import evaluate

from src.config import (
    CLASS_WEIGHTS,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    OUTPUT_DIR,
    get_device,
)


class RobustWeightedTrainer(Trainer):
    """Custom Hugging Face Trainer implementing cost-sensitive CrossEntropyLoss.
    
    Applies inverse-frequency class weights to counteract imbalance in minority
    acoustic classes (e.g., wind sounds).
    """

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
            self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_compute_metrics():
    """Build evaluation metric callback calculating Macro F1 and Accuracy."""
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        return {"f1": f1, "accuracy": acc}

    return compute_metrics


def train_model(
    model: ASTForAudioClassification,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str = str(OUTPUT_DIR),
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    device: Optional[torch.device] = None,
) -> RobustWeightedTrainer:
    """Execute end-to-end fine-tuning pipeline for Audio Spectrogram Transformer.
    
    Args:
        model: AST classification model.
        train_dataset: Preprocessed training dataset.
        eval_dataset: Preprocessed validation dataset.
        output_dir: Destination path for checkpoints.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Optimizer initial learning rate.
        weight_decay: L2 regularization strength for AdamW.
        device: Target compute device.
        
    Returns:
        Trained RobustWeightedTrainer instance with log history.
    """
    if device is None:
        device = get_device()

    weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        dataloader_pin_memory=False,  # Essential for Apple MPS stability
        dataloader_num_workers=0,     # Prevent multithreading deadlock on macOS
        fp16=False,                   # Full float32 precision for MPS compatibility
        report_to="none",
    )

    trainer = RobustWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics(),
        class_weights=weights_tensor,
    )

    print(f"Initiating training on {device} ({epochs} epochs, batch size {batch_size}, lr {learning_rate})...")
    trainer.train()

    # Save best checkpoint explicitly
    final_model_path = Path(output_dir) / "best_model"
    trainer.save_model(str(final_model_path))
    print(f"Best model checkpoint successfully saved to {final_model_path}")

    return trainer
