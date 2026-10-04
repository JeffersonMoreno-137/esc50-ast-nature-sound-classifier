"""Evaluation routines, confusion matrix visualization, and single-audio inference."""

from pathlib import Path
from typing import Dict, Any, List, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio
from sklearn.metrics import classification_report, confusion_matrix
from transformers import Trainer, ASTForAudioClassification, ASTFeatureExtractor
from datasets import Dataset

from src.config import (
    TARGET_NAMES,
    TARGET_SAMPLE_RATE,
    FIGURES_DIR,
    get_device,
)
from src.data import get_feature_extractor


def evaluate_dataset(
    trainer_or_model: Union[Trainer, ASTForAudioClassification],
    eval_dataset: Dataset,
    target_names: List[str] = TARGET_NAMES,
    save_confusion_matrix_path: Union[str, Path] = None,
) -> Dict[str, Any]:
    """Run model inference on evaluation dataset and compute classification metrics.
    
    Args:
        trainer_or_model: Hugging Face Trainer instance or AST model.
        eval_dataset: Tokenized/spectrogram evaluation dataset.
        target_names: List of class names.
        save_confusion_matrix_path: Optional path to save confusion matrix image.
        
    Returns:
        Dictionary containing predictions, targets, report string, and metrics dict.
    """
    if isinstance(trainer_or_model, Trainer):
        pred_output = trainer_or_model.predict(eval_dataset)
        logits = pred_output.predictions
        y_true = pred_output.label_ids
    else:
        device = next(trainer_or_model.parameters()).device
        trainer_or_model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for item in eval_dataset:
                inputs = torch.tensor(item["input_values"]).unsqueeze(0).to(device)
                out = trainer_or_model(input_values=inputs)
                all_logits.append(out.logits.cpu().numpy())
                all_labels.append(item["label"])
        logits = np.concatenate(all_logits, axis=0)
        y_true = np.array(all_labels)

    y_pred = np.argmax(logits, axis=-1)

    # Classification report
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, zero_division=0
    )

    print("\n" + "=" * 55)
    print("           EVALUATION CLASSIFICATION REPORT")
    print("=" * 55)
    print(report_text)
    print("=" * 55)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if save_confusion_matrix_path:
        save_path = Path(save_confusion_matrix_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
        )
        plt.title("Confusion Matrix: Nature Sound Classification (AST)", fontsize=12, fontweight="bold")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "report_dict": report_dict,
        "report_text": report_text,
        "confusion_matrix": cm,
    }


def predict_audio_file(
    model: ASTForAudioClassification,
    audio_path: Union[str, Path],
    feature_extractor: ASTFeatureExtractor = None,
    target_names: List[str] = TARGET_NAMES,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Perform single-file acoustic inference from a waveform audio file.
    
    Args:
        model: Trained AST model.
        audio_path: Path to target audio file (.wav, .mp3, .flac).
        feature_extractor: Preconfigured ASTFeatureExtractor.
        target_names: Class labels list.
        device: Target compute device.
        
    Returns:
        Dictionary containing predicted class, confidence, and all class probabilities.
    """
    if device is None:
        device = get_device()
    if feature_extractor is None:
        feature_extractor = get_feature_extractor()

    model = model.to(device)
    model.eval()

    # Load audio waveform
    waveform, sample_rate = torchaudio.load(str(audio_path))

    # Convert to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16,000 Hz if necessary
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
        waveform = resampler(waveform)

    audio_array = waveform.squeeze().numpy()

    # Extract Mel-spectrogram
    inputs = feature_extractor(
        audio_array,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )

    input_values = inputs["input_values"].to(device)
    if input_values.ndim == 4:
        input_values = input_values.squeeze(1)

    with torch.no_grad():
        outputs = model(input_values=input_values)
        probabilities = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

    pred_idx = int(np.argmax(probabilities))
    pred_label = target_names[pred_idx]
    confidence = float(probabilities[pred_idx])

    all_probs = {target_names[i]: float(probabilities[i]) for i in range(len(target_names))}

    return {
        "predicted_class": pred_label,
        "confidence": confidence,
        "probabilities": all_probs,
    }
