"""Model initialization, selective layer unfreezing, and parameter inspection."""

from typing import Tuple, Dict
import torch
import torch.nn as nn
from transformers import ASTForAudioClassification

from src.config import (
    MODEL_CHECKPOINT,
    NUM_CLASSES,
    LABEL2ID,
    ID2LABEL,
    get_device,
)


def build_model(
    checkpoint: str = MODEL_CHECKPOINT,
    unfreeze_last_layer: bool = True,
    device: torch.device = None
) -> ASTForAudioClassification:
    """Build and configure the Audio Spectrogram Transformer for 4-class nature sound classification.
    
    Args:
        checkpoint: Pretrained Hugging Face checkpoint.
        unfreeze_last_layer: Whether to selectively unfreeze encoder layer 11 alongside head.
        device: Torch compute device.
        
    Returns:
        Configured ASTForAudioClassification instance.
    """
    if device is None:
        device = get_device()

    model = ASTForAudioClassification.from_pretrained(
        checkpoint,
        num_labels=NUM_CLASSES,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Selectively unfreeze classification head, normalization layers, and last transformer layer
    for name, param in model.named_parameters():
        if "classifier" in name or "layernorm" in name:
            param.requires_grad = True
        elif unfreeze_last_layer and "encoder.layer.11" in name:
            param.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculate total and trainable parameters in the model.
    
    Args:
        model: PyTorch model module.
        
    Returns:
        Dictionary with 'total', 'trainable', and 'frozen' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": round((trainable / total) * 100, 2) if total > 0 else 0.0,
    }


def print_model_summary(model: nn.Module) -> None:
    """Print formatted parameter summary and trainable submodules."""
    stats = count_parameters(model)
    print("\n" + "=" * 55)
    print("      AUDIO SPECTROGRAM TRANSFORMER (AST) SUMMARY")
    print("=" * 55)
    print(f"Total Parameters:      {stats['total']:,}")
    print(f"Trainable Parameters:  {stats['trainable']:,} ({stats['trainable_percent']}%)")
    print(f"Frozen Parameters:     {stats['frozen']:,}")
    print("-" * 55)
    print("Trainable Submodules:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  [+] {name} (shape: {list(param.shape)})")
    print("=" * 55 + "\n")
