"""Utilities for reproducibility, hardware profiling, and publication-ready plots."""

import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.config import (
    DEFAULT_SEED,
    TARGET_NAMES,
    FIGURES_DIR,
    get_device,
)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seed across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_dataset_distribution(
    train_ds,
    val_ds,
    target_names: List[str] = TARGET_NAMES,
    save_path: Optional[Path] = None,
) -> None:
    """Generate side-by-side countplots for train and validation class distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (ds, split_name) in enumerate([(train_ds, "Training Set"), (val_ds, "Validation Set")]):
        names = [target_names[idx] for idx in ds["label"]]
        df_plot = pd.DataFrame(names, columns=["Class"])

        sns.countplot(
            data=df_plot,
            x="Class",
            ax=axes[i],
            order=target_names,
            palette="viridis",
            hue="Class",
            legend=False,
        )
        axes[i].set_title(f"Class Balance: {split_name} (n={len(ds)})", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Taxonomic Macro-Class")
        axes[i].set_ylabel("Sample Count")

        for p in axes[i].patches:
            height = int(p.get_height())
            axes[i].annotate(
                f"{height}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="center",
                xytext=(0, 6),
                textcoords="offset points",
                fontweight="bold",
            )

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Dataset distribution plot saved to {save_path}")
    plt.close()


def plot_training_history(
    history: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
) -> None:
    """Generate dual training history plots: CrossEntropy Loss and Macro F1 progress."""
    df = pd.DataFrame(history)
    train_logs = df[df["loss"].notna()]
    eval_logs = df[df["eval_loss"].notna()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curves
    if "epoch" in train_logs.columns:
        ax1.plot(train_logs["epoch"], train_logs["loss"], label="Training Loss", color="royalblue", alpha=0.7)
    if "epoch" in eval_logs.columns and "eval_loss" in eval_logs.columns:
        ax1.plot(eval_logs["epoch"], eval_logs["eval_loss"], label="Validation Loss", color="crimson", marker="o")
    ax1.set_title("Cross-Entropy Loss Dynamics", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Macro F1 Progress
    if "epoch" in eval_logs.columns and "eval_f1" in eval_logs.columns:
        ax2.plot(eval_logs["epoch"], eval_logs["eval_f1"], label="Validation Macro F1", color="forestgreen", marker="s")
    ax2.set_title("Validation Macro F1 Progression", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Training curves saved to {save_path}")
    plt.close()
