#!/usr/bin/env python3
"""Unified CLI entrypoint for Audio Spectrogram Transformer (AST) Nature Sound Classification.

Usage:
    # Full fine-tuning and evaluation:
    python main.py --mode train --epochs 5 --batch-size 4 --lr 2e-5

    # Evaluation on test set:
    python main.py --mode evaluate --checkpoint ./results_ast_nature/best_model

    # Single-sample inference on audio file:
    python main.py --mode predict --audio sample.wav --checkpoint ./results_ast_nature/best_model
"""

import argparse
import sys
from pathlib import Path

from src.config import (
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    MODEL_CHECKPOINT,
    OUTPUT_DIR,
    TARGET_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio Spectrogram Transformer (AST) for Environmental Nature Sound Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "predict"],
        help="Pipeline execution mode.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Per-device batch size for training and evaluation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Initial learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint or fine-tuned directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save model checkpoints and training logs.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file for prediction (required in predict mode).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from src.config import (
        DEFAULT_EPOCHS,
        DEFAULT_BATCH_SIZE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_SEED,
        MODEL_CHECKPOINT,
        OUTPUT_DIR,
        FIGURES_DIR,
        TARGET_NAMES,
        get_device,
    )
    from src.utils import set_seed, plot_dataset_distribution, plot_training_history
    from src.data import (
        load_and_filter_datasets,
        check_dataset_integrity,
        prepare_encoded_datasets,
        get_feature_extractor,
    )
    from src.model import build_model, print_model_summary
    from src.train import train_model
    from src.evaluate import evaluate_dataset, predict_audio_file

    set_seed(args.seed)
    device = get_device()
    print(f"\n[Environment] Device: {device} | Seed: {args.seed} | Mode: {args.mode.upper()}")

    checkpoint_to_load = args.checkpoint or MODEL_CHECKPOINT

    if args.mode == "train":
        print("\n--- 1. Data Ingestion and Integrity Check ---")
        train_raw, val_raw = load_and_filter_datasets()
        check_dataset_integrity(train_raw, "Train")
        check_dataset_integrity(val_raw, "Validation")

        # Save class distribution figure
        dist_plot_path = FIGURES_DIR / "01_dataset_distribution.png"
        plot_dataset_distribution(train_raw, val_raw, TARGET_NAMES, save_path=dist_plot_path)

        print("\n--- 2. Feature Extraction (Mel-Spectrograms) ---")
        feature_extractor = get_feature_extractor()
        encoded_train, encoded_val = prepare_encoded_datasets(train_raw, val_raw, feature_extractor)

        print("\n--- 3. Model Architecture Construction ---")
        model = build_model(checkpoint_to_load, unfreeze_last_layer=True, device=device)
        print_model_summary(model)

        print("\n--- 4. Cost-Sensitive Training Engine ---")
        trainer = train_model(
            model=model,
            train_dataset=encoded_train,
            eval_dataset=encoded_val,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
        )

        # Plot training curves
        curves_path = FIGURES_DIR / "03_training_validation_curves.png"
        plot_training_history(trainer.state.log_history, save_path=curves_path)

        print("\n--- 5. Validation Set Evaluation ---")
        cm_path = FIGURES_DIR / "04_confusion_matrix.png"
        evaluate_dataset(trainer, encoded_val, TARGET_NAMES, save_confusion_matrix_path=cm_path)

    elif args.mode == "evaluate":
        print("\n--- 1. Loading Evaluation Data ---")
        _, val_raw = load_and_filter_datasets()
        feature_extractor = get_feature_extractor()
        _, encoded_val = prepare_encoded_datasets(val_raw, val_raw, feature_extractor)

        print(f"\n--- 2. Loading Checkpoint: {checkpoint_to_load} ---")
        model = build_model(checkpoint_to_load, unfreeze_last_layer=False, device=device)

        print("\n--- 3. Running Evaluation ---")
        cm_path = FIGURES_DIR / "04_confusion_matrix.png"
        evaluate_dataset(model, encoded_val, TARGET_NAMES, save_confusion_matrix_path=cm_path)

    elif args.mode == "predict":
        if not args.audio:
            print("Error: --audio argument is required in 'predict' mode.", file=sys.stderr)
            sys.exit(1)

        audio_file = Path(args.audio)
        if not audio_file.exists():
            print(f"Error: Audio file '{audio_file}' not found.", file=sys.stderr)
            sys.exit(1)

        print(f"\n--- Loading Checkpoint: {checkpoint_to_load} ---")
        model = build_model(checkpoint_to_load, unfreeze_last_layer=False, device=device)
        feature_extractor = get_feature_extractor()

        print(f"--- Running Inference on '{audio_file}' ---")
        result = predict_audio_file(
            model=model,
            audio_path=audio_file,
            feature_extractor=feature_extractor,
            target_names=TARGET_NAMES,
            device=device,
        )

        print("\n" + "=" * 45)
        print("          ACOUSTIC PREDICTION")
        print("=" * 45)
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence:      {result['confidence']:.4f}")
        print("-" * 45)
        print("Class Probabilities:")
        for cls_name, prob in result["probabilities"].items():
            print(f"  {cls_name:<15}: {prob:.4f}")
        print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
