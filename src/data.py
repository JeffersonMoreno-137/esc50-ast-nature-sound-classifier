"""Data ingestion, deduplication, taxonomic filtering, and feature extraction for ESC-50."""

import hashlib
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import ASTFeatureExtractor

from src.config import (
    DATASET_NAME,
    DATASET_FOLD,
    TARGET_SAMPLE_RATE,
    NUM_MEL_BINS,
    MAX_SPECTROGRAM_LENGTH,
    NORMALIZE_AUDIO,
    NORM_MEAN,
    NORM_STD,
    MODEL_CHECKPOINT,
    NATURE_MAP,
    TARGET_NAMES,
)


def get_feature_extractor() -> ASTFeatureExtractor:
    """Instantiate and return the Audio Spectrogram Transformer feature extractor."""
    return ASTFeatureExtractor.from_pretrained(
        MODEL_CHECKPOINT,
        sampling_rate=TARGET_SAMPLE_RATE,
        num_mel_bins=NUM_MEL_BINS,
        max_length=MAX_SPECTROGRAM_LENGTH,
        do_normalize=NORMALIZE_AUDIO,
        mean=NORM_MEAN,
        std=NORM_STD,
    )


def check_dataset_integrity(ds: Dataset, split_name: str = "Dataset") -> Dataset:
    """Verify dataset metadata integrity and check duplicate audio waveforms via SHA-256 hashing.
    
    Args:
        ds: Hugging Face Dataset instance.
        split_name: Name of the partition (e.g., 'Train', 'Validation').
        
    Returns:
        Dataset augmented with 'audio_hash' column.
    """
    print(f"\n[Integrity Check] Partition: {split_name}")
    meta_df = pd.DataFrame(ds.remove_columns(["audio"]))
    null_counts = meta_df.isnull().sum().sum()
    print(f"  Null metadata values: {null_counts}")

    def generate_audio_hash(example: Dict[str, Any]) -> Dict[str, str]:
        audio_bytes = example["audio"]["array"].tobytes()
        return {"audio_hash": hashlib.sha256(audio_bytes).hexdigest()}

    ds_with_hashes = ds.map(generate_audio_hash, batched=False)
    hash_list = ds_with_hashes["audio_hash"]
    duplicates = len(hash_list) - len(set(hash_list))
    print(f"  Duplicate audio waveforms detected: {duplicates}")

    return ds_with_hashes


def load_and_filter_datasets() -> Tuple[Dataset, Dataset]:
    """Load ESC-50 fold 1 splits, filter to nature classes, and assign dense labels.
    
    Returns:
        Tuple of (filtered_train, filtered_val) datasets.
    """
    print(f"Loading '{DATASET_NAME}' ({DATASET_FOLD}) from Hugging Face Hub...")
    raw_train = load_dataset(DATASET_NAME, DATASET_FOLD, split="train")
    raw_val = load_dataset(DATASET_NAME, DATASET_FOLD, split="test")

    # Filter only relevant nature categories
    def filter_nature(example: Dict[str, Any]) -> bool:
        return example["sound"] in NATURE_MAP

    train_filtered = raw_train.filter(filter_nature)
    val_filtered = raw_val.filter(filter_nature)

    # Remap label column to dense 0..3 targets
    train_mapped = train_filtered.map(lambda x: {"label": NATURE_MAP[x["sound"]]})
    val_mapped = val_filtered.map(lambda x: {"label": NATURE_MAP[x["sound"]]})

    print(f"Filtered Nature Subsets -> Train: {len(train_mapped)}, Val: {len(val_mapped)}")
    return train_mapped, val_mapped


def prepare_encoded_datasets(
    train_ds: Dataset,
    val_ds: Dataset,
    feature_extractor: ASTFeatureExtractor = None
) -> Tuple[Dataset, Dataset]:
    """Transform raw audio waveforms into AST-compliant Mel-spectrogram tensors.
    
    Args:
        train_ds: Filtered training dataset.
        val_ds: Filtered validation dataset.
        feature_extractor: ASTFeatureExtractor instance (optional).
        
    Returns:
        Tuple of (encoded_train, encoded_val) with shape (batch, 1024, 128).
    """
    if feature_extractor is None:
        feature_extractor = get_feature_extractor()

    def preprocess_audio(examples: Dict[str, Any]) -> Dict[str, Any]:
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        if inputs["input_values"].ndim == 4:
            inputs["input_values"] = inputs["input_values"].squeeze(1)
        return inputs

    encoded_train = train_ds.map(
        preprocess_audio,
        batched=True,
        batch_size=10,
        remove_columns=["audio"]
    )
    encoded_val = val_ds.map(
        preprocess_audio,
        batched=True,
        batch_size=10,
        remove_columns=["audio"]
    )

    return encoded_train, encoded_val
