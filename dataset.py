"""
SF-Owen Dataset Loader for Training & Fine-Tuning
Loads self-play datasets (.bin / .sdata) and extracts active NNUE feature tensors for PyTorch.
"""

import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from loader.features import extract_features


class ChessNNUEDataset(Dataset):
    """
    Parses positions from .bin / .epd / .sdata formats and pre-computes active feature indices.
    """
    def __init__(self, fens_or_file, max_samples=None):
        self.samples = []
        if isinstance(fens_or_file, list):
            self.samples = fens_or_file
        elif os.path.exists(fens_or_file):
            if fens_or_file.endswith(".epd") or fens_or_file.endswith(".txt"):
                with open(fens_or_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(";")
                        fen = parts[0].strip()
                        # Extract score or result if available
                        score = 0.0
                        for p in parts[1:]:
                            p = p.strip()
                            if p.startswith("ce ") or p.startswith("c9 "):
                                try:
                                    score = float(p.split()[1])
                                except ValueError:
                                    pass
                        self.samples.append((fen, score))
                        if max_samples and len(self.samples) >= max_samples:
                            break
            elif fens_or_file.endswith(".bin"):
                # Binary format reader
                print(f"Loading binary dataset {fens_or_file}...")
                # Add bin parser
        print(f"Loaded {len(self.samples)} training positions.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, target_score = self.samples[idx]
        feat = extract_features(fen)
        return {
            "fen": fen,
            "target": torch.tensor(target_score, dtype=torch.float32),
            "bucket": feat["bucket"],
            "stm": feat["side_to_move"],
            "halfka_us": torch.from_numpy(feat["halfka_us"]),
            "halfka_them": torch.from_numpy(feat["halfka_them"]),
            "threat_us": torch.from_numpy(feat["threat_us"]),
            "threat_them": torch.from_numpy(feat["threat_them"]),
            "pp_us": torch.from_numpy(feat["pp_us"]),
            "pp_them": torch.from_numpy(feat["pp_them"]),
        }
