"""
SF-Owen Dataset Loader for Training & Fine-Tuning
Supports .epd text and .bin (69-byte SData binary) memory-mapped datasets.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from loader.features import extract_features, extract_features_from_board

SDATA_DTYPE_69 = np.dtype([
    ('board', np.uint8, (64,)),
    ('stm', np.uint8),
    ('eval', np.int16),
    ('result', np.uint8),
    ('ply', np.uint8)
])


class ChessNNUEDataset(Dataset):
    def __init__(self, fens_or_file, max_samples=None):
        self.is_bin = False
        self.file_path = None
        self.samples = []

        if isinstance(fens_or_file, list):
            self.samples = fens_or_file
        elif os.path.exists(fens_or_file):
            self.file_path = fens_or_file
            if fens_or_file.endswith(".bin"):
                self.is_bin = True
                file_size = os.path.getsize(fens_or_file)
                total_records = file_size // 69
                self.num_samples = total_records
                if max_samples and max_samples < self.num_samples:
                    self.num_samples = max_samples
                self.bin_data = np.memmap(fens_or_file, dtype=SDATA_DTYPE_69, mode='r')
                print(f"Loaded memory-mapped binary dataset: {self.num_samples:,} positions.")
            elif fens_or_file.endswith(".epd") or fens_or_file.endswith(".txt"):
                with open(fens_or_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(";")
                        fen = parts[0].strip()
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
                print(f"Loaded {len(self.samples):,} training positions from text.")

    def __len__(self):
        if self.is_bin:
            return self.num_samples
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_bin:
            row = self.bin_data[idx]
            board_bytes = row['board'].tobytes()
            stm = int(row['stm'])
            target_eval = float(row['eval'])
            feat = extract_features_from_board(board_bytes, stm)
            return {
                "target": torch.tensor(target_eval, dtype=torch.float32),
                "bucket": feat["bucket"],
                "stm": feat["side_to_move"],
                "halfka_us": torch.from_numpy(feat["halfka_us"]),
                "halfka_them": torch.from_numpy(feat["halfka_them"]),
                "threat_us": torch.from_numpy(feat["threat_us"]),
                "threat_them": torch.from_numpy(feat["threat_them"]),
                "pp_us": torch.from_numpy(feat["pp_us"]),
                "pp_them": torch.from_numpy(feat["pp_them"]),
            }

        fen, target_score = self.samples[idx]
        feat = extract_features(fen)
        return {
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
