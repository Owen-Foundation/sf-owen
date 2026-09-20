"""
SF-Owen Fine-Tuning & Knowledge Distillation Trainer.
Fine-tunes Stockfish NNUE networks on self-play / distillation datasets in PyTorch
and exports deployment-ready .nnue network binaries.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loader.nnue_loader import load_sf_nnue, save_sf_nnue
from loader.sf_model import StockfishNNUE
from dataset import ChessNNUEDataset


def collate_fn(batch):
    return batch


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Base Net
    print(f"Loading Base NNUE Net: {args.base_net}...")
    weights_dict = load_sf_nnue(args.base_net)

    model = StockfishNNUE().to(device)
    model.load_from_dict(weights_dict)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Setup Dataset
    print(f"Loading dataset: {args.data}...")
    dataset = ChessNNUEDataset(args.data, max_samples=args.max_samples)
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers
    )

    # 3. Optimizer & Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.SmoothL1Loss()

    # 4. Training Loop
    print(f"\nStarting Fine-Tuning for {args.epochs} Epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()
            batch_loss = 0.0

            for sample in batch:
                pred = model.forward_from_features(
                    sample["halfka_us"].to(device),
                    sample["halfka_them"].to(device),
                    sample["threat_us"].to(device),
                    sample["threat_them"].to(device),
                    sample["pp_us"].to(device),
                    sample["pp_them"].to(device),
                    sample["bucket"],
                    us_is_white=(sample["stm"] == 0)
                )
                target = sample["target"].to(device)
                loss = criterion(pred, target)
                batch_loss += loss

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Weight clamping to maintain integer quantization bounds
            with torch.no_grad():
                model.threat_weights.clamp_(-128, 127)
                model.pp_weights.clamp_(-128, 127)
                model.halfka_weights.clamp_(-32768, 32767)
                for stack in model.stacks:
                    stack.fc0.weight.clamp_(-128, 127)
                    stack.fc1.weight.clamp_(-128, 127)
                    stack.fc2.weight.clamp_(-128, 127)

            total_loss += batch_loss.item()
            num_batches += 1

            if (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == len(loader):
                avg_loss = total_loss / num_batches
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx+1}/{len(loader)}] Loss: {avg_loss:.4f} ({time.time()-t0:.1f}s)")

        scheduler.step()

    # 5. Export Fine-Tuned Net
    print(f"\nExporting Fine-Tuned Network to {args.output_net}...")
    # Update weights dict from PyTorch parameters
    with torch.no_grad():
        weights_dict["ft_biases"] = model.ft_biases.cpu().numpy().astype(np.int16)
        weights_dict["threat_weights"] = model.threat_weights.cpu().numpy().astype(np.int8)
        weights_dict["threat_psqt"] = model.threat_psqt.cpu().numpy().astype(np.int32)
        weights_dict["pp_weights"] = model.pp_weights.cpu().numpy().astype(np.int8)
        weights_dict["pp_psqt"] = model.pp_psqt.cpu().numpy().astype(np.int32)
        weights_dict["halfka_weights"] = model.halfka_weights.cpu().numpy().astype(np.int16)
        weights_dict["halfka_psqt"] = model.halfka_psqt.cpu().numpy().astype(np.int32)

        for i, stack in enumerate(model.stacks):
            weights_dict["layer_stacks"][i]["fc0_bias"] = stack.fc0.bias.cpu().numpy().astype(np.int32)
            weights_dict["layer_stacks"][i]["fc0_weight"] = stack.fc0.weight.cpu().numpy().astype(np.int8)
            weights_dict["layer_stacks"][i]["fc1_bias"] = stack.fc1.bias.cpu().numpy().astype(np.int32)
            weights_dict["layer_stacks"][i]["fc1_weight"] = stack.fc1.weight.cpu().numpy().astype(np.int8)
            weights_dict["layer_stacks"][i]["fc2_bias"] = stack.fc2.bias.cpu().numpy().astype(np.int32)
            weights_dict["layer_stacks"][i]["fc2_weight"] = stack.fc2.weight.cpu().numpy().astype(np.int8)

    save_sf_nnue(weights_dict, args.output_net, description=f"SF-Owen fine-tuned from {os.path.basename(args.base_net)}")
    print(f"Successfully saved {args.output_net} ({os.path.getsize(args.output_net):,} bytes)!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SF-Owen Fine-Tuning & Distillation Trainer")
    parser.add_argument("--base-net", type=str, default="/home/hemesh/sf-nets/nn-134a887f4c8f.nnue", help="Path to base .nnue")
    parser.add_argument("--data", type=str, default="/home/hemesh/Videos/Owen/tools/book-200.epd", help="Training dataset")
    parser.add_argument("--output-net", type=str, default="/home/hemesh/Videos/sf-owen/sf-owen-v1.nnue", help="Output .nnue path")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")

    args = parser.parse_args()
    train(args)
