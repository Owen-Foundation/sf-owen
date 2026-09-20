"""
PyTorch SF-NNUE Architecture Implementation for SF-Owen.
Matches official Stockfish NNUE architecture with 8 layer stacks,
SqrClippedReLU + ClippedReLU activations, and PSQT + Positional outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LayerStack(nn.Module):
    def __init__(self, stack_idx: int):
        super().__init__()
        self.stack_idx = stack_idx
        # fc0: 1024 -> 32
        self.fc0 = nn.Linear(1024, 32, bias=True)
        # fc1: 64 -> 32
        self.fc1 = nn.Linear(64, 32, bias=True)
        # fc2: 128 -> 1
        self.fc2 = nn.Linear(128, 1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, 1024] in range [0, 127] (or float equivalent)
        """
        # fc_0
        out_fc0 = self.fc0(x)  # [batch, 32]
        
        # ac_0 & ac_sqr_0 (WeightScaleBitsLocal = 7)
        # Sqr: clamp((x^2) >> 21, 0, 127) -> in float: clamp(x^2 / (2^21), 0, 127)
        # Clip: clamp(x >> 7, 0, 127) -> in float: clamp(x / (2^7), 0, 127)
        # In quantized integer arithmetic:
        sqr0 = torch.clamp((out_fc0 ** 2) / (1 << 21), 0.0, 127.0)
        clip0 = torch.clamp(out_fc0 / (1 << 7), 0.0, 127.0)
        concat0 = torch.cat([sqr0, clip0], dim=-1)  # [batch, 64]
        
        # fc_1
        out_fc1 = self.fc1(concat0)  # [batch, 32]
        
        # ac_1 & ac_sqr_1 (WeightScaleBitsLocal = 6)
        sqr1 = torch.clamp((out_fc1 ** 2) / (1 << 19), 0.0, 127.0)
        clip1 = torch.clamp(out_fc1 / (1 << 6), 0.0, 127.0)
        
        # Concat all activations into 128 inputs for fc_2
        concat_all = torch.cat([concat0, sqr1, clip1], dim=-1)  # [batch, 128]
        
        out_fc2 = self.fc2(concat_all)  # [batch, 1]
        
        # Skip connection from fc0
        skip = (out_fc0[:, 30:31] - out_fc0[:, 31:32])
        fwd_out = out_fc2 + skip
        
        # Positional scaling: (fwd_out * 600 * 16) / (128 * 64 * 2) = fwd_out * 9600 / 16384
        positional = (fwd_out * 9600.0) / 16384.0
        return positional


class StockfishNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature Transformer Weights
        self.ft_biases = nn.Parameter(torch.zeros(1024), requires_grad=True)
        self.threat_weights = nn.Parameter(torch.zeros(59808, 1024), requires_grad=True)
        self.threat_psqt = nn.Parameter(torch.zeros(59808, 8), requires_grad=True)
        
        self.pp_weights = nn.Parameter(torch.zeros(4560, 1024), requires_grad=True)
        self.pp_psqt = nn.Parameter(torch.zeros(4560, 8), requires_grad=True)
        
        self.halfka_weights = nn.Parameter(torch.zeros(22528, 1024), requires_grad=True)
        self.halfka_psqt = nn.Parameter(torch.zeros(22528, 8), requires_grad=True)
        
        # 8 Stacks
        self.stacks = nn.ModuleList([LayerStack(i) for i in range(8)])

    def load_from_dict(self, weights: dict):
        with torch.no_grad():
            self.ft_biases.copy_(torch.from_numpy(weights["ft_biases"]).float())
            self.threat_weights.copy_(torch.from_numpy(weights["threat_weights"]).float())
            self.threat_psqt.copy_(torch.from_numpy(weights["threat_psqt"]).float())
            
            self.pp_weights.copy_(torch.from_numpy(weights["pp_weights"]).float())
            self.pp_psqt.copy_(torch.from_numpy(weights["pp_psqt"]).float())
            
            self.halfka_weights.copy_(torch.from_numpy(weights["halfka_weights"]).float())
            self.halfka_psqt.copy_(torch.from_numpy(weights["halfka_psqt"]).float())
            
            for i, stack_dict in enumerate(weights["layer_stacks"]):
                stack = self.stacks[i]
                stack.fc0.weight.copy_(torch.from_numpy(stack_dict["fc0_weight"]).float())
                stack.fc0.bias.copy_(torch.from_numpy(stack_dict["fc0_bias"]).float())
                
                stack.fc1.weight.copy_(torch.from_numpy(stack_dict["fc1_weight"]).float())
                stack.fc1.bias.copy_(torch.from_numpy(stack_dict["fc1_bias"]).float())
                
                stack.fc2.weight.copy_(torch.from_numpy(stack_dict["fc2_weight"]).float())
                stack.fc2.bias.copy_(torch.from_numpy(stack_dict["fc2_bias"]).float())

    def forward_from_features(self, us_halfka, them_halfka, us_threat, them_threat, us_pp, them_pp, bucket: int, us_is_white: bool = True):
        """
        Runs full forward pass given sparse feature indices.
        """
        # 1. Accumulate HalfKA
        acc_us = self.ft_biases.clone()
        acc_them = self.ft_biases.clone()
        
        psqt_us = torch.zeros(8, device=self.ft_biases.device)
        psqt_them = torch.zeros(8, device=self.ft_biases.device)
        
        if len(us_halfka) > 0:
            acc_us = acc_us + self.halfka_weights[us_halfka].sum(dim=0)
            psqt_us = psqt_us + self.halfka_psqt[us_halfka].sum(dim=0)
        if len(them_halfka) > 0:
            acc_them = acc_them + self.halfka_weights[them_halfka].sum(dim=0)
            psqt_them = psqt_them + self.halfka_psqt[them_halfka].sum(dim=0)
            
        if len(us_threat) > 0:
            acc_us = acc_us + self.threat_weights[us_threat].sum(dim=0)
            psqt_us = psqt_us + self.threat_psqt[us_threat].sum(dim=0)
        if len(them_threat) > 0:
            acc_them = acc_them + self.threat_weights[them_threat].sum(dim=0)
            psqt_them = psqt_them + self.threat_psqt[them_threat].sum(dim=0)
            
        if len(us_pp) > 0:
            us_pp_idx = us_pp - 59808
            acc_us = acc_us + self.pp_weights[us_pp_idx].sum(dim=0)
            psqt_us = psqt_us + self.pp_psqt[us_pp_idx].sum(dim=0)
        if len(them_pp) > 0:
            them_pp_idx = them_pp - 59808
            acc_them = acc_them + self.pp_weights[them_pp_idx].sum(dim=0)
            psqt_them = psqt_them + self.pp_psqt[them_pp_idx].sum(dim=0)
            
        # Transform perspective (pairwise multiplication)
        # us
        us0 = torch.clamp(acc_us[:512], 0.0, 255.0)
        us1 = torch.clamp(acc_us[512:], 0.0, 255.0)
        out_us = torch.clamp((us0 * us1) / 512.0, 0.0, 127.0)
        
        # them
        them0 = torch.clamp(acc_them[:512], 0.0, 255.0)
        them1 = torch.clamp(acc_them[512:], 0.0, 255.0)
        out_them = torch.clamp((them0 * them1) / 512.0, 0.0, 127.0)
        
        # Combined 1024 feature vector
        features = torch.cat([out_us, out_them]).unsqueeze(0)
        
        # Positional propagation through bucketed stack
        positional = self.stacks[bucket](features)
        
        # PSQT
        psqt_score = (psqt_us[bucket] - psqt_them[bucket]) / 2.0
        
        # Total score in centipawns / internal units
        total_eval = (psqt_score + positional.squeeze()) / 16.0
        return total_eval
