"""
CUDAForge Post-Training Quantization (PTQ) Toolkit

Provides calibration (MinMax, KL-Divergence, SmoothQuant) and packing utilities
for FP16/BF16 to INT8/INT4 weight and activation quantization targeted at custom
fused CUDA/Triton inference kernels.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class QuantMode(str, Enum):
    INT8_PER_TENSOR = "int8_per_tensor"
    INT8_PER_CHANNEL = "int8_per_channel"
    INT4_PER_CHANNEL = "int4_per_channel"
    SMOOTH_QUANT_INT8 = "smooth_quant_int8"


@dataclass
class QuantizationScale:
    scale: torch.Tensor
    zero_point: Optional[torch.Tensor] = None
    smooth_scale: Optional[torch.Tensor] = None


@dataclass
class NumericalDriftReport:
    mse: float
    rmse: float
    mae: float
    cosine_similarity: float
    snr_db: float
    max_absolute_error: float


class QuantizationCalibrator:
    """
    Post-Training Quantization (PTQ) Calibrator supporting MinMax,
    KL-Divergence histogram matching, and SmoothQuant scale computation.
    """

    def __init__(self, mode: QuantMode = QuantMode.INT8_PER_CHANNEL):
        self.mode = mode

    @staticmethod
    def compute_minmax_scale(
        tensor: torch.Tensor,
        num_bits: int = 8,
        axis: Optional[int] = 0,
        symmetric: bool = True,
    ) -> QuantizationScale:
        """Computes per-tensor or per-channel scales using absolute maximum thresholds."""
        qmin = -(2 ** (num_bits - 1)) if symmetric else 0
        qmax = (2 ** (num_bits - 1)) - 1 if symmetric else (2**num_bits) - 1

        if axis is None:
            max_val = torch.max(torch.abs(tensor))
        else:
            # keepdim=True preserves tensor rank for seamless broadcasting
            max_val = torch.max(torch.abs(tensor), dim=axis, keepdim=True)[0]

        # Prevent division by zero
        scale = max_val / float(qmax)
        scale = torch.clamp(scale, min=1e-8)

        return QuantizationScale(scale=scale)

    @staticmethod
    def compute_kl_divergence_scale(
        activation_histogram: torch.Tensor,
        bin_edges: torch.Tensor,
        num_bits: int = 8,
        target_bins: int = 128,
    ) -> float:
        """
        Computes optimal quantization threshold by minimizing KL divergence
        between FP16 histogram and quantized/dequantized INT8 distribution.
        """
        histogram = activation_histogram.float().cpu().numpy()
        edges = bin_edges.cpu().numpy()
        
        # Zero out zero-bin
        histogram[0] = 0
        total_data = histogram.sum()
        
        if total_data == 0:
            return float(edges[-1])

        best_kl = float("inf")
        best_threshold = edges[-1]

        # Scan potential saturation thresholds
        for i in range(target_bins, len(histogram)):
            reference_dist = histogram[:i].copy()
            outliers = histogram[i:].sum()
            reference_dist[-1] += outliers

            # Normalize reference distribution
            p = reference_dist / reference_dist.sum()
            p = np.clip(p, 1e-12, 1.0)

            # Quantize reference into target_bins
            quantized_bins = np.zeros(target_bins, dtype=np.float64)
            num_merged_bins = i / target_bins

            for j in range(target_bins):
                start = j * num_merged_bins
                end = (j + 1) * num_merged_bins
                
                start_idx = int(np.floor(start))
                end_idx = int(np.ceil(end))
                
                for k in range(start_idx, min(end_idx, i)):
                    weight = 1.0
                    if k == start_idx:
                        weight -= (start - start_idx)
                    if k == end_idx - 1:
                        weight -= (end_idx - end)
                    quantized_bins[j] += reference_dist[k] * weight

            # Expand quantized distribution back to size i
            q = np.zeros(i, dtype=np.float64)
            for j in range(target_bins):
                start = j * num_merged_bins
                end = (j + 1) * num_merged_bins
                count = 0.0
                
                start_idx = int(np.floor(start))
                end_idx = int(np.ceil(end))
                
                for k in range(start_idx, min(end_idx, i)):
                    if reference_dist[k] > 0:
                        count += 1.0

                if count > 0:
                    normalized_val = quantized_bins[j] / count
                    for k in range(start_idx, min(end_idx, i)):
                        if reference_dist[k] > 0:
                            q[k] = normalized_val

            q = np.clip(q, 1e-12, 1.0)
            q = q / q.sum()

            # Compute KL Divergence: sum(P * log(P / Q))
            kl_div = np.sum(p * np.log(p / q))
            if kl_div < best_kl:
                best_kl = kl_div
                best_threshold = (edges[i] + edges[i + 1]) / 2.0

        return float(best_threshold)

    @staticmethod
    def compute_smoothquant_scales(
        activations_abs_max: torch.Tensor,
        weights_abs_max: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Computes SmoothQuant migration scales s = (max(|X|)^alpha) / (max(|W|)^(1-alpha))
        to balance activation outlier difficulty into weights before INT8 quantization.
        """
        act_pow = torch.pow(activations_abs_max, alpha)
        weight_pow = torch.pow(weights_abs_max, 1.0 - alpha)
        scales = act_pow / torch.clamp(weight_pow, min=1e-8)
        return torch.clamp(scales, min=1e-5)


class Quantizer:
    """
    Quantization conversion and bit-packing engine for INT8 and INT4 targets.
    """

    @staticmethod
    def quantize_int8(
        tensor: torch.Tensor, scale: torch.Tensor, symmetric: bool = True
    ) -> torch.Tensor:
        """Quantizes float tensor to int8 tensor given precomputed scale."""
        scaled = tensor / scale
        if symmetric:
            qtensor = torch.clamp(torch.round(scaled), -128, 127).to(torch.int8)
        else:
            qtensor = torch.clamp(torch.round(scaled), 0, 255).to(torch.uint8)
        return qtensor

    @staticmethod
    def dequantize_int8(
        qtensor: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Dequantizes int8 tensor back to floating point."""
        return qtensor.to(torch.float32) * scale

    @staticmethod
    def pack_int4_gemm(
        weight_tensor: torch.Tensor, scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes FP16 weight matrix into packed 4-bit unsigned integers (2x INT4 per uint8 element)
        for custom Ampere INT4 GEMM kernels.
        
        Input: Weight tensor [M, N] in float16/float32.
        Output: Packed uint8 tensor [M, N // 2] and FP16 scale vector [M, 1].
        """
        assert weight_tensor.dim() == 2, "INT4 packing expects a 2D matrix [M, N]."
        M, N = weight_tensor.shape
        assert N % 2 == 0, "Column dimension N must be even for byte packing."

        # Quantize to signed range [-8, 7] and map to unsigned [0, 15]
        q_weight = torch.clamp(torch.round(weight_tensor / scale), -8, 7).to(torch.int8)
        q_u4 = (q_weight + 8).to(torch.uint8)  # Range 0..15

        # Split into low nibbles and high nibbles
        low_nibble = q_u4[:, 0::2] & 0x0F
        high_nibble = (q_u4[:, 1::2] & 0x0F) << 4

        # Pack into uint8 bytes
        packed_bytes = low_nibble | high_nibble
        return packed_bytes.contiguous(), scale.to(torch.float16).contiguous()

    @staticmethod
    def unpack_int4_gemm(
        packed_bytes: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Unpacks 4-bit packed uint8 tensor back to float32 representation."""
        M, N_packed = packed_bytes.shape
        N = N_packed * 2

        low_nibbles = (packed_bytes & 0x0F).to(torch.int8) - 8
        high_nibbles = ((packed_bytes >> 4) & 0x0F).to(torch.int8) - 8

        unpacked = torch.empty((M, N), dtype=torch.float32, device=packed_bytes.device)
        unpacked[:, 0::2] = low_nibbles.to(torch.float32)
        unpacked[:, 1::2] = high_nibbles.to(torch.float32)

        return unpacked * scale


def evaluate_numerical_drift(
    baseline: torch.Tensor, quantized_dequantized: torch.Tensor
) -> NumericalDriftReport:
    """
    Quantifies numerical accuracy loss between float baseline and quantized-dequantized tensors.
    """
    orig = baseline.detach().to(torch.float64)
    dequant = quantized_dequantized.detach().to(torch.float64)

    diff = orig - dequant
    mse = float(torch.mean(diff ** 2).item())
    rmse = math.sqrt(mse)
    mae = float(torch.mean(torch.abs(diff)).item())
    max_err = float(torch.max(torch.abs(diff)).item())

    # Cosine Similarity
    orig_flat = orig.reshape(-1)
    dequant_flat = dequant.reshape(-1)
    cos_sim = float(
        torch.dot(orig_flat, dequant_flat)
        / (torch.norm(orig_flat) * torch.norm(dequant_flat) + 1e-12)
    )

    # Signal-to-Noise Ratio (SNR) in dB
    signal_power = torch.mean(orig ** 2)
    noise_power = torch.mean(diff ** 2)
    snr = 10.0 * math.log10((signal_power / (noise_power + 1e-12)).item() + 1e-12)

    return NumericalDriftReport(
        mse=mse,
        rmse=rmse,
        mae=mae,
        cosine_similarity=cos_sim,
        snr_db=snr,
        max_absolute_error=max_err,
    )