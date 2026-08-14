"""
Unit tests for cudaforge.quantization (PTQ Calibration & INT4 Packing)
"""

import math
import pytest
import torch
import numpy as np

from cudaforge.quantization import (
    QuantMode,
    QuantizationCalibrator,
    Quantizer,
    evaluate_numerical_drift,
)


def test_minmax_calibration_per_tensor():
    tensor = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=torch.float32)
    qscale = QuantizationCalibrator.compute_minmax_scale(
        tensor, num_bits=8, axis=None, symmetric=True
    )
    # qmax for signed 8-bit is 127. scale = 4.0 / 127
    assert pytest.approx(qscale.scale.item(), abs=1e-4) == (4.0 / 127.0)


def test_int8_quantize_dequantize_roundtrip():
    torch.manual_seed(42)
    x = torch.randn(128, 256, dtype=torch.float32)
    qscale = QuantizationCalibrator.compute_minmax_scale(x, num_bits=8, axis=-1)

    qx = Quantizer.quantize_int8(x, qscale.scale, symmetric=True)
    assert qx.dtype == torch.int8
    assert qx.shape == (128, 256)

    dqx = Quantizer.dequantize_int8(qx, qscale.scale)
    drift = evaluate_numerical_drift(x, dqx)

    assert drift.cosine_similarity > 0.99
    assert drift.snr_db > 30.0  # High signal quality for INT8


def test_int4_gemm_pack_unpack():
    torch.manual_seed(42)
    # Weight shape: M=64, N=128
    weight = torch.randn(64, 128, dtype=torch.float32)
    scale = torch.max(torch.abs(weight), dim=1, keepdim=True)[0] / 7.0

    packed_bytes, float_scale = Quantizer.pack_int4_gemm(weight, scale)

    # Packed N dimension should be halved (2x INT4 per uint8 byte)
    assert packed_bytes.shape == (64, 64)
    assert packed_bytes.dtype == torch.uint8
    assert float_scale.dtype == torch.float16

    unpacked = Quantizer.unpack_int4_gemm(packed_bytes, float_scale.to(torch.float32))
    assert unpacked.shape == (64, 128)

    drift = evaluate_numerical_drift(weight, unpacked)
    assert drift.cosine_similarity > 0.95


def test_smoothquant_scales_computation():
    act_max = torch.tensor([10.0, 20.0, 5.0], dtype=torch.float32)
    weight_max = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32)

    scales = QuantizationCalibrator.compute_smoothquant_scales(
        act_max, weight_max, alpha=0.5
    )
    assert scales.shape == (3,)
    # expected: sqrt(10/2)=sqrt(5), sqrt(20/4)=sqrt(5), sqrt(5/8)
    assert pytest.approx(scales[0].item(), abs=1e-3) == math.sqrt(5.0)