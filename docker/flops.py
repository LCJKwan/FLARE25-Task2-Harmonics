#!/usr/bin/env python3
"""
Count FLOPs for AttnUNet6 with custom handlers for unsupported ops.

Usage:
  python count_flops.py --device cpu
  python count_flops.py --device cuda --shape 1 1 256 256 64
"""

import argparse
import json
from math import prod
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

# ---------------------------------------------------------------------
# Model import (expects ./model/model.json and ./model/model.pth to exist)
# ---------------------------------------------------------------------
from model.AttnUNet6 import AttnUNet6


# ---------------------------------------------------------------------
# Custom FLOP handlers
# ---------------------------------------------------------------------
def flop_mul_jit(inputs, outputs):
    """Elementwise multiplication: 1 multiply per output element."""
    out_shape = outputs[0].type().sizes()
    num_elems = prod([d for d in out_shape if d is not None])
    return num_elems


def flop_sum_jit(inputs, outputs):
    """Summation (reduction): N elements → (N - 1) adds."""
    in_shape = inputs[0].type().sizes()
    num_elems = prod([d for d in in_shape if d is not None])
    return (num_elems - 1) if num_elems > 0 else 0


def flop_add_jit(inputs, outputs):
    """Elementwise addition: 1 add per output element."""
    out_shape = outputs[0].type().sizes()
    num_elems = prod([d for d in out_shape if d is not None])
    return num_elems


def flop_gelu_jit(inputs, outputs):
    """
    GeLU ~8 FLOPs per element (approximation).
    """
    out_shape = outputs[0].type().sizes()
    num_elems = 1
    for d in out_shape:
        if d is not None:
            num_elems *= d
    return 8 * num_elems


def flop_silu_jit(inputs, outputs):
    """
    SiLU (Swish) ~8 FLOPs per element (approximation).
    """
    out_shape = outputs[0].type().sizes()
    num_elems = 1
    for d in out_shape:
        if d is not None:
            num_elems *= d
    return 8 * num_elems


def flop_add__jit(inputs, outputs):
    """In-place add: 1 FLOP per element."""
    out_shape = outputs[0].type().sizes()
    num_elems = 1
    for d in out_shape:
        if d is not None:
            num_elems *= d
    return num_elems


def _safe_dim(d):
    return d if (isinstance(d, int) and d is not None) else 1


def flop_div_jit(inputs, outputs):
    """Elementwise division: 1 division per output element."""
    out_shape = outputs[0].type().sizes()
    num_elems = 1
    for d in out_shape:
        if d is not None:
            num_elems *= d
    return num_elems


def flop_unflatten_jit(inputs, outputs):
    """Unflatten / reshape: view-only → 0 FLOPs."""
    return 0


def flop_scaled_dot_product_attention_jit(inputs, outputs):
    """
    SDPA rough FLOP estimate.

    Shapes:
      Q: [B, H, Lq, D], K: [B, H, Lk, D], V: [B, H, Lk, D]

    Counted components:
      Q @ K^T   : ~2 * B * H * Lq * Lk * D
      scale     : ~1 * B * H * Lq * Lk
      softmax   : ~6 * B * H * Lq * Lk
      P @ V     : ~2 * B * H * Lq * Lk * D

    Total:
      4 * B * H * Lq * Lk * D  +  7 * B * H * Lq * Lk
    """
    q_shape = inputs[0].type().sizes()
    k_shape = inputs[1].type().sizes()

    B = _safe_dim(q_shape[0]) if len(q_shape) > 0 else 1
    H = _safe_dim(q_shape[1]) if len(q_shape) > 1 else 1
    Lq = _safe_dim(q_shape[2]) if len(q_shape) > 2 else 1
    D = _safe_dim(q_shape[3]) if len(q_shape) > 3 else 1
    Lk = _safe_dim(k_shape[2]) if len(k_shape) > 2 else Lq

    attn_scores = 2 * B * H * Lq * Lk * D
    scaling = 1 * B * H * Lq * Lk
    softmax = 6 * B * H * Lq * Lk
    attn_v = 2 * B * H * Lq * Lk * D

    return attn_scores + scaling + softmax + attn_v


custom_handles = {
    "aten::mul": flop_mul_jit,
    "aten::sum": flop_sum_jit,
    "aten::add": flop_add_jit,
    "aten::gelu": flop_gelu_jit,
    "aten::silu": flop_silu_jit,
    "aten::add_": flop_add__jit,
    "aten::div": flop_div_jit,
    "aten::unflatten": flop_unflatten_jit,
    "aten::scaled_dot_product_attention": flop_scaled_dot_product_attention_jit,
    # If your graph uses the underscored variant, map it as well:
    # "aten::_scaled_dot_product_attention": flop_scaled_dot_product_attention_jit,
}


# ---------------------------------------------------------------------
# FLOP runner
# ---------------------------------------------------------------------
@torch.inference_mode()
def run(device: str, shape: list[int], model_dir: Path):
    # Load model definition + weights
    model_json = model_dir / "model.json"
    model_pth = model_dir / "model.pth"

    if not model_json.exists():
        raise FileNotFoundError(f"Missing model config: {model_json}")
    if not model_pth.exists():
        raise FileNotFoundError(f"Missing model weights: {model_pth}")

    cfg = json.loads(model_json.read_text())
    model = AttnUNet6(cfg)
    # weights_only=True is available in newer torch; if not, drop the kwarg.
    # state = torch.load(model_pth, map_location="cpu", weights_only=True)
    # model.load_state_dict(state)
    model.eval().to(device)

    # Dummy input
    dummy = torch.randn(*shape, device=device)

    # FLOPs
    flops = FlopCountAnalysis(model, dummy)
    for op_name, fn in custom_handles.items():
        flops.set_op_handle(op_name, fn)

    total_flops = flops.total()
    gflops = total_flops / 1e9

    # --- Printing (kept identical style) ---
    print(f"Total FLOPs: {total_flops:.0f}")
    print(f"GFLOPs: {gflops:.3f}")
    print(flop_count_table(flops, max_depth=3))
    print(parameter_count_table(model))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="FLOP counter for AttnUNet6.")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
        help="Device to run the analysis on.",
    )
    p.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1, 1, 224, 224, 112],
        help="Dummy input shape, e.g. --shape 1 1 256 256 64",
    )
    p.add_argument(
        "--model_dir",
        type=Path,
        default=Path("./model"),
        help="Directory containing model.json and model.pth",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run(device=args.device, shape=args.shape, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
