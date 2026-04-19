#!/usr/bin/env python3
"""
Generate multi-epoch DenseLayer_Test_3 reference tensors using NumPy.

This script mirrors the math used in:
  - test/unit/dense_layer.unit.cpp (DenseLayer_Test_3)
  - LearnDeep/losses/squared_error.cpp
  - LearnDeep/optimizers/sgd.cpp

Important:
By default it reproduces current C++ behavior where manually set dense weights
are re-initialized from the original values at each epoch, while bias carries
the previous updated value. Pass --carry-weight-across-epochs to emulate
standard SGD behavior instead.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class DenseTest3Data:
    x: np.ndarray
    w: np.ndarray
    b: np.ndarray
    y: np.ndarray


def parse_cpp_array(text: str, name: str) -> np.ndarray:
    pattern = (
        r"std::float64_t\s+"
        + re.escape(name)
        + r"\[\]\s*(?:=)?\s*\{(.*?)\};"
    )
    match = re.search(pattern, text, re.S)
    if not match:
        raise ValueError(f"Array not found in header: {name}")
    values = np.fromstring(match.group(1).replace("\n", " "), sep=",", dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"Array parsed empty: {name}")
    return values


def load_dense_test_3(header_path: Path) -> DenseTest3Data:
    text = header_path.read_text(encoding="utf-8")
    # Fortran-order reshape matches i + j * rows indexing used in C++ tests.
    x = parse_cpp_array(text, "dense_layer_Test_3_input_data").reshape((43, 128), order="F")
    w = parse_cpp_array(text, "dense_layer_Test_3_weight_data").reshape((32, 43), order="F")
    b = parse_cpp_array(text, "dense_layer_Test_3_bias_data").reshape((32, 1), order="F")
    y = parse_cpp_array(text, "dense_layer_Test_3_target_data").reshape((32, 128), order="F")
    return DenseTest3Data(x=x, w=w, b=b, y=y)


def flatten_f(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(-1, order="F")


def format_cpp_array(name: str, values: np.ndarray, per_line: int = 8) -> str:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    lines: List[str] = []
    for i in range(0, vals.size, per_line):
        chunk = ", ".join(f"{v:.8f}" for v in vals[i : i + per_line])
        lines.append(f"    {chunk}")
    body = ",\n".join(lines)
    return f"std::float64_t {name}[] = {{\n{body}\n}};"


def simulate_epochs(
    data: DenseTest3Data,
    epochs: int,
    batch_size: int | None,
    learning_rate: float,
    carry_weight_across_epochs: bool,
) -> Dict[str, np.ndarray]:
    x, w0, b0, y = data.x, data.w, data.b, data.y
    if batch_size is None:
        batch_size = x.shape[1]
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if batch_size > x.shape[1]:
        raise ValueError(
            f"batch_size ({batch_size}) cannot exceed available samples ({x.shape[1]})"
        )

    x_batch = x[:, :batch_size]
    y_batch = y[:, :batch_size]

    all_output = []
    all_loss = []
    all_grad_loss = []
    all_grad_weight = []
    all_grad_bias = []
    all_updated_weight = []
    all_updated_bias = []

    curr_w = w0.copy()
    curr_b = b0.copy()

    for _ in range(epochs):
        if carry_weight_across_epochs:
            training_w = curr_w.copy()
        else:
            # Matches current C++ flow for manual weight init.
            training_w = w0.copy()
        training_b = curr_b.copy()

        output = training_w @ x_batch + training_b
        diff = output - y_batch
        loss = np.mean(diff * diff, axis=1, keepdims=True)
        grad_loss = (2.0 / batch_size) * diff
        grad_weight = grad_loss @ x_batch.T
        grad_bias = np.sum(grad_loss, axis=1, keepdims=True)
        updated_w = training_w - learning_rate * grad_weight
        updated_b = training_b - learning_rate * grad_bias

        all_output.append(flatten_f(output))
        all_loss.append(flatten_f(loss))
        all_grad_loss.append(flatten_f(grad_loss))
        all_grad_weight.append(flatten_f(grad_weight))
        all_grad_bias.append(flatten_f(grad_bias))
        all_updated_weight.append(flatten_f(updated_w))
        all_updated_bias.append(flatten_f(updated_b))

        curr_w = updated_w
        curr_b = updated_b

    return {
        "output": np.concatenate(all_output),
        "loss": np.concatenate(all_loss),
        "grad_loss": np.concatenate(all_grad_loss),
        "grad_weight": np.concatenate(all_grad_weight),
        "grad_bias": np.concatenate(all_grad_bias),
        "updated_weight": np.concatenate(all_updated_weight),
        "updated_bias": np.concatenate(all_updated_bias),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate NumPy reference arrays for DenseLayer_Test_3 with multiple epochs."
    )
    parser.add_argument(
        "--header",
        type=Path,
        default=Path("test/unit/dense_layer_data.hpp"),
        help="Path to dense_layer_data.hpp",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to emulate")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size used in loss/gradient computation (default: use all samples in input)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="SGD learning rate (default matches SGD::SGD)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="dense_layer_Test_3_multi_epoch",
        help="Prefix for generated C++ array names",
    )
    parser.add_argument(
        "--carry-weight-across-epochs",
        action="store_true",
        help="Use standard SGD carry-over for weights instead of current C++ reset behavior",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output .hpp path (prints to stdout if omitted)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")

    data = load_dense_test_3(args.header)
    history = simulate_epochs(
        data=data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        carry_weight_across_epochs=args.carry_weight_across_epochs,
    )

    text_blocks = [
        f"// Generated by {Path(__file__).name}",
        f"// epochs={args.epochs}, batch_size={args.batch_size or data.x.shape[1]}, learning_rate={args.learning_rate}",
        (
            "// behavior=standard_sgd_weight_carry"
            if args.carry_weight_across_epochs
            else "// behavior=match_current_cpp_weight_reset_each_epoch"
        ),
        format_cpp_array(f"{args.prefix}_output_data", history["output"]),
        format_cpp_array(f"{args.prefix}_loss_data", history["loss"]),
        format_cpp_array(f"{args.prefix}_grad_loss_data", history["grad_loss"]),
        format_cpp_array(f"{args.prefix}_grad_weight_data", history["grad_weight"]),
        format_cpp_array(f"{args.prefix}_grad_bias_data", history["grad_bias"]),
        format_cpp_array(f"{args.prefix}_updated_weights_data", history["updated_weight"]),
        format_cpp_array(f"{args.prefix}_updated_bias_data", history["updated_bias"]),
    ]
    final_text = "\n\n".join(text_blocks) + "\n"

    if args.out:
        args.out.write_text(final_text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(final_text)


if __name__ == "__main__":
    main()
