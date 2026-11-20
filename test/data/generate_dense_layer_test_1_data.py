#!/usr/bin/env python3
"""Generate the binary reference data consumed by FrameworkTest.DenseLayer_Test_1.

The canonical data generator is cell 141 of
``test/python_validation/numpy_unit_tests.ipynb``. This script executes that
cell and stores the resulting NumPy arrays as little-endian float64 ``.bin``
files, matching the ``load_bin()`` convention used by DenseLayer_Test_2 and the
other tests in ``test/unit``.

Run from the repository root:

    .python_virtualenv/bin/python test/data/generate_dense_layer_test_1_data.py
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "test/python_validation/numpy_unit_tests.ipynb"
DATA_DIR = REPO_ROOT / "test/data"

# Files are named after the test that consumes them, following the
# DenseLayer_Test_2_* convention.
ARRAYS = {
    "DenseLayer_Test_1_input_data.bin": "x_input",
    "DenseLayer_Test_1_target_data.bin": "y_output",
    "DenseLayer_Test_1_weights_data.bin": "weights",
    "DenseLayer_Test_1_bias_data.bin": "bias",
    "DenseLayer_Test_1_output_data.bin": "y_hat",
    "DenseLayer_Test_1_loss_data.bin": "loss",
    "DenseLayer_Test_1_grad_loss_data.bin": "grad_loss",
    "DenseLayer_Test_1_grad_weight_data.bin": "grad_weight",
    "DenseLayer_Test_1_grad_bias_data.bin": "grad_bias",
    "DenseLayer_Test_1_updated_weights_data.bin": "updated_weight",
    "DenseLayer_Test_1_updated_bias_data.bin": "updated_bias",
}


def notebook_cell_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "np.random.seed(32)" in source and "grad_loss = (2.0 / (batch_size * y_output.shape[1]))" in source:
            return source
    raise RuntimeError("Could not find the DenseLayer_Test_1 generator cell in the notebook")


def main() -> None:
    namespace: dict[str, object] = {"np": np}
    with redirect_stdout(io.StringIO()):
        exec(compile(notebook_cell_source(), "<notebook-cell>", "exec"), namespace)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, variable in ARRAYS.items():
        # The notebook prints these arrays with numpy's default C-order
        # flatten(), which is also the layout the C++ tests index with
        # (i + j * rows).
        values = np.asarray(namespace[variable], dtype=np.float64).reshape(-1)
        path = DATA_DIR / filename
        with path.open("wb") as handle:
            np.asarray(values, dtype="<f8").tofile(handle)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({values.size} float64 values)")


if __name__ == "__main__":
    main()
