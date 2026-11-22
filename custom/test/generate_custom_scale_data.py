#!/usr/bin/env python3
"""Generate reference data consumed by custom_scale_integration.cpp.

The canonical data generator is the ``CustomScaleLayer_Test_1`` cell of
``test/python_validation/numpy_unit_tests.ipynb``. This script executes that
cell and stores the arrays as little-endian float64 ``.bin`` files, matching
the framework's ``load_bin()`` convention.

Run from the repository root:

    .python_virtualenv/bin/python custom/test/generate_custom_scale_data.py
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "test/python_validation/numpy_unit_tests.ipynb"
DATA_DIR = REPO_ROOT / "custom/test/data"

ARRAYS = {
    "CustomScale_Test_1_input_data.bin": "x_input",
    "CustomScale_Test_1_target_data.bin": "y_output",
    "CustomScale_Test_1_scale_data.bin": "scale",
    "CustomScale_Test_1_output_data.bin": "y_hat",
    "CustomScale_Test_1_loss_data.bin": "loss",
    "CustomScale_Test_1_grad_loss_data.bin": "grad_loss",
    "CustomScale_Test_1_grad_input_data.bin": "grad_input",
}


def notebook_cell_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if ("np.random.seed(4242)" in source
                and "grad_input = scale * grad_loss" in source):
            return source
    raise RuntimeError(
        "Could not find the CustomScaleLayer_Test_1 cell in the notebook")


def main() -> None:
    namespace: dict[str, object] = {"np": np}
    with redirect_stdout(io.StringIO()):
        exec(compile(notebook_cell_source(), "<notebook-cell>", "exec"),
             namespace)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, variable in ARRAYS.items():
        values = np.asarray(namespace[variable], dtype=np.float64).reshape(-1)
        path = DATA_DIR / filename
        with path.open("wb") as handle:
            np.asarray(values, dtype="<f8").tofile(handle)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({values.size} float64 values)")


if __name__ == "__main__":
    main()
