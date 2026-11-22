# `custom/` — out-of-framework experiments

This directory is intentionally **not** part of the framework CMake build. It
is configured separately, after the framework has been built:

```bash
cmake -S custom -B build/custom \
  -DLD_FRAMEWORK_ROOT=/path/to/LearnDeep \
  -DLD_FRAMEWORK_BUILD_DIR=/path/to/LearnDeep/build/debug
cmake --build build/custom
ctest --test-dir build/custom --output-on-failure
```

It consumes the framework only as compiled libraries (`libtensor_api.so`,
`liblayer.so`, `libcore.so`, …) plus the public header `<api/tensor.h>`. No
framework `CMakeLists.txt` is edited to add an experiment here.

## Layout

- `layer/` — self-registering custom layer plugin (`custom_layer` shared library).
- `loss/` — self-registering custom loss plugin (`custom_loss` library):
  `custom_scaled_mse`, a mean squared error with a configurable scale.
- `test/` — standalone tests, one executable per `.cpp` file. Each includes
  **only** `<api/tensor.h>` and links the built libraries and plugins:
  - `custom_layer_test.cpp` — registration, handle validity, parameter access.
  - `custom_loss_test.cpp` — registration and id/type, `setParameter("scale",
    ...)`, `calculate()`, and `compile()` + `fit()` with `custom_scaled_mse`.
  - `custom_loss_integration.cpp` — full `tf::model` training pipeline with
    `custom_scaled_mse` configured to scale 1.0, reusing the NumPy squared-error
    fixtures and checking the loss tensor, gradient tensor, and scalar loss.
  - `custom_scale_integration.cpp` — full `tf::model` training pipeline: the
    custom `scale` layer feeds `squared_error`, callbacks record
    `training_output`, `grad_input`, the loss tensor and its gradient, and all
    values are checked against NumPy reference data.
  - `data/` — float64 `.bin` reference data used by the integration test.
  - `generate_custom_scale_data.py` — regenerates that data from the notebook.
- A future `optimizer/` experiment area can be added in
  `custom/CMakeLists.txt` the same way, incrementally.

Adding a source file to `layer/`, `loss/` or `test/` is picked up automatically
by the `CONFIGURE_DEPENDS` globs: layer/loss sources join their plugins, test
sources become new CTest entries.

## Regenerating the integration reference data

The canonical generator is the `CustomScaleLayer_Test_1` cell of
`test/python_validation/numpy_unit_tests.ipynb`. From the repository root:

```bash
.python_virtualenv/bin/python custom/test/generate_custom_scale_data.py
```

## How registration works

`REGISTER_LAYER` in a custom layer source creates a static initializer that
calls `LayerRegistry::instance().registerLayer(...)` when the plugin shared
library is loaded. The registry is a function-local singleton living in
`liblayer.so`, so the plugin and the framework share one registry instance.

`REGISTER_LOSS` is a public macro for `custom/loss/scaled_squared_error.cpp`.
The plugin implements `tf::loss::behavior`, uses `tf::loss::config` for
construction parameters, and the framework adapts that object to the internal
`Loss` ABI inside `libloss.so`. Its third argument is a public parameter setter,
so `tf::loss::loss::setParameter("scale", 3.0)` works for the generic handle.
`model.compile(tf::optimizer::type::sgd, "custom_scaled_mse")` creates the loss by its
registry key, and callbacks refer to it only by `loss_id`.

`custom/test/custom_layer_test.cpp`, `custom/test/custom_loss_test.cpp`, and
`custom/test/custom_loss_integration.cpp` call the tiny exported anchors
`ld_custom_layers_load()` and `ld_custom_losses_load()` so each plugin is
actually linked and loaded before a handle is created; without them,
`--as-needed` linkers may drop an otherwise unreferenced `DT_NEEDED` entry and
the static initializer would never run.

## Limitations of the "libraries + public header only" boundary

- **The custom layer implementation cannot be written against only
  `<api/tensor.h>`.** That header forward-declares `Layer` and `LayerConfig`;
  it does not define the layer base class, the `REGISTER_LAYER` macro, or the
  registry. `custom/layer/test_scale_layer.hpp` therefore includes the internal
  `<layers/layers.hpp>` and `<layers/layer_registry.hpp>` headers. The custom
  *test* has no such dependency.
- Consequently a custom layer is compiled against internal C++ headers and is
  tied to the internal `Layer` ABI (`Tensor<std::float64_t>*`, `Optimizer*`,
  pure-virtual layout). An `Optimizer`/`Tensor` change requires recompiling it.
- The custom loss has **no** internal-header dependency: it implements the
  public `tf::loss::behavior` contract and registers through the public
  `REGISTER_LOSS` macro. It still links against the framework libraries, so an
  ABI change in the public header requires recompilation. Gradient access is
  provided by `tf::loss::gradient_of()` instead of internal graph headers.
- The public header itself includes internal headers
  (`core/framework/MathLibrary.h`, `core/graph/graph_context.hpp`,
  `core/ops/opskernel.h`), so "public header
  only" still requires the
  framework's source-tree include directory at compile time.
- There is no install/export package yet; `custom/CMakeLists.txt` locates
  libraries directly in the framework build tree via `find_library`.
- No unregister API exists, and symbol visibility is not pinned down. A plugin
  loaded via `dlclose`, or built with `-fvisibility=hidden` without export
  macros, can leave stale factories in the registry.
- A first-class facade such as `tf::layer::dense` still needs a hand-written
  wrapper struct in `api/tensor.h`. The generic `tf::layer::layer("name", cfg)`
  path needs none of that.

The loss extension point is now public: `tf::loss::behavior`,
`tf::loss::config`, `tf::loss::registerBehavior()`, and `REGISTER_LOSS()` are
declared in `<api/tensor.h>`. The same treatment for layers is still a separate
design change; custom layers continue to use internal headers.
