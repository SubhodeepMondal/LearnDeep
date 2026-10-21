<p align="center">
  <img src="docs/images/LearnDeep_ Neural Networks in Motion.png"
       alt="LearnDeep — Deep Learning Framework From Scratch"
       width="100%">
</p>

<h1 align="center">LearnDeep</h1>

<p align="center">
  A Deep Learning Framework Built From Scratch in C++
</p>

# LearnDeep — Deep Learning Framework From Scratch

LearnDeep is a deep learning framework written in **C++ from scratch**, built to explore the internals of modern deep learning systems, from low-level tensor operations and hardware-specific kernels to automatic differentiation, neural network layers, and model training.

The project focuses on understanding and implementing the fundamental components of a deep learning framework rather than relying on an existing framework for its core computational logic.

## Features

- **N-dimensional tensors** for representing and manipulating numerical data.
- **Multiple execution modes** — eager execution and graph-based execution.
- **Automatic differentiation (AutoDiff)** using reverse-mode differentiation.
- **Hardware-accelerated computation** through CPU, AVX2, and CUDA kernels.
- **Tensor operations** including arithmetic, broadcasting, reductions, and activation functions.
- **Neural network layers** for constructing composite computations.
- **Model prediction and training** workflows.
- **Loss functions** for measuring prediction error.
- **Unit testing and numerical verification** against NumPy reference implementations.

The framework is under active development. Features and operations may have different levels of testing and backend support.

---

## Architecture

LearnDeep is organized into modular components that separate tensor representation, mathematical operations, execution, differentiation, and model-level abstractions.

| Component | Responsibility |
|---|---|
| Tensor API | Tensor representation, shapes, and user-facing operations |
| Linear Algebra Subroutines (LAS) | Numerical operations and their derivatives |
| Kernel layer | CPU, AVX2, and CUDA implementations of supported operations |
| Execution engine | Eager and graph-based execution |
| Automatic differentiation | Gradient propagation through computational graphs |
| Layers | Reusable neural network building blocks |
| Models | Composition of layers and model prediction/training workflows |
| Loss functions | Quantification of the difference between predictions and targets |
| Optimizers and callbacks | Parameter updates and training-time monitoring |

The separation between mathematical operations and their underlying kernels allows higher-level computations to use different hardware implementations without changing the model-level API.

## Execution Modes

### Eager execution

In eager mode, operations are executed as they are invoked. This provides a straightforward way to perform tensor computations and inspect intermediate results.

Eager execution is useful for debugging, numerical verification, and understanding individual operations.

### Graph execution

In graph mode, operations participate in a computational graph that represents dependencies between computations.

The graph provides the structure needed for reverse-mode automatic differentiation, allowing gradients to propagate from outputs to the tensors and parameters that contributed to them.

Graph execution also establishes a foundation for future graph-level optimizations.

## Automatic Differentiation

LearnDeep implements **reverse-mode automatic differentiation** to calculate gradients through computational graphs.

Rather than manually deriving and implementing the gradient of an entire neural network, individual operations define their backward behavior. These derivatives can then be composed to calculate gradients for larger computations.

The implementation includes gradient propagation for supported tensor operations, including broadcasting-aware operations.

This functionality is fundamental to training neural networks using gradient-based optimization.

---

## Tensor Operations and Linear Algebra Subroutines

The LAS component provides the numerical building blocks used by higher-level layers and models.

The operation set includes:

| Category | Operations |
|---|---|
| Arithmetic | Addition, subtraction, multiplication, scaling |
| Broadcasting | Shape-aware elementwise operations |
| Matrix operations | Matrix multiplication, transpose |
| Elementwise functions | Power, square root, exponential, logarithm |
| Activations | ReLU, sigmoid, softmax |
| Comparisons | Greater-than-zero |
| Reductions | Sum and mean along supported axes |
| Differentiation | Backward operations for supported computations |

The implementation of each operation is separated from its execution backend wherever applicable.

### Kernel backends

- **CPU:** General-purpose CPU execution.
- **AVX2:** Vectorized CPU kernels for supported operations.
- **CUDA:** GPU kernels for supported operations.

Backend availability and operation coverage depend on the specific kernel implementation. The presence of an operation in the public API does not necessarily imply identical support across every backend.

---

## Neural Network Layers

Layers combine lower-level tensor operations into reusable neural network components.

### Dense layer

The dense (fully connected) layer performs an affine transformation using learnable weights and biases:

\[
Y = W^\mathsf{T}X + b
\]

The framework uses a feature-by-batch input convention for dense computations, with tensors organized as:

- Input: `[features, batch]`
- Weights: `[input_features, output_units]`
- Output: `[output_units, batch]`

This convention is important when validating forward and backward computations against reference implementations.

### Activation functions

Activation functions introduce nonlinear transformations into neural networks.

The current operation set includes ReLU, sigmoid, and softmax, which can be composed with dense layers to construct more expressive models.

The supported layer and activation APIs continue to evolve alongside the underlying tensor operations.

---

## Models and Training

The model abstraction allows multiple layers to be composed into a neural network and evaluated as a complete computation.

The framework's model-level functionality includes:

- Forward prediction.
- Training through graph-based gradient computation.
- Gradient-based parameter updates.
- Batch-based training workflows.
- Training callbacks for observing and managing the training process.

### Training callbacks

Callbacks provide a mechanism for observing training progress and applying training-time control logic.

Implemented callback functionality includes:

- **TraceCallback:** Monitoring training progress.
- **HistoryCallback:** Recording training history.
- **EarlyStopping:** Stopping training when the configured stopping condition is met.

The exact callback behavior depends on its configuration and the current training API.

---

## Loss Functions and Optimization

### Categorical Cross-Entropy

Categorical Cross-Entropy (CCE) measures the difference between predicted class probabilities and target class labels or distributions.

For a batch of \(N\) samples, its standard form is:

\[
L = -\frac{1}{N}\sum_{i=1}^{N}
\sum_{c=1}^{C} y_{ic}\log(p_{ic})
\]

where:

- \(y_{ic}\) represents the target for class \(c\) in sample \(i\).
- \(p_{ic}\) represents the predicted probability for that class.
- \(C\) is the number of classes.

The implementation includes the tensor operations required for the loss calculation and its gradient propagation.

### Optimization

LearnDeep includes stochastic gradient descent (SGD) for updating learnable parameters using calculated gradients.

The optimizer, loss function, model, and execution engine work together to form the training pipeline.

---

## Testing and Numerical Verification

Testing is an important part of the development process. Numerical operations and their gradients are validated using unit tests, with NumPy used as a reference for selected computations.

The test suite covers areas including:

- Forward-operation correctness.
- Eager and graph execution.
- Gradient calculation.
- Composite computations.
- Neural network prediction and training.
- CPU, AVX2, and CUDA kernel implementations where supported.

Individual tests may cover different combinations of execution modes, backends, and gradient paths. Test coverage should therefore be evaluated per operation rather than assumed to be uniform across the library.

### Running the tests

From the configured build directory, run the test suite using the project's available CTest configuration:

```bash
ctest --output-on-failure
```

To execute an individual GoogleTest case, use the test executable generated by your build:

```bash
./path/to/test_executable \
    --gtest_filter="*TestName*"
```

Replace `path/to/test_executable` and `TestName` with the actual executable path and test name from your build.

---

## Build Instructions

### Requirements

- GCC/G++ 13 or a compatible supported compiler.
- CMake.
- OpenMP.
- CUDA Toolkit 12.4 or later for CUDA-enabled builds, subject to the project's configured CUDA requirements.
- GoogleTest and other project dependencies as configured by CMake.

### Build with the default configuration

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

### Build without CUDA

```bash
cmake -S . -B build-cpu \
    -DENABLE_CUDA=OFF

cmake --build build-cpu -j$(nproc)
```

**Note:** Use the project's current CMake options and presets if the configuration has changed. CPU, AVX2, and CUDA builds should be configured separately when required by the selected backend.

---

## Development Roadmap

LearnDeep is an ongoing effort to build a deep learning framework from the ground up.

Areas for continued development include:

- Expanding tensor functionality and shape semantics.
- Improving consistency across CPU, AVX2, and CUDA backends.
- Extending layer and loss-function support.
- Strengthening numerical and gradient validation.
- Improving memory management and GPU allocation efficiency.
- Introducing graph-level execution optimizations.
- Expanding model-training and parameter-management capabilities.

These are development directions, not claims that every listed capability is already implemented.

---

## Project Philosophy

LearnDeep is both a software engineering project and a learning exercise in understanding how deep learning frameworks work internally.

The goal is to build the framework layer by layer, validating each component numerically and understanding the relationship between tensor mathematics, hardware execution, automatic differentiation, and neural network training.

**Built from scratch, one operation at a time.**