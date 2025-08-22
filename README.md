# Deep Learning Library

## Description
This library uses **CUDA** and **AVX/AVX2 kernels** to implement basic linear algebra subroutines for deep learning.

With this library:
- Linear algebra forward pass can be implemented in **eager mode** and **graph mode**.
- Graph mode provides flexibility to add backpropagation functionality in future.

**This Library is intented to function with tensors of n dimentions**

---

## Linear Algebra Subroutines (LAS)
Currently, a small set of operations are implemented:

1. Matrix addition
2. Matrix multiplication
3. Matrix Hadamard multiplication
4. Exponential operation (`x^e`, where *e* is an integer)
5. Reduction sum along the *n*-th dimension
6. Scaling operation

---

## Build
Follow these steps to build the library:

### Requirement
- To compile this library with CUDA support you must have nvcc compiler installed (cuda-toolkit 12.4 or later)
- GCC/G++ : 13.x is required


```bash
mkdir -p build
cd build/
cmake -S .. -B .
make
```

---

## Tests
Tests are currently available only for the implemented operations.
Each operation works on a 2D matrix of size 4 × 4.

Run all test cases:

```bash
./test/deeplearingLA_test
```
Run a particular test case using --gtest_filter. Example:

```bash
./test/deeplearingLA_test --gtest_filter="*MatrixMultiplication_2D*"
```
