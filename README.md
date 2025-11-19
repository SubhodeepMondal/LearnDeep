# Deep Learning Library

## Description
This library uses **CUDA** and **AVX/AVX2 kernels** to implement basic linear algebra subroutines for deep learning.

With this library:
- Linear algebra forward pass can be implemented in **eager mode** and **graph mode**.
- Graph mode provides Backpropagation capibility for all ops.

**This Library is intented to function with tensors of n dimentions**

---

## Linear Algebra Subroutines (LAS)
Currently, a small set of operations are implemented:

|   #   | Operation Name                 | Forward Propagation   | Backward Propagation  | Unit Tests:Eager Mode | Unit Tests:Graph Mode | Unit Tests:Grad   | Unit Tests:Grad Pipe Line     |
| ---:  | :----------------------------  | :-------------------: | :-------------------: | :-------------------: | :-------------------: | :---------------: | :---------------------------: |
| 1     | Matrix addition                | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;          | &#x2713;                      |
| 2     | Matrix multiplication          | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;          | &#x2713;                      |
| 3     | Matrix Hadamard multiplication | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;          | &#10007;                      |
| 4     | Matrix Power                   | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;          | &#x2713;                      |
| 5     | Matrix Mean on n-th axis       | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#10007;          | &#10007;                      |
| 6     | Matrix Transpose               | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#10007;          | &#10007;                      |
| 7     | Matrix Subtraction             | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#10007;          | &#10007;                      |
| 8     | Matrix Scale                   | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#10007;          | &#10007;                      |
| 9     | Matrix Sigmoid                 | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#10007;          | &#10007;                      |
| 10    | Matrix sqrt                    | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#10007;          | &#10007;                      |
| 11    | Matrix Relu                    | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#10007;          | &#10007;                      |
| 12    | Matrix Reduction Sum           | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#10007;          | &#10007;                      |
| 13    | Matrix Exponentiation (`e^x`)  | &#x2713;              | &#x2713;              | &#x2713;              | &#x2713;              | &#10007;          | &#10007;                      |

---

## Build
Follow these steps to build the library:

### Requirements
- GCC/G++ : 13.x is required
- OpenMP
- Not mandetory: To compile this library with CUDA support you must have nvcc compiler installed (cuda-toolkit 12.4 or later)


```bash
mkdir -p build
cd build/
cmake -S .. -B .
make
```
- To compile this library without CUDA support -DENABLE_CUDA=OFF

```bash
mkdir -p build
cd build/
cmake -DENABLE_CUDA=OFF -S .. -B .
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
