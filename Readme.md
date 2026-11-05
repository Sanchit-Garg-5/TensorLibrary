# Mini Tensor Library

A lightweight, high-performance, header-only C++ tensor library designed for multidimensional array manipulation, parallel linear algebra, and neural network data handling. Built from scratch with contiguous memory management, stride-based index mapping, cache-friendly loop reordering, and OpenMP acceleration.

---

## Features

* **Header-Only Architecture:** Easy integration into any C++ project by simply including `class.h`.
* **Contiguous Memory Storage:** Internal data is flattened into a single contiguous `std::vector<T>` for cache locality and fast memory access.
* **Stride-Based Indexing:** Efficient $O(N_d)$ index-to-offset calculation for arbitrary multi-dimensional tensors.
* **Cache-Optimized Matrix Multiplication:** Implements $i-k-j$ inner loop ordering to maximize spatial locality and CPU cache hits.
* **Batched MatMul & Broadcasting:** Supports 2D matrix multiplication as well as multi-dimensional batched tensor multiplication with automatic batch dimension broadcasting.
* **OpenMP Parallel Acceleration:** Automatically parallelizes matrix multiplication and large element-wise operations across multiple CPU threads using conditional workload thresholds.
* **Modern C++ Design:** Templated (`mt::Tensor<T>`) for type flexibility (`float`, `double`, `int`), move semantics, and const-correct memory access.

---

## Supported Operations

| Operation | Description | Syntax | Time Complexity |
| :--- | :--- | :--- | :--- |
| **Element Access** | Multidimensional coordinate lookup/modification | `A({i, j, k})` | $O(N_d)$ |
| **Addition** | Element-wise tensor addition | `A + B` | $O(E)$ |
| **Subtraction** | Element-wise tensor subtraction | `A - B` | $O(E)$ |
| **Element-wise Multiplication** | Hadamard product | `A.mul(B)` | $O(E)$ |
| **Matrix Multiplication** | 2D / Batched MatMul with broadcasting | `A.matmul(B)` | $O(B \times M \times N \times P)$ |
| **Reshape** | Alters dimensions without copying memory layout | `A.reshape(newShape)` | $O(E)$ |
| **Transpose** | Reverses tensor dimensions | `A.transpose()` | $O(E \times N_d)$ |

*(Where $E$ = total elements, $N_d$ = number of dimensions, $B$ = total batch count)*

---

## Prerequisites

* **OS:** Linux (Ubuntu 22.04+ recommended), macOS, or Windows.
* **Compiler:** GCC or Clang with C++17 support.
* **Build System:** CMake 3.10 or higher.
* **Libraries:** OpenMP (usually bundled with GCC/Clang).

---

## Build & Test Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/MiniTensorLibrary.git
cd MiniTensorLibrary
```

### 2. Build with CMake
```bash
mkdir -p build && cd build
cmake ..
make
```

### 3. Run Benchmarks & Unit Tests
```bash
./tensor_test
```

---

## Quick Start Example

```cpp
#include "class.h"
#include <iostream>
#include <vector>

int main() {
    // 1. Initialize a 2D Tensor (1000x1000) filled with 1.0f
    std::vector<float> data_A(1000000, 1.0f);
    std::vector<float> data_B(1000000, 2.0f);

    mt::Tensor<float> A({1000, 1000}, data_A);
    mt::Tensor<float> B({1000, 1000}, data_B);

    // 2. Element-wise operations
    mt::Tensor<float> C_add = A + B;       // Addition
    mt::Tensor<float> C_mul = A.mul(B);     // Element-wise multiplication

    // 3. Batched Matrix Multiplication (A @ B)
    mt::Tensor<float> C_mat = A.matmul(B);  // MatMul

    // 4. Element Access (0-indexed)
    std::cout << "Element at (0, 0): " << C_mat({0, 0}) << "\n"; // Outputs 2000.0f

    // 5. Reshape and Transpose
    mt::Tensor<float> Reshaped = C_mat.reshape({500, 2000});
    mt::Tensor<float> Transposed = A.transpose();

    return 0;
}
```

---

## Performance Optimization & OpenMP Tuning

The library leverages OpenMP for multi-threaded parallel execution. Thread scaling is controlled conditionally to ensure small operations avoid multi-threading overhead while large tensor workloads gain maximum multi-core throughput:

* **Element-wise operations (`+`, `-`, `mul`):** Parallelized when total workload exceeds **10,000 elements**.
* **Matrix Multiplication (`matmul`):** Parallelized when inner multiply operations exceed **65,536 iterations**.

You can adjust thread execution dynamically at runtime in your test suite:

```cpp
#include <omp.h>

// Force single-threaded sequential execution
omp_set_num_threads(1);

// Enable maximum available CPU cores for parallel speedup
omp_set_num_threads(omp_get_max_threads());
```

---

## Project Structure

```text
MiniTensorLibrary/
├── CMakeLists.txt        # Build system configuration
├── class.h               # Header-only Tensor class implementation
├── tensor_test.cpp       # Unit tests and OpenMP timing benchmarks
└── README.md             # Project documentation
```
