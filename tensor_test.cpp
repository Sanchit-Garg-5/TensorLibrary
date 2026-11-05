#include "class_2.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h> 


template<typename Func>
double benchmark(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func(); 
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    std::cout << "--- Mini Tensor Library Benchmark ---\n\n";

    int max_threads = omp_get_max_threads();

    std::vector<float> data1(10000000, 1.5f);
    std::vector<float> data2(10000000, 2.0f);
    mt::Tensor<float> A({1000, 1000}, data1);
    mt::Tensor<float> B({1000, 1000}, data2);

    std::cout << "[Addition Test: 10M Elements]\n";
    omp_set_num_threads(1);
    double add_seq = benchmark([&]() { A + B; });
    omp_set_num_threads(max_threads);
    double add_par = benchmark([&]() { A+B; });
    std::cout << "Sequential (1 Thread): " << add_seq << " s\n";
    std::cout << "Parallel (" << max_threads << " Threads): " << add_par << " s\n\n";

    std::cout << "[Multiplication Test: 10M Elements]\n";
    omp_set_num_threads(1);
    double mul_seq = benchmark([&]() { A.mul(B); });
    omp_set_num_threads(max_threads);
    double mul_par = benchmark([&]() { A.mul(B); });
    std::cout << "Sequential (1 Thread): " << mul_seq << " s\n";
    std::cout << "Parallel (" << max_threads << " Threads): " << mul_par << " s\n\n";


    std::cout << "[MatMul Test: 1000x1000 @ 1000x1000]\n";
    omp_set_num_threads(1);
    double matmul_seq = benchmark([&]() { A.matmul(B); });
    omp_set_num_threads(max_threads);
    double matmul_par = benchmark([&]() { A.matmul(B); });
    std::cout << "Sequential (1 Thread): " << matmul_seq << " s\n";
    std::cout << "Parallel (" << max_threads << " Threads): " << matmul_par << " s\n\n";

    return 0;
}
