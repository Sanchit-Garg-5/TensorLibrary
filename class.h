#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mt {
template<typename T = float>
class Tensor {

    std::vector<T> data_;
    std::vector<std::size_t> shape_, strides_;
    std::size_t total_element = 1;
    std::size_t ndim = 0;

public:

    Tensor(const std::vector<std::size_t>& shape, const std::vector<T>& data):
    data_(data), shape_(shape){
        ndim = shape.size();
        total_element = 1;
        for (std::size_t s : shape_) { total_element *= s; }
        if (data_.size() != total_element) {
            throw std::invalid_argument("Error: Data size does not match shape");
        }
        strides_compute();
    }
    Tensor(std::vector<std::size_t> shape) : shape_(shape) {
        ndim = shape_.size();
        total_element = 1;
        for (std::size_t s : shape_) {
            total_element *= s;
        }
//there is 1 element even in a 0d Tensor
        strides_compute();                 // sizes and fills strides_
        data_.resize(total_element, T{});  // zero-filled
    }

    void strides_compute() {
        strides_.resize(ndim, 0);
        std::size_t prod = 1;
        for (std::size_t i = ndim; i-- > 0; ) {   
            strides_[i] = prod;
            prod *= shape_[i];
        }
    }

    // ELEMENT ACCESS
    T& operator()(const std::vector<std::size_t>& idx){ //0 based indexes
        if (idx.size() != ndim) {
            throw std::invalid_argument("Error: Incorrect number of indices");
        }
        std::size_t offset = 0;
        for (std::size_t i = 0; i < ndim; i++){
            if (idx[i] >= shape_[i]) {
                throw std::out_of_range("Error: Index out of bounds");
            }
            offset += idx[i] * strides_[i];
        }
        return data_[offset];
    }

    void read(){
        for (std::size_t i = 0; i < total_element; i++) {
            std::cin >> data_[i];
        }
    }

    // ADDITION
    Tensor operator+(const Tensor& other) const {
        if (shape_ != other.shape_) throw std::invalid_argument("Error: Shape Mismatch");
        Tensor result(shape_);
        T* r_ptr = result.data_.data();
        const T* d_ptr = data_.data();
        const T* o_ptr = other.data_.data();
        const std::size_t n = data_.size();

        #pragma omp parallel for simd if(n > 10000)
        for (std::size_t i = 0; i < n; i++) {
            r_ptr[i] = d_ptr[i] + o_ptr[i];
        }
        return result;
    }

    // SUBTRACTION
    Tensor operator-(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Error: Shape Mismatch");
        }
        T* r_ptr = result.data_.data();
        const T* d_ptr = data_.data();
        const T* o_ptr = other.data_.data();
        const std::size_t n = data_.size();

        #pragma omp parallel for simd if(n > 10000)
        Tensor result(shape_);
        for (std::size_t i = 0; i < n; i++) {
            r_ptr[i] = d_ptr[i] - o_ptr[i];
        }
        return result;
    }


    // ELEMENT-WISE MULTIPLICATION
    Tensor mul(const Tensor& other) const {

        if (shape_ != other.shape_) {
            throw std::invalid_argument("Error: Shape Mismatch");
        }

        Tensor result(shape_);
        T* r_ptr = result.data_.data();
        const T* d_ptr = data_.data();
        const T* o_ptr = other.data_.data();
        const std::size_t n = data_.size();

        #pragma omp parallel for simd if(n > 10000)
        for (std::size_t i = 0; i < n; i++) {
            r_ptr[i] = d_ptr[i] * o_ptr[i];
        }
        return result;
    }


    // MATRIX MULTIPLICATION (batched, with broadcasting on the batch dims)
    Tensor matmul(const Tensor& other) const {
        std::size_t ondim = other.ndim;
        if (ndim < 2 || ondim < 2) {
            throw std::invalid_argument("Error:The tensors aren't even 2D");
        }
        if (shape_[ndim - 1] != other.shape_[ondim - 2]) {
            throw std::invalid_argument("Error:The dimensions reqd are not satisfied");
        }
        std::size_t rb = std::max(ndim, ondim) - 2;
        std::vector<std::size_t> resShape(rb + 2);
        std::size_t BatchCount = 1;
        std::vector<std::size_t> aBatchShape(rb, 1), bBatchShape(rb, 1);

        for (std::size_t d = 0; d < rb; d++) {
            std::size_t da = (d < ndim  - 2) ? shape_[ndim - 3 - d] : 1;
            std::size_t db = (d < ondim - 2) ? other.shape_[ondim - 3 - d] : 1;
            if (da != db && da != 1 && db != 1) {
                throw std::invalid_argument("Error: batch dimensions not compatible");
            }
            resShape[rb - 1 - d] = std::max(da, db);
            aBatchShape[rb - 1 - d] = da;
            bBatchShape[rb - 1 - d] = db;
            BatchCount *= std::max(da, db);
        }

        const std::size_t m = shape_[ndim - 2];
        const std::size_t n = shape_[ndim - 1];
        const std::size_t p = other.shape_[ondim - 1];
        resShape[rb] = m;
        resShape[rb + 1] = p;
        Tensor result(resShape);  

        const T* A = data_.data();
        const T* B = other.data_.data();
        T* C = result.data_.data();

        // batch strides (0 where a dimension is broadcast)
        std::vector<std::size_t> aStride(rb, 0), bStride(rb, 0);
        std::size_t a_prod = 1, b_prod = 1;
        for (std::size_t d = rb; d-- > 0; ) {
            aStride[d] = (aBatchShape[d] > 1) ? a_prod : 0;
            bStride[d] = (bBatchShape[d] > 1) ? b_prod : 0;
            a_prod *= aBatchShape[d];
            b_prod *= bBatchShape[d];
        }

        // Starting offset of each matrix pair, computed once (serial, cheap)
        std::vector<std::size_t> aOffs(BatchCount), bOffs(BatchCount);
        for (std::size_t t = 0; t < BatchCount; t++) {
            std::size_t aOff = 0, bOff = 0, temp_t = t;
            for (std::size_t d = rb; d-- > 0; ) {
                std::size_t cur = temp_t % resShape[d];
                aOff += cur * aStride[d];
                bOff += cur * bStride[d];
                temp_t /= resShape[d];
            }
            aOffs[t] = aOff * (m * n);   // scale batch offsets by the matrix size
            bOffs[t] = bOff * (n * p);
        }

        const std::size_t totalRows = BatchCount * m;
        const std::size_t work = totalRows * n * p;   
        (void)work;       

        #pragma omp parallel for if(work > 65536)
        for (std::size_t r = 0; r < totalRows; r++) {
            const std::size_t t = r / m;   // which matrix pair
            const std::size_t i = r % m;   // which row inside it

            const T* Arow = A + aOffs[t] + i * n;
            const T* Bmat = B + bOffs[t];
            T* Crow = C + r * p;           // equals t*m*p + i*p

            for (std::size_t k = 0; k < n; k++) {
                const T aik = Arow[k];
                const T* Brow = Bmat + k * p;
                for (std::size_t j = 0; j < p; j++) {
                    Crow[j] += aik * Brow[j];
                }
            }
        }
        return result;
    }

    Tensor reshape(const std::vector<std::size_t>& newShape) const {
        return Tensor(newShape, data_);
    }
    // trailing const: promises not to modify the Tensor the function is called on
    // const on the parameter: 
    // lets a temporary vector like {2,3,4} in int main() to bind to the reference

    Tensor transpose() const {
        std::vector<std::size_t> transdim(shape_.rbegin(), shape_.rend());
        Tensor res(transdim);
        std::vector<std::size_t> idx(ndim, 0);
        for (std::size_t i = 0; i < total_element; i++) {
            std::size_t offset = 0;

            for (std::size_t d = 0; d < ndim; d++) {
                offset += idx[d] * strides_[ndim - d - 1];
            }
            res.data_[i] = data_[offset];

            for (std::size_t d = ndim; d-- > 0; ) {
                if (++idx[d] < transdim[d]) break;
                idx[d] = 0;
            }
        }
        return res;
    }
};

} // namespace mt
