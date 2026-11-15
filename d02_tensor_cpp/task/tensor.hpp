// Part B: a minimal strided tensor (float32) with shared storage.
// Fill in every TODO. Compile + test:  g++ -std=c++17 -O1 -Wall -Wextra test_tensor.cpp -o test_tensor && ./test_tensor
#pragma once
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

using Shape = std::vector<int64_t>;

// Thrown by the unimplemented stubs below; delete each throw as you implement the function.
struct NotImplemented : std::exception {
    const char* what() const noexcept override { return "not implemented yet"; }
};

// Owns a flat, contiguous buffer. Many Tensor views may share one Storage.
class Storage {
public:
    explicit Storage(size_t n) : size_(n) {
        // TODO: allocate n floats, zero-initialised, owned by data_ (a shared_ptr<float[]>).
        //       Hint: data_ = std::shared_ptr<float[]>(new float[n]());
        throw NotImplemented{};   // <- delete this line once implemented
    }
    float* data() const { return data_.get(); }
    size_t size() const { return size_; }

private:
    std::shared_ptr<float[]> data_;
    size_t size_;
};

class Tensor {
public:
    // Allocates NEW storage and computes row-major (C-order) strides.
    explicit Tensor(Shape shape) : shape_(std::move(shape)) {
        // TODO: numel = product of shape; storage_ = make_shared<Storage>(numel);
        //       strides_[last] = 1, strides_[i] = strides_[i+1] * shape_[i+1]; offset_ = 0.
        throw NotImplemented{};   // <- delete this line once implemented
    }

    // ---- given ----
    const Shape& shape() const { return shape_; }
    const Shape& strides() const { return strides_; }
    int64_t offset() const { return offset_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const {
        return std::accumulate(shape_.begin(), shape_.end(), int64_t{1}, std::multiplies<int64_t>());
    }
    const float* storage_ptr() const { return storage_->data(); }   // for identity checks in tests

    // ---- TODO ----
    // Element access: storage[offset + sum_k idx[k] * strides[k]]. Throw std::out_of_range if
    // idx.size() != ndim or any idx[k] is outside [0, shape[k]).
    float& at(const std::vector<int64_t>& idx) {
        // TODO
        throw NotImplemented{};
    }

    // True if strides match row-major strides for this shape (size-1 dims may be ignored, but
    // the simple exact comparison is fine for this task).
    bool is_contiguous() const {
        // TODO
        throw NotImplemented{};
    }

    // A VIEW: same storage and offset, shape/strides of dims d0 and d1 swapped. O(1), no copy.
    Tensor transpose(int64_t d0, int64_t d1) const {
        // TODO
        throw NotImplemented{};
    }

    // A VIEW of the sub-tensor at `index` along `dim` (that dimension disappears).
    // offset grows by index * strides[dim]. Throw std::out_of_range for bad dim/index.
    Tensor select(int64_t dim, int64_t index) const {
        // TODO
        throw NotImplemented{};
    }

    // A fresh, contiguous COPY with new storage (works for any strided view).
    // Hint: walk every multi-index with an "odometer" counter and call at().
    Tensor contiguous() const {
        // TODO
        throw NotImplemented{};
    }

    // A VIEW with a new shape (same numel). Only legal if is_contiguous(); otherwise throw
    // std::runtime_error (the caller should call contiguous() first).
    Tensor reshape(Shape new_shape) const {
        // TODO
        throw NotImplemented{};
    }

private:
    Tensor() = default;   // used internally to build views
    std::shared_ptr<Storage> storage_;
    Shape shape_;
    Shape strides_;
    int64_t offset_ = 0;
};
