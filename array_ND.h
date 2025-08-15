#pragma once
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <limits>
#include <cstddef>  // for size_t

template<typename T>
class ArrayND {
private:
    std::vector<size_t> dims_;    // sizes of each dimension
    std::vector<size_t> strides_; // precomputed strides
    std::vector<T> data_;         // contiguous storage

public:
    // Constructor using initializer_list
    ArrayND(std::initializer_list<size_t> dims) 
        : dims_(dims) 
    {
        if (dims_.empty())
            throw std::invalid_argument("ArrayND must have at least one dimension");

        // Compute strides safely
        strides_.resize(dims_.size());
        size_t stride = 1;
        for (size_t i = dims_.size(); i-- > 0;) {
            if (dims_[i] != 0 && stride > std::numeric_limits<size_t>::max() / dims_[i])
                throw std::overflow_error("ArrayND size too large");
            strides_[i] = stride;
            stride *= dims_[i];
        }

        data_.resize(stride); // fills with default-constructed T (0.0 for double)
    }

    // Access operator for 2D/3D/... using variadic template
    template<typename... Args>
    T& operator()(Args... args) {
        return data_[compute_index(args...)];
    }

    template<typename... Args>
    const T& operator()(Args... args) const {
        return data_[compute_index(args...)];
    }

    // Return raw pointer for MPI
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    // Number of dimensions
    size_t ndim() const { return dims_.size(); }

    // Get size of a dimension
    size_t size(size_t i) const { return dims_.at(i); }

private:
    // Helper to compute flat index
    template<typename... Args>
    size_t compute_index(Args... args) const {
        if (sizeof...(Args) != dims_.size())
            throw std::out_of_range("Wrong number of indices in ArrayND");
        size_t idx = 0;
        size_t indices[] = {static_cast<size_t>(args)...};
        for (size_t i = 0; i < dims_.size(); ++i) {
            if (indices[i] >= dims_[i])
                throw std::out_of_range("ArrayND index out of bounds");
            idx += indices[i] * strides_[i];
        }
        return idx;
    }
};
