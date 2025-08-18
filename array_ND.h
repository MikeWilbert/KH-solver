#pragma once
#include <vector>
#include <initializer_list>
#include <cassert>
#include <cstddef>

template<typename T>
class ArrayND {
public:
    std::vector<T> data;        // contiguous storage
    std::vector<size_t> dims;   // dimensions

    // --- Default constructor ---
    ArrayND() {}

    // --- Constructor from initializer list ---
    ArrayND(std::initializer_list<size_t> dims_) : dims(dims_) {
        size_t total = 1;
        for (auto d : dims) total *= d;
        data.resize(total, T(0));  // initialize with zeros
    }

    // --- Resize method for delayed allocation ---
    void resize(std::initializer_list<size_t> new_dims) {
        dims = new_dims;
        size_t total = 1;
        for (auto d : dims) total *= d;
        data.resize(total, T(0));
    }

    // --- 2D indexing operator ---
    T& operator()(size_t i, size_t j) {
        assert(dims.size() == 2);
        return data[i*dims[1] + j];
    }

    const T& operator()(size_t i, size_t j) const {
        assert(dims.size() == 2);
        return data[i*dims[1] + j];
    }

    // --- 3D indexing operator ---
    T& operator()(size_t i, size_t j, size_t k) {
        assert(dims.size() == 3);
        return data[i*dims[1]*dims[2] + j*dims[2] + k];
    }

    const T& operator()(size_t i, size_t j, size_t k) const {
        assert(dims.size() == 3);
        return data[i*dims[1]*dims[2] + j*dims[2] + k];
    }

    // --- Utility: total number of elements ---
    size_t size() const { return data.size(); }
};
