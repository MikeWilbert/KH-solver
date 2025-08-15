#include <vector>
#include <cassert>
#include <initializer_list>

template<typename T>
class ArrayND {
private:
    std::vector<T> data_;
    std::vector<size_t> dims_;
    std::vector<size_t> strides_;

public:
    // Constructor from a list of dimensions
    ArrayND(std::initializer_list<size_t> dims) : dims_(dims) {
        size_t N = dims_.size();
        strides_.resize(N);
        size_t stride = 1;
        for (size_t i = N; i-- > 0;) {
            strides_[i] = stride;
            stride *= dims_[i];
        }
        data_.resize(stride);
    }

    // Access element (2D/3D)
    template<typename... Args>
    T& operator()(Args... args) {
        static_assert(sizeof...(Args) == 2 || sizeof...(Args) == 3, 
                      "Only 2D or 3D arrays supported for now");
        std::array<size_t, sizeof...(Args)> idx{static_cast<size_t>(args)...};
        size_t offset = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            assert(idx[i] < dims_[i]);
            offset += idx[i] * strides_[i];
        }
        return data_[offset];
    }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    size_t size() const { return data_.size(); }
};
