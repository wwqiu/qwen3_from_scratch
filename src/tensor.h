#pragma once


class Tensor {
public:
    Tensor() : data_(nullptr) {}
    ~Tensor() {
        if (data_) {
            delete[] data_;
            data_ = nullptr;
        }
    }

    Tensor (std::vector<int64_t> shape, size_t elem_size) : shape_{shape}, elem_size_(elem_size) {
        size_t total_size = elem_size;
        for (int64_t dim : shape_) {
            total_size *= dim;
        }
        data_ = new uint8_t[total_size];
        if (!data_) {
            throw std::runtime_error("Failed to allocate memory for tensor data.");
        }
        // printf("Allocated tensor with element size %zu bytes (total size: %zu bytes)\n", elem_size, total_size);
    }

    Tensor (int n, int c, int h, int w, int elem_size) : shape_{n, c, h, w}, elem_size_(elem_size) {
        data_ = new uint8_t[n * c * h * w * elem_size];
    }

    // Add move constructor and move assignment operator for efficient resource management
    Tensor(Tensor&& other) noexcept : shape_(std::move(other.shape_)), data_(other.data_), elem_size_(other.elem_size_) {
        other.data_ = nullptr; // Prevent double deletion
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            shape_ = std::move(other.shape_);
            data_ = other.data_;
            elem_size_ = other.elem_size_;
            other.data_ = nullptr;
        }
        return *this;
    }

    Tensor clone() const {
        Tensor copy(shape_, elem_size_);
        size_t total_size = elem_size_;
        for (int64_t dim : shape_) {
            total_size *= dim;
        }
        memcpy(copy.data_, data_, total_size);
        return copy;
    }

    std::vector<int64_t> shape_;
    uint8_t* data_;
    size_t elem_size_;
};