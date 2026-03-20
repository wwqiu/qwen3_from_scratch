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

    friend Tensor operator+(const Tensor& a, const Tensor& b) {
        if (a.shape_ != b.shape_ || a.elem_size_ != b.elem_size_) {
            throw std::invalid_argument("Tensors must have the same shape and element size for addition.");
        }
        Tensor result(a.shape_, a.elem_size_);
        size_t total_size = a.elem_size_;
        for (int64_t dim : a.shape_) {
            total_size *= dim;
        }
        for (size_t i = 0; i < total_size / a.elem_size_; ++i) {
            float* a_ptr = (float*)(a.data_ + i * a.elem_size_);
            float* b_ptr = (float*)(b.data_ + i * b.elem_size_);
            float* res_ptr = (float*)(result.data_ + i * result.elem_size_);
            *res_ptr = *a_ptr + *b_ptr;
        }
        return result;
    }

    std::vector<int64_t> shape_;
    uint8_t* data_;
    size_t elem_size_;
};