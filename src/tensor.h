#pragma once
/*
 * ============================================================================
 * Tensor - 多维张量基础类
 * ============================================================================
 *
 * 目录：
 *   1. 构造 / 析构
 *      - Tensor(shape, elem_size)   - 按 shape 分配内存
 *      - Tensor(n, c, h, w, ...)    - 4D 快捷构造
 *      - clone()                    - 深拷贝
 *      - Tensor(const Tensor&)      - 拷贝构造（共享底层数据）
 *      - operator=(const Tensor&)   - 拷贝赋值
 *      - Tensor(Tensor&&)           - 移动构造
 *      - operator=(Tensor&&)        - 移动赋值
 *   2. 数据访问
 *      - data<T>()                  - 返回类型化裸指针
 *      - shape()                    - 返回维度列表
 *      - elem_size()                - 返回单元素字节数
 *   3. 运算符重载
 *      - operator+                  - 逐元素加法（相同 shape）
 */

class Tensor {
   public:
    Tensor() : data_(nullptr) {}
    ~Tensor() {}

    Tensor(std::vector<size_t> shape, size_t elem_size) : shape_{shape}, elem_size_(elem_size) {
        size_t total_size = elem_size;
        for (size_t dim : shape_) {
            total_size *= dim;
        }
        data_ = std::shared_ptr<uint8_t[]>(new uint8_t[total_size]);
        if (!data_) {
            throw std::runtime_error("Failed to allocate memory for tensor data.");
        }
    }

    Tensor(size_t n, size_t c, size_t h, size_t w, size_t elem_size) : shape_{n, c, h, w}, elem_size_(elem_size) {
        data_ = std::shared_ptr<uint8_t[]>(new uint8_t[n * c * h * w * elem_size]);
    }

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_)), data_(other.data_), elem_size_(other.elem_size_) {
        other.data_ = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
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
        for (size_t dim : shape_) {
            total_size *= dim;
        }
        memcpy(copy.data_.get(), data_.get(), total_size);
        return copy;
    }

    template <typename T>
    T* data() {
        return reinterpret_cast<T*>(data_.get());
    }

    const std::vector<size_t>& shape() const { return shape_; }

    size_t elem_size() const { return elem_size_; }

    friend Tensor operator+(Tensor& a, Tensor& b) {
        if (a.shape_ != b.shape_ || a.elem_size_ != b.elem_size_) {
            throw std::invalid_argument("Tensors must have the same shape and element size for addition.");
        }
        Tensor result(a.shape_, a.elem_size_);
        size_t num_elements = 1;
        for (size_t dim : a.shape_) {
            num_elements *= dim;
        }

        const float* a_data = a.data<float>();
        const float* b_data = b.data<float>();
        float* out_data = result.data<float>();
        for (size_t i = 0; i < num_elements; ++i) {
            out_data[i] = a_data[i] + b_data[i];
        }
        return result;
    }

   private:
    std::vector<size_t> shape_;
    std::shared_ptr<uint8_t[]> data_;
    size_t elem_size_;
};
