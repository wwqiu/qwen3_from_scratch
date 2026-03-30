# 第 3 章：张量（Tensor）—— 推理引擎的数据容器

在上一章，我们搭建好了开发环境。从这一章开始，我们要正式动手写推理引擎的第一个核心模块：**张量（Tensor）**。

张量是整个引擎的"血液"——模型的权重是张量，每一层的输入输出是张量，注意力分数是张量。理解了张量，你就理解了数据在引擎里流动的方式。

---

## 3.1 张量是什么？

### 3.1.1 从向量到张量

你可能已经熟悉这些概念：

| 名称 | 维度 | 例子 | 在推理中的用途 |
|------|------|------|----------------|
| 标量 | 0 维 | `3.14` | 温度参数、学习率 |
| 向量 | 1 维 | `[1.0, 2.0, 3.0]` | 单个 token 的 embedding |
| 矩阵 | 2 维 | shape = `[行, 列]` | 权重矩阵、attention 分数 |
| 张量 | N 维 | shape = `[d0, d1, ..., dN]` | 批量数据、多头注意力 |

张量就是对以上所有概念的统一抽象。

### 3.1.2 张量在 Qwen3 推理中的实际应用

在我们的推理引擎里，张量无处不在：

**1. Embedding 权重表**
```
shape: [151936, 1024]
含义：词表有 151936 个 token，每个 token 对应一个 896 维的向量
用途：把 token ID 转换成向量
```

**2. 输入序列**
```
shape: [seq_len, 1024]
含义：输入了 seq_len 个 token，每个 token 是 896 维向量
用途：送入 Transformer 的数据
```

**3. Attention 权重矩阵**
```
shape: [1024, 1024]
含义：Query/Key/Value 的线性变换矩阵
用途：计算注意力
```

**4. 多头注意力的中间结果**
```
shape: [seq_len, num_heads, head_dim]
含义：16 个注意力头，每个头处理 128 维的数据
用途：并行计算多个注意力视角
```

---

## 3.2 内存布局：多维数组如何存储？

在实现 Tensor 类之前，我们需要先搞清楚一件事：**多维数组在内存里是怎么存的？**

这直接关系到我们后续如何高效地访问张量数据。

### 3.2.1 内存是一条线

计算机的内存本质上是一个**一维的字节数组**。不管你的数据是几维的，最终都要"压平"成一条线存进去。

```
地址   0x1000   0x1004   0x1008   0x100C   0x1010   0x1014
     ┌────────┬────────┬────────┬────────┬────────┬────────┐
     │  1.0   │  2.0   │  3.0   │  4.0   │  5.0   │  6.0   │
     └────────┴────────┴────────┴────────┴────────┴────────┘
       [0]      [1]      [2]      [3]      [4]      [5]
       ↑                 ↑
      ptr             ptr + 2（跳过 2×4=8 字节）
```

用 C++ 代码来看：

```cpp
float arr[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
float* ptr = arr;

// 以下两种写法完全等价
float val1 = arr[2];      // 传统下标访问
float val2 = *(ptr + 2);  // 指针偏移访问
```

`ptr + 2` 的含义是：从 `ptr` 指向的地址出发，向后移动 `2 * sizeof(float)` = 8 个字节。

### 3.2.2 二维数组：行优先存储

现在把这 6 个数想象成一个 2 行 3 列的矩阵：

```
逻辑视图（2×3 矩阵）：
[1.0, 2.0, 3.0]  ← 第 0 行
[4.0, 5.0, 6.0]  ← 第 1 行

内存视图（行优先存储）：
索引    0     1     2     3     4     5
     ┌─────┬─────┬─────┬─────┬─────┬─────┐
     │ 1.0 │ 2.0 │ 3.0 │ 4.0 │ 5.0 │ 6.0 │
     └─────┴─────┴─────┴─────┴─────┴─────┘
      └─── 第 0 行 ───┘ └─── 第 1 行 ───┘
```

**行优先（Row-major）存储**：先把第 0 行存完，再存第 1 行。

要访问元素 `[i][j]`，偏移量的计算公式是：

```
offset = i * cols + j
```

**推理中的实际应用：Embedding 查表**

```cpp
// embedding_weight.shape() = [151936, 896]
// 取出 token_id 对应的 embedding 向量
size_t hidden_size = 896;
float* embed_vec = embedding_weight.data<float>() + token_id * hidden_size;
// embed_vec 现在指向第 token_id 行的起始位置，长度为 hidden_size
```

### 3.2.3 三维数组：推广到多维

在 Attention 计算中，我们经常处理 3D 张量，比如 `[seq_len, num_heads, head_dim]`。

访问元素 `[i][j][k]` 的偏移量：

```
offset = i * (num_heads * head_dim) + j * head_dim + k
```

**推理中的实际应用：多头注意力**

```cpp
// qkv.shape() = [seq_len, num_heads, head_dim]
size_t num_heads = 14, head_dim = 64;

// 取出位置 pos、第 h 个 head 的 query 向量
float* q_vec = qkv.data<float>() + pos * (num_heads * head_dim) + h * head_dim;
```

内存布局可视化：

```
逻辑视图（3D）：shape = [2, 3, 4]
seq=0 ┌─────────────────┐
      │ head=0: [0][1][2][3]
      │ head=1: [4][5][6][7]
      │ head=2: [8][9][A][B]
      └─────────────────┘
seq=1 ┌─────────────────┐
      │ head=0: [C][D][E][F]
      │ head=1: [G][H][I][J]
      │ head=2: [K][L][M][N]
      └─────────────────┘

内存视图（1D）：
[0][1][2][3][4][5][6][7][8][9][A][B][C][D][E][F][G][H][I][J][K][L][M][N]
 └─ seq=0─┘                          └─ seq=1─┘
    head=0                              head=0
```

---

## 3.3 Tensor 类的设计

### 3.3.1 设计目标

我们的 Tensor 要做到：
- **任意维度**：1D 向量、2D 矩阵、3D/4D 张量都能表示
- **类型无关**：底层存字节，用的时候再转换成 `float`、`int` 等
- **轻量级**：不依赖任何第三方库
- **共享内存**：多个 Tensor 可以指向同一块数据，避免不必要的拷贝

### 3.3.2 核心数据成员

```cpp
class Tensor {
private:
    std::vector<size_t> shape_;           // 各维度大小，如 {28, 896, 128}
    std::shared_ptr<uint8_t[]> data_;     // 底层字节缓冲区
    size_t elem_size_;                    // 单个元素的字节数，float 是 4
};
```

**为什么用 `uint8_t*` 作为底层存储？**

因为我们需要支持多种数据类型（`float`、`int`、`uint8_t`），用字节作为最小单位，可以在使用时再转换成目标类型：

```cpp
// 底层是字节数组
uint8_t* raw = data_.get();

// 用的时候转换成 float*
float* fptr = reinterpret_cast<float*>(raw);
fptr[0] = 3.14f;
```

**为什么用 `shared_ptr`？**

`shared_ptr` 会自动追踪有多少个对象在共享同一块内存，当最后一个持有者销毁时，内存才会被释放。

```
浅拷贝（共享内存）：
Tensor A ──┐
           ├──→ [内存块]（引用计数=2）
Tensor B ──┘

深拷贝（独立内存）：
Tensor A ──→ [内存块 1]
Tensor B ──→ [内存块 2]
```

---

## 3.4 实现 Tensor 类

### 3.4.1 构造函数

最核心的构造函数接受 `shape` 和 `elem_size`，计算总字节数并分配内存：

```cpp
Tensor(std::vector<size_t> shape, size_t elem_size)
    : shape_(shape), elem_size_(elem_size) {
    size_t total_bytes = elem_size;
    for (size_t dim : shape_) {
        total_bytes *= dim;
    }
    data_ = std::shared_ptr<uint8_t[]>(new uint8_t[total_bytes]);
}
```

使用示例：

```cpp
// 创建一个 [10, 896] 的 float 张量
Tensor t({10, 896}, sizeof(float));
// 分配 10 * 896 * 4 = 35840 字节
```

### 3.4.2 拷贝与移动

**浅拷贝（默认行为）**：共享 `shared_ptr`，不复制数据。

```cpp
Tensor(const Tensor&) = default;
Tensor& operator=(const Tensor&) = default;
```

**移动构造**：把另一个 Tensor 的资源"偷"过来。

```cpp
Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      data_(other.data_),
      elem_size_(other.elem_size_) {
    other.data_ = nullptr;
}
```

**深拷贝 `clone()`**：分配新内存，用 `memcpy` 复制数据。

```cpp
Tensor clone() const {
    Tensor copy(shape_, elem_size_);
    size_t total_bytes = elem_size_;
    for (size_t dim : shape_) total_bytes *= dim;
    memcpy(copy.data_.get(), data_.get(), total_bytes);
    return copy;
}
```

### 3.4.3 数据访问接口

`data<T>()` 是使用最频繁的接口，它把底层的 `uint8_t*` 转换成目标类型的指针：

```cpp
template <typename T>
T* data() {
    return reinterpret_cast<T*>(data_.get());
}
```

**推理中的实际使用**：

```cpp
// 访问 Embedding 表的第 token_id 行
float* row = embedding.data<float>() + token_id * hidden_size;

// 访问 Attention 的第 h 个 head
float* head = qkv.data<float>() + pos * (num_heads * head_dim) + h * head_dim;
```

### 3.4.4 运算符重载

`operator+` 实现逐元素加法，在 Transformer 的残差连接（Residual Connection）中会用到：

```cpp
friend Tensor operator+(Tensor& a, Tensor& b) {
    Tensor result(a.shape_, a.elem_size_);
    size_t num_elements = 1;
    for (size_t dim : a.shape_) num_elements *= dim;

    float* a_data = a.data<float>();
    float* b_data = b.data<float>();
    float* out    = result.data<float>();
    for (size_t i = 0; i < num_elements; ++i) {
        out[i] = a_data[i] + b_data[i];
    }
    return result;
}
```

**推理中的实际应用**：

```cpp
// Transformer Block 的残差连接
Tensor output = attention_output + input;  // 逐元素相加
```

---

## 3.5 完整代码

完整的 `src/tensor.h` 代码见：[tensor.h 链接待补充]

---

## 3.6 测试

### 3.6.1 目录结构

在项目根目录新建 `test/` 目录，每个模块一个测试文件：

```
qwen3.cpp/
├── CMakeLists.txt
├── src/
│   └── tensor.h
└── test/
    ├── test_environment.cpp   # 验证第 2 章的环境配置
    └── test_tensor.cpp        # 验证本章的 Tensor 实现
```

### 3.6.2 更新 CMakeLists.txt

在 `CMakeLists.txt` 中添加两个测试目标：

```cmake
# 环境测试
add_executable(test_environment test/test_environment.cpp)
target_include_directories(test_environment PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_environment PRIVATE Boost::regex)

# Tensor 测试
add_executable(test_tensor test/test_tensor.cpp)
target_include_directories(test_tensor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
```

### 3.6.3 测试用例示例

下面是几个简单的测试用例，帮助你验证 Tensor 的实现。

**测试：基本构造和访问**

```cpp
#include <iostream>
#include "tensor.h"

int main() {
    // 创建一个 [3, 4] 的 float 张量
    Tensor t({3, 4}, sizeof(float));

    // 验证 shape
    std::cout << "Shape: [" << t.shape()[0] << ", " << t.shape()[1] << "]" << std::endl;

    // 写入数据
    float* data = t.data<float>();
    for (int i = 0; i < 12; ++i) {
        data[i] = i * 1.0f;
    }

    // 读取第 1 行（索引从 0 开始）
    float* row1 = t.data<float>() + 1 * 4;  // offset = 1 * cols
    std::cout << "Row 1: [" << row1[0] << ", " << row1[1] << ", "
              << row1[2] << ", " << row1[3] << "]" << std::endl;
    // 预期输出：Row 1: [4, 5, 6, 7]

    return 0;
}
```

完整的 `test/test_tensor.cpp` 代码见：[test_tensor.cpp 链接待补充]

编译并运行：

```bash
cd build
cmake ..
make test_tensor
./test_tensor
```

---

## 3.7 小结

本章我们实现了一个轻量级的 `Tensor` 类，它是整个推理引擎的数据基础。

**核心要点**：
- 张量在内存里是连续存储的，通过 `offset = i * stride[0] + j * stride[1] + ...` 访问任意元素
- 用 `shared_ptr<uint8_t[]>` 管理内存，用 `reinterpret_cast` 在字节和具体类型之间转换
- 在推理中，最常见的操作是：从 2D 权重矩阵取某一行，从 3D 张量取某个 head 的数据

下一章，我们来实现**分词引擎（Tokenizer）**——把用户输入的文字，变成模型能理解的 token ID 序列。
