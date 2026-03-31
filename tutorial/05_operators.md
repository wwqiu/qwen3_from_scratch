# 第五章：核心算子（Operators）—— Transformer 的基本构件

在前面的章节中，我们实现了 Tensor（张量）和 Tokenizer（分词器），现在我们已经可以：
- 将文本转换为 Token IDs（Tokenizer）
- 用多维数组存储和操作数据（Tensor）

但是，要真正运行一个 Transformer 模型，我们还需要实现一些基本的**算子（Operators）**。

## 5.1 算子概述：Transformer 的基本构件

### 什么是算子？

**算子（Operator）** 就是神经网络中的基本计算单元，类似于数学中的函数。给定输入，算子执行某种计算，然后输出结果。

例如：
- **加法算子**：输入两个数，输出它们的和
- **矩阵乘法算子**：输入两个矩阵，输出它们的乘积
- **归一化算子**：输入一个向量，输出归一化后的向量

### 本章涉及的三个核心算子

在 Transformer 模型中，有三个最基础的算子：

| 算子 | 作用 | 输入 | 输出 |
|------|------|------|------|
| **Embedding** | 将 Token ID 转换为向量 | Token ID（整数） | 向量（浮点数数组） |
| **RMSNorm** | 归一化向量，让数值稳定 | 向量 | 归一化后的向量 |
| **Linear** | 线性变换（矩阵乘法） | 向量 | 变换后的向量 |

### 算子在推理流程中的位置

让我们看看这三个算子在 Qwen3 推理流程中的位置：

```
用户输入文本
    ↓
[Tokenizer] 分词
    ↓
Token IDs: [1, 2, 3, ...]
    ↓
[Embedding] 查表，转换为向量
    ↓
向量: [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
    ↓
[RMSNorm] 归一化
    ↓
归一化后的向量
    ↓
[Linear] 线性变换
    ↓
变换后的向量
    ↓
[Attention、FFN 等更复杂的模块...]
    ↓
最终输出
```

可以看到，这三个算子是 Transformer 的基础，后续所有复杂的模块都是基于它们构建的。

---

## 5.2 Embedding：从 Token ID 到向量

### 5.2.1 原理：查表操作

在第 4 章中，我们已经可以将文本转换为 Token IDs，例如：

```
"Hello" → [9906, 0]
```

但是，神经网络无法直接处理这些整数，它需要的是**浮点数向量**。这就是 Embedding 的作用。

**Embedding 的本质就是一个大表格**，表格的每一行对应一个 Token ID，每一行存储一个向量。

例如，假设我们的词表大小是 5（只有 5 个 Token），每个向量的维度是 3，那么 Embedding 矩阵就是一个 5×3 的表格：

```
Token ID    向量（3 维）
   0     →  [0.1, 0.2, 0.3]
   1     →  [0.4, 0.5, 0.6]
   2     →  [0.7, 0.8, 0.9]
   3     →  [1.0, 1.1, 1.2]
   4     →  [1.3, 1.4, 1.5]
```

**查表过程**：
- 输入 Token ID = 1
- 查表，找到第 1 行（从 0 开始计数）
- 输出向量 = [0.4, 0.5, 0.6]

就这么简单！Embedding 就是一个查表操作。

### 5.2.2 代码实现

现在让我们用 C++ 实现 Embedding。

**数据结构**：

```cpp
struct Embedding {
    Tensor weight;  // 形状: [vocab_size, hidden_dim]

    // 前向传播：给定 Token IDs，返回对应的向量
    Tensor Forward(const std::vector<int>& token_ids);
};
```

**Forward 函数实现**：

```cpp
Tensor Embedding::Forward(const std::vector<int>& token_ids) {
    int seq_len = token_ids.size();
    int hidden_dim = weight.shape[1];

    // 创建输出 Tensor，形状: [seq_len, hidden_dim]
    Tensor output({seq_len, hidden_dim});

    // 对每个 Token ID，查表取出对应的向量
    for (int i = 0; i < seq_len; i++) {
        int token_id = token_ids[i];

        // 从 weight 中取出第 token_id 行
        for (int j = 0; j < hidden_dim; j++) {
            output.data[i * hidden_dim + j] = weight.data[token_id * hidden_dim + j];
        }
    }

    return output;
}
```

**代码解释**：
1. 输入是一个 Token IDs 数组，例如 `[1, 2, 3]`
2. 输出是一个二维 Tensor，形状是 `[seq_len, hidden_dim]`
3. 对于每个 Token ID，我们从 `weight` 矩阵中取出对应的行
4. 就是这么简单，没有任何复杂的计算！

---

## 5.3 RMSNorm：归一化

### 5.3.1 原理：让数值稳定

在神经网络中，数值可能会变得很大或很小，导致计算不稳定。**归一化（Normalization）** 的作用就是让数值保持在一个合理的范围内。

**RMSNorm（Root Mean Square Normalization）** 是一种简化的归一化方法，它的计算步骤非常简单：

**步骤 1：计算均方根（RMS）**

给定一个向量 `x = [x₁, x₂, x₃, ..., xₙ]`，计算它的均方根：

```
RMS = sqrt((x₁² + x₂² + x₃² + ... + xₙ²) / n)
```

**步骤 2：归一化**

用每个元素除以 RMS：

```
x̂ᵢ = xᵢ / RMS
```

**步骤 3：缩放**

乘以可学习的权重 `wᵢ`：

```
yᵢ = wᵢ × x̂ᵢ
```

**手工计算示例**：

假设我们有一个向量 `x = [1.0, 2.0, 3.0]`，权重 `w = [0.5, 0.5, 0.5]`，`eps = 1e-6`（防止除零）。

**步骤 1：计算 RMS**

```
平方和 = 1.0² + 2.0² + 3.0² = 1 + 4 + 9 = 14
均值 = 14 / 3 = 4.6667
RMS = sqrt(4.6667 + 1e-6) ≈ 2.1602
```

**步骤 2：归一化**

```
x̂₁ = 1.0 / 2.1602 ≈ 0.4629
x̂₂ = 2.0 / 2.1602 ≈ 0.9258
x̂₃ = 3.0 / 2.1602 ≈ 1.3887
```

**步骤 3：缩放**

```
y₁ = 0.5 × 0.4629 ≈ 0.2315
y₂ = 0.5 × 0.9258 ≈ 0.4629
y₃ = 0.5 × 1.3887 ≈ 0.6944
```

最终输出：`y = [0.2315, 0.4629, 0.6944]`

### 5.3.2 代码实现

现在让我们用 C++ 实现 RMSNorm。

**数据结构**：

```cpp
struct RMSNorm {
    Tensor weight;  // 形状: [hidden_dim]
    float eps;      // 防止除零，通常是 1e-6

    // 前向传播：归一化输入向量
    Tensor Forward(const Tensor& input);
};
```

**Forward 函数实现**：

```cpp
Tensor RMSNorm::Forward(const Tensor& input) {
    int seq_len = input.shape[0];
    int hidden_dim = input.shape[1];

    // 创建输出 Tensor，形状与输入相同
    Tensor output(input.shape);

    // 对每个位置的向量进行归一化
    for (int i = 0; i < seq_len; i++) {
        // 步骤 1：计算均方根（RMS）
        float sum_squares = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            float val = input.data[i * hidden_dim + j];
            sum_squares += val * val;
        }
        float rms = std::sqrt(sum_squares / hidden_dim + eps);

        // 步骤 2 & 3：归一化并缩放
        for (int j = 0; j < hidden_dim; j++) {
            float val = input.data[i * hidden_dim + j];
            float normalized = val / rms;
            output.data[i * hidden_dim + j] = weight.data[j] * normalized;
        }
    }

    return output;
}
```

**代码解释**：
1. 对输入的每个位置（每一行）进行归一化
2. 先计算该位置向量的均方根（RMS）
3. 然后用每个元素除以 RMS，再乘以对应的权重
4. 就是按照数学公式一步步计算，没有任何技巧！

---

## 5.4 Linear：矩阵乘法

### 5.4.1 原理：线性变换

**Linear（线性层）** 的作用是对向量进行线性变换，本质上就是**矩阵乘法**。

假设我们有一个输入向量 `x`（维度是 3），我们想把它变换成一个新的向量 `y`（维度是 2）。我们需要一个权重矩阵 `W`（形状是 2×3）：

```
        [w₁₁  w₁₂  w₁₃]
W =     [w₂₁  w₂₂  w₂₃]

x = [x₁, x₂, x₃]

y = W × x = [w₁₁×x₁ + w₁₂×x₂ + w₁₃×x₃,
             w₂₁×x₁ + w₂₂×x₂ + w₂₃×x₃]
```

**手工计算示例**：

假设：
```
W = [1.0  2.0  3.0]
    [4.0  5.0  6.0]

x = [1.0, 2.0, 3.0]

bias = [0.1, 0.2]
```

计算 `y = W × x + bias`：

**步骤 1：矩阵乘法**

```
y₁ = 1.0×1.0 + 2.0×2.0 + 3.0×3.0 = 1.0 + 4.0 + 9.0 = 14.0
y₂ = 4.0×1.0 + 5.0×2.0 + 6.0×3.0 = 4.0 + 10.0 + 18.0 = 32.0
```

**步骤 2：加上偏置**

```
y₁ = 14.0 + 0.1 = 14.1
y₂ = 32.0 + 0.2 = 32.2
```

最终输出：`y = [14.1, 32.2]`

### 5.4.2 代码实现

现在让我们用 C++ 实现 Linear。

**数据结构**：

```cpp
struct Linear {
    Tensor weight;  // 形状: [out_dim, in_dim]
    Tensor bias;    // 形状: [out_dim]（可选）

    // 前向传播：线性变换
    Tensor Forward(const Tensor& input);
};
```

**Forward 函数实现**：

```cpp
Tensor Linear::Forward(const Tensor& input) {
    int seq_len = input.shape[0];
    int in_dim = input.shape[1];
    int out_dim = weight.shape[0];

    // 创建输出 Tensor，形状: [seq_len, out_dim]
    Tensor output({seq_len, out_dim});

    // 对每个位置进行矩阵乘法
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < out_dim; j++) {
            float sum = 0.0f;

            // 计算点积：weight[j] · input[i]
            for (int k = 0; k < in_dim; k++) {
                sum += weight.data[j * in_dim + k] * input.data[i * in_dim + k];
            }

            // 加上偏置（如果有的话）
            if (bias.data.size() > 0) {
                sum += bias.data[j];
            }

            output.data[i * out_dim + j] = sum;
        }
    }

    return output;
}
```

**代码解释**：
1. 三层循环：外层遍历序列位置，中层遍历输出维度，内层计算点积
2. 对于每个输出位置 `(i, j)`，计算 `weight[j]` 和 `input[i]` 的点积
3. 如果有偏置，就加上 `bias[j]`
4. 就是最朴素的矩阵乘法实现，没有任何优化！

---

## 5.5 完整代码

现在让我们把三个算子的代码整合到一起。

### 5.5.1 头文件：operators.h

```cpp
#pragma once

#include "tensor.h"
#include <vector>
#include <cmath>

// Embedding：将 Token ID 转换为向量
struct Embedding {
    Tensor weight;  // 形状: [vocab_size, hidden_dim]

    Tensor Forward(const std::vector<int>& token_ids);
};

// RMSNorm：归一化
struct RMSNorm {
    Tensor weight;  // 形状: [hidden_dim]
    float eps;      // 防止除零，通常是 1e-6

    RMSNorm() : eps(1e-6f) {}

    Tensor Forward(const Tensor& input);
};

// Linear：线性变换（矩阵乘法）
struct Linear {
    Tensor weight;  // 形状: [out_dim, in_dim]
    Tensor bias;    // 形状: [out_dim]（可选）

    Tensor Forward(const Tensor& input);
};
```

### 5.5.2 实现文件：operators.cpp

```cpp
#include "operators.h"

// ============================================================
// Embedding 实现
// ============================================================

Tensor Embedding::Forward(const std::vector<int>& token_ids) {
    int seq_len = token_ids.size();
    int hidden_dim = weight.shape[1];

    // 创建输出 Tensor，形状: [seq_len, hidden_dim]
    Tensor output({seq_len, hidden_dim});

    // 对每个 Token ID，查表取出对应的向量
    for (int i = 0; i < seq_len; i++) {
        int token_id = token_ids[i];

        // 从 weight 中取出第 token_id 行
        for (int j = 0; j < hidden_dim; j++) {
            output.data[i * hidden_dim + j] = weight.data[token_id * hidden_dim + j];
        }
    }

    return output;
}

// ============================================================
// RMSNorm 实现
// ============================================================

Tensor RMSNorm::Forward(const Tensor& input) {
    int seq_len = input.shape[0];
    int hidden_dim = input.shape[1];

    // 创建输出 Tensor，形状与输入相同
    Tensor output(input.shape);

    // 对每个位置的向量进行归一化
    for (int i = 0; i < seq_len; i++) {
        // 步骤 1：计算均方根（RMS）
        float sum_squares = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            float val = input.data[i * hidden_dim + j];
            sum_squares += val * val;
        }
        float rms = std::sqrt(sum_squares / hidden_dim + eps);

        // 步骤 2 & 3：归一化并缩放
        for (int j = 0; j < hidden_dim; j++) {
            float val = input.data[i * hidden_dim + j];
            float normalized = val / rms;
            output.data[i * hidden_dim + j] = weight.data[j] * normalized;
        }
    }

    return output;
}

// ============================================================
// Linear 实现
// ============================================================

Tensor Linear::Forward(const Tensor& input) {
    int seq_len = input.shape[0];
    int in_dim = input.shape[1];
    int out_dim = weight.shape[0];

    // 创建输出 Tensor，形状: [seq_len, out_dim]
    Tensor output({seq_len, out_dim});

    // 对每个位置进行矩阵乘法
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < out_dim; j++) {
            float sum = 0.0f;

            // 计算点积：weight[j] · input[i]
            for (int k = 0; k < in_dim; k++) {
                sum += weight.data[j * in_dim + k] * input.data[i * in_dim + k];
            }

            // 加上偏置（如果有的话）
            if (bias.data.size() > 0) {
                sum += bias.data[j];
            }

            output.data[i * out_dim + j] = sum;
        }
    }

    return output;
}
```

**代码总结**：
- 三个算子的实现总共不到 100 行代码
- 每个算子都只有一个 Forward 函数
- 代码逻辑清晰，直接按照数学公式实现
- 没有任何复杂的优化或封装

---

## 5.6 测试与验证

现在让我们编写测试程序，验证三个算子是否正确实现。

### 5.6.1 创建测试程序

创建文件 `test_operators.cpp`：

```cpp
#include "operators.h"
#include <iostream>
#include <iomanip>

void TestEmbedding() {
    std::cout << "=== 测试 Embedding ===" << std::endl;

    // 创建一个简单的 Embedding 矩阵：5 个 Token，每个 3 维
    Embedding emb;
    emb.weight = Tensor({5, 3});
    
    // 手工填充权重
    float weights[] = {
        0.1f, 0.2f, 0.3f,  // Token 0
        0.4f, 0.5f, 0.6f,  // Token 1
        0.7f, 0.8f, 0.9f,  // Token 2
        1.0f, 1.1f, 1.2f,  // Token 3
        1.3f, 1.4f, 1.5f   // Token 4
    };
    for (int i = 0; i < 15; i++) {
        emb.weight.data[i] = weights[i];
    }

    // 测试：输入 Token IDs [1, 2, 3]
    std::vector<int> token_ids = {1, 2, 3};
    Tensor output = emb.Forward(token_ids);

    // 打印输出
    std::cout << "输入 Token IDs: [1, 2, 3]" << std::endl;
    std::cout << "输出向量:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  Token " << token_ids[i] << ": [";
        for (int j = 0; j < 3; j++) {
            std::cout << std::fixed << std::setprecision(1) 
                      << output.data[i * 3 + j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

void TestRMSNorm() {
    std::cout << "=== 测试 RMSNorm ===" << std::endl;

    // 创建 RMSNorm
    RMSNorm norm;
    norm.weight = Tensor({3});
    norm.weight.data = {0.5f, 0.5f, 0.5f};  // 权重全为 0.5
    norm.eps = 1e-6f;

    // 创建输入：1 个位置，3 维向量 [1.0, 2.0, 3.0]
    Tensor input({1, 3});
    input.data = {1.0f, 2.0f, 3.0f};

    // 前向传播
    Tensor output = norm.Forward(input);

    // 打印输出
    std::cout << "输入向量: [1.0, 2.0, 3.0]" << std::endl;
    std::cout << "权重: [0.5, 0.5, 0.5]" << std::endl;
    std::cout << "输出向量: [";
    for (int i = 0; i < 3; i++) {
        std::cout << std::fixed << std::setprecision(4) << output.data[i];
        if (i < 2) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "（预期约为: [0.2315, 0.4629, 0.6944]）" << std::endl;
    std::cout << std::endl;
}

void TestLinear() {
    std::cout << "=== 测试 Linear ===" << std::endl;

    // 创建 Linear：输入 3 维，输出 2 维
    Linear linear;
    linear.weight = Tensor({2, 3});
    linear.bias = Tensor({2});

    // 手工填充权重和偏置
    linear.weight.data = {
        1.0f, 2.0f, 3.0f,  // 第 0 行
        4.0f, 5.0f, 6.0f   // 第 1 行
    };
    linear.bias.data = {0.1f, 0.2f};

    // 创建输入：1 个位置，3 维向量 [1.0, 2.0, 3.0]
    Tensor input({1, 3});
    input.data = {1.0f, 2.0f, 3.0f};

    // 前向传播
    Tensor output = linear.Forward(input);

    // 打印输出
    std::cout << "输入向量: [1.0, 2.0, 3.0]" << std::endl;
    std::cout << "权重矩阵:" << std::endl;
    std::cout << "  [1.0, 2.0, 3.0]" << std::endl;
    std::cout << "  [4.0, 5.0, 6.0]" << std::endl;
    std::cout << "偏置: [0.1, 0.2]" << std::endl;
    std::cout << "输出向量: [";
    for (int i = 0; i < 2; i++) {
        std::cout << std::fixed << std::setprecision(1) << output.data[i];
        if (i < 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "（预期: [14.1, 32.2]）" << std::endl;
    std::cout << std::endl;
}

int main() {
    TestEmbedding();
    TestRMSNorm();
    TestLinear();

    std::cout << "所有测试完成！" << std::endl;
    return 0;
}
```

### 5.6.2 更新 CMakeLists.txt

在 `CMakeLists.txt` 中添加测试目标：

```cmake
# 测试算子
add_executable(test_operators test_operators.cpp operators.cpp tensor.cpp)
```

### 5.6.3 编译并运行测试

```bash
# 编译
mkdir -p build
cd build
cmake ..
make test_operators

# 运行测试
./test_operators
```

**预期输出**：

```
=== 测试 Embedding ===
输入 Token IDs: [1, 2, 3]
输出向量:
  Token 1: [0.4, 0.5, 0.6]
  Token 2: [0.7, 0.8, 0.9]
  Token 3: [1.0, 1.1, 1.2]

=== 测试 RMSNorm ===
输入向量: [1.0, 2.0, 3.0]
权重: [0.5, 0.5, 0.5]
输出向量: [0.2315, 0.4629, 0.6944]
（预期约为: [0.2315, 0.4629, 0.6944]）

=== 测试 Linear ===
输入向量: [1.0, 2.0, 3.0]
权重矩阵:
  [1.0, 2.0, 3.0]
  [4.0, 5.0, 6.0]
偏置: [0.1, 0.2]
输出向量: [14.1, 32.2]
（预期: [14.1, 32.2]）

所有测试完成！
```

---

## 5.7 小结

在本章中，我们实现了三个核心算子：

1. **Embedding**：将 Token ID 转换为向量，本质是查表操作
2. **RMSNorm**：归一化向量，让数值稳定，计算均方根并缩放
3. **Linear**：线性变换，本质是矩阵乘法

这三个算子是 Transformer 的基础构件，后续的复杂模块（如 Attention、FFN）都是基于它们构建的。

**下一章预告**：

在第 6 章中，我们将实现 **RoPE 位置编码**，它是 Transformer 中用于表示位置信息的关键技术。
