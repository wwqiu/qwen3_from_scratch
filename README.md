# Qwen3.cpp

Qwen3 语言模型的 C++ 实现教程项目（循序渐进，从基础到优化）

## 📚 学习路线

本项目采用**分阶段**的方式，帮助你从零开始理解和实现 Transformer 模型：

### 🎯 第一阶段：基础实现（从这里开始）
- 📖 [实现指南](IMPLEMENTATION_GUIDE.md) - 学习路线图
- 📖 [基础架构](BASIC_ARCHITECTURE.md) - 不带优化的完整实现
- 🎯 目标：理解 Transformer 原理
- ✅ 特点：代码简单、易于调试

### 🚀 第二阶段：性能优化
- 📖 [KV Cache 优化](KVCACHE_ARCHITECTURE.md) - 加速 50 倍
- 🎯 目标：提升推理性能
- ⚡ 效果：O(N²) → O(N)

## 快速开始

### 1. 安装依赖

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libboost-regex-dev
```

### 2. 编译

#### 使用 Makefile

```bash
make
```

#### 使用 CMake

```bash
mkdir build
cd build
cmake ..
make
```

### 3. 运行

```bash
./qwen3 [tokenizer.json] [model.safetensors]
```

## 项目结构

```
qwen3.cpp/
├── 📖 文档
│   ├── IMPLEMENTATION_GUIDE.md    # 学习路线图
│   ├── BASIC_ARCHITECTURE.md      # 基础架构（无优化）
│   └── KVCACHE_ARCHITECTURE.md    # KV Cache 优化
├── 💻 代码
│   ├── main.cpp                   # 主程序
│   ├── tokenizer.h/cpp            # Tokenizer 实现
│   ├── type.h                     # 基础数据结构
│   └── nlohmann/                  # JSON 库
└── 🔧 构建
    ├── Makefile                   # Make 构建
    └── CMakeLists.txt             # CMake 构建
```

## 核心特性

- ✅ **BPE Tokenizer**：完整的 Byte-Pair Encoding 实现
- 🚧 **Transformer**：多层 Transformer 架构（待实现）
  - RMSNorm 归一化
  - Multi-Head Attention (GQA)
  - SwiGLU 激活函数
  - RoPE 位置编码
- 🚧 **KV Cache**：推理加速优化（待实现）
- 🚧 **采样策略**：Temperature、Top-P（待实现）

## 学习建议

1. **先读文档**：[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
2. **理解基础**：[BASIC_ARCHITECTURE.md](BASIC_ARCHITECTURE.md)
3. **动手实现**：按照文档逐步编写代码
4. **测试验证**：确保每个模块正确
5. **性能优化**：[KVCACHE_ARCHITECTURE.md](KVCACHE_ARCHITECTURE.md)

## 依赖说明

- **C++17**：标准库特性
- **Boost.Regex**：正则表达式（预分词）
- **nlohmann/json**：JSON 解析（已包含）

## 清理

```bash
make clean
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可

MIT License

