# Qwen3.cpp

用纯 C++（无第三方推理框架依赖）从零实现 Qwen3 语言模型的完整推理流程。

> 个人学习项目，目标是深入理解 LLM 推理引擎的底层原理。

## 功能

- BPE Tokenizer（含 byte-level 解码及特殊 token 处理）
- RMSNorm、GQA Multi-Head Attention、SwiGLU MLP、RoPE 位置编码
- KV Cache 增量推理（显著提升多轮对话性能）
- safetensors 权重加载（支持 BFloat16 → Float32 转换）
- Greedy 采样 + 文本解码
- 完整前向传播：Embedding → N × Decoder → Final Norm → LM Head → Softmax → Sampler
- 交互式聊天Demo

## 构建

**依赖**

- C++17
- CMake >= 3.15
- Boost.Regex（预分词用）

```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake libboost-regex-dev
```

**编译**

```bash
mkdir build && cd build
cmake ..
make -j4
```

## 运行

**单次推理模式**

```bash
./qwen3 /path/to/tokenizer.json /path/to/model.safetensors
```

**聊天模式**

```bash
./qwen3_chat /path/to/tokenizer.json /path/to/model.safetensors [max_new_tokens]
```

支持命令：
- `/clear` - 清空对话历史
- `/exit` 或 `quit` - 退出程序

模型文件从 Hugging Face 下载 Qwen3 系列（如 Qwen3-0.6B）即可。

## 运行截图

![运行效果图（Qwen3-0.6B)](docs/image/test_demo.png)

## 项目结构

```
qwen3_from_scratch/
├── src/
│   ├── main.cpp          # 单次推理入口
│   ├── qwen3_chat.cpp    # 交互式聊天入口（支持 KV cache）
│   ├── qwen3.h/cpp       # 模型主体
│   ├── operator.hpp      # 算子实现（Attention/MLP/RMSNorm/KVCache/Sampler 等）
│   ├── tokenizer.h/cpp   # BPE Tokenizer + Decode + 特殊 token 处理
│   ├── tensor.h          # Tensor 数据结构（shared_ptr 内存管理）
│   ├── type.h            # 基础类型定义
│   └── logger.h          # 日志工具
├── docs/
│   ├── architecture.md   # 架构说明
│   └── operators.md      # 算子说明
├── thirdparty/
│   └── nlohmann/         # JSON 解析库
└── CMakeLists.txt
```

## 架构文档
- **[architecture.md](docs/architecture.md)** - 架构说明
- **[operators.md](docs/operators.md)** - 算子说明
