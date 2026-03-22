# Qwen3.cpp Architecture

## 1. 项目目标
`qwen3_from_scratch` 的核心目标是：用纯 C++17 实现一个可读、可跑通的 Qwen3 推理最小系统，帮助理解 LLM 从输入文本到下一个 token 输出的全过程。


## 2. 模块划分

### 2.1 `src/tensor.h`
职责：定义最基础的 `Tensor` 容器。
- 保存 `shape_`、`data_`、`elem_size_`
- 提供 `clone()`
- 提供逐元素加法 `operator+`

### 2.2 `src/tokenizer.h/.cpp`
职责：完成文本与 token id 的双向转换。
- `LoadConfig`：加载 tokenizer 配置
- `Encode`：文本 -> token IDs
- `Decode`：token IDs -> 文本

### 2.3 `src/operator.hpp`
职责：实现推理中所有核心算子。
- `Embedding`
- `RMSNorm`
- `LinearProjection`
- `Attention`（含 RoPE + causal mask + GQA 逻辑）
- `MLP`（SwiGLU）
- `Decoder`
- `SoftMax`
- `Sampler`（greedy）

### 2.4 `src/qwen3.h/.cpp`
职责：定义 `Qwen3Model` 并连接模型结构与权重加载。
- `Load`：解析配置、读取 safetensors、装配算子权重
- `Forward`：执行完整前向

### 2.5 `src/main.cpp`
职责：教学示例入口。
- 分步骤演示加载、编码、前向、采样
- 支持 `--verbose` 打印 logits 信息
- 支持 `--dump` 输出 logits 到文件
- 输出简易耗时统计

## 3. 运行时数据流

完整链路如下：

`Text -> Tokenizer::Encode -> token_ids -> Qwen3Model::Forward -> logits -> Sampler::Sample -> next_token_id -> Tokenizer::Decode`

模型内部前向：

`Embedding -> N x Decoder -> Final RMSNorm -> LM Head (LinearProjection) -> Softmax`


## 4. 后续扩展方向

- 性能：KV Cache、算子融合、SIMD、并行化
- 功能：Top-k/Top-p/Temperature 采样
