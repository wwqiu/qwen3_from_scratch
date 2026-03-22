# Qwen3.cpp Operators Guide

## 1. 约定

- 本文中的 `seq_len` 表示序列长度。
- `hidden_dim` 表示隐藏维度。
- `num_heads` 是 Query 头数，`num_kv_heads` 是 Key/Value 头数（GQA）。
- `head_dim` 是每个头的维度。

## 2. Embedding

职责：将 token id 映射到词向量。

输入输出：
- input: `token_ids` `[seq_len]`
- weight: `[vocab_size, hidden_dim]`
- output: `[seq_len, hidden_dim]`

公式：
- `output[i] = weight[token_id[i]]`

## 3. RMSNorm

职责：对每个位置向量做 RMS 归一化并缩放。

输入输出：
- input: `[seq_len, dim]`
- weight: `[hidden_dim]`（在实现中可按 head 维度复用）
- output: `[seq_len, dim]`

公式：
- `RMS(x) = sqrt(mean(x^2) + eps)`
- `y = x / RMS(x) * weight`

## 4. LinearProjection

职责：线性投影（无 bias）。

输入输出：
- input: `[seq_len, input_dim]`
- weight: `[output_dim, input_dim]`
- output: `[seq_len, output_dim]`

公式：
- `output = input * weight^T`

## 5. Attention

职责：执行带 RoPE 与 causal mask 的自注意力。

### 5.1 子步骤
1. `q_proj/k_proj/v_proj` 线性映射
2. `q_norm/k_norm`（按 `head_dim` 做 RMSNorm）
3. 应用 RoPE
4. 计算分数并做 softmax
5. 与 `V` 加权求和
6. `output_proj` 回到 `hidden_dim`

### 5.2 Shape
- input: `[seq_len, hidden_dim]`
- q: `[seq_len, num_heads * head_dim]`
- k/v: `[seq_len, num_kv_heads * head_dim]`
- output: `[seq_len, hidden_dim]`

### 5.3 公式
- `Attention(Q,K,V) = Softmax((QK^T)/sqrt(head_dim) + mask) V`
- causal mask: `j > i` 位置置为极小值（近似 `-inf`）

### 5.4 RoPE
按前后半维做二维旋转：
- `x' = x cos(theta) - y sin(theta)`
- `y' = x sin(theta) + y cos(theta)`

## 6. MLP (SwiGLU)

职责：前馈网络，使用门控激活。

子步骤：
1. `up = up_proj(input)`
2. `gate = gate_proj(input)`
3. `up = up * SiLU(gate)`
4. `output = down_proj(up)`

输入输出：
- input: `[seq_len, hidden_dim]`
- up/gate: `[seq_len, intermediate_size]`
- output: `[seq_len, hidden_dim]`

公式：
- `sigmoid(x) = 1 / (1 + exp(-x))`
- `SiLU(x) = x * sigmoid(x)`

## 7. Decoder

职责：一个标准 Transformer decoder block（pre-norm 结构）。

流程：
1. `x1 = input_norm(x)`
2. `a = attention(x1)`
3. `x2 = x + a`（残差 1）
4. `x3 = post_attention_norm(x2)`
5. `m = mlp(x3)`
6. `output = x2 + m`（残差 2）

输入输出：
- input: `[seq_len, hidden_dim]`
- output: `[seq_len, hidden_dim]`

## 8. SoftMax

职责：将每行 logits 归一化为概率分布。

输入输出：
- input/output: `[seq_len, vocab_size]`

公式：
- `softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`

## 9. Sampler

职责：从最后一个位置的概率中选出 token。

当前策略：
- greedy（取最大概率 id）

输入输出：
- logits: `[seq_len, vocab_size]`
- result: `uint32_t token_id`
