# 🎓 Qwen3-From-Scratch

> **"What I cannot create, I do not understand." — Richard Feynman**

## 📖 项目简介 (Introduction)

本项目旨在**从零开始（From Scratch）**实现 Qwen3 大语言模型的推理过程。

与 vLLM、HuggingFace Transformers 等工业级库不同，本仓库**完全不考虑性能优化**。

**只关心一件事：代码是否足够简单，逻辑是否足够清晰。**

如果你想知道当调用 `model.generate()` 时底层到底发生了什么，但被复杂的封装和性能优化代码劝退，那么这个项目就是为你准备的。

## 🎯 核心目标 (Goals)

1.  **去黑盒化**：拆解 Qwen3 的每一个组件（RMSNorm, RoPE, SwiGLU, GQA 等）。
2.  **极致易读**：变量命名与论文公式对应，逻辑流线性化，尽量减少类继承和复杂的抽象。
3.  **零依赖（理想情况下）**：除了 ‘Boost::Regex’用于字符串处理外，不使用任何第三方依赖。

## 🛠️ 你将学到 (What You Will Learn)

通过阅读本项目代码，你将直观地理解：

*   **Tokenizer**: 文本是如何变成数字 ID 的？
*   **Embedding**: 数字 ID 是如何变成向量的？
*   **RoPE (旋转位置编码)**: 它是如何通过旋转向量来实现位置信息的注入？
*   **Attention (GQA)**: Qwen3 的 Grouped Query Attention 是如何计算的？KV Cache 是如何运作的？
*   **FFN (SwiGLU)**: 激活函数层面的数学细节。
*   **Sampling**: Top-k, Top-p (Nucleus), Temperature 采样策略的代码实现。

