#pragma once
/*
 * ============================================================================
 * Qwen3Model - 模型加载与推理
 * ============================================================================
 *
 * 目录：
 *   1. HeaderInfo        - safetensors 权重元信息（name / dtype / shape / offsets）
 *   2. Qwen3Model        - 主模型类
 *      - Load            - 加载 safetensors 模型文件与配置
 *      - Forward         - 前向推理：token IDs → logits Tensor
 *      - ClearCache      - 清空所有 Decoder 的 KV Cache
 *      - ParseSafetensorsHeader - 解析权重文件头
 *      - ParseConfig     - 解析 config.json 模型配置
 *      - LoadWeight      - 从文件读取单个权重张量
 *   3. 模型组件（私有成员）
 *      - embedding_      - 词嵌入层
 *      - decoders_       - Transformer 解码层列表
 *      - final_norms_    - 最终 RMSNorm
 *      - lm_head_        - 语言模型输出投影
 *      - softmax_        - Softmax
 */

#include <string>

#include "nlohmann/json.hpp"
#include "operator.hpp"
#include "tokenizer.h"
using json = nlohmann::json;

struct HeaderInfo {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    std::vector<uint64_t> data_offsets;
};

class Qwen3Model {
   public:
    Qwen3Model() = default;
    ~Qwen3Model() = default;

    bool Load(const std::string& model_path);

    void ClearCache() {
        for (auto& decoder : decoders_) {
            decoder->ClearCache();
        }
    }

    Tensor Forward(const std::vector<uint32_t>& token_ids, size_t position = 0);

   private:
    bool ParseSafetensorsHeader(const std::string& model_path);

    bool ParseConfig(const std::string& model_path, json& config);

    bool LoadWeight(std::ifstream& file, const HeaderInfo& info, Tensor& weight);

    Embedding::Ptr embedding_;

    std::vector<Decoder::Ptr> decoders_;

    std::shared_ptr<RMSNorm> final_norms_;

    std::shared_ptr<LinearProjection> lm_head_;

    std::map<std::string, HeaderInfo> headers_;

    size_t data_offset_;
};