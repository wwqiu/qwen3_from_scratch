#pragma once

#include <string>
#include "tokenizer.h"
#include "operator.hpp"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

struct HeaderInfo {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    std::vector<uint64_t> data_offsets;

    std::string ToString() const {
        std::string shape_str = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            shape_str += std::to_string(shape[i]);
            if (i < shape.size() - 1) shape_str += ", ";
        }
        shape_str += "]";
        std::string offsets_str = "[";
        for (size_t i = 0; i < data_offsets.size(); ++i) {
            offsets_str += std::to_string(data_offsets[i]);
            if (i < data_offsets.size() - 1) offsets_str += ", ";
        }
        offsets_str += "]";
        return "Tensor: " + name + " | dtype: " + dtype + " | shape: " + shape_str + " | data_offsets: " + offsets_str;
    }
};


class Qwen3Model
{
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

        std::shared_ptr<SoftMax> softmax_;

        std::map<std::string, HeaderInfo> headers_;

        size_t data_offset_;
};