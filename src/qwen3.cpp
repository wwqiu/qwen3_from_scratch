#include "qwen3.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "nlohmann/json.hpp"
#include "logger.h"

using json = nlohmann::json;

Tensor Qwen3Model::Forward(const std::vector<uint32_t>& token_ids) {
    Tensor hidden_state = embedding_->Forward(token_ids);
    
    return hidden_state; // Placeholder
    
}

bool Qwen3Model::Load(const std::string& model_path) {
    ParseSafetensorsHeader(model_path + "/model.safetensors");
    json config;
    if (!ParseConfig(model_path + "/config.json", config)) {
        LOG_ERROR("Failed to parse model config.");
        return false;
    }
    size_t vocab_size = config["vocab_size"];
    size_t hidden_dim = config["hidden_size"];
    size_t num_heads = config["num_attention_heads"];
    size_t num_kv_heads = config["num_key_value_heads"];
    size_t num_hidden = config["num_hidden_layers"];
    size_t head_dim = config["head_dim"];

    std::ifstream file(model_path + "/model.safetensors", std::ios::binary);
    // load embedding weight
    HeaderInfo& embedding_info = headers_["model.embed_tokens.weight"];
    embedding_ = std::make_shared<Embedding>(vocab_size, hidden_dim);
    LOG_INFO("Loading embedding layer: %s | dtype: %s | shape: [%ld, %ld]", embedding_info.name.c_str(), embedding_info.dtype.c_str(), embedding_info.shape[0], embedding_info.shape[1]);
    if (!LoadWeight(file, embedding_info, embedding_->weight_)) {
        LOG_ERROR("Failed to load embedding layer.");
        return false;
    }

    // load decoder layers
    LOG_INFO("Loading %zu decoder layers...", num_hidden);
    for (size_t i = 0; i < num_hidden; ++i) {
        // LOG_INFO("Loading decoder layer %zu", i);
        char layer_prefix[64];
        sprintf(layer_prefix, "model.layers.%zu", i);
        HeaderInfo& input_norm_info = headers_[std::string(layer_prefix) + ".input_layernorm.weight"];
        Decoder::Ptr decoder = std::make_shared<Decoder>(hidden_dim, num_kv_heads, num_heads, head_dim);
        LoadWeight(file, input_norm_info, decoder->input_norm_->weight_);
    }

    return true;
}


bool Qwen3Model::LoadWeight(std::ifstream& file, const HeaderInfo& info, Tensor& weight) {
    // LOG_INFO("Loading weight: %s | dtype: %s ", info.name.c_str(), info.dtype.c_str());
    size_t elem_size = 0;
    if (info.dtype != "BF16") {
        LOG_ERROR("Unsupported data type: %s", info.dtype.c_str());
        return false;
    }
    elem_size = 2;
    std::vector<int64_t> shape;
    switch (info.shape.size()) {
        case 1:
            shape = {info.shape[0], 1, 1, 1};
            break;
        case 2:
            shape = {info.shape[0], info.shape[1], 1, 1};
            break;
        case 3:
            shape = {info.shape[0], info.shape[1], info.shape[2], 1};
            break;
        case 4:
            shape = {info.shape[0], info.shape[1], info.shape[2], info.shape[3]};
            break;
        default:
            LOG_ERROR("Unsupported tensor shape with %zu dimensions.", info.shape.size());
            return false;   
    }
    int64_t num_elements = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    Tensor weight_bf16(shape, elem_size);
    file.seekg(info.data_offsets[0] + data_offset_, std::ios::beg);
    file.read((char*)(weight_bf16.data_), num_elements * elem_size);
    if (file.gcount() != static_cast<std::streamsize>(num_elements * elem_size)) {
        LOG_ERROR("Failed to read tensor data for: %s", info.name.c_str());
        return false;
    }
    uint16_t* bf16_data = (uint16_t*)weight_bf16.data_;
    float* float_data = (float*)weight.data_;
    for (int64_t i = 0; i < num_elements; ++i) {
        uint16_t bf16_val = bf16_data[i];
        uint32_t float_bits = (uint32_t)bf16_val << 16;
        float_data[i] = *(float*)&float_bits;
    }
    // LOG_INFO("Successfully loaded weight: %s", info.name.c_str());
    return true;
}

bool Qwen3Model::ParseSafetensorsHeader(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open file: %s", filepath.c_str());
        return false;
    }

    uint64_t header_len = 0;
    file.read((char*)(&header_len), sizeof(header_len));
    if (file.gcount() != sizeof(header_len)) {
        LOG_ERROR("Failed to read header length.");
        return false;
    }

    std::string header_str(header_len, '\0');
    file.read(&header_str[0], header_len);

    data_offset_ = sizeof(header_len) + header_len;

    try {
        json header = json::parse(header_str);
        LOG_INFO("--- Safetensors Header ---");

        for (auto& it : header.items()) {
            if (it.key() == "__metadata__") {
                LOG_INFO("(Metadata): %s", it.value().dump(2).c_str());
            }
            else {
                std::string tensor_name = it.key();
                auto info = it.value();
                std::string dtype = info["dtype"];
                std::vector<int64_t> shape = info["shape"];
                std::vector<uint64_t> offsets = info["data_offsets"];
                headers_[tensor_name] = {tensor_name, dtype, shape, offsets};
                LOG_DEBUG("%s", headers_[tensor_name].ToString().c_str());
            }
        }
    }
    catch (const json::parse_error& e) {
        LOG_ERROR("JSON Parse Failed: %s", e.what());
        return false;
    }

    return true;
}

bool Qwen3Model::ParseConfig(const std::string& model_path, json& config) {
    config = json::parse(std::ifstream(model_path));
    LOG_INFO("--- Model Config ---");
    LOG_INFO("%s", config.dump(2).c_str());
    return true;
}