#include "qwen3.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "nlohmann/json.hpp"
#include "logger.h"

using json = nlohmann::json;

Tensor Qwen3Model::Forward(const std::vector<uint32_t>& token_ids) {
    Tensor hidden_state = embedding_->Forward(token_ids);
    for (size_t i = 0; i < decoders_.size(); ++i) {
        hidden_state = decoders_[i]->Forward(hidden_state);
    }
    hidden_state = final_norms_->Forward(hidden_state);
    Tensor logits = lm_head_->Forward(hidden_state);
    softmax_->Forward(logits);    

    return logits;
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
    size_t intermediate_size = config["intermediate_size"];

    std::ifstream file(model_path + "/model.safetensors", std::ios::binary);
    // load output projection weight
    lm_head_ = std::make_shared<LinearProjection>(hidden_dim, vocab_size);
    LoadWeight(file, headers_["lm_head.weight"], lm_head_->weight_);

    // load embedding weight
    embedding_ = std::make_shared<Embedding>(vocab_size, hidden_dim);
    LoadWeight(file, headers_["model.embed_tokens.weight"], embedding_->weight_);

    // load decoder layers
    LOG_INFO("Loading %zu decoder layers...", num_hidden);
    for (size_t i = 0; i < num_hidden; ++i) {
        LOG_INFO("Loading decoder layer %zu", i);
        std::string layer_prefix = "model.layers." + std::to_string(i);
        Decoder::Ptr decoder = std::make_shared<Decoder>(hidden_dim, num_kv_heads, num_heads, head_dim, intermediate_size);
        // input norm
        LoadWeight(file, headers_[layer_prefix + ".input_layernorm.weight"], decoder->input_norm_->weight_);
        // attention
        LoadWeight(file, headers_[layer_prefix + ".self_attn.q_proj.weight"], decoder->attention_->q_proj_->weight_);
        LoadWeight(file, headers_[layer_prefix + ".self_attn.k_proj.weight"], decoder->attention_->k_proj_->weight_);
        LoadWeight(file, headers_[layer_prefix + ".self_attn.v_proj.weight"], decoder->attention_->v_proj_->weight_);
        LoadWeight(file, headers_[layer_prefix + ".self_attn.o_proj.weight"], decoder->attention_->output_proj_->weight_);
        LoadWeight(file, headers_[layer_prefix + ".self_attn.q_norm.weight"], decoder->attention_->q_norm_->weight_);
        LoadWeight(file, headers_[layer_prefix + ".self_attn.k_norm.weight"], decoder->attention_->k_norm_->weight_);
        // post attention norm
        LoadWeight(file, headers_[layer_prefix + ".post_attention_layernorm.weight"], decoder->post_attention_norm_->weight_);
        // mlp
        LoadWeight(file, headers_[layer_prefix + ".mlp.down_proj.weight"], decoder->mlp_->down_proj_->weight_);
        LoadWeight(file, headers_[layer_prefix + ".mlp.gate_proj.weight"], decoder->mlp_->gate_proj_->weight_);
        LoadWeight(file, headers_[layer_prefix + ".mlp.up_proj.weight"], decoder->mlp_->up_proj_->weight_);
        decoders_.push_back(decoder);
    }

    final_norms_ = std::make_shared<RMSNorm>(hidden_dim);
    LoadWeight(file, headers_["model.norm.weight"], final_norms_->weight_);

    softmax_ = std::make_shared<SoftMax>();

    return true;
}


bool Qwen3Model::LoadWeight(std::ifstream& file, const HeaderInfo& info, Tensor& weight) {
    // LOG_INFO("Loading weight: %s | dtype: %s ", info.name.c_str(), info.dtype.c_str());
    size_t elem_size = 0;
    // if (info.dtype != "BF16") {
    //     LOG_ERROR("Unsupported data type: %s", info.dtype.c_str());
    //     return false;
    // }
    elem_size = 2;
    std::vector<size_t> shape;
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
                std::vector<size_t> shape = info["shape"];
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