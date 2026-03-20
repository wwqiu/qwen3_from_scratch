#pragma once

#include <string>
#include "type.h"
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

class Embedding {
public:
    using Ptr = std::shared_ptr<Embedding>;

    Embedding(size_t vocab_size, size_t hidden_dim) : vocab_size_(vocab_size), hidden_dim_(hidden_dim) {
        weight_ = Tensor({vocab_size_, hidden_dim_}, sizeof(float));
    }

    Tensor Forward(const std::vector<uint32_t>& token_ids) {
        Tensor output({token_ids.size(), hidden_dim_}, sizeof(float));
        for (size_t i = 0; i < token_ids.size(); ++i) {
            uint32_t token_id = token_ids[i];
            float* weight_row = (float*)weight_.data_ + token_id * hidden_dim_;
            float* output_row = (float*)output.data_ + i * hidden_dim_;
            memcpy(output_row, weight_row, hidden_dim_ * sizeof(float));
        }
        return output;
    }

    Tensor weight_;  // [vocab_size, hidden_dim]

    size_t vocab_size_;
    size_t hidden_dim_;
};

class RMSNorm {
public:
    RMSNorm(size_t hidden_dim) : hidden_dim_(hidden_dim) {
        weight_ = Tensor({hidden_dim_}, sizeof(float));
    }

    Tensor Forward(const Tensor& input) {
        float rms = 0.0f;
        float* input_data = (float*)input.data_;

        size_t seq_len = input.shape_[0];
        size_t dim = input.shape_[1];
        size_t num_head = dim / hidden_dim_;

        Tensor output(input.shape_, sizeof(float));

        float* weight_data = (float*)weight_.data_;
        for (size_t i = 0; i < seq_len; ++i) {
            float* input_row = (float*)input.data_ + i * dim;
            float* output_row = (float*)output.data_ + i * dim;
            for (size_t h = 0; h < num_head; ++h) {
                float* input_head = input_row + h * hidden_dim_;
                float* output_head = output_row + h * hidden_dim_;
                float head_rms = 0.0f;
                // RMS(a) = sqrt(mean(a^2) + eps)
                for (size_t d = 0; d < hidden_dim_; ++d) {
                    float val = input_head[d];
                    head_rms += val * val;
                }
                head_rms = std::sqrt(head_rms / hidden_dim_ + eps_);

                // output = input / RMS(a) * weight
                for (size_t d = 0; d < hidden_dim_; ++d) {
                    output_head[d] = input_head[d] / head_rms * weight_data[d];
                }
            }
        }

        return output;
    }

    Tensor weight_;  // [hidden_dim]

    float eps_ = 1e-6;
    
    size_t hidden_dim_;
};

class LinearProjection {
public:
    LinearProjection(size_t input_dim, size_t output_dim) : input_dim_(input_dim), output_dim_(output_dim) {
        weight_ = Tensor({output_dim_, input_dim_}, sizeof(float));
    }

    Tensor Forward(const Tensor& input) {
        size_t seq_len = input.shape_[0];
        Tensor output({seq_len, output_dim_}, sizeof(float));
        // matmul : output = input * weight
        for (size_t i = 0; i < seq_len; ++i) {
            float* input_row = (float*)input.data_ + i * input_dim_;
            float* output_row = (float*)output.data_ + i * output_dim_;
            for (size_t j = 0; j < output_dim_; ++j) {
                output_row[j] = 0.0f;
                float* weight_data = (float*)weight_.data_ + j * input_dim_;
                for (size_t k = 0; k < input_dim_; ++k) {
                    output_row[j] += input_row[k] * weight_data[k];
                }
            }
        }
        return output; 
    }

    Tensor weight_;  // [output_dim, input_dim]

    size_t input_dim_;
    size_t output_dim_;
};


class Attention {
public:
    Attention(size_t hidden_dim, size_t num_kv_heads, size_t num_heads, size_t head_dim) 
        : hidden_dim_(hidden_dim), num_kv_heads_(num_kv_heads), num_heads_(num_heads), head_dim_(head_dim) {
        q_norm_ = std::make_shared<RMSNorm>(head_dim_);
        k_norm_ = std::make_shared<RMSNorm>(head_dim_);
        q_proj_ = std::make_shared<LinearProjection>(hidden_dim_, num_heads_ * head_dim_);
        k_proj_ = std::make_shared<LinearProjection>(hidden_dim_, num_kv_heads_ * head_dim_);
        v_proj_ = std::make_shared<LinearProjection>(hidden_dim_, num_kv_heads_ * head_dim_);
        output_proj_ = std::make_shared<LinearProjection>(num_heads_ * head_dim_, hidden_dim_);
    }   

    
   /**
    * @brief Attention forward pass with RoPE and causal masking
    *
    *            input
    *              │
    *    ┌─────────┼───────┐
    *    ↓         ↓       ↓
    *  Q-Proj    K-Proj  V-Proj
    *    ↓         ↓       │
    *  Q-Norm    K-Norm    │
    *    ↓         ↓       │
    *    └──RoPE───┘       │
    *        ↓             ↓ 
    *   ComputeAttention(Q, K, V)
    *              ↓
    *         Output-Proj
    *              ↓
    *            return
    */
    Tensor Forward(const Tensor& input) {
        // q k v projection [seq, hidden] -> [seq, heads, head_dim]
        Tensor q = q_proj_->Forward(input);
        Tensor k = k_proj_->Forward(input);
        Tensor v = v_proj_->Forward(input);
          
        // q k norm
        k = k_norm_->Forward(k);
        q = q_norm_->Forward(q);

        // RoPE
        ApplyRoPE(q, k);

        // attention scores and weighted sum
        Tensor attention_output = ComputeAttention(q, k, v);

        // output projection
        attention_output = output_proj_->Forward(attention_output);

        return attention_output;
    }

    void ApplyRoPE(Tensor& q, Tensor& k) {
        size_t seq_lens = q.shape_[0];
        int half_dim = head_dim_ / 2;
        
        for (size_t i = 0; i < seq_lens; ++i) {
            // half spilt
            for (int d = 0; d < half_dim; ++d) {
                float freq = 1.0f / std::pow(rope_theta_, (float)(2 * d) / head_dim_);
                float angle = i * freq;
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);

                // apply RoPE to q
                for (size_t h = 0; h < num_heads_; ++h) {
                    float* head_ptr = (float*)q.data_ + i * num_heads_ * head_dim_ + h * head_dim_;
                    float x = head_ptr[d];
                    float y = head_ptr[d + half_dim];
                    head_ptr[d] = x * cos_val - y * sin_val;
                    head_ptr[d + half_dim] = x * sin_val + y * cos_val;
                }

                // apply RoPE to k
                for (size_t h = 0; h < num_kv_heads_; ++h) {
                    float* head_ptr = (float*)k.data_ + i * num_kv_heads_ * head_dim_ + h * head_dim_;
                    float x = head_ptr[d];
                    float y = head_ptr[d + half_dim];
                    head_ptr[d] = x * cos_val - y * sin_val;
                    head_ptr[d + half_dim] = x * sin_val + y * cos_val;
                }
            }
        }
    }

    Tensor ComputeAttention(const Tensor& q, const Tensor& k, const Tensor& v) {
        size_t seq_len = q.shape_[0];
        float scale = 1.0f / std::sqrt((float)head_dim_);
        Tensor output({seq_len, num_heads_ * head_dim_}, sizeof(float));
        for (size_t h = 0; h < num_heads_; ++h) {
            size_t kv_h = h / (num_heads_ / num_kv_heads_);
            for (size_t i = 0; i < seq_len; ++i) {
                // score = Q * K^T
                std::vector<float> attention_scores(seq_len);
                for (size_t j = 0; j < seq_len; ++j) {
                    // causal mask is applied here by setting attention scores to -inf for j > i
                    if (j > i) { 
                        attention_scores[j] = -1e9; 
                        continue;
                    }
                    float dot = 0.0f;
                    float* q_ptr = (float*)q.data_ + (i * num_heads_ + h) * head_dim_;
                    float* k_ptr = (float*)k.data_ + (j * num_kv_heads_ + kv_h) * head_dim_;
                    
                    for (int d = 0; d < head_dim_; ++d) {
                        dot += q_ptr[d] * k_ptr[d];
                    }
                    attention_scores[j] = dot * scale;
                }
                
                SoftMax(attention_scores.data(), seq_len);

                // weighted sum: output = sum(attention_scores * V)
                float* out_ptr = (float*)output.data_ + (i * num_heads_ + h) * head_dim_;
                memset(out_ptr, 0, head_dim_ * sizeof(float));
                for (size_t j = 0; j < seq_len; ++j) {
                    float weight = attention_scores[j];
                    float* v_ptr = (float*)v.data_ + (j * num_kv_heads_ + kv_h) * head_dim_;
                    for (int d = 0; d < head_dim_; ++d) {
                        out_ptr[d] += weight * v_ptr[d];
                    }
                }
            }
        }
        return output;
    }

    void SoftMax(float* data, size_t len) {
        // output = exp(x - max) / sum(exp(x - max))
        float max_val = *std::max_element(data, data + len);
        float sum = 0.0f;
        for (size_t i = 0; i < len; ++i) {
            data[i] = std::exp(data[i] - max_val);
            sum += data[i];
        }
        for (size_t i = 0; i < len; ++i) {
            data[i] /= sum;
        }
    }

    size_t hidden_dim_;
    size_t num_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;
    float rope_theta_ = 100000.0f;

    std::shared_ptr<RMSNorm> q_norm_;
    std::shared_ptr<RMSNorm> k_norm_;
    std::shared_ptr<LinearProjection> q_proj_;
    std::shared_ptr<LinearProjection> k_proj_;
    std::shared_ptr<LinearProjection> v_proj_;
    std::shared_ptr<LinearProjection> output_proj_;

    std::vector<float> cos_table_k_;
    std::vector<float> sin_table_k_;
    std::vector<float> cos_table_q_;
    std::vector<float> sin_table_q_;
    
};

class MLP {
public:
    MLP(size_t hidden_size, size_t intermediate_size) : intermediate_size_(intermediate_size) {
        up_proj_ = std::make_shared<LinearProjection>(hidden_size, intermediate_size);
        down_proj_ = std::make_shared<LinearProjection>(intermediate_size, hidden_size);
        gate_proj_ = std::make_shared<LinearProjection>(hidden_size, intermediate_size);
    }

   /**
    * @brief MLP forward pass with gated activation
    *
    *      input
    *        │
    *  ┌─────┴──────┐
    *  │            ↓
    *  │         gate proj
    *  │            ↓
    *  │          SiLU (gate * sigmoid(gate))
    *  ↓            ↓
    * up proj ──→  (*) ─ (Element-wise Multiply)
    *               │
    *               ↓
    *            down proj
    *               ↓
    *             output
    */
    void Forward(const Tensor& input, Tensor& output) {
        Tensor up = up_proj_->Forward(input);
        Tensor gate = gate_proj_->Forward(input);
        size_t seq_len = input.shape_[0];
        for (size_t i = 0; i < seq_len; ++i) {
            float* up_ptr = (float*)up.data_ + i * intermediate_size_;
            float* gate_ptr = (float*)gate.data_ + i * intermediate_size_;
            for (size_t j = 0; j < intermediate_size_; ++j) {
                // SiLU activation: x * sigmoid(x)
                up_ptr[j] =  (gate_ptr[j] * Sigmoid(gate_ptr[j])) * up_ptr[j];
            }   
        }
        output = down_proj_->Forward(up);
    }

    float Sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    std::shared_ptr<LinearProjection> up_proj_;
    std::shared_ptr<LinearProjection> down_proj_;
    std::shared_ptr<LinearProjection> gate_proj_;

    size_t intermediate_size_;

};

class Decoder {
public:
    using Ptr = std::shared_ptr<Decoder>;

    Decoder(size_t hidden_dim, size_t num_kv_heads, size_t num_heads, size_t head_dim, size_t intermediate_size) : 
                hidden_dim_(hidden_dim), num_kv_heads_(num_kv_heads), num_heads_(num_heads), head_dim_(head_dim), intermediate_size_(intermediate_size)
    {
        input_norm_ = std::make_shared<RMSNorm>(hidden_dim_);
        attention_ = std::make_shared<Attention>(hidden_dim_, num_kv_heads_, num_heads_, head_dim_);
        post_attention_norm_ = std::make_shared<RMSNorm>(hidden_dim_);
        mlp_ = std::make_shared<MLP>(hidden_dim_, intermediate_size_);
    }

    
   /**
    * @brief Decoder layer forward pass with attention and MLP
    *      input
    *        │
    *  ┌─────┴──────┐
    *  │            ↓
    *  │         input norm
    *  │            ↓
    *  │         attention
    *  │            ↓
    *  └─→ residual add 1
    *        │
    *  ┌─────┴──────┐
    *  │            ↓
    *  │         post attention norm
    *  │            ↓
    *  │           MLP (Feed Forward)
    *  │            ↓
    *  └─→ residual add 2
    *        ↓
    *      output
    */
    Tensor Forward(const Tensor& input) {
        Tensor residual = input.clone();
        Tensor hidden_states = input_norm_->Forward(input);
        // attention
        hidden_states = attention_->Forward(hidden_states);
        hidden_states = residual + hidden_states;
        residual = hidden_states.clone();
        // mlp
        hidden_states = post_attention_norm_->Forward(hidden_states);
        mlp_->Forward(hidden_states, hidden_states);
        hidden_states = residual + hidden_states;
        return hidden_states;
    }

    std::shared_ptr<RMSNorm> input_norm_;
    std::shared_ptr<Attention> attention_;
    std::shared_ptr<RMSNorm> post_attention_norm_;
    std::shared_ptr<MLP> mlp_;

    size_t hidden_dim_;
    size_t num_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;
    size_t intermediate_size_;
};

class Qwen3Model
{
    public:
        Qwen3Model() = default;
        ~Qwen3Model() = default;

        bool Load(const std::string& model_path);

        Tensor Forward(const std::vector<uint32_t>& token_ids);

        bool ParseSafetensorsHeader(const std::string& model_path);

        bool ParseConfig(const std::string& model_path, json& config);

    private:
        bool LoadWeight(std::ifstream& file, const HeaderInfo& info, Tensor& weight);

        Embedding::Ptr embedding_;

        std::vector<Decoder::Ptr> decoders_;

        std::map<std::string, HeaderInfo> headers_;

        size_t data_offset_;
};