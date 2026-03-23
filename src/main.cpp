#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "nlohmann/json.hpp"
#include "tokenizer.h"
#include "qwen3_model.h"


void PrintTokenIds(const std::vector<uint32_t>& ids) {
    std::cout << "Token IDs: [";
    for (size_t i = 0; i < ids.size(); ++i) {
        std::cout << ids[i];
        if (i < ids.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void DumpTensor(const std::string& filename, Tensor& tensor) {
    size_t rows = tensor.shape()[0];
    size_t cols = tensor.shape()[1];
    std::ofstream file(filename);
    for (size_t i = 0; i < rows; ++i) {
        float* data = tensor.data<float>() + i * cols;
        for (size_t j = 0; j < cols; ++j) {
            file << data[j] << " ";
        }
        file << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::string model_path = "../../Qwen3-0.6B";
    std::string tokenizer_path = "../../Qwen3-0.6B/tokenizer.json";

    if (argc > 1) tokenizer_path = argv[1];
    if (argc > 2) model_path = argv[2];

    // Parse model header
    std::cout << "Loading model from: " << model_path << std::endl;
    Qwen3Model model;
    if (!model.Load(model_path)) {
        std::cerr << "Failed to load model." << std::endl;
        return -1;
    }

    // Load tokenizer
    std::cout << "\nLoading tokenizer from: " << tokenizer_path << std::endl;
    Tokenizer tokenizer;
    tokenizer.LoadConfig(tokenizer_path);

    // Test tokenization
    std::vector<std::string> test_strings = {
        "hello ",
        "1+1=",
        "广州最出名的地标是广州"
    };

    std::cout << "\n--- Tokenization Test ---" << std::endl;

    Sampler sampler;

    for (const auto& input : test_strings) {
        std::cout << "Input: " << input << std::endl;

        std::vector<uint32_t> ids = tokenizer.Encode(input);
        
        model.ClearCache();

        Tensor tensor = model.Forward(ids);

        PrintTokenIds(ids);
        
        uint32_t sampled_token = sampler.Sample(tensor);
        std::cout << "Sampled Token ID: " << sampled_token << std::endl;
        std::cout << "Sampled Token Text: " << tokenizer.Decode({sampled_token}) << std::endl;
    }

    return 0;
}

