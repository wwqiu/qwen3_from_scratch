#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "nlohmann/json.hpp"
#include "tokenizer.h"
#include "type.h"

using json = nlohmann::json;

int ParseSafetensorsHeader(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return -1;
    }

    uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (file.gcount() != sizeof(header_len)) {
        std::cerr << "Failed to read header length." << std::endl;
        return -1;
    }

    std::string header_str(header_len, '\0');
    file.read(&header_str[0], header_len);
    if (file.gcount() != static_cast<std::streamsize>(header_len)) {
        std::cerr << "Failed to read header JSON." << std::endl;
        return -1;
    }

    try {
        json header = json::parse(header_str);
        std::cout << "--- Safetensors Header ---" << std::endl;

        for (auto& it : header.items()) {
            if (it.key() == "__metadata__") {
                std::cout << "(Metadata): " << it.value().dump(2) << std::endl;
            }
            else {
                std::string tensor_name = it.key();
                auto info = it.value();

                std::string dtype = info["dtype"];
                std::vector<int64_t> shape = info["shape"];
                std::vector<uint64_t> offsets = info["data_offsets"];

                std::cout << "Name: " << tensor_name
                    << " | DataType: " << dtype
                    << " | Shape: [";
                for (size_t i = 0; i < shape.size(); ++i)
                    std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
                std::cout << "] | Offsets: " << offsets[0] << " - " << offsets[1] << std::endl;
            }
        }
    }
    catch (const json::parse_error& e) {
        std::cerr << "JSON Parse Failed: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

int main(int argc, char* argv[]) {
    std::string model_path = "../Qwen3-0.6B/model.safetensors";
    std::string tokenizer_path = "../Qwen3-0.6B/tokenizer.json";

    if (argc > 1) tokenizer_path = argv[1];
    if (argc > 2) model_path = argv[2];

    // Parse model header
    std::cout << "Loading model from: " << model_path << std::endl;
    ParseSafetensorsHeader(model_path);

    // Load tokenizer
    std::cout << "\nLoading tokenizer from: " << tokenizer_path << std::endl;
    Tokenizer tokenizer;
    tokenizer.LoadConfig(tokenizer_path);

    // Test tokenization
    std::vector<std::string> test_strings = {
        "hello",
        "广州有哪些好玩的景点？",
        "Café",
        "Hello world"
    };

    std::cout << "\n--- Tokenization Test ---" << std::endl;
    for (const auto& input : test_strings) {
        std::cout << "Input: " << input << std::endl;
        std::vector<uint32_t> ids = tokenizer.Encode(input);

        std::cout << "Token IDs: [";
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i];
            if (i < ids.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl << std::endl;
    }

    return 0;
}

