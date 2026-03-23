#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "type.h"

class Tokenizer {
public:
    Tokenizer() = default;

    void LoadConfig(const std::string& config_file);
    std::vector<uint32_t> Encode(const std::string& text);

    std::string Decode(const std::vector<uint32_t>& token_ids);

private:
    TokenizerConfig config_;
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::unordered_map<uint32_t, std::string> id_to_token_;
    std::unordered_map<std::string, uint32_t> special_token_to_id_;
    std::unordered_set<uint32_t> special_token_ids_;
    std::vector<std::string> special_tokens_sorted_;

    std::vector<std::wstring> PreTokenize(const std::wstring& text);
    void EncodeNormalText(const std::string& text, std::vector<uint32_t>& ids);
    std::string DecodeByteLevelText(const std::string& byte_level_text);
    void BPETokenize(const std::string& utf8_word, std::vector<uint32_t>& tokens);
};

// Utility functions
std::string WideToUTF8(const std::wstring& wstr);
std::wstring UTF8ToWide(const std::string& str);
std::map<unsigned char, std::string> CreateByteToUnicodeMap();
