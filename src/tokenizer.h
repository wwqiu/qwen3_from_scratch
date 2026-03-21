#pragma once
#include <string>
#include <vector>
#include "type.h"

class Tokenizer {
public:
    Tokenizer() = default;

    void LoadConfig(const std::string& config_file);
    std::vector<uint32_t> Encode(const std::string& text);

    std::string Decode(const std::vector<uint32_t>& token_ids);

private:
    TokenizerConfig config_;

    std::vector<std::wstring> PreTokenize(const std::wstring& text);
    void BPETokenize(const std::string& utf8_word, std::vector<uint32_t>& tokens);
};

// Utility functions
std::string WideToUTF8(const std::wstring& wstr);
std::wstring UTF8ToWide(const std::string& str);
std::map<unsigned char, std::string> CreateByteToUnicodeMap();
