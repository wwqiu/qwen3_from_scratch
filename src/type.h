#pragma once
#include <array>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
#include <codecvt>
#include <locale>

struct AddedToken {
    int id;
    std::string content;
    bool special;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AddedToken, id, content, special)

using Vocab = std::unordered_map<std::string, uint32_t>;
using Merges = std::vector<std::pair<std::string, std::string>>;

struct TokenizerConfig {
    Vocab vocab;
    Merges merges;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    std::vector<AddedToken> added_tokens;
    std::wstring pre_tokenizer_pattern;
    std::map<unsigned char, std::string> byte_encoder;
};
