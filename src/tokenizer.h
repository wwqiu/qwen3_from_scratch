#pragma once
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"

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

class Tokenizer {
   public:
    Tokenizer() = default;

    void LoadConfig(const std::string& config_file);
    std::vector<uint32_t> Encode(const std::string& text);

    std::string Decode(const std::vector<uint32_t>& token_ids);

   private:
    TokenizerConfig config_;
    std::unordered_map<uint32_t, std::string> id_to_token_;
    std::unordered_set<uint32_t> special_token_ids_;

    std::vector<std::wstring> PreTokenize(const std::wstring& text);
    void EncodeNormalText(const std::string& text, std::vector<uint32_t>& ids);
    std::string DecodeByteLevelText(const std::string& byte_level_text);
    void BPETokenize(const std::string& utf8_word, std::vector<uint32_t>& tokens);
};

// Utility functions
std::string WideToUTF8(const std::wstring& wstr);
std::wstring UTF8ToWide(const std::string& str);
std::map<unsigned char, std::string> CreateByteToUnicodeMap();
