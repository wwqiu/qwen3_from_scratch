#pragma once
#include <array>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
#include <codecvt>
#include <locale>

namespace nlohmann {
    template <>
    struct adl_serializer<std::wstring> {
        static void from_json(const json& j, std::wstring& wstr) {
            std::string s_utf8 = j.get<std::string>();
            std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
            wstr = converter.from_bytes(s_utf8);
        }
    };

    template <>
    struct adl_serializer<std::unordered_map<std::wstring, uint32_t>> {
        static void from_json(const json& j, std::unordered_map<std::wstring, uint32_t>& m) {
            for (auto& element : j.items()) {
                std::string key_utf8 = element.key();
                std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
                std::wstring key_w = converter.from_bytes(key_utf8);
                uint32_t val = element.value().get<uint32_t>();
                m[key_w] = val;
            }
        }
    };
}

struct AddedToken {
    int id;
    std::wstring content;
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

struct Tensor {
    std::array<int, 4> shape;
    uint8_t* data;
};
