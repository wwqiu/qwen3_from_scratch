#include "tokenizer.h"
#include "nlohmann/json.hpp"
#include <boost/regex.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <codecvt>
#include <locale>

using json = nlohmann::json;

// Utility: Convert wstring to UTF-8 string (cross-platform)
std::string WideToUTF8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.to_bytes(wstr);
}

// Utility: Convert UTF-8 string to wstring (cross-platform)
std::wstring UTF8ToWide(const std::string& str) {
    if (str.empty()) return std::wstring();
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(str);
}

// Create byte-to-unicode mapping for BPE
std::map<unsigned char, std::string> CreateByteToUnicodeMap() {
    std::map<unsigned char, std::string> byte_encoder;
    std::vector<int> bs;

    // Include printable ASCII characters (33-126)
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
    // Include Latin-1 Supplement printable characters
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;

    // Fill remaining bytes with unique Unicode characters
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    // Build mapping: byte_value -> UTF-8 string
    for (size_t i = 0; i < bs.size(); ++i) {
        unsigned char byte_val = static_cast<unsigned char>(bs[i]);
        int unicode_val = cs[i];

        std::string utf8_char;
        if (unicode_val < 0x80) {
            utf8_char += (char)unicode_val;
        }
        else if (unicode_val < 0x800) {
            utf8_char += (char)(0xC0 | (unicode_val >> 6));
            utf8_char += (char)(0x80 | (unicode_val & 0x3F));
        }
        else {
            utf8_char += (char)(0xE0 | (unicode_val >> 12));
            utf8_char += (char)(0x80 | ((unicode_val >> 6) & 0x3F));
            utf8_char += (char)(0x80 | (unicode_val & 0x3F));
        }

        byte_encoder[byte_val] = utf8_char;
    }
    return byte_encoder;
}

void Tokenizer::LoadConfig(const std::string& config_file) {
    try {
        std::ifstream ifs(config_file);
        if (!ifs.good()) {
            std::cerr << "Failed to open config file: " << config_file << std::endl;
            return;
        }
        auto j = json::parse(ifs);

        config_.byte_encoder = CreateByteToUnicodeMap();
        config_.added_tokens = j.at("added_tokens").get<std::vector<AddedToken>>();
        config_.merges = j.at("model").at("merges").get<Merges>();

        for (size_t i = 0; i < config_.merges.size(); ++i) {
            std::string p1 = config_.merges[i].first;
            std::string p2 = config_.merges[i].second;
            config_.bpe_ranks[{p1, p2}] = (int)i;
        }

        config_.vocab = j.at("model").at("vocab").get<Vocab>();
        token_to_id_.clear();
        id_to_token_.clear();
        special_token_to_id_.clear();
        special_token_ids_.clear();
        special_tokens_sorted_.clear();
        for (const auto& kv : config_.vocab) {
            token_to_id_[kv.first] = kv.second;
            id_to_token_[kv.second] = kv.first;
        }
        for (const auto& token : config_.added_tokens) {
            const std::string content = WideToUTF8(token.content);
            token_to_id_[content] = static_cast<uint32_t>(token.id);
            id_to_token_[static_cast<uint32_t>(token.id)] = content;
            if (token.special) {
                special_token_to_id_[content] = static_cast<uint32_t>(token.id);
                special_token_ids_.insert(static_cast<uint32_t>(token.id));
                special_tokens_sorted_.push_back(content);
            }
        }
        std::sort(special_tokens_sorted_.begin(), special_tokens_sorted_.end(),
                  [](const std::string& a, const std::string& b) {
                      return a.size() > b.size();
                  });

        std::wstring w_regex = j.at("pre_tokenizer")
            .at("pretokenizers").at(0)
            .at("pattern")
            .at("Regex").get<std::wstring>();

        config_.pre_tokenizer_pattern = w_regex;
    }
    catch (std::exception& e) {
        std::cerr << "Error loading tokenizer config: " << e.what() << std::endl;
    }
}

std::vector<std::wstring> Tokenizer::PreTokenize(const std::wstring& wtext) {
    const std::wstring pattern_str =
        L"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        L"[^\\r\\n[:alpha:][:digit:]]?[[:alpha:]]+|"
        L"[[:digit:]]|"
        L" ?[^\\s[:alpha:][:digit:]]+[\\r\\n]*|"
        L"\\s*[\\r\\n]+|"
        L"\\s+(?!\\S)|"
        L"\\s+";

    boost::wregex re;
    try {
        re.assign(pattern_str, boost::regex::perl);
    }
    catch (const boost::regex_error& e) {
        std::cerr << "Regex Error: " << e.what() << std::endl;
        return {};
    }

    std::vector<std::wstring> tokens;
    boost::wsregex_token_iterator it(wtext.begin(), wtext.end(), re, 0);
    boost::wsregex_token_iterator end;

    for (; it != end; ++it) {
        tokens.push_back(it->str());
    }

    return tokens;
}

void Tokenizer::BPETokenize(const std::string& utf8_word, std::vector<uint32_t>& tokens) {
    if (utf8_word.empty()) return;

    std::vector<std::string> word_parts;
    word_parts.reserve(utf8_word.size());

    std::string full_mapped_word = "";

    for (unsigned char c : utf8_word) {
        std::string mapped_char = config_.byte_encoder.at(c);
        word_parts.push_back(mapped_char);
        full_mapped_word += mapped_char;
    }

    // Try direct lookup
    if (config_.vocab.find(full_mapped_word) != config_.vocab.end()) {
        tokens.push_back(config_.vocab.at(full_mapped_word));
        return;
    }

    // BPE merge loop
    while (word_parts.size() > 1) {
        int min_rank = -1;
        int best_idx = -1;
        std::pair<std::string, std::string> best_pair;

        for (size_t i = 0; i < word_parts.size() - 1; ++i) {
            std::pair<std::string, std::string> pair = { word_parts[i], word_parts[i + 1] };
            auto it = config_.bpe_ranks.find(pair);
            if (it != config_.bpe_ranks.end()) {
                int rank = it->second;
                if (min_rank == -1 || rank < min_rank) {
                    min_rank = rank;
                    best_idx = i;
                    best_pair = pair;
                }
            }
        }

        if (best_idx == -1) break;

        word_parts[best_idx] = best_pair.first + best_pair.second;
        word_parts.erase(word_parts.begin() + best_idx + 1);
    }

    // Convert to token IDs
    for (const auto& part : word_parts) {
        auto it = config_.vocab.find(part);
        if (it != config_.vocab.end()) {
            tokens.push_back(it->second);
        }
        else {
            std::cerr << "[UNK: " << part << "] ";
            if (config_.vocab.count("<unk>")) tokens.push_back(config_.vocab.at("<unk>"));
            else if (config_.vocab.count("<|endoftext|>")) tokens.push_back(config_.vocab.at("<|endoftext|>"));
            else tokens.push_back(0);
        }
    }
}

void Tokenizer::EncodeNormalText(const std::string& text, std::vector<uint32_t>& ids) {
    if (text.empty()) return;
    std::wstring wtext = UTF8ToWide(text);
    std::vector<std::wstring> pre_tokens = PreTokenize(wtext);

    for (const auto& w : pre_tokens) {
        std::string utf8_part = WideToUTF8(w);
        BPETokenize(utf8_part, ids);
    }
}

std::vector<uint32_t> Tokenizer::Encode(const std::string& text) {
    std::vector<uint32_t> ids;
    std::string normal_buffer;
    size_t pos = 0;
    while (pos < text.size()) {
        bool matched_special = false;
        for (const auto& special : special_tokens_sorted_) {
            if (special.empty() || pos + special.size() > text.size()) {
                continue;
            }
            if (text.compare(pos, special.size(), special) == 0) {
                EncodeNormalText(normal_buffer, ids);
                normal_buffer.clear();
                ids.push_back(special_token_to_id_.at(special));
                pos += special.size();
                matched_special = true;
                break;
            }
        }
        if (!matched_special) {
            normal_buffer.push_back(text[pos]);
            ++pos;
        }
    }
    EncodeNormalText(normal_buffer, ids);
    return ids;
}

std::string Tokenizer::DecodeByteLevelText(const std::string& byte_level_text) {
    static std::unordered_map<uint32_t, uint8_t> unicode_to_byte;
    if (unicode_to_byte.empty()) {
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool is_direct = (b >= 33 && b <= 126) || 
                             (b >= 161 && b <= 172) || 
                             (b >= 174 && b <= 255);
            if (is_direct) {
                unicode_to_byte[b] = static_cast<uint8_t>(b);
            } else {
                unicode_to_byte[256 + n] = static_cast<uint8_t>(b);
                n++;
            }
        }
    }

    std::string raw_bytes_text;
    for (size_t i = 0; i < byte_level_text.size(); ) {
        unsigned char c = byte_level_text[i];
        uint32_t cp = 0;
        int bytes = 0;
        
        if (c <= 0x7F) {
            cp = c;
            bytes = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            bytes = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            bytes = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            bytes = 4;
        } else {
            cp = c;
            bytes = 1;
        }

        for (int j = 1; j < bytes && i + j < byte_level_text.size(); ++j) {
            cp = (cp << 6) | (byte_level_text[i + j] & 0x3F);
        }

        if (unicode_to_byte.count(cp)) {
            raw_bytes_text.push_back(static_cast<char>(unicode_to_byte[cp]));
        }

        i += bytes;
    }

    return raw_bytes_text;
}

std::string Tokenizer::Decode(const std::vector<uint32_t>& token_ids) {
    std::string output;
    std::string byte_level_text;
    for (uint32_t id : token_ids) {
        auto it = id_to_token_.find(id);
        if (it == id_to_token_.end()) {
            continue;
        }
        if (special_token_ids_.count(id)) {
            output += DecodeByteLevelText(byte_level_text);
            byte_level_text.clear();
            output += it->second;
        }
        else {
            byte_level_text += it->second;
        }
    }
    output += DecodeByteLevelText(byte_level_text);
    return output;
}
