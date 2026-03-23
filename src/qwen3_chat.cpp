#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "qwen3.h"
#include "tokenizer.h"

struct ChatMessage {
    std::string role;
    std::string content;
};

static std::string BuildPrompt(const std::vector<ChatMessage>& messages, bool add_generation_prompt, bool enable_think = false) {
    std::string prompt;
    for (const auto& msg : messages) {
        prompt += "<|im_start|>" + msg.role + "\n";
        prompt += msg.content;
        prompt += "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        prompt += "<|im_start|>assistant\n";
    }
    if (!enable_think) {
        prompt += "<think>\n\n</think>\n\n";
    }
    return prompt;
}

static bool EndsWithTokenSequence(const std::vector<uint32_t>& generated,
                                  const std::vector<uint32_t>& suffix) {
    if (suffix.empty() || generated.size() < suffix.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), generated.rbegin());
}

static std::string GenerateReply(Qwen3Model& model,
                                 Tokenizer& tokenizer,
                                 const std::vector<ChatMessage>& messages,
                                 size_t max_new_tokens,
                                 const std::vector<uint32_t>& im_end_ids,
                                 bool enable_think = false) {
    const std::string prompt = BuildPrompt(messages, true, enable_think);
    // std::cout << prompt << std::endl;
    const std::vector<uint32_t> prompt_ids = tokenizer.Encode(prompt);
    std::vector<uint32_t> generated_ids;
    generated_ids.reserve(max_new_tokens);

    Sampler sampler;
    std::string reply;
    bool use_cache = true;
    std::cout << std::endl << "Assistant: " << std::endl;
    for (size_t step = 0; step < max_new_tokens; ++step) {
        std::vector<uint32_t> input_ids;
        input_ids.reserve(prompt_ids.size() + generated_ids.size());
        input_ids.insert(input_ids.end(), prompt_ids.begin(), prompt_ids.end());
        input_ids.insert(input_ids.end(), generated_ids.begin(), generated_ids.end());
        size_t position = 0;
        if (use_cache) {
            position = generated_ids.empty() ? 0 : prompt_ids.size() + generated_ids.size() - 1;
            input_ids = generated_ids.empty() ? prompt_ids : std::vector<uint32_t>{generated_ids.back()};
        }
        Tensor logits = model.Forward(input_ids, position);
        uint32_t next_id = sampler.Sample(logits);
        generated_ids.push_back(next_id);

        std::string piece = tokenizer.Decode({next_id});
        reply += piece;

        if (EndsWithTokenSequence(generated_ids, im_end_ids)) {
            break;
        }

        const size_t end_pos = reply.find("<|im_end|>");
        if (end_pos != std::string::npos) {
            reply.resize(end_pos);
            break;
        }

        std::cout << piece << std::flush;
    }
    std::cout << std::endl;

    const size_t end_pos = reply.find("<|im_end|>");
    if (end_pos != std::string::npos) {
        reply.resize(end_pos);
    }

    return reply;
}

int main(int argc, char* argv[]) {
    std::string model_path = "../../Qwen3-0.6B";
    std::string tokenizer_path = "../../Qwen3-0.6B/tokenizer.json";
    size_t max_new_tokens = 1024;

    if (argc > 1) tokenizer_path = argv[1];
    if (argc > 2) model_path = argv[2];
    if (argc > 3) max_new_tokens = static_cast<size_t>(std::stoul(argv[3]));
    bool enable_think = false;

    std::cout << "Loading model from: " << model_path << std::endl;
    Qwen3Model model;
    if (!model.Load(model_path)) {
        std::cerr << "Failed to load model." << std::endl;
        return -1;
    }

    std::cout << "Loading tokenizer from: " << tokenizer_path << std::endl;
    Tokenizer tokenizer;
    tokenizer.LoadConfig(tokenizer_path);

    std::vector<ChatMessage> messages = {
        {"system", "You are a helpful assistant."}
    };

    const std::vector<uint32_t> im_end_ids = tokenizer.Encode("<|im_end|>");

    std::cout << "Commands: /clear to clear history, /exit to quit" << std::endl;

    while (true) {
        std::cout << "You: " << std::endl;
        std::string user_input;
        if (!std::getline(std::cin, user_input)) {
            std::cout << "Bye!" << std::endl;
            break;
        }

        if (user_input.empty()) {
            continue;
        }

        if (user_input == "/exit" || user_input == "exit" || user_input == "quit" || user_input == "/quit") {
            std::cout << "Bye!" << std::endl;
            break;
        }

        if (user_input == "/clear") {
            messages.resize(1);
            std::cout << "History cleared." << std::endl;
            continue;
        }

        messages.push_back({"user", user_input});

        std::string reply = GenerateReply(model, tokenizer, messages, max_new_tokens, im_end_ids);
        messages.push_back({"assistant", reply});
    }

    return 0;
}
