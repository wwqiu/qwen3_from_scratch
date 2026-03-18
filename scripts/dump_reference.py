"""
Dump reference intermediate values from Qwen3 inference using transformers.
Used to validate the C++ implementation.

Usage:
    python dump_reference.py [model_dir]

Default model_dir: ../../Qwen3-0.6B

Requirements:
    pip install transformers torch
"""

import sys
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def save_embedding_output(text: str, ids: list, emb_out: np.ndarray, output_file: str):
    """Save embedding output to txt file, one token per line."""
    with open(output_file, "w") as f:
        f.write(f"# Input: {text}\n")
        f.write(f"# Token IDs: {ids}\n")
        f.write(f"# Shape: {list(emb_out.shape)}\n")
        f.write("#" + "=" * 60 + "\n")
        for token_idx, token_id in enumerate(ids):
            vec = emb_out[token_idx]  # [hidden_dim]
            f.write(f"# Token {token_idx}: id={token_id}\n")
            f.write(" ".join(f"{v:.8f}" for v in vec) + "\n")
    print(f"  → Saved to {output_file}")


def main():
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../Qwen3-0.6B")

    print(f"Model dir: {model_dir.resolve()}")

    # ── Load tokenizer & model ─────────────────
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    print("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    model.eval()

    # ── Test cases (same as C++ main.cpp) ─────
    test_inputs = [
        "hello",
        "广州有哪些好玩的景点？",
        "Café",
        "Hello world",
    ]

    for idx, text in enumerate(test_inputs):
        print("=" * 60)
        print(f"Input: {repr(text)}")

        # Tokenize (add_special_tokens=False to match C++ behavior)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"][0].tolist()
        print(f"Token IDs : {input_ids}")
        print(f"Tokens    : {tokenizer.convert_ids_to_tokens(input_ids)}")

        # Embedding lookup
        with torch.no_grad():
            emb_out = model.model.embed_tokens(inputs["input_ids"])  # [1, seq_len, hidden_dim]

        emb_np = emb_out[0].float().numpy()  # [seq_len, hidden_dim]

        flat = emb_np.flatten()
        print(f"  shape={list(emb_np.shape)}")
        print(f"  first 8: {flat[:8].tolist()}")
        print(f"  last  8: {flat[-8:].tolist()}")
        print(f"  min={flat.min():.6f}  max={flat.max():.6f}  mean={flat.mean():.6f}")

        output_file = f"embedding_output_{idx}.txt"
        save_embedding_output(text, input_ids, emb_np, output_file)
        print()


if __name__ == "__main__":
    main()
