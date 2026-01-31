#!/usr/bin/env python3
"""
Generation script for trained Transformer Language Model.

Usage:
    python generate.py --prompt "Once upon a time" --max_new_tokens 100
"""
import argparse
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cs336_basics.layer.transformer import TransformerLM
from cs336_basics.tokenizer.bpe_tokenizer import bpe_tokenizer
from cs336_basics.tokenizer.decoding import decode


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("--checkpoint", type=str, default="final.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to config file")
    parser.add_argument("--vocab", type=str, default="../data/vocab_TinyStoriesV2-GPT4-train_myself.json",
                        help="Path to vocab file")
    parser.add_argument("--merges", type=str, default="../data/merges_TinyStoriesV2-GPT4-train_myself.json",
                        help="Path to merges file")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt text to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling threshold")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ===== 1) Load config =====
    with open(args.config, "r") as f:
        config = json.load(f)
    model_config = config["model_config"]
    print(f"Model config: {model_config}")

    # ===== 2) Build model =====
    model = TransformerLM(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config.get("rope_theta", 10000.0),
        device=device,
        dtype=torch.float32,
    )
    model.eval()

    # ===== 3) Load checkpoint =====
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)  # Ensure all parameters are on device
    print("Checkpoint loaded successfully!")

    # ===== 4) Load tokenizer =====
    print(f"Loading tokenizer from {args.vocab} and {args.merges}")
    
    # Import the byte decoder for GPT-2 encoding
    from cs336_basics.tokenizer.bpe_tokenizer import bytes_to_unicode
    
    # Build decoder: unicode char -> byte
    _byte_encoder = bytes_to_unicode()  # byte -> unicode char
    _byte_decoder = {ch: b for b, ch in _byte_encoder.items()}  # unicode char -> byte
    
    def token_str_to_bytes_safe(token: str) -> bytes:
        """Convert GPT-2 encoded token string to raw bytes, handling special tokens."""
        # Special tokens like <|endoftext|> are stored as-is (UTF-8)
        if token.startswith("<|") and token.endswith("|>"):
            return token.encode("utf-8")
        # Regular tokens use GPT-2 byte encoding
        try:
            return bytes(_byte_decoder[c] for c in token)
        except KeyError:
            # Fallback: treat as UTF-8
            return token.encode("utf-8")
    
    # Load vocab: {token_str: id} 
    with open(args.vocab, "r", encoding="utf-8") as f:
        vocab_str_to_id: dict[str, int] = json.load(f)
    
    # Convert to {id: bytes}
    vocab: dict[int, bytes] = {}
    for tok_str, tok_id in vocab_str_to_id.items():
        vocab[int(tok_id)] = token_str_to_bytes_safe(tok_str)
    
    # Load merges: text file with "a b" per line (GPT-2 byte encoding)
    merges: list[tuple[bytes, bytes]] = []
    with open(args.merges, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split(' ')
            if len(parts) >= 2:
                a = token_str_to_bytes_safe(parts[0])
                b = token_str_to_bytes_safe(parts[1])
                merges.append((a, b))
    
    # Create tokenizer
    tokenizer = bpe_tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])
    
    # Get EOS token ID
    eos_token_id = tokenizer.special_token_to_id.get("<|endoftext|>", 0)
    print(f"EOS token ID: {eos_token_id}")

    # ===== 5) Encode prompt =====
    print(f"\nPrompt: {args.prompt}")
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"Prompt token IDs: {prompt_ids}")
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    # ===== 6) Generate =====
    print(f"\nGenerating with temperature={args.temperature}, top_p={args.top_p}...")
    with torch.no_grad():
        output_ids = decode(
            model,
            prompt_tensor,
            max_new_tokens=args.max_new_tokens,
            context_length=model_config["context_length"],
            eos_token_id=eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    # ===== 7) Decode output =====
    output_list = output_ids.tolist()
    print(f"\nGenerated token IDs: {output_list}")
    
    output_text = tokenizer.decode(output_list)
    print(f"\n{'='*50}")
    print("Generated text:")
    print(f"{'='*50}")
    print(output_text)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
