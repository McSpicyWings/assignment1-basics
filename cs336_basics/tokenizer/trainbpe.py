# export_tokenizer_artifacts.py
import json
from pathlib import Path
from cs336_basics.tokenizer.bpe_utilis import train_bpe_func
from cs336_basics.tokenizer.bpe_tokenizer import bytes_to_unicode

def bytes_to_token_str(bs: bytes) -> str:
    b2u = bytes_to_unicode()
    return "".join(b2u[b] for b in bs)

def main():
    input_path = "path/to/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    out_dir = Path("path/to/out")

    vocab, merges = train_bpe_func(input_path, vocab_size, special_tokens)

    # vocab.json: token_str -> id
    encoder = {bytes_to_token_str(tok_bytes): int(tok_id) for tok_id, tok_bytes in vocab.items()}
    (out_dir / "vocab.json").write_text(json.dumps(encoder, ensure_ascii=False), encoding="utf-8")

    # merges.txt: token_str_a token_str_b
    with (out_dir / "merges.txt").open("w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(bytes_to_token_str(a) + " " + bytes_to_token_str(b) + "\n")

    print("saved to:", out_dir)

if __name__ == "__main__":
    main()
