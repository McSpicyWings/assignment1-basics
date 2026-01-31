# export_tokenizer_artifacts.py
import argparse
import json
from pathlib import Path
from cs336_basics.tokenizer.bpe_utilis import train_bpe_func
from cs336_basics.tokenizer.bpe_tokenizer import bytes_to_unicode

def bytes_to_token_str(bs: bytes) -> str:
    b2u = bytes_to_unicode()
    return "".join(b2u[b] for b in bs)

def main():
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer and export vocab/merges files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cs336_basics.tokenizer.trainbpe -i data/TinyStoriesV2-GPT4-train.txt -o data/output
  python -m cs336_basics.tokenizer.trainbpe -i data/owt_train.txt -o tokenizer/ --vocab_size 32000
        """,
    )
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True,
        help="Path to input text file for training"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        required=True,
        help="Output directory for vocab.json and merges.txt"
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=10000,
        help="Target vocabulary size (default: 10000)"
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to include (default: ['<|endoftext|>'])"
    )
    args = parser.parse_args()

    input_path = args.input
    vocab_size = args.vocab_size
    special_tokens = args.special_tokens
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training BPE tokenizer...")
    print(f"  Input: {input_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Special tokens: {special_tokens}")

    vocab, merges = train_bpe_func(input_path, vocab_size, special_tokens)

    # vocab.json: token_str -> id
    encoder = {bytes_to_token_str(tok_bytes): int(tok_id) for tok_id, tok_bytes in vocab.items()}
    (out_dir / "vocab.json").write_text(json.dumps(encoder, ensure_ascii=False), encoding="utf-8")

    # merges.txt: token_str_a token_str_b
    with (out_dir / "merges.txt").open("w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(bytes_to_token_str(a) + " " + bytes_to_token_str(b) + "\n")

    print(f"\nSaved to: {out_dir}")
    print(f"  vocab.json: {len(encoder)} tokens")
    print(f"  merges.txt: {len(merges)} merges")

if __name__ == "__main__":
    main()
