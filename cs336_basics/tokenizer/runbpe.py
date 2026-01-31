from cs336_basics.tokenizer.bpe_utilis import train_bpe_func

# vocab,merges = run_train_bpe_func("data/TinyStoriesV2-GPT4-train.txt",10000,["<|endoftext|>"])
# import json
# from pathlib import Path
# out_dir = Path("../data/artifacts")
# out_dir.mkdir(exist_ok=True, parents=True)
# # 1) vocab: id -> bytes (先转成 str)
# vocab_str = {str(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()}
# (out_dir / "vocab.json").write_text(json.dumps(vocab_str, ensure_ascii=False, indent=2), encoding="utf-8")
# # 2) merges: list[(bytes, bytes)] -> 写成文本，每行两个 token 以空格分隔
# with (out_dir / "merges.txt").open("w", encoding="utf-8") as f:
#     for a, b in merges:
#         f.write(f"{a.decode('utf-8', errors='replace')} {b.decode('utf-8', errors='replace')}\n")
        
vocab,merges = train_bpe_func("data/TinyStoriesV2-GPT4-train.txt",10000,["<|endoftext|>"])
print(len(vocab),len(merges))