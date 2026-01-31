import os
import numpy as np
from pathlib import Path
from cs336_basics.tokenizer.bpe_tokenizer import bpe_tokenizer

def txt_to_bin_and_npy(
    txt_path: str,
    vocab_json: str,
    merges_txt: str,
    out_prefix: str,
    special_tokens = ["<|endoftext|>"],
    dtype = np.uint16,
    chunk_lines: int = 2000,
):
    tok = bpe_tokenizer.from_files(vocab_json, merges_txt, special_tokens=special_tokens)

    txt_path = Path(txt_path)
    out_prefix = Path(out_prefix)
    bin_path = out_prefix.with_suffix(".bin")
    npy_path = out_prefix.with_suffix(".npy")

    # 1) 写 .bin（连续 token ids）
    with txt_path.open("r", encoding="utf-8") as f_in, open(bin_path, "wb") as f_out:
        buf = []
        for line in f_in:
            buf.append(line)  # 保留原换行（line 自带 \n）
            if len(buf) >= chunk_lines:
                ids = list(tok.encode_iterable(buf))
                np.asarray(ids, dtype=dtype).tofile(f_out)
                buf.clear()

        if buf:
            ids = list(tok.encode_iterable(buf))
            np.asarray(ids, dtype=dtype).tofile(f_out)

    # 2) bin -> memmap -> 保存为 npy
    n = os.path.getsize(bin_path) // np.dtype(dtype).itemsize
    mm = np.memmap(bin_path, dtype=dtype, mode="r", shape=(n,))
    np.save(npy_path, mm)  # 会按顺序写出 npy

    print("done:")
    print("  bin:", bin_path, "tokens:", n, "dtype:", dtype)
    print("  npy:", npy_path)

if __name__ == "__main__":
    txt_path = "path/to/TinyStoriesV2-GPT4-train.txt"
    vocab_json = "path/to/vocab.json"
    merges_txt = "path/to/merges.txt"
    out_prefix = "path/to/tinystories_train"  

    txt_to_bin_and_npy(txt_path, vocab_json, merges_txt, out_prefix)
