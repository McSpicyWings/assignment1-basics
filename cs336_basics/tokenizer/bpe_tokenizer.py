# class bpe_tokenizer:
#     def __init__(self,vocab,merges,special_tokens=None):
#         """Construct a tokenizer from a given vocabulary, list of merges,
#         and (optionally) a list of special tokens.

#         Args:
#             vocab (dict[int,bytes]) :
#             merges (list[tuple[bytes, bytes]]):
#             special_tokens (list[str] |None=None):
#         """
        
        
#     def from_files(cls, vocab_filepath:str,merges_filepath:str,special_tokens:str|None):
#         '''
#         Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
#         (in the same format that your BPE training code output) and (optionally) a list of special
#         tokens. This method should accept the following additional parameters:
        
#         Args:
#             vocab_filepath (str):
#             merges_filepath (str):
#             special_tokens (list[str] | None = None) :
#         '''

#     def encode(self, text: str)-> list[int]:
#         """Encode an input text into a sequence of token IDs.
#         """
        
#     def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:'''
#     Docstring for encode_iterable
#     '''
    
#     def decode(self, ids:list[int])-> str:
#         """ Decode a sequence of token IDs into text.
#         """
from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable, Iterator

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def bytes_to_unicode() -> dict[int, str]:
    """
    把 0..255 映射到一批“可打印的 unicode 字符”，保证 merges/vocab 可用文本存储。
    """
    bs = list(range(ord("!"), ord("~") + 1)) \
       + list(range(ord("¡"), ord("¬") + 1)) \
       + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]  # codepoints
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# 反向表：unicode字符 -> byte(0..255)
_BYTE_DECODER = {ch: b for b, ch in bytes_to_unicode().items()}


def token_str_to_bytes(token: str) -> bytes:
    """
    把 vocab/merges 文件里的 token（unicode 字符串）还原成真实 bytes。
    其中每个字符都对应一个 byte。
    """
    return bytes(_BYTE_DECODER[c] for c in token)


class bpe_tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        # id -> bytes
        self.id_to_token: dict[int, bytes] = {int(i): bytes(b) for i, b in vocab.items()}
        # bytes -> id
        self.token_to_id: dict[bytes, int] = {b: i for i, b in self.id_to_token.items()}
        if len(self.token_to_id) != len(self.id_to_token):
            raise ValueError("vocab 中存在重复的 token bytes（bytes->id 不能一一对应）")

        # merges + rank（编码时用 rank 查表，比逐条扫 merges 快）
        self.merges = [(bytes(a), bytes(b)) for a, b in merges]
        self.merge_ranks = {pair: r for r, pair in enumerate(self.merges)}

        # special tokens：如果不在 vocab 里就追加
        self.special_tokens: list[str] = []
        self.special_token_to_id: dict[str, int] = {}
        if special_tokens:
            seen = set()
            for t in special_tokens:
                if t not in seen:
                    seen.add(t)
                    self.special_tokens.append(t)

            next_id = (max(self.id_to_token) + 1) if self.id_to_token else 0
            for t in self.special_tokens:
                tb = t.encode("utf-8")
                if tb not in self.token_to_id:
                    self.id_to_token[next_id] = tb
                    self.token_to_id[tb] = next_id
                    next_id += 1
                self.special_token_to_id[t] = self.token_to_id[tb]

        self._pat = re.compile(PAT)

        # 用于识别 special token（encode/encode_iterable）
        if self.special_tokens:
            toks = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile("|".join(re.escape(t) for t in toks))
            self._max_special_len = max(len(t) for t in toks)
            self._special_prefixes = set()
            for t in toks:
                for k in range(1, len(t)):
                    self._special_prefixes.add(t[:k])
        else:
            self._special_re = None
            self._max_special_len = 0
            self._special_prefixes = set()

        # cache：同一个 pre-token 反复出现时很赚
        self._cache: OrderedDict[bytes, tuple[int, ...]] = OrderedDict()
        self._cache_max = 50_000

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "bpe_tokenizer":
        """
        vocab: JSON 文件，格式为 {token_str: id}（如 gpt2_vocab.json）:contentReference[oaicite:3]{index=3}
        merges: 文本文件，每行两个 token_str（空白分隔），如 gpt2_merges.txt :contentReference[oaicite:4]{index=4}
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            encoder: dict[str, int] = json.load(f)

        # 转成作业要求的 id -> bytes
        vocab: dict[int, bytes] = {int(i): token_str_to_bytes(tok) for tok, i in encoder.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split()
                merges.append((token_str_to_bytes(a), token_str_to_bytes(b)))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    # --------- BPE 核心：对一个 pre-token(bytes) 做合并并转成 ids ---------
    def _bpe_bytes_to_ids(self, bs: bytes) -> tuple[int, ...]:
        cached = self._cache.get(bs)
        if cached is not None:
            self._cache.move_to_end(bs)
            return cached

        tokens = [bytes([b]) for b in bs]

        if len(tokens) >= 2 and self.merge_ranks:
            while True:
                best_rank = None
                best_pair = None
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    r = self.merge_ranks.get(pair)
                    if r is None:
                        continue
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best_pair = pair
                if best_pair is None:
                    break

                a, b = best_pair
                merged = a + b
                new_tokens = []
                i = 0
                n = len(tokens)
                while i < n:
                    if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

        ids = []
        for t in tokens:
            tid = self.token_to_id.get(t)
            if tid is None:
                raise KeyError(f"token bytes not in vocab: {t!r}")
            ids.append(tid)

        out = tuple(ids)

        self._cache[bs] = out
        self._cache.move_to_end(bs)
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

        return out

    def _encode_normal_full(self, text: str) -> list[int]:
        out: list[int] = []
        for m in self._pat.finditer(text):
            out.extend(self._bpe_bytes_to_ids(m.group(0).encode("utf-8")))
        return out

    def _encode_normal_stream(self, text: str, final: bool) -> tuple[list[int], str]:
        """
        final=False 时：保留最后一个 pre-token 不输出，避免 chunk 边界改变“是否包含前导空格”等行为。
        """
        if not text:
            return [], ""

        out: list[int] = []
        prev = None
        for m in self._pat.finditer(text):
            if prev is not None:
                out.extend(self._bpe_bytes_to_ids(text[prev.start():prev.end()].encode("utf-8")))
            prev = m

        if prev is None:
            return ([], text) if not final else ([], "")

        if final:
            out.extend(self._bpe_bytes_to_ids(text[prev.start():prev.end()].encode("utf-8")))
            return out, ""
        else:
            return out, text[prev.start():]

    def _encode_mixed_stream(self, text: str, final: bool) -> tuple[list[int], str]:
        if not text:
            return [], ""

        if self._special_re is None:
            return self._encode_normal_stream(text, final)

        out: list[int] = []
        pos = 0
        for m in self._special_re.finditer(text):
            normal = text[pos:m.start()]
            if normal:
                out.extend(self._encode_normal_full(normal))
            out.append(self.special_token_to_id[m.group(0)])
            pos = m.end()

        tail = text[pos:]
        if final:
            if tail:
                out.extend(self._encode_normal_full(tail))
            return out, ""
        else:
            tail_ids, tail_left = self._encode_normal_stream(tail, final=False)
            out.extend(tail_ids)
            return out, tail_left

    def encode(self, text: str) -> list[int]:
        ids, left = self._encode_mixed_stream(text, final=True)
        assert left == ""
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buf = ""
        for chunk in iterable:
            if not chunk:
                continue
            buf += chunk

            # 防止 special token 被切断：保留末尾“可能是 special token 前缀”的部分
            keep_special = 0
            if self._special_prefixes:
                max_check = min(self._max_special_len - 1, len(buf))
                for L in range(1, max_check + 1):
                    if buf[-L:] in self._special_prefixes:
                        keep_special = L

            process = buf[:-keep_special] if keep_special else buf
            suffix_special = buf[-keep_special:] if keep_special else ""

            out_ids, leftover = self._encode_mixed_stream(process, final=False)
            for i in out_ids:
                yield i

            buf = leftover + suffix_special

        if buf:
            for i in self.encode(buf):
                yield i

    def decode(self, ids: list[int]) -> str:
        b = bytearray()
        for i in ids:
            tok = self.id_to_token.get(int(i))
            if tok is None:
                raise KeyError(f"Unknown token id: {i}")
            b.extend(tok)
        return bytes(b).decode("utf-8", errors="replace")