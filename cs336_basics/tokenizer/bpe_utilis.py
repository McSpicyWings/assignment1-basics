from __future__ import annotations

import os
import heapq
from collections import Counter, defaultdict

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries


# pre-tokenizer pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _compile_special_splitter(special_tokens: list[str]) -> re.Pattern | None:
    """把 special tokens 编成一个 split 用的 regex，防止跨 special token 发生统计/合并。
    """
    toks = [t for t in special_tokens if t]
    if not toks:
        return None
    # 用 | 拼接时必须 escape
    return re.compile("|".join(re.escape(t) for t in toks))


def _iter_pairs(word: tuple[bytes, ...]):
    """生成器函数。给出一个word内所有相邻的token pair组合，用于merge"""
    for i in range(len(word) - 1):
        yield (word[i], word[i + 1])


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes], merged_tok: bytes) -> tuple[bytes, ...]:
    """处理单个word，把所有非重叠的 pair=(a,b) 合并成 merged_tok=a+b"""
    a, b = pair
    out: list[bytes] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(merged_tok)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def _count_chunk_pretokens(
    path: str,
    start: int,
    end: int,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:
    """worker: 读取 [start,end) 这段文本，做 special token 切分 + PAT 预分词，返回 word_counts"""
    pat = re.compile(PAT)
    special_splitter = _compile_special_splitter(special_tokens)

    with open(path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    text = chunk_bytes.decode("utf-8", errors="ignore")

    pieces: list[str]
    if special_splitter is None:
        pieces = [text]
    else:
        # 训练阶段：把 special token 当边界切开，避免跨它发生 pre-tokenize / merge
        pieces = special_splitter.split(text)

    counts: Counter[tuple[bytes, ...]] = Counter()
    for piece in pieces:
        if not piece:
            continue
        # finditer避免储存所有匹配
        for m in pat.finditer(piece):
            s = m.group(0)
            bs = s.encode("utf-8")
            # 一个 pre-token 表示为 tuple[bytes]，每个元素是单字节 bytes([b])
            word = tuple(bytes([b]) for b in bs)
            counts[word] += 1
    return counts


def train_bpe_func(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    input_path = os.fspath(input_path)

    # ---------- 0) 参数与初始 vocab ----------
    if vocab_size < 0:
        raise ValueError("vocab_size must be positive")

    vocab: dict[int, bytes] = {}
    token_to_id: dict[bytes, int] = {}

    next_id = 0
    # special tokens 加入 vocab（不参与 merge 的内部拆分；训练统计时作为边界切开）
    # 按输入顺序去重
    seen_special: set[str] = set()
    special_tokens_dedup: list[str] = []
    for t in special_tokens:
        if t not in seen_special:
            seen_special.add(t)
            special_tokens_dedup.append(t)

    for t in special_tokens_dedup:
        tb = t.encode("utf-8")
        if tb not in token_to_id:
            vocab[next_id] = tb
            token_to_id[tb] = next_id
            next_id += 1
    
    # 256 个单字节
    for b in range(256):
        tok = bytes([b])
        vocab[next_id] = tok
        token_to_id[tok] = next_id
        next_id += 1
    
    if vocab_size < next_id:
        raise ValueError(
            f"vocab_size={vocab_size} is too small: need at least {next_id} to fit bytes(256)+special_tokens"
        )

    # 每次 merge 理论上新增 1 个 token, id从0开始
    merges_to_do = vocab_size - next_id  
    merges: list[tuple[bytes, bytes]] = []
    if merges_to_do == 0:
        return vocab, merges

    # ---------- 1) 并行预分词计数 ----------
    num_processes = int(kwargs.get("num_processes", os.cpu_count() or 1))
    num_processes = max(1, num_processes)

    # 用某个 special token 作为 chunk 对齐点；优先用 <|endoftext|>（TinyStories 常用）
    split_tok: bytes | None = None
    if special_tokens_dedup:
        if "<|endoftext|>" in special_tokens_dedup:
            split_tok = b"<|endoftext|>"
        else:
            split_tok = special_tokens_dedup[0].encode("utf-8")

    with open(input_path, "rb") as f:
        if split_tok is not None:
            boundaries = find_chunk_boundaries(
                f,
                desired_num_chunks=num_processes,
                split_special_token=split_tok,
            )
        else:
            # 没有 special token 时的保底分块（不如对齐 special token 稳，但足够用于小测试）
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = max(1, size // num_processes)
            boundaries = [0]
            for i in range(1, num_processes):
                boundaries.append(min(size, i * chunk))
            boundaries.append(size)
            boundaries = sorted(set(boundaries))

    tasks = [(input_path, s, e, special_tokens_dedup) for s, e in zip(boundaries[:-1], boundaries[1:])]

    word_counts: Counter[tuple[bytes, ...]] = Counter()
    if num_processes == 1 or len(tasks) <= 1:
        for (p, s, e, stoks) in tasks:
            word_counts.update(_count_chunk_pretokens(p, s, e, stoks))
    else:
        import multiprocessing as mp

        # pytest 环境一般是 Linux/fork； 这里尽量选择 fork，避免 spawn 反复 import 太慢
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()

        with ctx.Pool(processes=num_processes) as pool:
            for c in pool.starmap(_count_chunk_pretokens, tasks, chunksize=1):
                word_counts.update(c)

    if not word_counts:
        return vocab, merges

    # ---------- 2) 初始化 pair 统计 + 反向索引 ----------
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for word, wcnt in word_counts.items():
        if len(word) < 2:
            continue
        pf = Counter(_iter_pairs(word))  # 统计该 word 内每个 pair 出现次数
        for p, freq in pf.items():
            pair_counts[p] += wcnt * freq
            pair_to_words[p].add(word)

    # ---------- 3) 用 lazy heap 快速取“最高频 + 字典序最大”的 pair ----------
    # lazy:不执行pop
    inv_cache: dict[bytes, tuple[int, ...]] = {}

    # 辅助函数
    # 让“原始 bytes 字典序越大” -> “inv_key 越小”
    # invert
    def inv_key_bytes(tok: bytes) -> tuple[int, ...]:
        k = inv_cache.get(tok)
        if k is None:
            k = tuple(255 - b for b in tok) + (256,)  # 256 作为终止符，反转 prefix 规则
            inv_cache[tok] = k
        return k

    def inv_key_pair(p: tuple[bytes, bytes]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (inv_key_bytes(p[0]), inv_key_bytes(p[1]))

    heap: list[tuple[int, tuple[tuple[int, ...], tuple[int, ...]], tuple[bytes, bytes]]] = []
    for p, cnt in pair_counts.items():
        if cnt > 0:
            # -cnt用于比较次数（heapq默认小根堆） ，inv_key_pair 返回p[0],p[1]的字典序反转，
            heapq.heappush(heap, (-cnt, inv_key_pair(p), p))

    def pop_best_pair() -> tuple[bytes, bytes] | None:
        while heap:
            neg_cnt, _inv, p = heapq.heappop(heap)
            cnt_now = pair_counts.get(p, 0)
            if cnt_now <= 0:
                continue
            if -neg_cnt != cnt_now:
                continue  # lazy：旧条目，跳过
            return p
        return None

    # ---------- 4) 迭代合并 ----------
    for _ in range(merges_to_do):
        best = pop_best_pair()
        if best is None:
            break

        a, b = best
        merged_tok = a + b
        merges.append(best)

        # 新 token 加入 vocab
        if merged_tok not in token_to_id:
            vocab[next_id] = merged_tok
            token_to_id[merged_tok] = next_id
            next_id += 1
            if next_id >= vocab_size:
                # 已达上限（理论上 merges_to_do 已控制，这里只是保险）
                pass

        # 更新所有包含该 pair 的 words
        affected_words = list(pair_to_words.get(best, ()))
        if not affected_words:
            # 可能是 lazy 堆里残留的 pair（已被更新掉）
            pair_counts.pop(best, None)
            pair_to_words.pop(best, None)
            continue

        # best pair 不应该在 merge 后继续出现，所以我们最终会清掉它的反向索引
        for word in affected_words:
            
            # 1) 从统计中移除 old word
            wcnt = word_counts.get(word)
            if wcnt is None:
                continue
            del word_counts[word]

            # old word 里每个 pair 的反向索引里移除，并更新 pair_counts（减去该 word 的贡献）
            old_pf = Counter(_iter_pairs(word))
            for p, freq in old_pf.items():
                # 反向索引移除
                s = pair_to_words.get(p)
                if s is not None:
                    s.discard(word)
                    # 可选：空集合就删，节省内存
                    if not s:
                        pair_to_words.pop(p, None)

                # 计数更新
                new_cnt = pair_counts.get(p, 0) - wcnt * freq
                if new_cnt <= 0:
                    pair_counts.pop(p, None)
                else:
                    pair_counts[p] = new_cnt
                    heapq.heappush(heap, (-new_cnt, inv_key_pair(p), p))

            # 2) 生成 merged 后的新 word
            new_word = _merge_word(word, best, merged_tok)

            # 3) 写回 word_counts（可能与其它 merged 结果碰撞，直接累加即可）
            word_counts[new_word] = word_counts.get(new_word, 0) + wcnt

            # 4) 把 new word 的 pair 贡献加回 pair_counts / pair_to_words
            if len(new_word) >= 2:
                new_pf = Counter(_iter_pairs(new_word))
                for p, freq in new_pf.items():
                    pair_to_words[p].add(new_word)
                    new_cnt = pair_counts.get(p, 0) + wcnt * freq
                    pair_counts[p] = new_cnt
                    heapq.heappush(heap, (-new_cnt, inv_key_pair(p), p))

        # best pair 合并后应消失：清理索引与计数（如果还残留）
        pair_counts.pop(best, None)
        pair_to_words.pop(best, None)

    return vocab, merges
