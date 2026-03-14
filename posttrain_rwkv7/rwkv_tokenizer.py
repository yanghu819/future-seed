# Minimal RWKV world tokenizer (Trie-based), extracted from HF RWKV tokenizer.
# No transformers dependency.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


class _Trie:
    __slots__ = ("ch", "to", "values", "front")

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for _ in range(256)]
        self.values = set()
        self.front = front

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = _Trie(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        u = self
        ch = key[idx]
        ret = None
        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1
            if u.values:
                ret = (idx, u, u.values)
            if idx == len(key):
                break
            ch = key[idx]
        if ret is None:
            raise RuntimeError("Tokenizer trie failed to match any token")
        return ret


@dataclass
class RWKVWorldTokenizer:
    vocab_file: str

    def __post_init__(self):
        self.idx2token: Dict[int, bytes] = {}
        with open(self.vocab_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        sorted_tokens: List[bytes] = []
        for line in lines:
            idx = int(line[: line.index(" ")])
            token = eval(line[line.index(" ") : line.rindex(" ")])
            token = token.encode("utf-8") if isinstance(token, str) else token
            if not isinstance(token, (bytes, bytearray)):
                raise TypeError(f"Bad token type: {type(token)}")
            token = bytes(token)
            token_len = int(line[line.rindex(" ") :])
            assert len(token) == token_len
            sorted_tokens.append(token)
            self.idx2token[idx] = token

        self.token2idx: Dict[bytes, int] = {v: int(k) for k, v in self.idx2token.items()}
        self.root = _Trie()
        for token, idx in self.token2idx.items():
            _ = self.root.add(token, val=(token, idx))

        self.vocab_size = len(self.idx2token)
        self.eot_id = 0

    def encode_bytes(self, src: bytes) -> List[int]:
        idx = 0
        out: List[int] = []
        while idx < len(src):
            idx0 = idx
            idx, _, values = self.root.find_longest(src, idx)
            if idx == idx0:
                raise RuntimeError("Tokenizer did not advance")
            _, token_id = next(iter(values))
            out.append(int(token_id))
        return out

    def decode_bytes(self, tokens: Iterable[int]) -> bytes:
        return b"".join(self.idx2token[int(i)] for i in tokens)

    def encode(self, text: str) -> List[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def decode(self, tokens: List[int]) -> str:
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")
