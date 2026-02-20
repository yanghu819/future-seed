from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import List, Tuple


def _rand_digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))

_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _to_base36(n: int, width: int) -> str:
    if n < 0:
        raise ValueError(n)
    out = []
    while True:
        n, r = divmod(n, 36)
        out.append(_B36[r])
        if n == 0:
            break
    s = "".join(reversed(out))
    if len(s) > width:
        raise ValueError("width too small")
    return s.rjust(width, "0")


@dataclass
class Sample:
    prompt: str
    answer: str


@dataclass
class MaskedSample:
    text: str
    mask_start: int
    mask_len: int
    target: str


def sample_retrieval(
    rng: random.Random,
    *,
    k: int = 8,
    vlen: int = 6,
    key_width: int = 2,
    query_last: bool = False,
    q_first: bool = False,
) -> Sample:
    # fixed-width base36 keys (ASCII-safe, fixed-length)
    if k > (36**key_width):
        raise ValueError("k too large for key_width")
    keys = [_to_base36(i, key_width) for i in range(k)]
    vals = {kk: _rand_digits(rng, vlen) for kk in keys}
    q = rng.choice(keys)
    # doc first, question at end
    order = keys[:]
    rng.shuffle(order)
    if query_last:
        # Make the task trivially local: ensure the queried pair appears last in the doc.
        order.remove(q)
        order.append(q)
    doc = ";".join([f"{kk}={vals[kk]}" for kk in order])
    if q_first:
        prompt = f"Q={q};{doc};A="
    else:
        prompt = f"{doc};Q={q};A="
    return Sample(prompt=prompt, answer=vals[q])


def sample_retrieval_a2q(
    rng: random.Random,
    *,
    k: int = 8,
    vlen: int = 6,
    key_width: int = 2,
    mask_ch: str = "_",
) -> MaskedSample:
    """Given doc + answer, predict the question key (masked).

    Sequence:
      {doc};Q=__ ;A={value}
    Target: the true key (length=key_width).
    """
    if len(mask_ch) != 1:
        raise ValueError("mask_ch must be 1 char")

    if k > (36**key_width):
        raise ValueError("k too large for key_width")
    keys = [_to_base36(i, key_width) for i in range(k)]
    vals = {kk: _rand_digits(rng, vlen) for kk in keys}
    q = rng.choice(keys)
    v = vals[q]

    order = keys[:]
    rng.shuffle(order)
    doc = ";".join([f"{kk}={vals[kk]}" for kk in order])

    # Place Q at the very beginning so the supervised tokens are in the prefix.
    # This makes the task strongly future-dependent (needs doc + A on the right).
    q_mask = mask_ch * key_width
    prefix = "Q="
    mid = f"{doc};A={v}"
    text = prefix + q_mask + ";" + mid
    mask_start = len(prefix)
    mask_len = key_width
    return MaskedSample(text=text, mask_start=mask_start, mask_len=mask_len, target=q)


def sample_copy_a2q(
    rng: random.Random,
    *,
    key_width: int = 2,
    mask_ch: str = "_",
) -> MaskedSample:
    """Pure future-copy probe: predict masked prefix key from suffix."""
    if len(mask_ch) != 1:
        raise ValueError("mask_ch must be 1 char")
    key = _to_base36(rng.randrange(0, 36**key_width), key_width)
    prefix = "Q="
    text = prefix + (mask_ch * key_width) + f";A={key}"
    return MaskedSample(text=text, mask_start=len(prefix), mask_len=key_width, target=key)


def sample_kvsort(
    rng: random.Random,
    *,
    n: int = 10,
    vlen: int = 4,
    key_max: int = 99,
) -> Sample:
    # numeric keys with 2 digits to avoid ambiguity
    keys = rng.sample(range(0, key_max + 1), n)
    pairs = [(f"{k:02d}", _rand_digits(rng, vlen)) for k in keys]
    rng.shuffle(pairs)

    doc = ",".join([f"{k}:{v}" for k, v in pairs])
    prompt = f"{doc}|Q=SORT|A="

    out = ",".join([v for k, v in sorted(pairs, key=lambda x: x[0])])
    return Sample(prompt=prompt, answer=out)


def sample_nameindex(
    rng: random.Random,
    *,
    n: int = 16,
    name_width: int = 2,
    idx_width: int = 2,
    q_first: bool = False,
) -> Sample:
    """NameIndex from arXiv:2512.14982 (fixed-format ASCII).

    Given a list of "names", query asks for the name at index i (0-indexed).

    - Doc-first (default): N=..;Q=I=..;A=   (causal-unfriendly)
    - Q-first (--q_first): Q=I=..;N=..;A=   (control)
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if n > (36**name_width):
        raise ValueError("n too large for name_width")

    # Fixed-width "names" from base36 IDs.
    pool = list(range(36**name_width))
    rng.shuffle(pool)
    ids = pool[:n]
    names = [_to_base36(x, name_width) for x in ids]

    i = rng.randrange(0, n)
    idx = _to_base36(i, idx_width)
    doc = ",".join(names)

    if q_first:
        prompt = f"Q=I={idx};N={doc};A="
    else:
        prompt = f"N={doc};Q=I={idx};A="

    return Sample(prompt=prompt, answer=names[i])


def sample_middlematch(
    rng: random.Random,
    *,
    n: int = 24,
    name_width: int = 2,
    q_first: bool = False,
) -> Sample:
    """MiddleMatch from arXiv:2512.14982 (fixed-format ASCII).

    Given a list of "names" (a0,a1,...), query asks:
      Which name is between L and R in the list?
    We choose a random interior position k and set (L, M, R) = (names[k-1], names[k], names[k+1]).
    Answer is M.
    """
    if n < 3:
        raise ValueError("n must be >= 3")
    if n > (36**name_width):
        raise ValueError("n too large for name_width")

    pool = list(range(36**name_width))
    rng.shuffle(pool)
    ids = pool[:n]
    names = [_to_base36(x, name_width) for x in ids]

    k = rng.randrange(1, n - 1)
    left = names[k - 1]
    mid = names[k]
    right = names[k + 1]

    doc = ",".join(names)
    if q_first:
        prompt = f"Q=L={left};R={right};N={doc};A="
    else:
        prompt = f"N={doc};Q=L={left};R={right};A="
    return Sample(prompt=prompt, answer=mid)
