import os, time, math, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def env_int(name, default):
    return int(os.getenv(name, default))


def env_float(name, default):
    return float(os.getenv(name, default))


HEAD_SIZE = env_int("HEAD_SIZE", 64)
SEQ_LEN = env_int("SEQ_LEN", 1024)
N_LAYER = env_int("N_LAYER", 12)
N_EMBD = env_int("N_EMBD", 768)
BATCH_SIZE = env_int("BATCH_SIZE", 512)
DEVICE_BSZ = env_int("DEVICE_BSZ", 64)
MAX_ITERS = env_int("MAX_ITERS", 3200)
EVAL_INTERVAL = env_int("EVAL_INTERVAL", 125)
EVAL_ITERS = env_int("EVAL_ITERS", 200)
WARMUP_ITERS = env_int("WARMUP_ITERS", 0)
WARMDOWN_ITERS = env_int("WARMDOWN_ITERS", 914)
MUON_LR = env_float("MUON_LR", 0.02)
ADAM_LR = env_float("ADAM_LR", 0.0026)
LN_LR = env_float("LN_LR", 0.0090)
WTE_LR = env_float("WTE_LR", 0.3)
HEAD_LR = env_float("HEAD_LR", 0.002)
LAMBDA_LR = env_float("LAMBDA_LR", 0.02)
DW_JRT = env_int("DW_JRT", 0) == 1
DW_JRT_SCALE = env_float("DW_JRT_SCALE", 1.0)
DW_JRT_ALPHA_INIT = env_float("DW_JRT_ALPHA_INIT", 0.0)
TRAIN = env_int("TRAIN", 0) == 1
MEM_CHECK = env_int("MEM_CHECK", 0) == 1
REVERSE_TASK = env_int("REVERSE_TASK", 0) == 1
REVERSE_DIGIT_MAX = env_int("REVERSE_DIGIT_MAX", 60)
REVERSE_EVAL = env_int("REVERSE_EVAL", 0) == 1
REVERSE_MASK_REVERSE = env_int("REVERSE_MASK_REVERSE", 0) == 1
BIDIR_TASK = env_int("BIDIR_TASK", 0) == 1
BIDIR_LEN = env_int("BIDIR_LEN", 16)
BIDIR_EVAL = env_int("BIDIR_EVAL", 0) == 1
BIDIR_MASK_MIDDLE = env_int("BIDIR_MASK_MIDDLE", 1) == 1
BIDIR_BASE = env_int("BIDIR_BASE", 10)
ADD_TASK = env_int("ADD_TASK", 0) == 1
ADD_LEN = env_int("ADD_LEN", 8)
ADD_MASK_LEN = env_int("ADD_MASK_LEN", 4)
ADD_EVAL = env_int("ADD_EVAL", 0) == 1
PERM_TASK = env_int("PERM_TASK", 0) == 1
PERM_LEN = env_int("PERM_LEN", 8)
PERM_EVAL = env_int("PERM_EVAL", 0) == 1
INTER_TASK = env_int("INTER_TASK", 0) == 1
INTER_LEN = env_int("INTER_LEN", 8)
INTER_EVAL = env_int("INTER_EVAL", 0) == 1
MULTI_TASK = env_int("MULTI_TASK", 0) == 1
MULTI_LEN = env_int("MULTI_LEN", 8)
MULTI_EVAL = env_int("MULTI_EVAL", 0) == 1
PARITY_TASK = env_int("PARITY_TASK", 0) == 1
PARITY_BLOCK = env_int("PARITY_BLOCK", 4)
PARITY_BLOCKS = env_int("PARITY_BLOCKS", 8)
PARITY_MASK_POS = env_int("PARITY_MASK_POS", 1)
PARITY_EVAL = env_int("PARITY_EVAL", 0) == 1
STRUCT_TASK = env_int("STRUCT_TASK", 0) == 1
STRUCT_LEN = env_int("STRUCT_LEN", 8)
STRUCT_EVAL = env_int("STRUCT_EVAL", 0) == 1
STRUCT_EVAL_FIXED = env_int("STRUCT_EVAL_FIXED", 0) == 1
STRUCT_EVAL_SEED = env_int("STRUCT_EVAL_SEED", 1234)
STRUCT_EVAL_N = env_int("STRUCT_EVAL_N", 200)
RETR_TASK = env_int("RETR_TASK", 0) == 1
RETR_K = env_int("RETR_K", 4)
RETR_VLEN = env_int("RETR_VLEN", 4)
RETR_EVAL = env_int("RETR_EVAL", 0) == 1
SENT_TASK = env_int("SENT_TASK", 0) == 1
SENT_MASK_LEN = env_int("SENT_MASK_LEN", 16)
SENT_EVAL = env_int("SENT_EVAL", 0) == 1
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "weights/diffusion.pt")
DATA_PATH = os.getenv("DATA_PATH", "../tiny-diffusion/data.txt")
DATA_BIN = os.getenv("DATA_BIN", "")
DATA_VAL_BIN = os.getenv("DATA_VAL_BIN", "")
VOCAB_SIZE = env_int("VOCAB_SIZE", 50304)
GEN_TOKENS = env_int("GEN_TOKENS", 2000)
PROMPT_LEN = env_int("PROMPT_LEN", 16)

random_seed = env_int("RANDOM_SEED", 1337)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


device = (
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)
muon_dtype = torch.bfloat16 if device == "cuda" else torch.float32

use_bin = len(DATA_BIN) > 0 and not REVERSE_TASK and not BIDIR_TASK and not STRUCT_TASK and not RETR_TASK and not SENT_TASK
use_bin = (
    use_bin
    and not ADD_TASK
    and not PERM_TASK
    and not INTER_TASK
    and not MULTI_TASK
    and not PARITY_TASK
)


if STRUCT_TASK:
    vocab_base = [str(i) for i in range(10)] + ["a", "b", "c", "{", "}", ":", ",", "\"", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _struct_sample(n):
        a = "".join([str(random.randint(0, 9)) for _ in range(n)])
        c = "".join([str(random.randint(0, 9)) for _ in range(n)])
        b = a + c
        s = "{\"a\":" + a + ",\"b\":" + b + ",\"c\":" + c + "}"
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _struct_sample(STRUCT_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("\"b\":")
            p2 = s.find(",\"c\":")
            if p1 >= 0 and p2 > p1:
                mask[i, p1 + 4 : p2] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)

    STRUCT_EVAL_SET = None
elif RETR_TASK:
    keys = [chr(ord("a") + i) for i in range(RETR_K)]
    vocab_base = [str(i) for i in range(10)] + keys + ["=", ";", "|", "Q", "A", "N", "S", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _retr_sample():
        vals = {}
        for k in keys:
            vals[k] = "".join([str(random.randint(0, 9)) for _ in range(RETR_VLEN)])
        q = random.choice(keys)
        left = ";".join([f"{k}={vals[k]}" for k in keys])
        s = left + "|ANS=" + vals[q] + "|Q=" + q
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _retr_sample()
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("|ANS=")
            p2 = s.find("|Q=")
            if p1 >= 0 and p2 > p1:
                mask[i, p1 + 5 : p2] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif PERM_TASK:
    vocab_base = [str(i) for i in range(10)] + ["P", "Y", "A", "=", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _perm_sample(n):
        a = "".join([str(random.randint(0, 9)) for _ in range(n)])
        perm = list(range(n))
        random.shuffle(perm)
        p = "".join([str(i) for i in perm])
        y = "".join([a[i] for i in perm])
        s = "P=" + p + "|Y=" + y + "|A=" + a
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _perm_sample(PERM_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("|A=")
            if p1 >= 0:
                mask[i, p1 + 3 : p1 + 3 + PERM_LEN] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif ADD_TASK:
    vocab_base = [str(i) for i in range(10)] + ["+", "=", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _add_sample(n):
        a = "".join([str(random.randint(0, 9)) for _ in range(n)])
        b = "".join([str(random.randint(0, 9)) for _ in range(n)])
        c = str(int(a) + int(b)).zfill(n + 1)
        s = a + "+" + b + "=" + c
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _add_sample(ADD_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("=")
            if p1 >= 0:
                start = p1 + 1 + (ADD_LEN + 1 - ADD_MASK_LEN) // 2
                mask[i, start : start + ADD_MASK_LEN] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif INTER_TASK:
    vocab_base = [str(i) for i in range(10)] + ["A", "B", "M", "=", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _inter_sample(n):
        a = "".join([str(random.randint(0, 9)) for _ in range(n)])
        b = "".join([str(random.randint(0, 9)) for _ in range(n)])
        m = "".join([a[i] + b[i] for i in range(n)])
        s = "A=" + a + "|B=" + b + "|M=" + m
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _inter_sample(INTER_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("|M=")
            if p1 >= 0:
                mask[i, p1 + 3 : p1 + 3 + 2 * INTER_LEN] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif MULTI_TASK:
    vocab_base = [str(i) for i in range(10)] + ["A", "B", "C", "D", "=", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _multi_sample(n):
        a = "".join([str(random.randint(0, 9)) for _ in range(n)])
        c = "".join([str(random.randint(0, 9)) for _ in range(n)])
        b = a + c
        d = a[::-1] + c
        s = "A=" + a + "|C=" + c + "|B=" + b + "|D=" + d
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _multi_sample(MULTI_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("|B=")
            if p1 >= 0:
                mask[i, p1 + 3 : p1 + 3 + 2 * MULTI_LEN] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif PARITY_TASK:
    vocab_base = ["0", "1", "X", "P", "=", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _parity_sample(block, blocks):
        x = "".join([str(random.randint(0, 1)) for _ in range(block * blocks)])
        p = []
        for i in range(blocks):
            seg = x[i * block : (i + 1) * block]
            p.append(str(sum(int(ch) for ch in seg) % 2))
        p = "".join(p)
        s = "X=" + x + "|P=" + p
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _parity_sample(PARITY_BLOCK, PARITY_BLOCKS)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("X=")
            if p1 >= 0:
                start = p1 + 2
                for b in range(PARITY_BLOCKS):
                    pos = start + b * PARITY_BLOCK + PARITY_MASK_POS
                    mask[i, pos] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif BIDIR_TASK:
    vocab_base = [str(i) for i in range(BIDIR_BASE)] + ["|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _bidir_sample(n):
        a = [str(random.randint(0, BIDIR_BASE - 1)) for _ in range(n)]
        b = [str(random.randint(0, BIDIR_BASE - 1)) for _ in range(n)]
        x = [str((int(a[i]) + int(b[i])) % BIDIR_BASE) for i in range(n)]
        s = "".join(a) + "|" + "".join(x) + "|" + "".join(b)
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _bidir_sample(BIDIR_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            if BIDIR_MASK_MIDDLE:
                p1 = s.find("|")
                p2 = s.find("|", p1 + 1)
                if p1 >= 0 and p2 > p1:
                    mask[i, p1 + 1 : p2] = True
        y = x.clone()
        if not BIDIR_MASK_MIDDLE:
            mask_probs = torch.rand(DEVICE_BSZ, 1)
            mask = torch.rand(DEVICE_BSZ, SEQ_LEN) < mask_probs
            missing = ~mask.any(dim=1)
            if missing.any():
                idx = torch.randint(0, SEQ_LEN, (int(missing.sum()),))
                mask[missing, idx] = True
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif REVERSE_TASK:
    vocab_base = [str(i) for i in range(10)] + [",", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _reverse_sample(digits):
        lo = 0 if digits == 1 else 10 ** (digits - 1)
        raw = str(random.randint(lo, 10 ** digits - 1))
        s = raw + "," + raw[::-1]
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _reverse_sample(random.randint(1, REVERSE_DIGIT_MAX))
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            if REVERSE_MASK_REVERSE:
                comma = s.find(",")
                sharp = s.find("#")
                if comma >= 0 and sharp > comma:
                    mask[i, comma + 1 : sharp] = True
        y = x.clone()
        if not REVERSE_MASK_REVERSE:
            mask_probs = torch.rand(DEVICE_BSZ, 1)
            mask = torch.rand(DEVICE_BSZ, SEQ_LEN) < mask_probs
            missing = ~mask.any(dim=1)
            if missing.any():
                idx = torch.randint(0, SEQ_LEN, (int(missing.sum()),))
                mask[missing, idx] = True
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif use_bin:
    def _load_data_shard(filename):
        with open(filename, "rb") as f:
            _ = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            tokens = np.frombuffer(f.read(), dtype=np.uint16)
        return tokens

    class BinDataLoader:
        def __init__(self, filename_pattern, B, T):
            self.B = B
            self.T = T
            self.files = sorted(glob.glob(filename_pattern))
            self.current_shard = 0
            self.current_position = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])

        def advance(self):
            self.current_shard = (self.current_shard + 1) % len(self.files)
            self.current_position = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])

        def next_batch(self):
            buf = self.tokens[self.current_position : self.current_position + self.B * self.T]
            if buf.shape[0] < self.B * self.T:
                self.advance()
                buf = self.tokens[self.current_position : self.current_position + self.B * self.T]
            x = torch.tensor(buf.astype(np.int32), dtype=torch.long).view(self.B, self.T)
            self.current_position += self.B * self.T
            return x

    train_loader = BinDataLoader(DATA_BIN, DEVICE_BSZ, SEQ_LEN)
    val_loader = BinDataLoader(DATA_VAL_BIN if len(DATA_VAL_BIN) > 0 else DATA_BIN, DEVICE_BSZ, SEQ_LEN)
    vocab_size = VOCAB_SIZE + 1
    mask_token_id = VOCAB_SIZE

    def get_batch(split):
        data_loader = train_loader if split == "train" else val_loader
        x = data_loader.next_batch()
        y = x.clone()
        mask_probs = torch.rand(DEVICE_BSZ, 1)
        mask = torch.rand(DEVICE_BSZ, SEQ_LEN) < mask_probs
        missing = ~mask.any(dim=1)
        if missing.any():
            idx = torch.randint(0, SEQ_LEN, (int(missing.sum()),))
            mask[missing, idx] = True
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)

    def decode(tokens):
        return " ".join([str(int(t)) for t in tokens])
else:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chars = ["_"] + sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    mask_token_id = stoi["_"]

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        data = train_data if split == "train" else val_data
        idx = torch.randint(len(data) - SEQ_LEN, (DEVICE_BSZ,))
        x = torch.stack([data[i : i + SEQ_LEN] for i in idx])
        y = x.clone()
        if SENT_TASK:
            start = (SEQ_LEN - SENT_MASK_LEN) // 2
            mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
            mask[:, start : start + SENT_MASK_LEN] = True
        else:
            mask_probs = torch.rand(DEVICE_BSZ, 1)
            mask = torch.rand(DEVICE_BSZ, SEQ_LEN) < mask_probs
            missing = ~mask.any(dim=1)
            if missing.any():
                idx = torch.randint(0, SEQ_LEN, (int(missing.sum()),))
                mask[missing, idx] = True
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)


def rwkv7_recurrence(r, w, k, v, a, b, state, jrt_alpha):
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    r = r.view(B, T, H, N).float()
    w = w.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w))

    has_state = state is not None
    if state is None:
        state = torch.zeros(B, H, N, N, device=r.device, dtype=torch.float32)
        inject = 0.0
    else:
        base = state
        state = state * DW_JRT_SCALE
        inject = base * jrt_alpha

    y = torch.empty(B, T, H, N, device=r.device, dtype=torch.float32)
    s = state
    for t in range(T):
        a_t = a[:, t]
        b_t = b[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        r_t = r[:, t]
        w_t = w[:, t]
        sa = torch.einsum("bhij,bhj->bhi", s, a_t)
        s = s * w_t.unsqueeze(-2) + sa.unsqueeze(-1) * b_t.unsqueeze(-2) + v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
        s = s + inject
        y_t = torch.einsum("bhij,bhj->bhi", s, r_t)
        y[:, t] = y_t
    return y.view(B, T, C), s


class RWKV7(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_embd = args.n_embd
        args.dim_att = args.n_embd

        self.head_size = HEAD_SIZE
        self.n_head = args.dim_att // self.head_size

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0 ** 0.9))
            self.time_maa_rg = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.time_maa_wa = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -7 + 5 * (n / (args.dim_att - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att) + 0.5)

            self.time_faaaa = nn.Parameter(torch.zeros(1, 1, self.n_head, self.head_size))
            self.time_aaaaa = nn.Parameter(torch.zeros(1, 1, args.dim_att))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        return x
                    return x

            D_MIX_LORA = 28
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA * 4))
            self.time_maa_w2 = nn.Parameter(ortho_init(torch.zeros(4, D_MIX_LORA, args.n_embd), 0.1))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, args.dim_att), 0.1))

            D_AAA_LORA = 16
            self.time_aaa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_AAA_LORA))
            self.time_aaa_w2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, args.dim_att), 0.1))

            D_KKK_LORA = 16
            self.time_kkk_w1 = nn.Parameter(torch.zeros(args.n_embd, D_KKK_LORA))
            self.time_kkk_w2 = nn.Parameter(ortho_init(torch.zeros(D_KKK_LORA, args.dim_att), 0.1))

            D_GATE_LORA = 120
            self.gate_w1 = nn.Parameter(torch.zeros(args.n_embd, D_GATE_LORA))
            self.gate_w2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, args.dim_att), 0.1))

            D_MA_LORA = 16
            self.ma_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MA_LORA))
            self.ma_w2 = nn.Parameter(ortho_init(torch.zeros(D_MA_LORA, args.dim_att), 0.1))
            self.time_misc_a = nn.Parameter(torch.zeros(1, 1, args.n_embd))
            D_MK_LORA = 16
            self.mk_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MK_LORA))
            self.mk_w2 = nn.Parameter(ortho_init(torch.zeros(D_MK_LORA, args.dim_att), 0.1))
            self.time_misc_k = nn.Parameter(torch.zeros(1, 1, args.n_embd))
            if layer_id != 0:
                D_MV_LORA = 16
                self.mv_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MV_LORA))
                self.mv_w2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, args.dim_att), 0.1))
                self.time_misc_v = nn.Parameter(torch.zeros(1, 1, args.n_embd) + 1.0)

            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
            self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5 / (self.n_embd**0.5), 0.5 / (self.n_embd**0.5))
            self.key.weight.data.uniform_(-0.05 / (self.n_embd**0.5), 0.05 / (self.n_embd**0.5))
            self.value.weight.data.uniform_(-0.5 / (self.n_embd**0.5), 0.5 / (self.n_embd**0.5))
            self.output.weight.data.zero_()
            self.jrt_alpha = nn.Parameter(torch.tensor(DW_JRT_ALPHA_INIT))

    def forward(self, x, v1, state):
        B, T, C = x.size()
        H = self.n_head
        xx = torch.zeros_like(x)
        xx[:, 1:] = x[:, :-1]
        xx = xx - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(4, B, T, -1)
        xrg, xwa, xk, xv = xxx.unbind(dim=0)

        xrg = x + xx * (self.time_maa_rg + xrg)
        xwa = x + xx * (self.time_maa_wa + xwa)
        xk = x + xx * (self.time_maa_k + xk)
        xv = x + xx * (self.time_maa_v + xv)

        r = self.receptance(xrg)
        w = -F.softplus(-(self.time_decay + torch.tanh(xwa @ self.time_decay_w1) @ self.time_decay_w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v1 = v
        else:
            v = v + (v1 - v) * torch.sigmoid(self.time_misc_v + (xv @ self.mv_w1) @ self.mv_w2)
        g = torch.sigmoid(xrg @ self.gate_w1) @ self.gate_w2

        kk = k + torch.tanh(xk @ self.time_kkk_w1) @ self.time_kkk_w2
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        a = torch.sigmoid(self.time_aaaaa + (xwa @ self.time_aaa_w1) @ self.time_aaa_w2)

        ma = torch.sigmoid(self.time_misc_a + (xwa @ self.ma_w1) @ self.ma_w2)
        k = k * ma + k * a * (1 - ma)
        mk = torch.sigmoid(self.time_misc_k + (xk @ self.mk_w1) @ self.mk_w2)
        k = k * torch.clamp(w * mk, max=0).exp()

        jrt_alpha = torch.sigmoid(self.jrt_alpha)
        x, state = rwkv7_recurrence(r, w, k, v, -kk, kk * a, state, jrt_alpha)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.time_faaaa)
            .sum(dim=-1, keepdim=True)
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v1, state


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 7 * config.n_embd // 2, bias=False)
        self.c_proj = nn.Linear(7 * config.n_embd // 2, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.attn = RWKV7(config, layer_id)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, v1, x0, state0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1, state1 = self.attn(self.ln1(x), v1, state0)
        x = x + x1
        x = x + self.mlp(self.ln2(x))
        return x, v1, state1


class GPTConfig:
    def __init__(self, vocab_size, n_layer, n_embd):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd


class GPT(nn.Module):
    def __init__(self, config, dw_jrt=False):
        super().__init__()
        self.config = config
        self.dw_jrt = dw_jrt
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config, layer_id) for layer_id in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, mask=None):
        x = self.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        v1 = None
        state0 = None
        for block in self.transformer.h:
            x, v1, state1 = block(x, v1, x0, state0 if self.dw_jrt else None)
            if self.dw_jrt:
                state0 = state1
        x = F.rms_norm(x, (x.size(-1),))

        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            if mask is not None:
                mask_flat = mask.view(B * T).float()
                loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
                loss = (loss * mask_flat).sum() / mask_flat.sum()
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            total_params = sum(p.numel() for p in group["params"])
            device = group["params"][0].device
            updates_flat = torch.zeros(total_params, device=device, dtype=muon_dtype)
            curr_idx = 0
            for p in group["params"]:
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()

            curr_idx = 0
            for p in group["params"]:
                g = updates_flat[curr_idx : curr_idx + p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()


def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(muon_dtype)
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y, M = get_batch(split)
            _, loss = model(X, Y, M)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def reverse_eval(model, trials_per_digit=10):
    if not REVERSE_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for digits in range(1, REVERSE_DIGIT_MAX + 1):
        for _ in range(trials_per_digit):
            s = _reverse_sample(digits)
            comma = s.find(",")
            sharp = s.find("#")
            if comma < 0 or sharp < 0:
                continue
            tgt = torch.tensor(encode(s), device=device)
            x = tgt.clone().unsqueeze(0)
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[:, comma + 1 : sharp] = True
            x[mask] = mask_token_id
            logits, _ = model(x)
            pred = logits.argmax(dim=-1)[0]
            pred_seg = pred[comma + 1 : sharp]
            tgt_seg = tgt[comma + 1 : sharp]
            correct += (pred_seg == tgt_seg).sum().item()
            total += (sharp - comma - 1)
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def bidir_eval(model, trials=200):
    if not BIDIR_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _bidir_sample(BIDIR_LEN)
        p1 = s.find("|")
        p2 = s.find("|", p1 + 1)
        if p1 < 0 or p2 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 1 : p2] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 1 : p2]
        tgt_seg = tgt[p1 + 1 : p2]
        correct += (pred_seg == tgt_seg).sum().item()
        total += (p2 - p1 - 1)
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def add_eval(model, trials=200):
    if not ADD_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _add_sample(ADD_LEN)
        p1 = s.find("=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        start = p1 + 1 + (ADD_LEN + 1 - ADD_MASK_LEN) // 2
        mask[:, start : start + ADD_MASK_LEN] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[start : start + ADD_MASK_LEN]
        tgt_seg = tgt[start : start + ADD_MASK_LEN]
        correct += (pred_seg == tgt_seg).sum().item()
        total += ADD_MASK_LEN
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def perm_eval(model, trials=200):
    if not PERM_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _perm_sample(PERM_LEN)
        p1 = s.find("|A=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 3 : p1 + 3 + PERM_LEN] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 3 : p1 + 3 + PERM_LEN]
        tgt_seg = tgt[p1 + 3 : p1 + 3 + PERM_LEN]
        correct += (pred_seg == tgt_seg).sum().item()
        total += PERM_LEN
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def inter_eval(model, trials=200):
    if not INTER_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _inter_sample(INTER_LEN)
        p1 = s.find("|M=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 3 : p1 + 3 + 2 * INTER_LEN] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 3 : p1 + 3 + 2 * INTER_LEN]
        tgt_seg = tgt[p1 + 3 : p1 + 3 + 2 * INTER_LEN]
        correct += (pred_seg == tgt_seg).sum().item()
        total += 2 * INTER_LEN
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def multi_eval(model, trials=200):
    if not MULTI_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _multi_sample(MULTI_LEN)
        p1 = s.find("|B=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 3 : p1 + 3 + 2 * MULTI_LEN] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 3 : p1 + 3 + 2 * MULTI_LEN]
        tgt_seg = tgt[p1 + 3 : p1 + 3 + 2 * MULTI_LEN]
        correct += (pred_seg == tgt_seg).sum().item()
        total += 2 * MULTI_LEN
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def parity_eval(model, trials=200):
    if not PARITY_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _parity_sample(PARITY_BLOCK, PARITY_BLOCKS)
        p1 = s.find("X=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        start = p1 + 2
        for b in range(PARITY_BLOCKS):
            pos = start + b * PARITY_BLOCK + PARITY_MASK_POS
            mask[:, pos] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[mask[0]]
        tgt_seg = tgt[mask[0]]
        correct += (pred_seg == tgt_seg).sum().item()
        total += mask.sum().item()
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def struct_eval(model, trials=200):
    if not STRUCT_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    exact = 0
    used = 0
    samples = None
    if STRUCT_EVAL_FIXED:
        global STRUCT_EVAL_SET
        if STRUCT_EVAL_SET is None:
            rng_state = random.getstate()
            random.seed(STRUCT_EVAL_SEED)
            STRUCT_EVAL_SET = []
            for _ in range(STRUCT_EVAL_N):
                s = _struct_sample(STRUCT_LEN)
                p1 = s.find("\"b\":")
                p2 = s.find(",\"c\":")
                if p1 < 0 or p2 < 0:
                    continue
                STRUCT_EVAL_SET.append((s, p1, p2))
            random.setstate(rng_state)
        samples = STRUCT_EVAL_SET
    if samples is None:
        samples = [None] * trials
    for item in samples:
        if item is None:
            s = _struct_sample(STRUCT_LEN)
            p1 = s.find("\"b\":")
            p2 = s.find(",\"c\":")
        else:
            s, p1, p2 = item
        if p1 < 0 or p2 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 4 : p2] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 4 : p2]
        tgt_seg = tgt[p1 + 4 : p2]
        correct += (pred_seg == tgt_seg).sum().item()
        total += (p2 - p1 - 4)
        exact += int((pred_seg == tgt_seg).all().item())
        used += 1
    acc = correct / total if total > 0 else 0.0
    exact_rate = exact / used if used > 0 else 0.0
    model.train()
    return acc, exact_rate


@torch.no_grad()
def retr_eval(model, trials=200):
    if not RETR_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _retr_sample()
        p1 = s.find("|ANS=")
        p2 = s.find("|Q=")
        if p1 < 0 or p2 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 5 : p2] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 5 : p2]
        tgt_seg = tgt[p1 + 5 : p2]
        correct += (pred_seg == tgt_seg).sum().item()
        total += (p2 - p1 - 5)
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def sent_eval(model, trials=200):
    if not SENT_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        x, y, m = get_batch("train")
        logits, _ = model(x, y, m)
        pred = logits.argmax(dim=-1)
        correct += (pred[m] == y[m]).sum().item()
        total += m.sum().item()
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def generate(model, max_new_tokens, prompt_len=16, temp=1.0, confidence_threshold=0.95, top_k=3):
    if STRUCT_TASK:
        s = _struct_sample(STRUCT_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif ADD_TASK:
        s = _add_sample(ADD_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif PERM_TASK:
        s = _perm_sample(PERM_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif INTER_TASK:
        s = _inter_sample(INTER_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif MULTI_TASK:
        s = _multi_sample(MULTI_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif PARITY_TASK:
        s = _parity_sample(PARITY_BLOCK, PARITY_BLOCKS)
        all_tokens = encode(s)[:prompt_len]
    elif RETR_TASK:
        s = _retr_sample()
        all_tokens = encode(s)[:prompt_len]
    elif BIDIR_TASK:
        s = _bidir_sample(BIDIR_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif REVERSE_TASK:
        s = _reverse_sample(REVERSE_DIGIT_MAX)
        all_tokens = encode(s)[:prompt_len]
    elif use_bin:
        all_tokens = [0] * prompt_len
    else:
        all_tokens = data[:prompt_len].tolist()
    total_steps = 0

    while len(all_tokens) - prompt_len < max_new_tokens:
        block_len = min(240, prompt_len + max_new_tokens - len(all_tokens))
        x = torch.full((1, SEQ_LEN), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(all_tokens[-prompt_len:], device=device)

        masked = torch.zeros(1, SEQ_LEN, dtype=torch.bool, device=device)
        masked[0, prompt_len : prompt_len + block_len] = True

        while masked.any():
            total_steps += 1
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)

            decode_mask = (confidences >= confidence_threshold) & masked
            if not decode_mask.any():
                masked_confidences = torch.where(masked, confidences, torch.tensor(-float("inf"), device=device))
                decode_mask.view(-1)[masked_confidences.argmax()] = True

            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(top_k_probs_norm.view(-1, top_k), 1).view(1, SEQ_LEN)
            sampled_tokens = torch.gather(top_k_indices, -1, sampled_k.unsqueeze(-1)).squeeze(-1)

            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask

        all_tokens.extend(x[0, prompt_len : prompt_len + block_len].tolist())

    tokens_generated = len(all_tokens) - prompt_len
    print(f"Total steps: {total_steps} for {tokens_generated} tokens")
    print(f"Avg decoded per step: {tokens_generated / total_steps:.2f}")
    return decode(all_tokens)


@torch.no_grad()
def mem_check(model, prompt_len=16):
    model.eval()
    if STRUCT_TASK:
        s = _struct_sample(STRUCT_LEN)
        y = torch.tensor(encode(s), dtype=torch.long, device=device).unsqueeze(0)
        x = y.clone()
    elif RETR_TASK:
        s = _retr_sample()
        y = torch.tensor(encode(s), dtype=torch.long, device=device).unsqueeze(0)
        x = y.clone()
    elif BIDIR_TASK:
        s = _bidir_sample(BIDIR_LEN)
        y = torch.tensor(encode(s), dtype=torch.long, device=device).unsqueeze(0)
        x = y.clone()
    elif REVERSE_TASK:
        s = _reverse_sample(REVERSE_DIGIT_MAX)
        y = torch.tensor(encode(s), dtype=torch.long, device=device).unsqueeze(0)
        x = y.clone()
    elif use_bin:
        x, y, _ = get_batch("train")
        x = x[:1].clone()
        y = y[:1].clone()
    else:
        y = data[:SEQ_LEN].unsqueeze(0).to(device)
        x = y.clone()
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, prompt_len:] = True
    x[mask] = mask_token_id
    logits, _ = model(x)
    pred = logits.argmax(dim=-1)
    out = torch.where(mask, pred, x)
    acc = (out[mask] == y[mask]).float().mean().item()
    print(f"MEM acc: {acc:.4f}")
    print(f"GT:\n{decode(y[0].tolist())}")
    print(f"PR:\n{decode(out[0].tolist())}")
    model.train()


if __name__ == "__main__":
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    model = GPT(GPTConfig(vocab_size=vocab_size, n_layer=N_LAYER, n_embd=N_EMBD), dw_jrt=DW_JRT)
    m = model.to(device)

    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    if os.path.exists(WEIGHTS_PATH) and not TRAIN:
        m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    else:
        fused_ok = device == "cuda"
        optimizer1 = torch.optim.Adam([m.transformer.wte.weight], lr=WTE_LR, betas=(0.9, 0.95), fused=fused_ok)
        optimizer2 = torch.optim.Adam([m.lm_head.weight], lr=HEAD_LR, betas=(0.9, 0.95), fused=fused_ok)

        params = list(m.transformer.h.named_parameters())
        optimizer3 = Muon([p for n, p in params if p.ndim == 2 and "_w1" not in n and "_w2" not in n], lr=MUON_LR, momentum=0.95)
        optimizer4 = torch.optim.Adam(
            [p for n, p in params if (p.ndim != 2 or "_w1" in n or "_w2" in n) and ("lambdas" not in n and "ln" not in n)],
            lr=ADAM_LR,
            betas=(0.9, 0.95),
            fused=fused_ok,
        )
        optimizer5 = torch.optim.Adam([p for n, p in params if "lambdas" in n], lr=LAMBDA_LR, betas=(0.9, 0.95), fused=fused_ok)
        optimizer6 = torch.optim.Adam([p for n, p in params if "ln" in n], lr=LN_LR, betas=(0.9, 0.95), fused=fused_ok)
        optimizers = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5, optimizer6]

        def get_lr(it):
            if it < WARMUP_ITERS:
                return (it + 1) / WARMUP_ITERS if WARMUP_ITERS > 0 else 1.0
            if it < MAX_ITERS - WARMDOWN_ITERS:
                return 1.0
            decay_ratio = (MAX_ITERS - it) / WARMDOWN_ITERS
            return decay_ratio

        schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

        start = time.time()
        for iter in range(MAX_ITERS):
            if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
                losses = estimate_loss(m)
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {time.time() - start:.2f} seconds"
                )
                if REVERSE_EVAL:
                    acc = reverse_eval(m)
                    print(f"reverse acc {acc:.4f}")
                if BIDIR_EVAL:
                    acc = bidir_eval(m)
                    print(f"bidir acc {acc:.4f}")
                if ADD_EVAL:
                    acc = add_eval(m)
                    print(f"add acc {acc:.4f}")
                if PERM_EVAL:
                    acc = perm_eval(m)
                    print(f"perm acc {acc:.4f}")
                if INTER_EVAL:
                    acc = inter_eval(m)
                    print(f"inter acc {acc:.4f}")
                if MULTI_EVAL:
                    acc = multi_eval(m)
                    print(f"multi acc {acc:.4f}")
                if PARITY_EVAL:
                    acc = parity_eval(m)
                    print(f"parity acc {acc:.4f}")
                if STRUCT_EVAL:
                    acc, exact = struct_eval(m)
                    print(f"struct acc {acc:.4f}, exact {exact:.4f}")
                if RETR_EVAL:
                    acc = retr_eval(m)
                    print(f"retr acc {acc:.4f}")
                if SENT_EVAL:
                    acc = sent_eval(m)
                    print(f"sent acc {acc:.4f}")
                if not use_bin:
                    sample = generate(m, max_new_tokens=240)
                    print(f"Sample:\n{sample}\n")

            m.train()
            m.zero_grad(set_to_none=True)
            grad_accum_steps = BATCH_SIZE // DEVICE_BSZ
            for _ in range(grad_accum_steps):
                xb, yb, mb = get_batch("train")
                _, loss = m(xb, yb, mb)
                (loss / grad_accum_steps).backward()

            frac = min(iter / 500, 1)
            optimizer3.param_groups[0]["momentum"] = (1 - frac) * 0.85 + frac * 0.95

            for opt, sched in zip(optimizers, schedulers):
                opt.step()
                sched.step()

        print(f"Total training time: {time.time() - start:.2f} seconds")
        torch.save(m.state_dict(), WEIGHTS_PATH)

    if MEM_CHECK:
        mem_check(m, prompt_len=PROMPT_LEN)
    output = generate(m, max_new_tokens=GEN_TOKENS, temp=0.8, confidence_threshold=0.95, top_k=2)
    print(f"\nOutput:\n{output}")
