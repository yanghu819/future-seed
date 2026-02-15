import os, time, math, glob, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load


def env_int(name, default):
    return int(os.getenv(name, default))


def env_float(name, default):
    return float(os.getenv(name, default))


MODEL = os.getenv("MODEL", "rwkv")  # rwkv|transformer|transformer_causal
DECODE = os.getenv("DECODE", "argmax")  # argmax|hungarian
REFINE_STEPS = env_int("REFINE_STEPS", 0)
REFINE_CONF = env_float("REFINE_CONF", 0.90)
BIN_MASK_MODE = os.getenv("BIN_MASK_MODE", "random")  # random|prefix|span
BIN_PREFIX_RATIO = env_float("BIN_PREFIX_RATIO", 0.50)
BIN_SPAN_LEN = env_int("BIN_SPAN_LEN", 128)
TRANS_N_HEAD = env_int("TRANS_N_HEAD", 8)
TRANS_DROPOUT = env_float("TRANS_DROPOUT", 0.0)
TRANS_FF_MULT = env_int("TRANS_FF_MULT", 4)
ATTN_FS = env_int("ATTN_FS", 0) == 1
ATTN_FS_COLLECTOR = os.getenv("ATTN_FS_COLLECTOR", "zero")  # zero|learned
ATTN_FS_GATING = env_int("ATTN_FS_GATING", 0) == 1
ATTN_FS_ALPHA_INIT = env_float("ATTN_FS_ALPHA_INIT", 0.0)

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
FUTURE_SEED = env_int("FUTURE_SEED", 0) == 1
FUTURE_SEED_SCALE = env_float("FUTURE_SEED_SCALE", 1.0)
FUTURE_SEED_ALPHA_INIT = env_float("FUTURE_SEED_ALPHA_INIT", 0.0)
FUTURE_SEED_LAYER_START = env_int("FUTURE_SEED_LAYER_START", 0)
FUTURE_SEED_S0_GATE = env_int("FUTURE_SEED_S0_GATE", 0) == 1
RWKV7_KERNEL = os.getenv("RWKV7_KERNEL", "python")
RWKV7_CUDA_SRC = os.getenv("RWKV7_CUDA_SRC", "")
TRAIN = env_int("TRAIN", 0) == 1
MEM_CHECK = env_int("MEM_CHECK", 0) == 1
LOG_SAMPLE = env_int("LOG_SAMPLE", 1) == 1
LOG_WIN = env_int("LOG_WIN", 80)
LOG_OUTPUT = env_int("LOG_OUTPUT", 0) == 1
LOG_JSONL = os.getenv("LOG_JSONL", "")
MASKACC_EVAL = env_int("MASKACC_EVAL", 0) == 1
MASKACC_SPLIT = os.getenv("MASKACC_SPLIT", "val")
MASKACC_ITERS = env_int("MASKACC_ITERS", 200)
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
RIGHTCOPY_TASK = env_int("RIGHTCOPY_TASK", 0) == 1
RIGHTCOPY_LEN = env_int("RIGHTCOPY_LEN", 8)
RIGHTCOPY_EVAL = env_int("RIGHTCOPY_EVAL", 0) == 1
RIGHTREV_TASK = env_int("RIGHTREV_TASK", 0) == 1
RIGHTREV_LEN = env_int("RIGHTREV_LEN", 8)
RIGHTREV_EVAL = env_int("RIGHTREV_EVAL", 0) == 1
KVSORT_TASK = env_int("KVSORT_TASK", 0) == 1
KVSORT_N_MIN = env_int("KVSORT_N_MIN", 3)
KVSORT_N_MAX = env_int("KVSORT_N_MAX", 6)
KVSORT_N_TEST = env_int("KVSORT_N_TEST", 10)
KVSORT_EVAL = env_int("KVSORT_EVAL", 0) == 1
KVSORT_PAD = env_int("KVSORT_PAD", 8)
KVSORT_NOISE = env_int("KVSORT_NOISE", 0)
KVSORT_USE_ORDER = env_int("KVSORT_USE_ORDER", 1) == 1
KVSORT_KEYS_ONLY = env_int("KVSORT_KEYS_ONLY", 0) == 1
KVSORT_KEYS_SEP = env_int("KVSORT_KEYS_SEP", 1) == 1
KVSORT_MASK_SEP = env_int("KVSORT_MASK_SEP", 1) == 1
PERMFILL_TASK = env_int("PERMFILL_TASK", 0) == 1
PERMFILL_N_MIN = env_int("PERMFILL_N_MIN", 3)
PERMFILL_N_MAX = env_int("PERMFILL_N_MAX", 6)
PERMFILL_N_TEST = env_int("PERMFILL_N_TEST", 10)
PERMFILL_EVAL = env_int("PERMFILL_EVAL", 0) == 1
PERMFILL_PAD = env_int("PERMFILL_PAD", 8)
PERMFILL_USE_ORDER = env_int("PERMFILL_USE_ORDER", 1) == 1
PERMFILL_ANCHOR = env_int("PERMFILL_ANCHOR", 0) == 1
PERMFILL_ANCHOR_K = env_int("PERMFILL_ANCHOR_K", 2)
INDEX_TASK = env_int("INDEX_TASK", 0) == 1
INDEX_LEN = env_int("INDEX_LEN", 16)
INDEX_EVAL = env_int("INDEX_EVAL", 0) == 1
RULE_TASK = env_int("RULE_TASK", 0) == 1
RULE_LEN = env_int("RULE_LEN", 8)
RULE_EVAL = env_int("RULE_EVAL", 0) == 1
CIPHER_TASK = env_int("CIPHER_TASK", 0) == 1
CIPHER_LEN = env_int("CIPHER_LEN", 16)
CIPHER_EVAL = env_int("CIPHER_EVAL", 0) == 1
SHIFT_TASK = env_int("SHIFT_TASK", 0) == 1
SHIFT_LEN = env_int("SHIFT_LEN", 16)
SHIFT_EVAL = env_int("SHIFT_EVAL", 0) == 1
POSCOPY_TASK = env_int("POSCOPY_TASK", 0) == 1
POSCOPY_LEN = env_int("POSCOPY_LEN", 8)
POSCOPY_REP = env_int("POSCOPY_REP", 3)
POSCOPY_EVAL = env_int("POSCOPY_EVAL", 0) == 1
CONSTR_TASK = env_int("CONSTR_TASK", 0) == 1
CONSTR_LEN = env_int("CONSTR_LEN", 8)
CONSTR_EVAL = env_int("CONSTR_EVAL", 0) == 1
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
QA_TASK = env_int("QA_TASK", 0) == 1
QA_EVAL = env_int("QA_EVAL", 0) == 1
QA_MODE = os.getenv("QA_MODE", "both")
QA_FILE = os.getenv("QA_FILE", "")
QA_SYNTH_A_MAX = env_int("QA_SYNTH_A_MAX", 99)
QA_SYNTH_B_MAX = env_int("QA_SYNTH_B_MAX", 99)
SUDOKU_TASK = env_int("SUDOKU_TASK", 0) == 1
SUDOKU_EVAL = env_int("SUDOKU_EVAL", 0) == 1
SUDOKU_PAD = env_int("SUDOKU_PAD", 8)
SUDOKU_HOLES_MIN = env_int("SUDOKU_HOLES_MIN", 4)
SUDOKU_HOLES_MAX = env_int("SUDOKU_HOLES_MAX", 8)
SUDOKU_HOLES_TEST = env_int("SUDOKU_HOLES_TEST", 8)
SUDOKU_MASK_MODE = os.getenv("SUDOKU_MASK_MODE", "random")  # random|prefix
SUDOKU_TRIALS = env_int("SUDOKU_TRIALS", 200)
SUDOKU_CONS_LAMBDA = env_float("SUDOKU_CONS_LAMBDA", 0.0)
FS_MASK_ONLY = env_int("FS_MASK_ONLY", 0) == 1
FS_ENCDEC = env_int("FS_ENCDEC", 0) == 1
FS_ENCDEC_AUX = env_int("FS_ENCDEC_AUX", 1) == 1
FS_ENCDEC_LAMBDA = env_float("FS_ENCDEC_LAMBDA", 0.3)
FS_ENCDEC_RATIO = env_float("FS_ENCDEC_RATIO", 0.5)
FS_ENCDEC_STATE_DROPOUT = env_float("FS_ENCDEC_STATE_DROPOUT", 0.1)
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "weights/diffusion.pt")
DATA_PATH = os.getenv("DATA_PATH", "../tiny-diffusion/data.txt")
DATA_BIN = os.getenv("DATA_BIN", "")
DATA_VAL_BIN = os.getenv("DATA_VAL_BIN", "")
VOCAB_SIZE = env_int("VOCAB_SIZE", 50304)
GEN_TOKENS = env_int("GEN_TOKENS", 2000)
PROMPT_LEN = env_int("PROMPT_LEN", 16)

if MODEL != "rwkv" and (FS_MASK_ONLY or FS_ENCDEC):
    raise RuntimeError("FS_MASK_ONLY/FS_ENCDEC are only supported for MODEL=rwkv")

random_seed = env_int("RANDOM_SEED", 1337)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


device = (
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)
muon_dtype = torch.bfloat16 if device == "cuda" else torch.float32

use_bin = (
    len(DATA_BIN) > 0
    and not REVERSE_TASK
    and not BIDIR_TASK
    and not STRUCT_TASK
    and not RETR_TASK
    and not SENT_TASK
    and not QA_TASK
    and not SUDOKU_TASK
)
use_bin = (
    use_bin
    and not ADD_TASK
    and not PERM_TASK
    and not INTER_TASK
    and not MULTI_TASK
    and not PARITY_TASK
    and not RIGHTCOPY_TASK
    and not RIGHTREV_TASK
    and not KVSORT_TASK
    and not PERMFILL_TASK
    and not INDEX_TASK
    and not RULE_TASK
    and not CIPHER_TASK
    and not SHIFT_TASK
    and not POSCOPY_TASK
    and not CONSTR_TASK
)


RUN_CUDA_RWKV7_STATE = None
if MODEL == "rwkv" and RWKV7_KERNEL == "cuda_wind":
    if device != "cuda":
        raise RuntimeError("RWKV7_KERNEL=cuda_wind requires CUDA device")
    cuda_src = RWKV7_CUDA_SRC
    if len(cuda_src) == 0:
        cuda_src = os.path.join(os.path.dirname(__file__), "..", "modded-nanogpt-rwkv", "rwkv_cuda_wind")
    wind_cu = os.path.join(cuda_src, "wind_rwkv7.cu")
    wind_cpp = os.path.join(cuda_src, "wind_rwkv7.cpp")
    if (not os.path.exists(wind_cu)) or (not os.path.exists(wind_cpp)):
        raise FileNotFoundError(f"rwkv cuda sources missing: {wind_cu} | {wind_cpp}")
    mod_name = f"wind_fs_h{HEAD_SIZE}"
    load(
        name=mod_name,
        sources=[wind_cu, wind_cpp],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_C_={HEAD_SIZE}",
        ],
    )

    class WindRWKV7State(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, s0):
            B, T, H, C = w.shape
            assert T % 16 == 0
            assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, a, b, s0])
            w, q, k, v, a, b, s0 = [i.contiguous() for i in [w, q, k, v, a, b, s0]]
            y = torch.empty_like(v)
            sT = torch.empty_like(s0)
            s = torch.zeros(B, H, T // 16, C, C, dtype=w.dtype, device=w.device)
            torch.ops.wind.forward(w, q, k, v, a, b, s0, y, s, sT)
            ctx.save_for_backward(w, q, k, v, a, b, s)
            return y, sT

        @staticmethod
        def backward(ctx, dy, dsT):
            w, q, k, v, a, b, s = ctx.saved_tensors
            B, T, H, C = w.shape
            dy = dy.contiguous()
            dsT = dsT.contiguous()
            dw, dq, dk, dv, da, db, ds0 = [torch.empty_like(x) for x in [w, q, k, v, a, b, dsT]]
            torch.ops.wind.backward(w, q, k, v, a, b, dy, s, dsT, dw, dq, dk, dv, da, db, ds0)
            return dw, dq, dk, dv, da, db, ds0

    def RUN_CUDA_RWKV7_STATE(w, q, k, v, a, b, s0):
        return WindRWKV7State.apply(w, q, k, v, a, b, s0)


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
elif RIGHTCOPY_TASK:
    vocab_base = [str(i) for i in range(10)] + ["L", "M", "R", "=", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _rightcopy_sample(n):
        l = "".join([str(random.randint(0, 9)) for _ in range(n)])
        r = "".join([str(random.randint(0, 9)) for _ in range(n)])
        m = r
        s = "L=" + l + "|M=" + m + "|R=" + r
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _rightcopy_sample(RIGHTCOPY_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("|M=")
            if p1 >= 0:
                mask[i, p1 + 3 : p1 + 3 + RIGHTCOPY_LEN] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif RIGHTREV_TASK:
    vocab_base = [str(i) for i in range(10)] + ["L", "M", "R", "=", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _rightrev_sample(n):
        l = "".join([str(random.randint(0, 9)) for _ in range(n)])
        r = "".join([str(random.randint(0, 9)) for _ in range(n)])
        m = r[::-1]
        s = "L=" + l + "|M=" + m + "|R=" + r
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _rightrev_sample(RIGHTREV_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("|M=")
            if p1 >= 0:
                mask[i, p1 + 3 : p1 + 3 + RIGHTREV_LEN] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif INDEX_TASK:
    vocab_base = [str(i) for i in range(10)] + ["A", "I", "Y", "=", "|", "#", ","]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _index_sample(n):
        a = "".join([str(random.randint(0, 9)) for _ in range(n)])
        i = random.randint(0, n - 4)
        j = i + 3
        y = a[i : j + 1]
        s = "A=" + a + "|Y=" + y + "|I=" + str(i) + "," + str(j)
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _index_sample(INDEX_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("|Y=")
            if p1 >= 0:
                mask[i, p1 + 3 : p1 + 3 + 4] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif RULE_TASK:
    vocab_base = [str(i) for i in range(10)] + ["A", "M", "R", "=", "|", "#", "C", "V"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _rule_sample(n):
        a = "".join([str(random.randint(0, 9)) for _ in range(n)])
        if random.random() < 0.5:
            r = "C"
            m = a
        else:
            r = "V"
            m = a[::-1]
        s = "A=" + a + "|M=" + m + "|R=" + r
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _rule_sample(RULE_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("|M=")
            if p1 >= 0:
                mask[i, p1 + 3 : p1 + 3 + RULE_LEN] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif CIPHER_TASK:
    vocab_base = [str(i) for i in range(10)] + ["A", "M", "K", "=", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _cipher_sample(n):
        a = "".join([str(random.randint(0, 9)) for _ in range(n)])
        perm = list(range(10))
        random.shuffle(perm)
        k = "".join([str(i) for i in perm])
        m = "".join([k[int(ch)] for ch in a])
        s = "A=" + a + "|M=" + m + "|K=" + k
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _cipher_sample(CIPHER_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("|M=")
            if p1 >= 0:
                mask[i, p1 + 3 : p1 + 3 + CIPHER_LEN] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif SHIFT_TASK:
    vocab_base = [str(i) for i in range(10)] + ["A", "M", "K", "=", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _shift_sample(n):
        a = "".join([str(random.randint(0, 9)) for _ in range(n)])
        k = random.randint(0, 9)
        m = "".join([str((int(ch) + k) % 10) for ch in a])
        s = "A=" + a + "|M=" + m + "|K=" + str(k)
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _shift_sample(SHIFT_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("|M=")
            if p1 >= 0:
                mask[i, p1 + 3 : p1 + 3 + SHIFT_LEN] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif POSCOPY_TASK:
    vocab_base = [str(i) for i in range(10)] + ["A", "M", "I", "=", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _poscopy_sample(n):
        a = "".join([str(random.randint(0, 9)) for _ in range(n)])
        i = random.randint(0, n - 1)
        m = a[i] * POSCOPY_REP
        s = "A=" + a + "|M=" + m + "|I=" + str(i)
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _poscopy_sample(POSCOPY_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("|M=")
            if p1 >= 0:
                mask[i, p1 + 3 : p1 + 3 + POSCOPY_REP] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif CONSTR_TASK:
    vocab_base = [str(i) for i in range(10)] + ["P", "M", "R", "=", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _constr_sample(n):
        p = "".join([str(random.randint(0, 9)) for _ in range(n)])
        d1 = str(random.randint(0, 9))
        d2 = str(random.randint(0, 9))
        pat = "".join([d1 if i % 2 == 0 else d2 for i in range(n - 1)])
        m = p[-1] + pat
        s = "P=" + p + "|M=" + m + "|R=" + d1 + d2
        s = s + "#" * (SEQ_LEN - len(s))
        return s

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            s = _constr_sample(CONSTR_LEN)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            p1 = s.find("|M=")
            if p1 >= 0:
                mask[i, p1 + 3 : p1 + 3 + CONSTR_LEN] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif QA_TASK:
    qa_pairs = []
    if len(QA_FILE) > 0 and os.path.exists(QA_FILE):
        with open(QA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" in line:
                    q, a = line.split("\t", 1)
                elif "|||" in line:
                    q, a = line.split("|||", 1)
                else:
                    continue
                q = q.strip()
                a = a.strip()
                if len(q) > 0 and len(a) > 0:
                    qa_pairs.append((q, a))

    def _qa_synth_pair():
        x = random.randint(0, QA_SYNTH_A_MAX)
        y = random.randint(0, QA_SYNTH_B_MAX)
        return f"{x}+{y}=?", str(x + y)

    if len(qa_pairs) == 0:
        qa_pairs = [_qa_synth_pair() for _ in range(max(4096, DEVICE_BSZ * 16))]

    vocab_set = set(["Q", "A", ":", "|", "#"])
    for q, a in qa_pairs:
        vocab_set.update(list(q))
        vocab_set.update(list(a))
    vocab_base = sorted(list(vocab_set))
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _qa_pack(q, a):
        raw = ("Q:" + q + "|A:" + a)[:SEQ_LEN]
        sep = raw.find("|A:")
        if sep < 0:
            sep = max(2, min(len(raw), SEQ_LEN // 2))
            raw = (raw[:sep] + "|A:" + raw[sep:])[:SEQ_LEN]
            sep = raw.find("|A:")
        valid_len = len(raw)
        q_start = 2
        q_end = sep
        a_start = min(sep + 3, valid_len)
        s = raw + "#" * (SEQ_LEN - valid_len)
        return s, q_start, q_end, a_start, valid_len

    def _qa_pick(split):
        return qa_pairs[random.randint(0, len(qa_pairs) - 1)]

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            q, a = _qa_pick(split)
            s, q_start, q_end, a_start, valid_len = _qa_pack(q, a)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            if QA_MODE == "qa":
                m_start, m_end = a_start, valid_len
            elif QA_MODE == "aq":
                m_start, m_end = q_start, q_end
            else:
                if random.random() < 0.5:
                    m_start, m_end = a_start, valid_len
                else:
                    m_start, m_end = q_start, q_end
            if m_end <= m_start:
                m_start = max(0, valid_len - 1)
                m_end = valid_len
            mask[i, m_start:m_end] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif SUDOKU_TASK:
    digits = ["1", "2", "3", "4"]
    vocab_base = digits + ["P", "M", "=", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _sudoku_sol_4x4():
        # Base valid 4x4 (2x2 subgrid) solution, then apply random symmetries.
        base = [
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
        ]

        perm = random.sample([1, 2, 3, 4], 4)
        mp = {i + 1: perm[i] for i in range(4)}
        g = [[mp[v] for v in row] for row in base]

        bands = [[0, 1], [2, 3]]
        random.shuffle(bands)
        rows = []
        for b in bands:
            bb = b[:]
            random.shuffle(bb)
            rows += bb

        stacks = [[0, 1], [2, 3]]
        random.shuffle(stacks)
        cols = []
        for s in stacks:
            ss = s[:]
            random.shuffle(ss)
            cols += ss

        out = []
        for r in rows:
            for c in cols:
                out.append(str(g[r][c]))
        return "".join(out)  # len=16

    def _sudoku_pack_4x4(holes):
        sol = _sudoku_sol_4x4()
        p = "#" * SUDOKU_PAD
        s = "P=" + p + "|M=" + sol
        s = s[:SEQ_LEN]
        s = s + "#" * (SEQ_LEN - len(s))
        m0 = s.find("|M=") + 3
        m1 = m0 + 16
        holes = max(1, min(16, int(holes)))
        if SUDOKU_MASK_MODE == "prefix":
            idx = list(range(holes))
        else:
            idx = random.sample(list(range(16)), holes)
        return s, sol, m0, m1, idx

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            if split == "train":
                holes = random.randint(SUDOKU_HOLES_MIN, SUDOKU_HOLES_MAX)
            else:
                holes = SUDOKU_HOLES_TEST
            s, _sol, m0, _m1, idx = _sudoku_pack_4x4(holes)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            for j in idx:
                mask[i, m0 + j] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif KVSORT_TASK:
    digits = [str(i) for i in range(10)]
    letters = [chr(ord("a") + i) for i in range(26)]
    key_pool = digits + letters
    vocab_base = digits + letters + ["P", "M", "R", "O", "=", ":", ";", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _kvsort_pack(n):
        keys = random.sample(key_pool, n)
        vals = {k: random.choice(letters) for k in keys}

        inp_keys = keys[:]
        random.shuffle(inp_keys)

        noise_keys = []
        if KVSORT_NOISE > 0:
            avail = [k for k in key_pool if k not in keys]
            if len(avail) > 0:
                noise_keys = random.sample(avail, min(KVSORT_NOISE, len(avail)))
        noise = {k: random.choice(letters) for k in noise_keys}

        if KVSORT_USE_ORDER:
            order = "".join(random.sample(key_pool, len(key_pool)))
        else:
            order = "".join(key_pool)
        rank = {ch: i for i, ch in enumerate(order)}

        pairs = inp_keys + noise_keys
        random.shuffle(pairs)
        right_pairs = []
        for k in pairs:
            if k in vals:
                right_pairs.append(f"{k}:{vals[k]}")
            else:
                right_pairs.append(f"{k}:{noise[k]}")
        right = ";".join(right_pairs)

        gt_keys = sorted(keys, key=lambda x: rank[x])
        if KVSORT_KEYS_ONLY:
            gt = (";".join(gt_keys)) if KVSORT_KEYS_SEP else ("".join(gt_keys))
        else:
            gt = ";".join([f"{k}:{vals[k]}" for k in gt_keys])
        p = "#" * KVSORT_PAD
        if KVSORT_USE_ORDER:
            s = "P=" + p + "|M=" + gt + "|R=O=" + order + ";" + right
        else:
            s = "P=" + p + "|M=" + gt + "|R=" + right
        s = s[:SEQ_LEN]
        s = s + "#" * (SEQ_LEN - len(s))
        m0 = s.find("|M=") + 3
        m1 = m0 + len(gt)
        return s, m0, m1

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            if split == "train":
                n = random.randint(KVSORT_N_MIN, KVSORT_N_MAX)
            else:
                n = KVSORT_N_TEST
            s, m0, m1 = _kvsort_pack(n)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            if KVSORT_KEYS_ONLY and KVSORT_KEYS_SEP and (not KVSORT_MASK_SEP):
                for j in range(m0, m1):
                    if s[j] in key_pool:
                        mask[i, j] = True
            else:
                mask[i, m0:m1] = True
        y = x.clone()
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)
elif PERMFILL_TASK:
    digits = [str(i) for i in range(10)]
    letters = [chr(ord("a") + i) for i in range(26)]
    key_pool = digits + letters
    vocab_base = digits + letters + ["P", "M", "R", "O", "=", ":", ";", "|", "#"]
    vocab_size = len(vocab_base) + 1
    mask_token_id = len(vocab_base)
    stoi = {ch: i for i, ch in enumerate(vocab_base)}
    itos = {i: ch for i, ch in enumerate(vocab_base)}
    itos[mask_token_id] = "_"

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[n] for n in l])

    def _permfill_pack(n):
        items = random.sample(key_pool, n)
        if PERMFILL_USE_ORDER:
            order = "".join(random.sample(key_pool, len(key_pool)))
        else:
            order = "".join(key_pool)
        rank = {ch: i for i, ch in enumerate(order)}
        gt = "".join(sorted(items, key=lambda x: rank[x]))
        right = "".join(items)
        p = "#" * PERMFILL_PAD
        s = "P=" + p + "|M=" + gt + "|R=O=" + order + ";" + right
        s = s[:SEQ_LEN]
        s = s + "#" * (SEQ_LEN - len(s))
        m0 = s.find("|M=") + 3
        m1 = m0 + len(gt)
        anchors = []
        if PERMFILL_ANCHOR:
            k = min(PERMFILL_ANCHOR_K, max(1, n))
            anchors = random.sample(list(range(n)), k)
        return s, m0, m1, anchors

    def get_batch(split):
        x = torch.empty(DEVICE_BSZ, SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
        for i in range(DEVICE_BSZ):
            if split == "train":
                n = random.randint(PERMFILL_N_MIN, PERMFILL_N_MAX)
            else:
                n = PERMFILL_N_TEST
            s, m0, m1, anchors = _permfill_pack(n)
            x[i] = torch.tensor(encode(s), dtype=torch.long)
            mask[i, m0:m1] = True
            for a in anchors:
                mask[i, m0 + a] = False
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
        if BIN_MASK_MODE == "prefix":
            prefix_len = max(1, min(SEQ_LEN, int(SEQ_LEN * BIN_PREFIX_RATIO)))
            mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
            mask[:, :prefix_len] = True
        elif BIN_MASK_MODE == "span":
            span_len = max(1, min(SEQ_LEN, int(BIN_SPAN_LEN)))
            start = max(0, (SEQ_LEN - span_len) // 2)
            mask = torch.zeros(DEVICE_BSZ, SEQ_LEN, dtype=torch.bool)
            mask[:, start : start + span_len] = True
        else:
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


def rwkv7_recurrence(r, w, k, v, a, b, state, future_seed_alpha):
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    if RUN_CUDA_RWKV7_STATE is not None:
        q4 = r.view(B, T, H, N).to(torch.bfloat16)
        w4 = w.view(B, T, H, N).to(torch.bfloat16)
        k4 = k.view(B, T, H, N).to(torch.bfloat16)
        v4 = v.view(B, T, H, N).to(torch.bfloat16)
        a4 = a.view(B, T, H, N).to(torch.bfloat16)
        b4 = b.view(B, T, H, N).to(torch.bfloat16)
        if state is None:
            s0 = torch.zeros(B, H, N, N, device=r.device, dtype=torch.bfloat16)
        else:
            s0f = state.float() * FUTURE_SEED_SCALE
            if FUTURE_SEED_S0_GATE:
                s0f = s0f * future_seed_alpha.float()
            s0 = s0f.to(torch.bfloat16)
        y, sT = RUN_CUDA_RWKV7_STATE(w4, q4, k4, v4, a4, b4, s0)
        return y.view(B, T, C).float(), sT.float()

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
        state = state * FUTURE_SEED_SCALE
        inject = base * future_seed_alpha

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
        s = s + inject * (1.0 - w_t.unsqueeze(-2))
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
            self.future_seed_alpha = nn.Parameter(torch.full((1, self.n_head, 1, 1), FUTURE_SEED_ALPHA_INIT))

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

        future_seed_alpha = torch.sigmoid(self.future_seed_alpha)
        x, state = rwkv7_recurrence(r, w, k, v, -kk, kk * a, state, future_seed_alpha)
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
        self.layer_id = layer_id

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
    def __init__(self, config, future_seed=False):
        super().__init__()
        self.config = config
        self.future_seed = future_seed
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config, layer_id) for layer_id in range(config.n_layer)]),
            )
        )
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, mask=None, return_states=False, seed_states=None, seed_dropout=0.0):
        x = self.transformer.wte(idx)
        x = x + (idx == mask_token_id).unsqueeze(-1).to(x.dtype) * self.mask_emb
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        v1 = None
        state0 = None
        states = [] if return_states else None
        for block in self.transformer.h:
            use_state = None
            if self.future_seed and block.layer_id >= FUTURE_SEED_LAYER_START:
                if seed_states is not None:
                    use_state = seed_states[block.layer_id]
                    if seed_dropout > 0:
                        use_state = F.dropout(use_state, p=seed_dropout, training=self.training)
                else:
                    use_state = state0
            x, v1, state1 = block(x, v1, x0, use_state)
            if self.future_seed and seed_states is None:
                state0 = state1
            if return_states:
                states.append(state1)
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
        if return_states:
            return logits, loss, states
        return logits, loss


class TransformerMLM(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head, dropout=0.0, ff_mult=4):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"TRANS_N_HEAD must divide N_EMBD: {n_head} | {n_embd}")
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(SEQ_LEN, n_embd)
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, n_embd))

        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=int(ff_mult) * n_embd,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None, mask=None, **_ignored):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.wte(idx) + self.wpe(pos)
        x = x + (idx == mask_token_id).unsqueeze(-1).to(x.dtype) * self.mask_emb
        x = self.enc(x)
        x = self.ln_f(x)
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


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"TRANS_N_HEAD must divide N_EMBD: {n_head} | {n_embd}")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = float(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,nh,T,hd]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(self.dropout if self.training else 0.0),
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y


class CausalMLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0, ff_mult=4):
        super().__init__()
        hidden = int(ff_mult) * n_embd
        self.fc1 = nn.Linear(n_embd, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerCausalBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0, ff_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = CausalMLP(n_embd, dropout=dropout, ff_mult=ff_mult)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerCausal(nn.Module):
    """
    Decoder-only causal Transformer.

    If ATTN_FS=1, we add a cross-layer "future-seed" path using two extra tokens per layer:
      - prefix memory token: carries z_{l-1} to all positions (causal past)
      - suffix collector token: reads the whole sequence (causal end) and becomes z_l
    """

    def __init__(
        self,
        vocab_size,
        n_layer,
        n_embd,
        n_head,
        dropout=0.0,
        ff_mult=4,
        attn_fs=False,
        attn_fs_collector="zero",
        attn_fs_gating=False,
        attn_fs_alpha_init=0.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_layer = int(n_layer)
        self.n_embd = int(n_embd)
        self.attn_fs = bool(attn_fs)
        self.attn_fs_collector = str(attn_fs_collector)
        self.attn_fs_gating = bool(attn_fs_gating)

        if self.attn_fs_collector not in ("zero", "learned"):
            raise ValueError(f"ATTN_FS_COLLECTOR must be zero|learned, got {self.attn_fs_collector}")

        if n_embd % n_head != 0:
            raise ValueError(f"TRANS_N_HEAD must divide N_EMBD: {n_head} | {n_embd}")

        self.wte = nn.Embedding(vocab_size, n_embd)
        # +2 for optional prefix/suffix tokens when ATTN_FS=1.
        self.wpe = nn.Embedding(SEQ_LEN + 2, n_embd)
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, n_embd))

        self.blocks = nn.ModuleList(
            [
                TransformerCausalBlock(
                    n_embd=n_embd, n_head=n_head, dropout=dropout, ff_mult=ff_mult
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        if self.attn_fs_collector == "learned":
            self.collector_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        else:
            self.collector_token = None

        if self.attn_fs and self.attn_fs_gating:
            init = float(attn_fs_alpha_init)
            self.attn_fs_alpha = nn.Parameter(torch.full((n_layer,), init))
        else:
            self.attn_fs_alpha = None

    def forward(self, idx, targets=None, mask=None, **_ignored):
        B, T = idx.shape
        if self.attn_fs:
            # Tokens are shifted right by 1 due to the prefix memory token.
            pos = torch.arange(1, T + 1, device=idx.device).unsqueeze(0).expand(B, T)
        else:
            pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)

        x = self.wte(idx) + self.wpe(pos)
        x = x + (idx == mask_token_id).unsqueeze(-1).to(x.dtype) * self.mask_emb

        if not self.attn_fs:
            for block in self.blocks:
                x = block(x)
        else:
            # z carries a summary of the full sequence from the previous layer (via suffix token output).
            z = torch.zeros(B, self.n_embd, device=idx.device, dtype=x.dtype)
            pe_prefix = self.wpe(torch.zeros(1, 1, device=idx.device, dtype=torch.long)).to(x.dtype)
            pe_suffix = self.wpe(torch.full((1, 1), T + 1, device=idx.device, dtype=torch.long)).to(x.dtype)
            if self.collector_token is None:
                collector_in = torch.zeros(B, 1, self.n_embd, device=idx.device, dtype=x.dtype)
            else:
                collector_in = self.collector_token.expand(B, 1, -1).to(x.dtype)

            for li, block in enumerate(self.blocks):
                p = z.unsqueeze(1)
                if self.attn_fs_alpha is not None:
                    gate = torch.sigmoid(self.attn_fs_alpha[li]).to(x.dtype)
                    p = p * gate
                p = p + pe_prefix
                c = collector_in + pe_suffix
                ext = torch.cat([p, x, c], dim=1)  # [B, T+2, C]
                ext = block(ext)
                x = ext[:, 1 : 1 + T, :]
                z = ext[:, -1, :]

        x = self.ln_f(x)
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


def batch_loss(model, X, Y, M):
    logits, loss_main = model(X, Y, M)
    if SUDOKU_TASK and SUDOKU_CONS_LAMBDA > 0:
        # 4x4 sudoku: penalize row/col/block digit-count deviation from 1.
        # Use ground-truth one-hot for unmasked clue positions to avoid double-counting uncertainty.
        m0 = SUDOKU_PAD + 5  # "P=" (2) + pad + "|M=" (3)
        seg_logits = logits[:, m0 : m0 + 16, :]  # [B,16,V]
        seg_mask = M[:, m0 : m0 + 16]  # [B,16] (True where we need to predict)
        seg_y = Y[:, m0 : m0 + 16]  # [B,16]

        digit_ids = torch.tensor([stoi[d] for d in ["1", "2", "3", "4"]], device=seg_logits.device, dtype=torch.long)
        p_model = torch.softmax(seg_logits[..., digit_ids], dim=-1)  # [B,16,4]

        # Map seg_y token ids -> digit index 0..3 for one-hot
        y_idx = torch.zeros_like(seg_y)
        for di in range(4):
            y_idx = torch.where(seg_y == digit_ids[di], torch.full_like(y_idx, di), y_idx)
        p_true = F.one_hot(y_idx, num_classes=4).to(p_model.dtype)  # [B,16,4]

        p = torch.where(seg_mask.unsqueeze(-1), p_model, p_true)

        groups = [
            # rows
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            # cols
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
            # 2x2 blocks
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15],
        ]
        cons = 0.0
        for g in groups:
            cnt = p[:, g, :].sum(dim=1)  # [B,4]
            cons = cons + (cnt - 1.0).pow(2).mean()
        cons = cons / len(groups)
        loss_main = loss_main + SUDOKU_CONS_LAMBDA * cons
    if not FS_ENCDEC:
        return loss_main
    r = torch.rand_like(X.float())
    M_enc = M & (r < FS_ENCDEC_RATIO)
    M_dec = M & (~M_enc)
    X_enc = Y.clone()
    X_enc[M_enc] = mask_token_id
    X_dec = Y.clone()
    X_dec[M_dec] = mask_token_id
    _, loss_enc, states_enc = model(X_enc, Y, M_enc, return_states=True)
    _, loss_dec = model(X_dec, Y, M_dec, seed_states=states_enc, seed_dropout=FS_ENCDEC_STATE_DROPOUT)
    if FS_ENCDEC_AUX:
        return loss_main + FS_ENCDEC_LAMBDA * (loss_enc + loss_dec)
    return loss_enc + FS_ENCDEC_LAMBDA * loss_dec


@torch.no_grad()
def maskacc_eval(model, split="val", iters=200):
    model.eval()
    correct = 0
    total = 0
    for _ in range(iters):
        xb, yb, mb = get_batch(split)
        logits, _ = model(xb, yb, mb)
        pred = logits.argmax(dim=-1)
        correct += (pred[mb] == yb[mb]).sum().item()
        total += mb.sum().item()
    model.train()
    return correct / total if total > 0 else 0.0


def _hungarian_min_cost(cost):
    """
    Solve min-cost assignment for cost matrix [n,n] or [n,m] (n<=m).
    Returns a list `assign` of length n with assigned column indices in [0,m).
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    transposed = False
    if n > m:
        cost = cost.T
        n, m = cost.shape
        transposed = True

    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=np.bool_)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # p[j] = assigned row for column j
    assign_row_to_col = np.full(n + 1, -1, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j] != 0:
            assign_row_to_col[p[j]] = j - 1
    assign = assign_row_to_col[1:].tolist()

    if transposed:
        # We solved for cost.T; convert to assignment for original rows (m of them).
        # For our use (square matrices), this case should not happen.
        raise RuntimeError("hungarian: unexpected rectangular transpose case")
    return assign


@torch.no_grad()
def _infer_fill_refine(model, x, mask):
    if REFINE_STEPS <= 0:
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)
        return torch.where(mask, pred, x)

    x = x.clone()
    m = mask.clone()
    for _ in range(int(REFINE_STEPS)):
        if not m.any():
            break
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=-1)
        conf, tok = probs.max(dim=-1)
        conf = conf.masked_fill(~m, -1.0)
        fill = (conf >= REFINE_CONF) & m
        if not fill.any():
            flat = conf.view(-1)
            idx = flat.argmax().item()
            fill.view(-1)[idx] = True
        x = torch.where(fill, tok, x)
        m = m & ~fill

    if m.any():
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)
        x = torch.where(m, pred, x)
    return x


def _jsonl_append(path, obj):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y, M = get_batch(split)
            loss = batch_loss(model, X, Y, M)
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
def qa_eval(model, trials=200, mode="qa"):
    if not QA_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        q, a = qa_pairs[random.randint(0, len(qa_pairs) - 1)]
        s, q_start, q_end, a_start, valid_len = _qa_pack(q, a)
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        if mode == "qa":
            m_start, m_end = a_start, valid_len
        else:
            m_start, m_end = q_start, q_end
        if m_end <= m_start:
            m_start = max(0, valid_len - 1)
            m_end = valid_len
        mask[:, m_start:m_end] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[m_start:m_end]
        tgt_seg = tgt[m_start:m_end]
        correct += (pred_seg == tgt_seg).sum().item()
        total += (m_end - m_start)
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def sudoku_eval(model, trials=200, holes=8):
    if not SUDOKU_TASK:
        return None

    def valid_4x4(seg):
        if len(seg) != 16:
            return False
        if any(ch not in "1234" for ch in seg):
            return False
        g = [int(ch) for ch in seg]
        need = {1, 2, 3, 4}
        # rows
        for r in range(4):
            if set(g[r * 4 : r * 4 + 4]) != need:
                return False
        # cols
        for c in range(4):
            if set(g[c + 4 * r] for r in range(4)) != need:
                return False
        # 2x2 blocks
        for br in [0, 2]:
            for bc in [0, 2]:
                blk = []
                for r in range(br, br + 2):
                    for c in range(bc, bc + 2):
                        blk.append(g[r * 4 + c])
                if set(blk) != need:
                    return False
        return True

    model.eval()
    ok_valid = 0
    ok_exact = 0
    for _ in range(trials):
        s, sol, m0, m1, idx = _sudoku_pack_4x4(holes)
        y = torch.tensor(encode(s), dtype=torch.long, device=device).unsqueeze(0)
        x = y.clone()
        mask = torch.zeros_like(x, dtype=torch.bool)
        for j in idx:
            mask[:, m0 + j] = True
        x[mask] = mask_token_id
        out = _infer_fill_refine(model, x, mask)
        out_s = decode(out[0].tolist())
        out_m = out_s[m0:m1]
        if out_m == sol:
            ok_exact += 1
        if valid_4x4(out_m):
            ok_valid += 1
    valid_rate = ok_valid / trials if trials > 0 else 0.0
    exact_rate = ok_exact / trials if trials > 0 else 0.0
    model.train()
    return valid_rate, exact_rate


def _kvsort_parse_keys_from_input(s):
    # Parse keys from right context. Works for KVSORT_NOISE=0 (recommended).
    p = s.find("|R=")
    if p < 0:
        return []
    right = s[p + 3 :]
    if right.startswith("O="):
        semi = right.find(";")
        if semi >= 0:
            right = right[semi + 1 :]
        else:
            right = ""
    sharp = right.find("#")
    if sharp >= 0:
        right = right[:sharp]
    keys = []
    for part in right.split(";"):
        if len(part) >= 2 and part[1] == ":":
            keys.append(part[0])
        elif ":" in part:
            k = part.split(":", 1)[0]
            if len(k) > 0:
                keys.append(k[0])
    # unique, keep order
    out = []
    seen = set()
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _permfill_parse_items_from_input(s, n):
    p = s.find("|R=O=")
    if p < 0:
        return []
    semi = s.find(";", p)
    if semi < 0:
        return []
    right = s[semi + 1 :]
    sharp = right.find("#")
    if sharp >= 0:
        right = right[:sharp]
    return list(right[:n])


@torch.no_grad()
def _decode_perm_hungarian(logits, pos_list, allowed_token_ids):
    # logits: [1,T,V]; pos_list: list[int]; allowed_token_ids: list[int] length n
    if len(pos_list) != len(allowed_token_ids):
        return None
    lp = F.log_softmax(logits[0, pos_list, :], dim=-1)  # [n,V]
    mat = lp[:, allowed_token_ids].detach().cpu().numpy()  # [n,n]
    cost = (-mat).astype(np.float64)
    assign = _hungarian_min_cost(cost)  # row -> col
    out = [allowed_token_ids[j] for j in assign]
    return out


@torch.no_grad()
def kvsort_eval(model, trials=200, mode="test"):
    model.eval()
    ok_exact = 0
    ok_key_valid = 0
    ok_key_order = 0
    pair_acc_sum = 0.0
    for _ in range(trials):
        if mode == "train":
            n = random.randint(KVSORT_N_MIN, KVSORT_N_MAX)
        else:
            n = KVSORT_N_TEST
        s, m0, m1 = _kvsort_pack(n)
        y = torch.tensor(encode(s), dtype=torch.long, device=device).unsqueeze(0)
        x = y.clone()
        mask = torch.zeros_like(x, dtype=torch.bool)
        if KVSORT_KEYS_ONLY and KVSORT_KEYS_SEP and (not KVSORT_MASK_SEP):
            s_chars = list(s)
            for j in range(m0, m1):
                if s_chars[j] in key_pool:
                    mask[:, j] = True
        else:
            mask[:, m0:m1] = True
        x[mask] = mask_token_id
        gt_s = decode(y[0].tolist())
        gt_m = gt_s[m0:m1]
        logits, _ = model(x)
        if DECODE == "hungarian" and KVSORT_KEYS_ONLY and (not KVSORT_KEYS_SEP):
            # Only supports the keys-only / no-sep permutation setting.
            n_pos = m1 - m0
            keys = _kvsort_parse_keys_from_input(s)
            if len(keys) != n_pos:
                gt_keys = [ch for ch in gt_m if ch in key_pool]
                keys = gt_keys
            allowed = [stoi[k] for k in keys]
            decoded = _decode_perm_hungarian(logits, list(range(m0, m1)), allowed)
            if decoded is None:
                pred = logits.argmax(dim=-1)
                out = torch.where(mask, pred, x)
            else:
                out = x.clone()
                out[0, m0:m1] = torch.tensor(decoded, device=out.device, dtype=out.dtype)
        else:
            pred = logits.argmax(dim=-1)
            out = torch.where(mask, pred, x)
        out_s = decode(out[0].tolist())
        out_m = out_s[m0:m1]
        if out_m == gt_m:
            ok_exact += 1

        def parse_pairs(seg):
            pairs = []
            parts = seg.split(";")
            for p in parts:
                if len(p) < 3:
                    continue
                if ":" not in p:
                    continue
                k, v = p.split(":", 1)
                if len(k) == 1 and len(v) >= 1:
                    pairs.append((k[0], v[0]))
            return pairs

        pred_pairs = parse_pairs(out_m)
        gt_pairs = parse_pairs(gt_m)
        if KVSORT_KEYS_ONLY:
            pred_keys = [ch for ch in out_m if ch in key_pool]
            gt_keys = [ch for ch in gt_m if ch in key_pool]
        else:
            pred_keys = [k for k, _ in pred_pairs]
            gt_keys = [k for k, _ in gt_pairs]

        if (
            len(pred_keys) == len(gt_keys)
            and sorted(pred_keys) == sorted(gt_keys)
            and len(set(pred_keys)) == len(pred_keys)
        ):
            ok_key_valid += 1
        if pred_keys == gt_keys:
            ok_key_order += 1

        if not KVSORT_KEYS_ONLY:
            gt_map = {k: v for k, v in gt_pairs}
            if len(gt_map) > 0 and len(pred_pairs) > 0:
                corr = 0
                for k, v in pred_pairs:
                    if k in gt_map and gt_map[k] == v:
                        corr += 1
                pair_acc_sum += corr / max(1, len(gt_map))
    model.train()
    return ok_exact / trials, ok_key_valid / trials, ok_key_order / trials, pair_acc_sum / trials


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
def permfill_eval(model, trials=200, mode="test"):
    if not PERMFILL_TASK:
        return None
    model.eval()
    ok_exact = 0
    ok_valid = 0
    ok_anchor = 0
    for _ in range(trials):
        if mode == "train":
            n = random.randint(PERMFILL_N_MIN, PERMFILL_N_MAX)
        else:
            n = PERMFILL_N_TEST
        s, m0, m1, anchors = _permfill_pack(n)
        y = torch.tensor(encode(s), dtype=torch.long, device=device).unsqueeze(0)
        x = y.clone()
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, m0:m1] = True
        for a in anchors:
            mask[:, m0 + a] = False
        x[mask] = mask_token_id
        gt_s = decode(y[0].tolist())
        gt_m = gt_s[m0:m1]
        logits, _ = model(x)
        if DECODE == "hungarian":
            n_pos = m1 - m0
            items = _permfill_parse_items_from_input(s, n_pos)
            if len(items) != n_pos:
                items = [ch for ch in gt_m if ch in key_pool][:n_pos]
            allowed = [stoi[k] for k in items]

            fixed_pos = sorted([m0 + a for a in anchors])
            fixed_tok = [int(y[0, p].item()) for p in fixed_pos]
            # remove fixed
            pos_list = [p for p in range(m0, m1) if p not in set(fixed_pos)]
            allowed_rem = [t for t in allowed if t not in set(fixed_tok)]
            decoded_rem = _decode_perm_hungarian(logits, pos_list, allowed_rem) if len(pos_list) > 0 else []
            if decoded_rem is None and len(pos_list) > 0:
                pred = logits.argmax(dim=-1)
                out = torch.where(mask, pred, x)
            else:
                out = x.clone()
                # fill fixed
                for p, t in zip(fixed_pos, fixed_tok):
                    out[0, p] = t
                # fill remaining (ordered by pos_list)
                if len(pos_list) > 0:
                    out_vals = torch.tensor(decoded_rem, device=out.device, dtype=out.dtype)
                    out[0, pos_list] = out_vals
        else:
            pred = logits.argmax(dim=-1)
            out = torch.where(mask, pred, x)
        out_s = decode(out[0].tolist())
        out_m = out_s[m0:m1]
        if out_m == gt_m:
            ok_exact += 1
        if PERMFILL_ANCHOR:
            good = True
            for a in anchors:
                if out_s[m0 + a] != gt_s[m0 + a]:
                    good = False
                    break
            if good:
                ok_anchor += 1
        pred_keys = [ch for ch in out_m if ch in key_pool]
        gt_keys = [ch for ch in gt_m if ch in key_pool]
        if (
            len(pred_keys) == len(gt_keys)
            and sorted(pred_keys) == sorted(gt_keys)
            and len(set(pred_keys)) == len(pred_keys)
        ):
            ok_valid += 1
    model.train()
    exact = ok_exact / trials
    valid = ok_valid / trials
    anchor = ok_anchor / trials if PERMFILL_ANCHOR else 0.0
    return exact, valid, anchor


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
def rightcopy_eval(model, trials=200):
    if not RIGHTCOPY_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _rightcopy_sample(RIGHTCOPY_LEN)
        p1 = s.find("|M=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 3 : p1 + 3 + RIGHTCOPY_LEN] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 3 : p1 + 3 + RIGHTCOPY_LEN]
        tgt_seg = tgt[p1 + 3 : p1 + 3 + RIGHTCOPY_LEN]
        correct += (pred_seg == tgt_seg).sum().item()
        total += RIGHTCOPY_LEN
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def rightrev_eval(model, trials=200):
    if not RIGHTREV_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _rightrev_sample(RIGHTREV_LEN)
        p1 = s.find("|M=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 3 : p1 + 3 + RIGHTREV_LEN] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 3 : p1 + 3 + RIGHTREV_LEN]
        tgt_seg = tgt[p1 + 3 : p1 + 3 + RIGHTREV_LEN]
        correct += (pred_seg == tgt_seg).sum().item()
        total += RIGHTREV_LEN
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def index_eval(model, trials=200):
    if not INDEX_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _index_sample(INDEX_LEN)
        p1 = s.find("|Y=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 3 : p1 + 3 + 4] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 3 : p1 + 3 + 4]
        tgt_seg = tgt[p1 + 3 : p1 + 3 + 4]
        correct += (pred_seg == tgt_seg).sum().item()
        total += 4
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def rule_eval(model, trials=200):
    if not RULE_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _rule_sample(RULE_LEN)
        p1 = s.find("|M=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 3 : p1 + 3 + RULE_LEN] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 3 : p1 + 3 + RULE_LEN]
        tgt_seg = tgt[p1 + 3 : p1 + 3 + RULE_LEN]
        correct += (pred_seg == tgt_seg).sum().item()
        total += RULE_LEN
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def cipher_eval(model, trials=200):
    if not CIPHER_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _cipher_sample(CIPHER_LEN)
        p1 = s.find("|M=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 3 : p1 + 3 + CIPHER_LEN] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 3 : p1 + 3 + CIPHER_LEN]
        tgt_seg = tgt[p1 + 3 : p1 + 3 + CIPHER_LEN]
        correct += (pred_seg == tgt_seg).sum().item()
        total += CIPHER_LEN
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def shift_eval(model, trials=200):
    if not SHIFT_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _shift_sample(SHIFT_LEN)
        p1 = s.find("|M=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 3 : p1 + 3 + SHIFT_LEN] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 3 : p1 + 3 + SHIFT_LEN]
        tgt_seg = tgt[p1 + 3 : p1 + 3 + SHIFT_LEN]
        correct += (pred_seg == tgt_seg).sum().item()
        total += SHIFT_LEN
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def poscopy_eval(model, trials=200):
    if not POSCOPY_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _poscopy_sample(POSCOPY_LEN)
        p1 = s.find("|M=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 3 : p1 + 3 + POSCOPY_REP] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 3 : p1 + 3 + POSCOPY_REP]
        tgt_seg = tgt[p1 + 3 : p1 + 3 + POSCOPY_REP]
        correct += (pred_seg == tgt_seg).sum().item()
        total += POSCOPY_REP
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


@torch.no_grad()
def constr_eval(model, trials=200):
    if not CONSTR_TASK:
        return None
    model.eval()
    correct = 0
    total = 0
    for _ in range(trials):
        s = _constr_sample(CONSTR_LEN)
        p1 = s.find("|M=")
        if p1 < 0:
            continue
        tgt = torch.tensor(encode(s), device=device)
        x = tgt.clone().unsqueeze(0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, p1 + 3 : p1 + 3 + CONSTR_LEN] = True
        x[mask] = mask_token_id
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)[0]
        pred_seg = pred[p1 + 3 : p1 + 3 + CONSTR_LEN]
        tgt_seg = tgt[p1 + 3 : p1 + 3 + CONSTR_LEN]
        correct += (pred_seg == tgt_seg).sum().item()
        total += CONSTR_LEN
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
    elif POSCOPY_TASK:
        s = _poscopy_sample(POSCOPY_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif CONSTR_TASK:
        s = _constr_sample(CONSTR_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif SHIFT_TASK:
        s = _shift_sample(SHIFT_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif CIPHER_TASK:
        s = _cipher_sample(CIPHER_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif RULE_TASK:
        s = _rule_sample(RULE_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif RIGHTCOPY_TASK:
        s = _rightcopy_sample(RIGHTCOPY_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif RIGHTREV_TASK:
        s = _rightrev_sample(RIGHTREV_LEN)
        all_tokens = encode(s)[:prompt_len]
    elif INDEX_TASK:
        s = _index_sample(INDEX_LEN)
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
    elif QA_TASK:
        q, a = qa_pairs[random.randint(0, len(qa_pairs) - 1)]
        s, _, _, _, _ = _qa_pack(q, a)
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
def log_eval_sample(model, split="val"):
    model.eval()
    xb, yb, mb = get_batch(split)
    xb = xb[:1]
    yb = yb[:1]
    mb = mb[:1]
    logits, _ = model(xb, yb, mb)
    pred = logits.argmax(dim=-1)
    out = torch.where(mb, pred, xb)
    idx = mb[0].nonzero().squeeze(-1)
    start = idx[0].item()
    end = idx[-1].item() + 1
    s = max(start - LOG_WIN, 0)
    e = min(end + LOG_WIN, xb.size(1))
    print(f"mask[{start}:{end}] len={end - start}")
    print(f"IN[{s}:{e}]: {decode(xb[0, s:e].tolist())}")
    print(f"GT[{start}:{end}]: {decode(yb[0, start:end].tolist())}")
    print(f"PR[{start}:{end}]: {decode(out[0, start:end].tolist())}")
    model.train()


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
    elif QA_TASK:
        q, a = qa_pairs[random.randint(0, len(qa_pairs) - 1)]
        s, _, _, _, _ = _qa_pack(q, a)
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
    if MODEL == "transformer":
        model = TransformerMLM(
            vocab_size=vocab_size,
            n_layer=N_LAYER,
            n_embd=N_EMBD,
            n_head=TRANS_N_HEAD,
            dropout=TRANS_DROPOUT,
            ff_mult=TRANS_FF_MULT,
        )
    elif MODEL == "transformer_causal":
        model = TransformerCausal(
            vocab_size=vocab_size,
            n_layer=N_LAYER,
            n_embd=N_EMBD,
            n_head=TRANS_N_HEAD,
            dropout=TRANS_DROPOUT,
            ff_mult=TRANS_FF_MULT,
            attn_fs=ATTN_FS,
            attn_fs_collector=ATTN_FS_COLLECTOR,
            attn_fs_gating=ATTN_FS_GATING,
            attn_fs_alpha_init=ATTN_FS_ALPHA_INIT,
        )
    else:
        model = GPT(GPTConfig(vocab_size=vocab_size, n_layer=N_LAYER, n_embd=N_EMBD), future_seed=FUTURE_SEED)
    m = model.to(device)

    if FS_MASK_ONLY and MODEL == "rwkv":
        for p in m.parameters():
            p.requires_grad = False
        for n, p in m.named_parameters():
            if "future_seed_alpha" in n or n == "mask_emb":
                p.requires_grad = True

    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")
    print(sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6, "M trainable")

    if os.path.exists(WEIGHTS_PATH):
        m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

    run_meta = dict(
        model=MODEL,
        future_seed=int(bool(FUTURE_SEED)),
        attn_fs=int(bool(ATTN_FS)),
        attn_fs_collector=str(ATTN_FS_COLLECTOR),
        attn_fs_gating=int(bool(ATTN_FS_GATING)),
        attn_fs_alpha_init=float(ATTN_FS_ALPHA_INIT),
        decode=DECODE,
        refine_steps=int(REFINE_STEPS),
        refine_conf=float(REFINE_CONF),
        bin_mask_mode=BIN_MASK_MODE,
        bin_prefix_ratio=float(BIN_PREFIX_RATIO),
        seq_len=int(SEQ_LEN),
        n_layer=int(N_LAYER),
        n_embd=int(N_EMBD),
        trans_n_head=int(TRANS_N_HEAD),
        trans_dropout=float(TRANS_DROPOUT),
        trans_ff_mult=int(TRANS_FF_MULT),
        head_size=int(HEAD_SIZE),
        random_seed=int(random_seed),
        weights_path=str(WEIGHTS_PATH),
        task=dict(
            rightcopy=bool(RIGHTCOPY_TASK),
            constr=bool(CONSTR_TASK),
            kvsort=bool(KVSORT_TASK),
            permfill=bool(PERMFILL_TASK),
            sudoku=bool(SUDOKU_TASK),
            struct=bool(STRUCT_TASK),
            sent=bool(SENT_TASK),
        ),
    )
    _jsonl_append(LOG_JSONL, {"event": "start", **run_meta})

    if TRAIN:
        fused_ok = device == "cuda"
        if MODEL in ("transformer", "transformer_causal"):
            optimizer = torch.optim.AdamW(
                [p for p in m.parameters() if p.requires_grad],
                lr=ADAM_LR,
                betas=(0.9, 0.95),
                fused=fused_ok,
            )
            optimizers = [optimizer]
        elif FS_MASK_ONLY:
            trainable = [p for p in m.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=ADAM_LR, betas=(0.9, 0.95), fused=fused_ok)
            optimizers = [optimizer]
        else:
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
                rec = {"event": "eval", "iter": int(iter), "train_loss": float(losses["train"]), "val_loss": float(losses["val"])}
                if REVERSE_EVAL:
                    acc = reverse_eval(m)
                    print(f"reverse acc {acc:.4f}")
                    rec["reverse_acc"] = float(acc)
                if BIDIR_EVAL:
                    acc = bidir_eval(m)
                    print(f"bidir acc {acc:.4f}")
                    rec["bidir_acc"] = float(acc)
                if ADD_EVAL:
                    acc = add_eval(m)
                    print(f"add acc {acc:.4f}")
                    rec["add_acc"] = float(acc)
                if PERM_EVAL:
                    acc = perm_eval(m)
                    print(f"perm acc {acc:.4f}")
                    rec["perm_acc"] = float(acc)
                if INTER_EVAL:
                    acc = inter_eval(m)
                    print(f"inter acc {acc:.4f}")
                    rec["inter_acc"] = float(acc)
                if MULTI_EVAL:
                    acc = multi_eval(m)
                    print(f"multi acc {acc:.4f}")
                    rec["multi_acc"] = float(acc)
                if PARITY_EVAL:
                    acc = parity_eval(m)
                    print(f"parity acc {acc:.4f}")
                    rec["parity_acc"] = float(acc)
                if RIGHTCOPY_EVAL:
                    acc = rightcopy_eval(m)
                    print(f"rightcopy acc {acc:.4f}")
                    rec["rightcopy_acc"] = float(acc)
                if RIGHTREV_EVAL:
                    acc = rightrev_eval(m)
                    print(f"rightrev acc {acc:.4f}")
                    rec["rightrev_acc"] = float(acc)
                if INDEX_EVAL:
                    acc = index_eval(m)
                    print(f"index acc {acc:.4f}")
                    rec["index_acc"] = float(acc)
                if RULE_EVAL:
                    acc = rule_eval(m)
                    print(f"rule acc {acc:.4f}")
                    rec["rule_acc"] = float(acc)
                if CIPHER_EVAL:
                    acc = cipher_eval(m)
                    print(f"cipher acc {acc:.4f}")
                    rec["cipher_acc"] = float(acc)
                if SHIFT_EVAL:
                    acc = shift_eval(m)
                    print(f"shift acc {acc:.4f}")
                    rec["shift_acc"] = float(acc)
                if POSCOPY_EVAL:
                    acc = poscopy_eval(m)
                    print(f"poscopy acc {acc:.4f}")
                    rec["poscopy_acc"] = float(acc)
                if CONSTR_EVAL:
                    acc = constr_eval(m)
                    print(f"constr acc {acc:.4f}")
                    rec["constr_acc"] = float(acc)
                if STRUCT_EVAL:
                    acc, exact = struct_eval(m)
                    print(f"struct acc {acc:.4f}, exact {exact:.4f}")
                    rec["struct_acc"] = float(acc)
                    rec["struct_exact"] = float(exact)
                if RETR_EVAL:
                    acc = retr_eval(m)
                    print(f"retr acc {acc:.4f}")
                    rec["retr_acc"] = float(acc)
                if SENT_EVAL:
                    acc = sent_eval(m)
                    print(f"sent acc {acc:.4f}")
                    rec["sent_acc"] = float(acc)
                if MASKACC_EVAL:
                    acc = maskacc_eval(m, split=MASKACC_SPLIT, iters=MASKACC_ITERS)
                    print(f"maskacc_{MASKACC_SPLIT} {acc:.4f}")
                    rec[f"maskacc_{MASKACC_SPLIT}"] = float(acc)
                if QA_EVAL:
                    acc_qa = qa_eval(m, mode="qa")
                    acc_aq = qa_eval(m, mode="aq")
                    print(f"qa->a acc {acc_qa:.4f}, a->q acc {acc_aq:.4f}")
                    rec["qa_to_a_acc"] = float(acc_qa)
                    rec["a_to_q_acc"] = float(acc_aq)
                if SUDOKU_EVAL:
                    res = sudoku_eval(m, trials=SUDOKU_TRIALS, holes=SUDOKU_HOLES_TEST)
                    if res is not None:
                        va, ex = res
                        print(f"sudoku holes {SUDOKU_HOLES_TEST} solve {va:.4f}, exact {ex:.4f}")
                        rec["sudoku_holes"] = int(SUDOKU_HOLES_TEST)
                        rec["sudoku_solve"] = float(va)
                        rec["sudoku_exact"] = float(ex)
                if KVSORT_EVAL:
                    ex, kv, ko, pa = kvsort_eval(m, mode="train")
                    ex2, kv2, ko2, pa2 = kvsort_eval(m, mode="test")
                    print(f"kvsort_id exact {ex:.4f}, key_valid {kv:.4f}, key_order {ko:.4f}, pair_acc {pa:.4f}")
                    print(f"kvsort_ood exact {ex2:.4f}, key_valid {kv2:.4f}, key_order {ko2:.4f}, pair_acc {pa2:.4f}")
                    rec["kvsort_id_exact"] = float(ex)
                    rec["kvsort_id_key_valid"] = float(kv)
                    rec["kvsort_id_key_order"] = float(ko)
                    rec["kvsort_ood_exact"] = float(ex2)
                    rec["kvsort_ood_key_valid"] = float(kv2)
                    rec["kvsort_ood_key_order"] = float(ko2)
                if PERMFILL_EVAL:
                    ex, va, an = permfill_eval(m, mode="train")
                    ex2, va2, an2 = permfill_eval(m, mode="test")
                    print(f"permfill_id exact {ex:.4f}, valid {va:.4f}, anchor {an:.4f}")
                    print(f"permfill_ood exact {ex2:.4f}, valid {va2:.4f}, anchor {an2:.4f}")
                    rec["permfill_id_exact"] = float(ex)
                    rec["permfill_id_valid"] = float(va)
                    rec["permfill_ood_exact"] = float(ex2)
                    rec["permfill_ood_valid"] = float(va2)
                    rec["permfill_anchor_ok"] = float(an2)
                if LOG_SAMPLE:
                    log_eval_sample(m, "val")
                _jsonl_append(LOG_JSONL, {**run_meta, **rec})

            m.train()
            m.zero_grad(set_to_none=True)
            grad_accum_steps = BATCH_SIZE // DEVICE_BSZ
            for _ in range(grad_accum_steps):
                xb, yb, mb = get_batch("train")
                loss = batch_loss(m, xb, yb, mb)
                (loss / grad_accum_steps).backward()

            if not FS_MASK_ONLY:
                frac = min(iter / 500, 1)
                if MODEL == "rwkv":
                    optimizer3.param_groups[0]["momentum"] = (1 - frac) * 0.85 + frac * 0.95

            for opt, sched in zip(optimizers, schedulers):
                opt.step()
                sched.step()

        print(f"Total training time: {time.time() - start:.2f} seconds")
        _jsonl_append(LOG_JSONL, {"event": "end", **run_meta, "time_sec": float(time.time() - start)})
        torch.save(m.state_dict(), WEIGHTS_PATH)
    elif not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"WEIGHTS_PATH not found and TRAIN=0: {WEIGHTS_PATH}")

    if (not TRAIN) and KVSORT_EVAL:
        ex, kv, ko, pa = kvsort_eval(m, mode="train")
        ex2, kv2, ko2, pa2 = kvsort_eval(m, mode="test")
        print(f"kvsort_id exact {ex:.4f}, key_valid {kv:.4f}, key_order {ko:.4f}, pair_acc {pa:.4f}")
        print(f"kvsort_ood exact {ex2:.4f}, key_valid {kv2:.4f}, key_order {ko2:.4f}, pair_acc {pa2:.4f}")
    if (not TRAIN) and SUDOKU_EVAL:
        res = sudoku_eval(m, trials=SUDOKU_TRIALS, holes=SUDOKU_HOLES_TEST)
        if res is not None:
            va, ex = res
            print(f"sudoku holes {SUDOKU_HOLES_TEST} solve {va:.4f}, exact {ex:.4f}")
    if (not TRAIN) and PERMFILL_EVAL:
        ex, va, an = permfill_eval(m, mode="train")
        ex2, va2, an2 = permfill_eval(m, mode="test")
        print(f"permfill_id exact {ex:.4f}, valid {va:.4f}, anchor {an:.4f}")
        print(f"permfill_ood exact {ex2:.4f}, valid {va2:.4f}, anchor {an2:.4f}")
    if (not TRAIN) and MASKACC_EVAL:
        acc = maskacc_eval(m, split=MASKACC_SPLIT, iters=MASKACC_ITERS)
        print(f"maskacc_{MASKACC_SPLIT} {acc:.4f}")

    if MEM_CHECK:
        mem_check(m, prompt_len=PROMPT_LEN)
    if LOG_OUTPUT:
        output = generate(m, max_new_tokens=GEN_TOKENS, temp=0.8, confidence_threshold=0.95, top_k=2)
        print(f"\nOutput:\n{output}")
