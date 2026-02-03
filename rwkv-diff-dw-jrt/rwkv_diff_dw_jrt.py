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
TRAIN = env_int("TRAIN", 0) == 1
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

use_bin = len(DATA_BIN) > 0


if use_bin:
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
        mask_probs = torch.rand(DEVICE_BSZ, 1)
        mask = torch.rand(DEVICE_BSZ, SEQ_LEN) < mask_probs
        missing = ~mask.any(dim=1)
        if missing.any():
            idx = torch.randint(0, SEQ_LEN, (int(missing.sum()),))
            mask[missing, idx] = True
        x[mask] = mask_token_id
        return x.to(device), y.to(device), mask.to(device)


def rwkv7_recurrence(r, w, k, v, a, b, state):
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
    if has_state:
        state = state * DW_JRT_SCALE

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

        x, state = rwkv7_recurrence(r, w, k, v, -kk, kk * a, state)
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
def generate(model, max_new_tokens, prompt_len=16, temp=1.0, confidence_threshold=0.95, top_k=3):
    if use_bin:
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

    output = generate(m, max_new_tokens=GEN_TOKENS, temp=0.8, confidence_threshold=0.95, top_k=2)
    print(f"\nOutput:\n{output}")
