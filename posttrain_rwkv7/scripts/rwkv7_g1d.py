from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_wind import rwkv7_wind


@dataclass
class RWKV7G1DConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    norm_eps: float = 1e-5  # matches HF config; GN uses head_dim*norm_eps


def _infer_num_layers(sd: Dict[str, torch.Tensor]) -> int:
    pat = re.compile(r"^blocks\.(\d+)\.")
    mx = -1
    for k in sd.keys():
        m = pat.match(k)
        if m:
            mx = max(mx, int(m.group(1)))
    if mx < 0:
        raise ValueError("No blocks.* keys found")
    return mx + 1


def infer_config_from_state_dict(sd: Dict[str, torch.Tensor]) -> RWKV7G1DConfig:
    emb = sd["emb.weight"]
    vocab_size, hidden = emb.shape
    n_layer = _infer_num_layers(sd)
    rk = sd["blocks.0.att.r_k"]
    num_heads, head_dim = rk.shape
    ffn_key = sd["blocks.0.ffn.key.weight"]
    intermediate, hidden2 = ffn_key.shape
    if hidden2 != hidden:
        raise ValueError(f"ffn key hidden mismatch: {hidden2} vs {hidden}")
    return RWKV7G1DConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_layers=n_layer,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=intermediate,
    )


def _fs_schedule_multiplier(
    schedule: str,
    layer_idx: int,
    layer_start: int,
    num_layers: int,
    vmin: float,
    vmax: float,
) -> float:
    """Layer-depth schedule multiplier for Future-Seed gate."""
    if schedule == "none":
        return 1.0

    if num_layers <= 0:
        return float(vmax)

    start = max(1, int(layer_start))
    i0 = max(start, 1)
    i1 = max(i0, num_layers - 1)
    if i1 <= i0:
        p = 1.0
    else:
        p = float(layer_idx - i0) / float(i1 - i0)
        p = max(0.0, min(1.0, p))

    lo = float(vmin)
    hi = float(vmax)
    if schedule == "linear":
        return lo + (hi - lo) * p
    if schedule == "cosine":
        # 0->1 smooth ramp.
        c = 0.5 - 0.5 * math.cos(math.pi * p)
        return lo + (hi - lo) * c
    raise ValueError(f"Unsupported fs_alpha_schedule: {schedule}")


class LowRankAdapter(nn.Module):
    """RWKV7 LoRA-style module, but named to match BlinkDL .pth keys.

    For weights:
      *2: first linear weight (low_rank, in)
      *1: second linear weight (out, low_rank)
      *0: bias (out,) optional, stored as (1,1,out) in .pth

    Activation is applied between the two linears.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        low_rank: int,
        activation: str,
        bias: bool,
        *,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.low_rank = low_rank
        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "sigmoid":
            self.act = torch.sigmoid
        elif activation == "none":
            self.act = None
        else:
            raise ValueError(f"bad activation: {activation}")

        # Store as parameters for state_dict compatibility.
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.has_bias = bias
        if bias:
            self.w0 = nn.Parameter(w0)
        else:
            self.register_parameter("w0", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,in]
        y = x @ self.w2.t()  # [B,T,low]
        if self.act is not None:
            y = self.act(y)
        y = y @ self.w1.t()  # [B,T,out]
        if self.has_bias:
            y = y + self.w0
        return y


class RWKV7Attention(nn.Module):
    def __init__(
        self,
        cfg: RWKV7G1DConfig,
        layer_idx: int,
        sd: Dict[str, torch.Tensor],
        *,
        cuda_src_dir: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.cuda_src_dir = cuda_src_dir

        p = f"blocks.{layer_idx}.att."

        # token-shift mixing
        self.x_r = nn.Parameter(sd[p + "x_r"].to(torch.bfloat16))
        self.x_w = nn.Parameter(sd[p + "x_w"].to(torch.bfloat16))
        self.x_k = nn.Parameter(sd[p + "x_k"].to(torch.bfloat16))
        self.x_v = nn.Parameter(sd[p + "x_v"].to(torch.bfloat16))
        self.x_a = nn.Parameter(sd[p + "x_a"].to(torch.bfloat16))
        self.x_g = nn.Parameter(sd[p + "x_g"].to(torch.bfloat16))

        self.k_k = nn.Parameter(sd[p + "k_k"].to(torch.bfloat16))  # (1,1,C)
        self.k_a = nn.Parameter(sd[p + "k_a"].to(torch.bfloat16))  # (1,1,C)
        self.r_k = nn.Parameter(sd[p + "r_k"].to(torch.bfloat16))  # (H,D)

        # Linear projections (bias=False)
        self.receptance_w = nn.Parameter(sd[p + "receptance.weight"].to(torch.bfloat16))
        self.key_w = nn.Parameter(sd[p + "key.weight"].to(torch.bfloat16))
        self.value_w = nn.Parameter(sd[p + "value.weight"].to(torch.bfloat16))
        self.output_w = nn.Parameter(sd[p + "output.weight"].to(torch.bfloat16))

        # LoRA-style adapters
        # decay: tanh -> sigmoid outside
        self.w_lora = LowRankAdapter(
            cfg.hidden_size,
            cfg.hidden_size,
            low_rank=sd[p + "w2"].shape[0],
            activation="tanh",
            bias=True,
            w0=sd[p + "w0"].view(1, 1, -1).to(torch.bfloat16),
            w1=sd[p + "w1"].to(torch.bfloat16),
            w2=sd[p + "w2"].to(torch.bfloat16),
        )
        # a: identity -> sigmoid outside
        self.a_lora = LowRankAdapter(
            cfg.hidden_size,
            cfg.hidden_size,
            low_rank=sd[p + "a2"].shape[0],
            activation="none",
            bias=True,
            w0=sd[p + "a0"].view(1, 1, -1).to(torch.bfloat16),
            w1=sd[p + "a1"].to(torch.bfloat16),
            w2=sd[p + "a2"].to(torch.bfloat16),
        )
        # gate: sigmoid activation inside, no bias
        self.g_lora = LowRankAdapter(
            cfg.hidden_size,
            cfg.hidden_size,
            low_rank=sd[p + "g2"].shape[0],
            activation="sigmoid",
            bias=False,
            w0=torch.zeros(1, 1, cfg.hidden_size, dtype=torch.bfloat16),
            w1=sd[p + "g1"].to(torch.bfloat16),
            w2=sd[p + "g2"].to(torch.bfloat16),
        )
        # v: identity -> sigmoid outside (only for non-first layers)
        if layer_idx != 0:
            self.v_lora = LowRankAdapter(
                cfg.hidden_size,
                cfg.hidden_size,
                low_rank=sd[p + "v2"].shape[0],
                activation="none",
                bias=True,
                w0=sd[p + "v0"].view(1, 1, -1).to(torch.bfloat16),
                w1=sd[p + "v1"].to(torch.bfloat16),
                w2=sd[p + "v2"].to(torch.bfloat16),
            )
        else:
            self.v_lora = None

        # GroupNorm (named ln_x in .pth)
        gn_eps = cfg.head_dim * cfg.norm_eps
        self.gn = nn.GroupNorm(
            num_groups=cfg.num_heads,
            num_channels=cfg.hidden_size,
            eps=gn_eps,
            affine=True,
        )
        self.gn.weight = nn.Parameter(sd[p + "ln_x.weight"].to(torch.float32))
        self.gn.bias = nn.Parameter(sd[p + "ln_x.bias"].to(torch.float32))

    def forward(
        self,
        x: torch.Tensor,  # [B,T,C] bf16
        v_first: Optional[torch.Tensor],
        s0: Optional[torch.Tensor],  # [B,H,D,D] float32
        *,
        return_state: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape
        H, D = self.cfg.num_heads, self.cfg.head_dim

        # token shift delta = x_{t-1} - x_t
        x_shift = torch.zeros_like(x)
        x_shift[:, 1:] = x[:, :-1]
        delta = x_shift - x

        xr = x + delta * self.x_r
        xw = x + delta * self.x_w
        xk = x + delta * self.x_k
        xv = x + delta * self.x_v
        xa = x + delta * self.x_a
        xg = x + delta * self.x_g

        r = F.linear(xr, self.receptance_w)  # [B,T,C]
        # Reference (BlinkDL train_rwkv7.py):
        #   w = -softplus(-(time_decay + tanh(xwa @ w1) @ w2)) - 0.5
        # Note: the CUDA kernel further applies exp(-exp(w)).
        w_pre = self.w_lora(xw).float()
        w = -F.softplus(-w_pre) - 0.5
        k = F.linear(xk, self.key_w)
        v = F.linear(xv, self.value_w)

        if self.layer_idx == 0:
            v_first = v
        else:
            assert v_first is not None
            gate_v = torch.sigmoid(self.v_lora(xv))
            v = torch.lerp(v, v_first, gate_v)

        a_gate = torch.sigmoid(self.a_lora(xa))
        g = self.g_lora(xg)

        kk = F.normalize((k * self.k_k).view(B, T, H, D), dim=-1, p=2.0)
        # k = k*(1 + (a-1)*k_a)
        k = k + (k * (a_gate - 1.0)) * self.k_a

        r4 = r.view(B, T, H, D)
        w4 = w.view(B, T, H, D)
        k4 = k.view(B, T, H, D)
        v4 = v.view(B, T, H, D)
        ag4 = a_gate.view(B, T, H, D)
        a_vec = (-kk).to(torch.bfloat16)
        b_vec = (kk * ag4).to(torch.bfloat16)

        if s0 is None:
            s0_bf16 = torch.zeros(B, H, D, D, device=x.device, dtype=torch.bfloat16)
        else:
            s0_bf16 = s0.to(torch.bfloat16)

        y4, sT = rwkv7_wind(
            w=w4.to(torch.bfloat16),
            q=r4.to(torch.bfloat16),
            k=k4.to(torch.bfloat16),
            v=v4.to(torch.bfloat16),
            a=a_vec,
            b=b_vec,
            s0=s0_bf16,
            head_size=D,
            cuda_src_dir=self.cuda_src_dir,
        )

        y = y4.reshape(B, T, C).float()
        # groupnorm over channels
        y = self.gn(y.reshape(B * T, C)).reshape(B, T, C)

        # gate + output correction
        # correction_term = ((r*k*r_k).sum(-1,keepdim=True) * v)
        corr_s = (r4.float() * k4.float() * self.r_k.float().unsqueeze(0).unsqueeze(0)).sum(-1, keepdim=True)
        corr = (corr_s * v4.float()).reshape(B, T, C)
        y = (y + corr) * g.float()

        # output projection
        y = F.linear(y, self.output_w.float())  # [B,T,C]

        if return_state:
            return y, v_first, sT.float()
        return y, v_first, None


class RWKV7FeedForward(nn.Module):
    def __init__(self, cfg: RWKV7G1DConfig, layer_idx: int, sd: Dict[str, torch.Tensor]):
        super().__init__()
        p = f"blocks.{layer_idx}.ffn."
        self.x_k = nn.Parameter(sd[p + "x_k"].to(torch.bfloat16))  # (1,1,C)
        self.key_w = nn.Parameter(sd[p + "key.weight"].to(torch.bfloat16))  # (inter, C)
        self.value_w = nn.Parameter(sd[p + "value.weight"].to(torch.bfloat16))  # (C, inter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shift = torch.zeros_like(x)
        x_shift[:, 1:] = x[:, :-1]
        delta = x_shift - x
        xx = x + delta * self.x_k
        h = F.linear(xx, self.key_w)
        h = F.relu(h).square()
        y = F.linear(h, self.value_w)
        return y


class RWKV7Block(nn.Module):
    def __init__(
        self,
        cfg: RWKV7G1DConfig,
        layer_idx: int,
        sd: Dict[str, torch.Tensor],
        *,
        cuda_src_dir: str,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # layernorm weights (ln0 exists only at layer0)
        if layer_idx == 0 and f"blocks.0.ln0.weight" in sd:
            self.pre_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, elementwise_affine=True)
            self.pre_ln.weight = nn.Parameter(sd["blocks.0.ln0.weight"].float())
            self.pre_ln.bias = nn.Parameter(sd["blocks.0.ln0.bias"].float())
        else:
            self.pre_ln = None

        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, elementwise_affine=True)
        self.ln1.weight = nn.Parameter(sd[f"blocks.{layer_idx}.ln1.weight"].float())
        self.ln1.bias = nn.Parameter(sd[f"blocks.{layer_idx}.ln1.bias"].float())

        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, elementwise_affine=True)
        self.ln2.weight = nn.Parameter(sd[f"blocks.{layer_idx}.ln2.weight"].float())
        self.ln2.bias = nn.Parameter(sd[f"blocks.{layer_idx}.ln2.bias"].float())

        self.att = RWKV7Attention(cfg, layer_idx, sd, cuda_src_dir=cuda_src_dir)
        self.ffn = RWKV7FeedForward(cfg, layer_idx, sd)

    def forward(
        self,
        x: torch.Tensor,
        v_first: Optional[torch.Tensor],
        s0: Optional[torch.Tensor],
        *,
        return_state: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.pre_ln is not None:
            x = self.pre_ln(x.float()).to(torch.bfloat16)
        residual = x
        att_in = self.ln1(x.float()).to(torch.bfloat16)
        att_out, v_first, sT = self.att(att_in, v_first, s0, return_state=return_state)
        x = residual + att_out
        residual = x
        ffn_out = self.ffn(self.ln2(x.float()).to(torch.bfloat16))
        x = residual + ffn_out
        return x, v_first, sT


class RWKV7G1DLM(nn.Module):
    def __init__(self, sd: Dict[str, torch.Tensor], cfg: RWKV7G1DConfig, *, cuda_src_dir: str):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.emb.weight = nn.Parameter(sd["emb.weight"].to(torch.bfloat16))

        self.blocks = nn.ModuleList(
            [RWKV7Block(cfg, i, sd, cuda_src_dir=cuda_src_dir) for i in range(cfg.num_layers)]
        )

        self.ln_out = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, elementwise_affine=True)
        self.ln_out.weight = nn.Parameter(sd["ln_out.weight"].float())
        self.ln_out.bias = nn.Parameter(sd["ln_out.bias"].float())

        # head.weight in .pth is (C, V)
        self.head_w = nn.Parameter(sd["head.weight"].to(torch.bfloat16))

    @classmethod
    def from_pth(cls, path: str, *, cuda_src_dir: str, device: str = "cuda") -> "RWKV7G1DLM":
        sd = torch.load(path, map_location="cpu")
        if not isinstance(sd, dict):
            raise TypeError("Expected state_dict")
        cfg = infer_config_from_state_dict(sd)
        m = cls(sd, cfg, cuda_src_dir=cuda_src_dir)
        return m.to(device)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to logits."""
        return F.linear(hidden, self.head_w).float()

    def forward(
        self,
        input_ids: torch.Tensor,  # [B,T]
        *,
        seed_states: Optional[List[Optional[torch.Tensor]]] = None,
        future_seed: bool = False,
        fs_alpha: Optional[torch.Tensor] = None,
        fs_alpha_head: Optional[torch.Tensor] = None,
        fs_in_w: Optional[torch.Tensor] = None,
        fs_in_b: Optional[torch.Tensor] = None,
        seed_scale: float = 1.0,
        fs_layer_start: int = 1,
        fs_alpha_schedule: str = "none",
        fs_alpha_min: float = 1.0,
        fs_alpha_max: float = 1.0,
        fs_norm: bool = False,
        fs_clip: float = 0.0,
        fs_detach: bool = False,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward.

        seed_states: per-layer initial attention state, each is [B,H,D,D] float32.
        future_seed: if True, layer i>0 uses the terminal attention state of layer i-1
          as its initial state (the "Future-Seed" mechanism).
        fs_alpha: optional per-layer gate (raw, sigmoid'ed) applied to the future seed.
        fs_alpha_head: optional per-layer per-head gate (raw, sigmoid'ed) applied to the future seed.
        fs_in_w: optional per-layer input-adaptive gate weights, shape [L,C].
        fs_in_b: optional per-layer input-adaptive gate bias, shape [L].
          If either is set, we compute a per-sample gate:
            gate = sigmoid(fs_alpha[i] + <x_last, fs_in_w[i]>/sqrt(C) + fs_in_b[i])
          where x_last is the layer input's last-token hidden state.
        seed_scale: scalar multiplier applied after gating.
        fs_layer_start: apply FS only for layers i >= fs_layer_start (i>0 always).
        fs_alpha_schedule: optional depth schedule for FS strength: {none, linear, cosine}.
        fs_alpha_min/fs_alpha_max: schedule output range (multiplier on seed gate).
        fs_norm: if True, RMS-normalize the future seed per head before gating (stabilizes scale drift).
        fs_clip: if >0, clip the gated FS tensor to [-fs_clip, +fs_clip] before injecting.
        fs_detach: if True, stop gradients through the future seed tensor (stabilizes post-training).
        """
        x = self.emb(input_ids)
        x = x.to(torch.bfloat16)

        B, T, C = x.shape
        v_first = None
        states: List[torch.Tensor] = []
        prev_sT: Optional[torch.Tensor] = None
        need_state = return_states or future_seed
        for i, block in enumerate(self.blocks):
            s0 = None
            if seed_states is not None:
                s0 = seed_states[i]
            if future_seed and i > 0 and i >= int(fs_layer_start):
                assert prev_sT is not None
                sched = _fs_schedule_multiplier(
                    schedule=str(fs_alpha_schedule),
                    layer_idx=i,
                    layer_start=int(fs_layer_start),
                    num_layers=len(self.blocks),
                    vmin=float(fs_alpha_min),
                    vmax=float(fs_alpha_max),
                )
                g: torch.Tensor = x.new_tensor(seed_scale * sched, dtype=torch.float32)

                # Input-adaptive gate (per sample) is the default when any of fs_in_* is provided.
                if (fs_in_w is not None) or (fs_in_b is not None):
                    x_last = x[:, -1, :].float()  # [B,C]
                    logit = torch.zeros((B,), device=x.device, dtype=torch.float32)
                    if fs_alpha is not None:
                        logit = logit + fs_alpha[i].to(torch.float32)
                    if fs_in_w is not None:
                        w = fs_in_w[i].to(torch.float32)  # [C]
                        logit = logit + (x_last * w).sum(dim=-1) / math.sqrt(C)
                    if fs_in_b is not None:
                        logit = logit + fs_in_b[i].to(torch.float32)
                    g = torch.sigmoid(logit).view(B, 1, 1, 1) * g
                elif fs_alpha is not None:
                    # Scalar gate per layer, keep in [0,1] for stability.
                    # Keep as tensor to allow post-training the gate parameters.
                    g = torch.sigmoid(fs_alpha[i]).to(torch.float32) * g
                if fs_alpha_head is not None:
                    # Per-head gate (tiny additional params) to stabilize long-context post-training.
                    # Shape: [L,H] or [H] per layer; expected [L,H].
                    gh = torch.sigmoid(fs_alpha_head[i]).to(torch.float32)  # [H]
                    g = g * gh.view(1, -1, 1, 1)

                fs_base = prev_sT.detach() if fs_detach else prev_sT
                if fs_norm:
                    # Normalize per head (over the state matrix dims) so gating is length/scale-robust.
                    denom = fs_base.square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp(min=1e-6)
                    fs_base = fs_base / denom

                fs = fs_base * g
                if fs_clip and float(fs_clip) > 0:
                    c = float(fs_clip)
                    fs = fs.clamp(min=-c, max=c)
                # If an explicit seed_states is provided, treat FS as an additive residual.
                # This keeps the mechanism composable (e.g., prompt-prefill seeding + FS).
                s0 = fs if s0 is None else (s0 + fs)

            x, v_first, sT = block(x, v_first, s0, return_state=need_state)
            if need_state:
                assert sT is not None
                if return_states:
                    states.append(sT)
                if future_seed:
                    prev_sT = sT

        x = self.ln_out(x.float()).to(torch.bfloat16)

        return x, (states if return_states else None)
