#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sudoku_rwkv_official import ensure_snapshot


@dataclass
class FutureSeedConfig:
    enabled: bool = False
    layer_start: int = 1
    seed_scale: float = 1.0
    fs_alpha: Sequence[float] | None = None
    fs_alpha_head: Sequence[Sequence[float]] | None = None
    fs_norm: bool = False
    fs_clip: float = 0.0
    residual: bool = True


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_official_rwkv_module(snapshot_dir: str | Path | None = None):
    root: Path
    if snapshot_dir is not None:
        candidate = Path(snapshot_dir).expanduser()
        if (candidate / "rwkv_model.py").exists():
            root = candidate
        else:
            root = ensure_snapshot(root=candidate, include_checkpoint=False, verbose=False)
            root = Path(root)
    else:
        root = ensure_snapshot(root=None, include_checkpoint=False, verbose=False)
        root = Path(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module_name = f"sudoku_rwkv_official_{root.name}"
    if module_name in sys.modules:
        return sys.modules[module_name], root
    mod = _load_module(module_name, root / "rwkv_model.py")
    return mod, root


def build_future_seed_class(snapshot_dir: str | Path | None = None):
    official, root = load_official_rwkv_module(snapshot_dir)
    F = official.F
    torch = official.torch
    mm8_one = official.mm8_one
    mm8_seq = official.mm8_seq

    class RWKVFutureSeed(official.RWKV):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.future_seed_cfg = FutureSeedConfig(enabled=False)

        def set_future_seed(self, cfg: FutureSeedConfig | None) -> None:
            self.future_seed_cfg = cfg or FutureSeedConfig(enabled=False)

        def _init_state_v6(self):
            args = self.args
            w = self.w
            state = [None] * args.n_layer * 3
            for i in range(args.n_layer):
                dd = self.strategy[i]
                dev = dd.device
                atype = dd.atype
                state[i * 3 + 0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
                if args.time_state:
                    state[i * 3 + 1] = w[f"blocks.{i}.att.time_state"].transpose(1, 2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()
                else:
                    state[i * 3 + 1] = torch.zeros((args.n_head, args.n_att // args.n_head, args.n_att // args.n_head), dtype=torch.float, requires_grad=False, device=dev).contiguous()
                state[i * 3 + 2] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
            return state

        def _layer_gate(self, layer_idx: int, device, dtype, n_head: int):
            cfg = self.future_seed_cfg
            gate = torch.tensor(float(cfg.seed_scale), device=device, dtype=dtype)
            if cfg.fs_alpha is not None:
                alpha = cfg.fs_alpha[layer_idx] if layer_idx < len(cfg.fs_alpha) else cfg.fs_alpha[-1]
                gate = gate * torch.sigmoid(torch.tensor(float(alpha), device=device, dtype=dtype))
            if cfg.fs_alpha_head is not None:
                raw = cfg.fs_alpha_head[layer_idx] if layer_idx < len(cfg.fs_alpha_head) else cfg.fs_alpha_head[-1]
                head = torch.as_tensor(list(raw), device=device, dtype=dtype)
                if head.numel() != n_head:
                    raise ValueError(f"fs_alpha_head for layer {layer_idx} has {head.numel()} heads, expected {n_head}")
                gate = gate.view(1, 1, 1) * torch.sigmoid(head).view(n_head, 1, 1)
            return gate

        def _inject_seed(self, current_state, prev_state, layer_idx: int):
            cfg = self.future_seed_cfg
            if (not cfg.enabled) or prev_state is None or layer_idx < int(cfg.layer_start):
                return current_state
            seed = prev_state.to(device=current_state.device, dtype=current_state.dtype)
            if cfg.fs_norm:
                denom = seed.square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp(min=1e-6)
                seed = seed / denom
            gate = self._layer_gate(layer_idx, current_state.device, current_state.dtype, current_state.shape[0])
            seeded = seed * gate
            if cfg.fs_clip and float(cfg.fs_clip) > 0:
                c = float(cfg.fs_clip)
                seeded = seeded.clamp(min=-c, max=c)
            return current_state + seeded if cfg.residual else seeded

        def forward(self, tokens, state, full_output=False):
            cfg = self.future_seed_cfg
            if (not cfg.enabled) or self.version != 6.0:
                return super().forward(tokens, state, full_output=full_output)

            with torch.no_grad():
                w = self.w
                args = self.args
                if state is None:
                    state = self._init_state_v6()

                seq_mode = len(tokens) > 1
                x = w["emb.weight"][tokens if seq_mode else tokens[0]]
                prev_att_state = None

                for i in range(args.n_layer):
                    bbb = f"blocks.{i}."
                    att = f"blocks.{i}.att."
                    ffn = f"blocks.{i}.ffn."
                    dd = self.strategy[i]
                    dev = dd.device
                    atype = dd.atype
                    wtype = dd.wtype
                    ATT = self.att_seq_v6_0 if seq_mode else self.att_one_v6_0
                    FFN = self.ffn_seq_v6 if seq_mode else self.ffn_one_v6

                    x = x.to(dtype=atype, device=dev)
                    state[i * 3 + 1] = self._inject_seed(state[i * 3 + 1], prev_att_state, i)

                    kw = w[f"{att}key.weight"]
                    vw = w[f"{att}value.weight"]
                    rw = w[f"{att}receptance.weight"]
                    ow = w[f"{att}output.weight"]
                    if dd.stream:
                        kw = kw.to(device=dev, non_blocking=True)
                        vw = vw.to(device=dev, non_blocking=True)
                        rw = rw.to(device=dev, non_blocking=True)
                        ow = ow.to(device=dev, non_blocking=True)
                    kmx = w[f"{att}key.weight_mx"] if wtype == torch.uint8 else x
                    krx = w[f"{att}key.weight_rx"] if wtype == torch.uint8 else x
                    kmy = w[f"{att}key.weight_my"] if wtype == torch.uint8 else x
                    kry = w[f"{att}key.weight_ry"] if wtype == torch.uint8 else x
                    vmx = w[f"{att}value.weight_mx"] if wtype == torch.uint8 else x
                    vrx = w[f"{att}value.weight_rx"] if wtype == torch.uint8 else x
                    vmy = w[f"{att}value.weight_my"] if wtype == torch.uint8 else x
                    vry = w[f"{att}value.weight_ry"] if wtype == torch.uint8 else x
                    rmx = w[f"{att}receptance.weight_mx"] if wtype == torch.uint8 else x
                    rrx = w[f"{att}receptance.weight_rx"] if wtype == torch.uint8 else x
                    rmy = w[f"{att}receptance.weight_my"] if wtype == torch.uint8 else x
                    rry = w[f"{att}receptance.weight_ry"] if wtype == torch.uint8 else x
                    omx = w[f"{att}output.weight_mx"] if wtype == torch.uint8 else x
                    orx = w[f"{att}output.weight_rx"] if wtype == torch.uint8 else x
                    omy = w[f"{att}output.weight_my"] if wtype == torch.uint8 else x
                    ory = w[f"{att}output.weight_ry"] if wtype == torch.uint8 else x
                    gw = w[f"{att}gate.weight"]
                    if dd.stream:
                        gw = gw.to(device=dev, non_blocking=True)
                    gmx = w[f"{att}gate.weight_mx"] if wtype == torch.uint8 else x
                    grx = w[f"{att}gate.weight_rx"] if wtype == torch.uint8 else x
                    gmy = w[f"{att}gate.weight_my"] if wtype == torch.uint8 else x
                    gry = w[f"{att}gate.weight_ry"] if wtype == torch.uint8 else x

                    x, state[i * 3 + 0], state[i * 3 + 1] = ATT(
                        x,
                        state[i * 3 + 0],
                        state[i * 3 + 1],
                        w[f"{bbb}ln1.weight"], w[f"{bbb}ln1.bias"],
                        w[f"{att}ln_x.weight"], w[f"{att}ln_x.bias"],
                        w[f"{att}time_maa_x"], w[f"{att}time_maa_w"], w[f"{att}time_maa_k"], w[f"{att}time_maa_v"], w[f"{att}time_maa_r"], w[f"{att}time_maa_g"],
                        w[f"{att}time_maa_w1"], w[f"{att}time_maa_w2"], w[f"{att}time_decay_w1"], w[f"{att}time_decay_w2"],
                        w[f"{att}time_decay"], w[f"{att}time_first"],
                        kw, vw, rw, gw, ow,
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx, rrx, rmy, rry,
                        gmx, grx, gmy, gry,
                        omx, orx, omy, ory,
                    )
                    prev_att_state = state[i * 3 + 1]
                    if dd.stream:
                        del kw, vw, rw, ow, gw

                    kw = w[f"{ffn}key.weight"]
                    vw = w[f"{ffn}value.weight"]
                    rw = w[f"{ffn}receptance.weight"]
                    if dd.stream:
                        kw = kw.to(device=dev, non_blocking=True)
                        vw = vw.to(device=dev, non_blocking=True)
                        rw = rw.to(device=dev, non_blocking=True)
                    kmx = w[f"{ffn}key.weight_mx"] if wtype == torch.uint8 else x
                    krx = w[f"{ffn}key.weight_rx"] if wtype == torch.uint8 else x
                    kmy = w[f"{ffn}key.weight_my"] if wtype == torch.uint8 else x
                    kry = w[f"{ffn}key.weight_ry"] if wtype == torch.uint8 else x
                    vmx = w[f"{ffn}value.weight_mx"] if wtype == torch.uint8 else x
                    vrx = w[f"{ffn}value.weight_rx"] if wtype == torch.uint8 else x
                    vmy = w[f"{ffn}value.weight_my"] if wtype == torch.uint8 else x
                    vry = w[f"{ffn}value.weight_ry"] if wtype == torch.uint8 else x
                    rmx = w[f"{ffn}receptance.weight_mx"] if wtype == torch.uint8 else x
                    rrx = w[f"{ffn}receptance.weight_rx"] if wtype == torch.uint8 else x
                    rmy = w[f"{ffn}receptance.weight_my"] if wtype == torch.uint8 else x
                    rry = w[f"{ffn}receptance.weight_ry"] if wtype == torch.uint8 else x
                    offset = i * 3 + 2
                    x, state[offset] = FFN(
                        x,
                        state[offset],
                        w[f"{bbb}ln2.weight"], w[f"{bbb}ln2.bias"],
                        w[f"{ffn}time_maa_k"], w[f"{ffn}time_maa_r"],
                        kw, vw, rw,
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx, rrx, rmy, rry,
                    )
                    if dd.stream:
                        del kw, vw, rw

                    if self.RESCALE_LAYER > 0 and (i + 1) % self.RESCALE_LAYER == 0:
                        x = x / 2

                dd = self.strategy[args.n_layer]
                x = x[-1, :] if (seq_mode and (not full_output)) else x
                x = x.to(dtype=dd.atype, device=dd.device)
                x = F.layer_norm(x, (args.n_embd,), weight=w["ln_out.weight"], bias=w["ln_out.bias"])
                if w["head.weight"].dtype != torch.uint8:
                    x = x @ w["head.weight"]
                else:
                    if seq_mode and full_output:
                        x = mm8_seq(x, w["head.weight"], w["head.weight_mx"], w["head.weight_rx"], w["head.weight_my"], w["head.weight_ry"])
                    else:
                        x = mm8_one(x, w["head.weight"], w["head.weight_mx"], w["head.weight_rx"], w["head.weight_my"], w["head.weight_ry"])
                return x.float(), state

    return RWKVFutureSeed, official, root


def parse_float_list(text: str | None) -> list[float] | None:
    if text is None or text.strip() == "":
        return None
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_head_matrix(text: str | None) -> list[list[float]] | None:
    if text is None or text.strip() == "":
        return None
    rows = []
    for row in text.split(";"):
        row = row.strip()
        if not row:
            continue
        rows.append([float(x.strip()) for x in row.split(",") if x.strip()])
    return rows or None
