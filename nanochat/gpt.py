"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Number of query heads to treat as global (full causal attention). Remaining heads use sliding window.
    # Set to 0 to disable global heads (all local), set to n_head for all global.
    n_global_head: int = 0
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Predictive subtraction layers for mixed-head attention.
    # Options: "all", "none", or comma-separated 0-based indices/ranges (e.g. "4-9,11").
    pred_sub_layers: str = "all"
    # If True, disable predictive subtraction on full-context (L) layers.
    pred_sub_skip_full_layers: bool = False
    # Auxiliary loss weight for minimizing predictive subtraction local error.
    pred_sub_error_weight: float = 0.0
    # Optional DINO-style auxiliary loss for split-head layers.
    # dino_layer: 0-based layer index, -1 disables.
    # dino_delta: predict teacher features at t + delta from student features at t.
    dino_layer: int = -1
    dino_delta: int = 1
    dino_weight: float = 0.0
    dino_student_temp: float = 0.1
    dino_teacher_temp: float = 0.04
    dino_mask_ratio: float = 0.0
    # If True, fuse masked DINO local features back into the main local stream.
    dino_fuse_main: bool = False
    # Initial scalar gate value for DINO fusion branch.
    dino_fuse_init: float = 0.0
    # Optional student bottleneck width before projecting to teacher space (0 disables).
    dino_bottleneck_dim: int = 0


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, enable_pred_sub, enable_dino):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_global_head = config.n_global_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        assert 0 <= self.n_global_head <= self.n_head
        if 0 < self.n_global_head < self.n_head:
            q_per_kv = self.n_head // self.n_kv_head
            assert self.n_global_head % q_per_kv == 0, "n_global_head must align to GQA groups"
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.pred_sub_proj = None
        self.pred_sub_scale = None
        self.pred_sub_gate = None
        self.dino_local_proj = None
        self.dino_teacher_dim = self.n_global_head * self.head_dim
        self.dino_mask_ratio = float(config.dino_mask_ratio)
        self.dino_fuse_main = bool(config.dino_fuse_main)
        self.dino_fuse_proj = None
        self.dino_fuse_gate = None
        self.dino_bottleneck_proj = None
        if enable_pred_sub and 0 < self.n_global_head < self.n_head:
            n_local_head = self.n_head - self.n_global_head
            local_dim = n_local_head * self.head_dim
            self.pred_sub_proj = nn.Linear(
                self.n_global_head * self.head_dim,
                local_dim,
                bias=False,
            )
            # Predictive gate: confidence for how much local prediction error to correct per local head.
            self.pred_sub_gate = nn.Linear(local_dim, n_local_head, bias=False)
            # Learnable scale for predictive subtraction (init small for stability).
            self.pred_sub_scale = nn.Parameter(torch.tensor(0.1))
        if enable_dino and 0 < self.n_global_head < self.n_head:
            n_local_head = self.n_head - self.n_global_head
            local_dim = n_local_head * self.head_dim
            if int(config.dino_bottleneck_dim) > 0:
                self.dino_bottleneck_proj = nn.Linear(local_dim, int(config.dino_bottleneck_dim), bias=False)
            self._ensure_dino_local_proj()
            if self.dino_fuse_main:
                self.dino_fuse_proj = nn.Linear(local_dim, local_dim, bias=False)
                self.dino_fuse_gate = nn.Parameter(torch.tensor(float(config.dino_fuse_init)))

    def _ensure_dino_local_proj(self):
        if self.dino_local_proj is not None:
            return
        if not (0 < self.n_global_head < self.n_head):
            return
        n_local_head = self.n_head - self.n_global_head
        local_dim = n_local_head * self.head_dim
        in_dim = self.dino_bottleneck_proj.out_features if self.dino_bottleneck_proj is not None else local_dim
        self.dino_local_proj = nn.Linear(in_dim, self.n_global_head * self.head_dim, bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, capture_dino=False, capture_pred_sub_error=False):
        B, T, C = x.size()
        dino_pair = None
        pred_sub_error_loss = None

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head.
        # Gate presence is static per layer, which helps torch.compile avoid None-based recompiles.
        if self.ve_gate is not None:
            assert ve is not None, "ve tensor is required when ve_gate is enabled"
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if self.n_global_head == 0:
            # All local heads
            if kv_cache is None:
                y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            else:
                k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
                y = flash_attn.flash_attn_with_kvcache(
                    q, k_cache, v_cache,
                    k=k, v=v,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True,
                    window_size=window_size,
                )
                if self.layer_idx == kv_cache.n_layers - 1:
                    kv_cache.advance(T)
        elif self.n_global_head == self.n_head:
            # All global heads (full causal context)
            global_window = (-1, 0)
            if kv_cache is None:
                y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=global_window)
            else:
                k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
                y = flash_attn.flash_attn_with_kvcache(
                    q, k_cache, v_cache,
                    k=k, v=v,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True,
                    window_size=global_window,
                )
                if self.layer_idx == kv_cache.n_layers - 1:
                    kv_cache.advance(T)
        else:
            # Split heads: global (full context) + local (sliding window)
            q_per_kv = self.n_head // self.n_kv_head
            n_global_kv = self.n_global_head // q_per_kv
            global_window = (-1, 0)
            q_global, q_local = q[:, :, :self.n_global_head, :], q[:, :, self.n_global_head:, :]
            k_global, k_local = k[:, :, :n_global_kv, :], k[:, :, n_global_kv:, :]
            v_global, v_local = v[:, :, :n_global_kv, :], v[:, :, n_global_kv:, :]
            if kv_cache is None:
                y = None
                if self.n_kv_head == self.n_head:
                    # Attempt single-call FlexAttention with per-head mask.
                    y = flash_attn.flash_attn_func_split_heads(
                        q, k, v, self.n_global_head, window_size=window_size
                    )
                if y is None:
                    y_global = flash_attn.flash_attn_func(
                        q_global, k_global, v_global,
                        causal=True, window_size=global_window,
                    )
                    y_local = flash_attn.flash_attn_func(
                        q_local, k_local, v_local,
                        causal=True, window_size=window_size,
                    )
                else:
                    y_global = y[:, :, :self.n_global_head, :]
                    y_local = y[:, :, self.n_global_head:, :]
            else:
                k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
                k_cache_global, k_cache_local = k_cache[:, :, :n_global_kv, :], k_cache[:, :, n_global_kv:, :]
                v_cache_global, v_cache_local = v_cache[:, :, :n_global_kv, :], v_cache[:, :, n_global_kv:, :]
                k_global_new, k_local_new = k[:, :, :n_global_kv, :], k[:, :, n_global_kv:, :]
                v_global_new, v_local_new = v[:, :, :n_global_kv, :], v[:, :, n_global_kv:, :]
                y_global = flash_attn.flash_attn_with_kvcache(
                    q_global, k_cache_global, v_cache_global,
                    k=k_global_new, v=v_global_new,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True,
                    window_size=global_window,
                )
                y_local = flash_attn.flash_attn_with_kvcache(
                    q_local, k_cache_local, v_cache_local,
                    k=k_local_new, v=v_local_new,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True,
                    window_size=window_size,
                )
                if self.layer_idx == kv_cache.n_layers - 1:
                    kv_cache.advance(T)
            if self.pred_sub_proj is not None:
                # Predict local features from global heads, then apply gated error correction in head space.
                y_global_flat = y_global.contiguous().view(B, T, -1)
                y_pred_local = self.pred_sub_proj(y_global_flat).view(
                    B, T, self.n_head - self.n_global_head, self.head_dim
                )
                error_local = y_local - y_pred_local
                if capture_pred_sub_error:
                    pred_sub_error_loss = error_local.float().pow(2).mean()
                gate = torch.sigmoid(self.pred_sub_gate(error_local.contiguous().view(B, T, -1))).unsqueeze(-1)
                y_local = y_local - self.pred_sub_scale * gate * error_local
            if capture_dino:
                local_for_dino = y_local
                if self.training and self.dino_mask_ratio > 0.0:
                    keep_prob = 1.0 - self.dino_mask_ratio
                    n_local_head = self.n_head - self.n_global_head
                    # Mask whole local heads in the aux branch so the student must infer missing structure.
                    head_mask = (torch.rand(B, T, n_local_head, 1, device=y_local.device) < keep_prob).to(y_local.dtype)
                    local_for_dino = local_for_dino * head_mask / max(keep_prob, 1e-6)
                local_flat = local_for_dino.contiguous().view(B, T, -1)
                if self.dino_fuse_proj is not None and self.dino_fuse_gate is not None:
                    # Gated residual from masked DINO branch into the main local stream.
                    fused = self.dino_fuse_proj(local_flat).view(B, T, self.n_head - self.n_global_head, self.head_dim)
                    y_local = y_local + torch.sigmoid(self.dino_fuse_gate) * fused
                global_flat = y_global.contiguous().view(B, T, -1)
                local_for_student = local_flat
                if self.dino_bottleneck_proj is not None:
                    local_for_student = self.dino_bottleneck_proj(local_for_student)
                if self.dino_local_proj is not None:
                    student_feat = self.dino_local_proj(local_for_student)
                else:
                    # Robust fallback: deterministic truncate/pad projection so DINO can still run.
                    # This should be rare; keep training alive and surface via dino_active metrics.
                    target_dim = global_flat.size(-1)
                    if local_for_student.size(-1) >= target_dim:
                        student_feat = local_for_student[..., :target_dim]
                    else:
                        pad = target_dim - local_for_student.size(-1)
                        student_feat = F.pad(local_for_student, (0, pad))
                dino_pair = {
                    "student": student_feat,
                    "teacher": global_flat,
                }
            y = torch.cat([y_global, y_local], dim=2)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        if capture_dino and self.n_global_head < self.n_head:
            if capture_pred_sub_error:
                return y, dino_pair, pred_sub_error_loss
            return y, dino_pair
        if capture_pred_sub_error:
            return y, pred_sub_error_loss
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx, enable_pred_sub, enable_dino):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx, enable_pred_sub, enable_dino)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, capture_dino=False, capture_pred_sub_error=False):
        attn_out = self.attn(
            norm(x),
            ve,
            cos_sin,
            window_size,
            kv_cache,
            capture_dino=capture_dino,
            capture_pred_sub_error=capture_pred_sub_error,
        )
        dino_pair = None
        pred_sub_error_loss = None
        if capture_dino and capture_pred_sub_error:
            attn_out, dino_pair, pred_sub_error_loss = attn_out
        elif capture_dino:
            attn_out, dino_pair = attn_out
        elif capture_pred_sub_error:
            attn_out, pred_sub_error_loss = attn_out
        x = x + attn_out
        x = x + self.mlp(norm(x))
        if capture_dino and capture_pred_sub_error:
            return x, dino_pair, pred_sub_error_loss
        if capture_dino:
            return x, dino_pair
        if capture_pred_sub_error:
            return x, pred_sub_error_loss
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        schedule = ", ".join(f"L{i}:{w[0]}" for i, w in enumerate(self.window_sizes))
        print0(f"Attention window schedule (left context): {schedule}")
        self.pred_sub_layers = self._compute_pred_sub_layers(config)
        if self.pred_sub_layers:
            pred_sub_str = ", ".join(str(i) for i in sorted(self.pred_sub_layers))
        else:
            pred_sub_str = "none"
        print0(f"Predictive subtraction layers: {pred_sub_str}")
        self.dino_layer = int(config.dino_layer)
        self.dino_enabled = config.dino_weight > 0.0 and self.dino_layer >= 0
        self.pred_sub_error_enabled = float(config.pred_sub_error_weight) > 0.0
        if self.dino_enabled:
            assert 0 <= self.dino_layer < config.n_layer, f"dino_layer out of bounds: {self.dino_layer}"
            assert 0 < config.n_global_head < config.n_head, (
                "DINO aux requires split heads: set global_head_pct so 0 < n_global_head < n_head"
            )
            assert config.dino_delta > 0, "dino_delta must be > 0"
            assert 0.0 <= config.dino_mask_ratio < 1.0, "dino_mask_ratio must be in [0, 1)"
            assert 0.0 <= config.dino_fuse_init <= 1.0, "dino_fuse_init must be in [0, 1]"
            assert int(config.dino_bottleneck_dim) >= 0, "dino_bottleneck_dim must be >= 0"
            print0(
                f"DINO aux enabled at layer {self.dino_layer} "
                f"(delta={config.dino_delta}, weight={config.dino_weight}, mask={config.dino_mask_ratio}, "
                f"fuse={config.dino_fuse_main}, bottleneck={config.dino_bottleneck_dim})"
            )
        else:
            print0("DINO aux disabled")
        if self.pred_sub_error_enabled:
            print0(f"Predictive subtraction error aux enabled (weight={config.pred_sub_error_weight})")
        else:
            print0("Predictive subtraction error aux disabled")
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([
                Block(
                    config,
                    layer_idx,
                    layer_idx in self.pred_sub_layers,
                    layer_idx == self.dino_layer and self.dino_enabled,
                )
                for layer_idx in range(config.n_layer)
            ]),
        })
        if self.dino_enabled:
            dino_attn = self.transformer.h[self.dino_layer].attn
            dino_attn._ensure_dino_local_proj()
            assert dino_attn.dino_local_proj is not None, "Failed to initialize DINO local projection"
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)
        # Runtime DINO weight can be updated during training (e.g., warmup) without recompiling.
        self.register_buffer("dino_weight_buffer", torch.tensor(float(config.dino_weight)), persistent=False)
        self.register_buffer("pred_sub_error_weight_buffer", torch.tensor(float(config.pred_sub_error_weight)), persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if block.attn.pred_sub_proj is not None:
                torch.nn.init.uniform_(block.attn.pred_sub_proj.weight, -s, s)
            if block.attn.pred_sub_gate is not None:
                torch.nn.init.zeros_(block.attn.pred_sub_gate.weight)
            if block.attn.pred_sub_scale is not None:
                block.attn.pred_sub_scale.fill_(0.1)
            if block.attn.dino_bottleneck_proj is not None:
                torch.nn.init.uniform_(block.attn.dino_bottleneck_proj.weight, -s, s)
            if block.attn.dino_local_proj is not None:
                torch.nn.init.uniform_(block.attn.dino_local_proj.weight, -s, s)
            if block.attn.dino_fuse_proj is not None:
                torch.nn.init.zeros_(block.attn.dino_fuse_proj.weight)  # neutral residual at init
            if block.attn.dino_fuse_gate is not None:
                block.attn.dino_fuse_gate.fill_(float(self.config.dino_fuse_init))
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.dino_weight_buffer.fill_(float(self.config.dino_weight))
        self.pred_sub_error_weight_buffer.fill_(float(self.config.pred_sub_error_weight))

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def _compute_pred_sub_layers(self, config):
        """
        Parse predictive subtraction layer spec into a set of 0-based layer indices.

        Supported specs:
        - "all": all layers
        - "none": no layers
        - Comma-separated indices/ranges, e.g. "4-9,11"
        """
        spec = str(config.pred_sub_layers).strip().lower()
        if spec in {"all", "*"}:
            layers = set(range(config.n_layer))
        elif spec in {"none", "off", ""}:
            layers = set()
        else:
            layers = set()
            for part in spec.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    lo_s, hi_s = part.split("-", 1)
                    lo, hi = int(lo_s), int(hi_s)
                    assert lo <= hi, f"Invalid pred_sub_layers range '{part}': start must be <= end"
                    for idx in range(lo, hi + 1):
                        assert 0 <= idx < config.n_layer, f"pred_sub_layers index {idx} out of bounds for n_layer={config.n_layer}"
                        layers.add(idx)
                else:
                    idx = int(part)
                    assert 0 <= idx < config.n_layer, f"pred_sub_layers index {idx} out of bounds for n_layer={config.n_layer}"
                    layers.add(idx)

        if config.pred_sub_skip_full_layers:
            # Full-context layers have long-window attention (left context >= sequence length).
            layers = {i for i in layers if self.window_sizes[i][0] < config.sequence_len}
        return layers

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        n_global = self.config.n_global_head
        n_local = h - n_global
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            local_seq = t if window < 0 else min(window, t)
            if n_global == 0:
                attn_flops += 12 * h * q * local_seq
            elif n_global == h:
                attn_flops += 12 * h * q * t
            else:
                attn_flops += 12 * q * (n_global * t + n_local * local_seq)
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        block_params = list(self.transformer.h.parameters())
        transformer_matrices = sum(p.numel() for p in block_params if p.ndim >= 2)
        transformer_scalars = sum(p.numel() for p in block_params if p.ndim < 2)
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + transformer_scalars
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        block_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in block_params if p.ndim >= 2]
        block_scalar_params = [p for p in block_params if p.ndim < 2]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(block_scalar_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=block_scalar_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_loss_breakdown=False):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx) # embed current token
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        dino_pair = None
        pred_sub_error_losses = []
        zero_ve = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx)
            else:
                # Keep ve argument tensor-typed across layers to reduce torch.compile recompiles.
                if zero_ve is None:
                    kv_dim = self.config.n_kv_head * (self.config.n_embd // self.config.n_head)
                    zero_ve = torch.zeros(B, T, kv_dim, dtype=x.dtype, device=x.device)
                ve = zero_ve
            capture_dino = self.dino_enabled and targets is not None and kv_cache is None and i == self.dino_layer
            capture_pred_sub_error = self.pred_sub_error_enabled and targets is not None and kv_cache is None
            if capture_dino and capture_pred_sub_error:
                x, dino_pair, pred_sub_error_loss_i = block(
                    x, ve, cos_sin, self.window_sizes[i], kv_cache, capture_dino=True, capture_pred_sub_error=True
                )
                if pred_sub_error_loss_i is not None:
                    pred_sub_error_losses.append(pred_sub_error_loss_i)
            elif capture_dino:
                x, dino_pair = block(
                    x, ve, cos_sin, self.window_sizes[i], kv_cache, capture_dino=True, capture_pred_sub_error=False
                )
            elif capture_pred_sub_error:
                x, pred_sub_error_loss_i = block(
                    x, ve, cos_sin, self.window_sizes[i], kv_cache, capture_dino=False, capture_pred_sub_error=True
                )
                if pred_sub_error_loss_i is not None:
                    pred_sub_error_losses.append(pred_sub_error_loss_i)
            else:
                x = block(
                    x, ve, cos_sin, self.window_sizes[i], kv_cache, capture_dino=False, capture_pred_sub_error=False
                )
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            dino_loss = None
            pred_sub_error_loss = None
            dino_active = ce_loss.detach().new_zeros(())
            pred_sub_error_active = ce_loss.detach().new_zeros(())
            loss = ce_loss
            if self.dino_enabled and loss_reduction == "mean":
                if dino_pair is not None:
                    dino_loss = self._compute_dino_aux_loss(dino_pair["student"], dino_pair["teacher"], targets)
                    loss = loss + self.dino_weight_buffer * dino_loss
                    dino_active = ce_loss.detach().new_ones(())
            if self.pred_sub_error_enabled and loss_reduction == "mean":
                if pred_sub_error_losses:
                    pred_sub_error_loss = torch.stack(pred_sub_error_losses).mean()
                    loss = loss + self.pred_sub_error_weight_buffer * pred_sub_error_loss
                    pred_sub_error_active = ce_loss.detach().new_ones(())
            if return_loss_breakdown:
                dino_detached = dino_loss.detach() if dino_loss is not None else ce_loss.detach().new_zeros(())
                pred_sub_detached = pred_sub_error_loss.detach() if pred_sub_error_loss is not None else ce_loss.detach().new_zeros(())
                return loss, {
                    "ce_loss": ce_loss.detach(),
                    "dino_aux_loss": dino_detached,
                    "dino_active": dino_active,
                    "dino_weight": self.dino_weight_buffer.detach(),
                    "pred_sub_error_loss": pred_sub_detached,
                    "pred_sub_error_active": pred_sub_error_active,
                    "pred_sub_error_weight": self.pred_sub_error_weight_buffer.detach(),
                }
            return loss
        else:
            # inference: just return the logits directly
            return logits

    def _compute_dino_aux_loss(self, student, teacher, targets=None):
        """
        DINO-style temporal distillation loss:
        student(t) predicts stop-grad teacher(t + delta).
        """
        delta = int(self.config.dino_delta)
        assert delta > 0, "dino_delta must be > 0"
        if student.size(1) <= delta:
            return student.new_zeros(())
        student = student[:, :-delta, :]
        teacher = teacher[:, delta:, :].detach()
        student = F.normalize(student.float(), dim=-1)
        teacher = F.normalize(teacher.float(), dim=-1)
        s_logits = student / max(float(self.config.dino_student_temp), 1e-6)
        t_logits = teacher / max(float(self.config.dino_teacher_temp), 1e-6)
        t_probs = F.softmax(t_logits, dim=-1)
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        token_loss = -(t_probs * s_log_probs).sum(dim=-1)  # (B, T-delta)
        if targets is not None:
            # Respect training mask: exclude positions after end-of-sample (targets == -1).
            # Require both source t and teacher target t+delta to be valid.
            valid = (targets >= 0)
            valid = valid[:, :-delta] & valid[:, delta:]
            if not valid.any():
                return student.new_zeros(())
            token_loss = token_loss.masked_select(valid)
        return token_loss.mean()

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
