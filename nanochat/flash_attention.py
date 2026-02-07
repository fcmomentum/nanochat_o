"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on non-Hopper GPUs (including Blackwell), MPS, and CPU.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import os
import torch
import torch.nn.functional as F

try:
    # Newer PyTorch exposes flex_attention as a module with create_block_mask inside.
    from torch.nn.attention import flex_attention as _flex_attention
    try:
        from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask
    except Exception:
        _create_block_mask = getattr(_flex_attention, "create_block_mask", None)
    if callable(_flex_attention):
        _flex_attention_fn = _flex_attention
    else:
        _flex_attention_fn = getattr(_flex_attention, "flex_attention", None)
    HAS_FLEX_ATTENTION = _create_block_mask is not None and _flex_attention_fn is not None
except Exception:
    _flex_attention = None
    _create_block_mask = None
    HAS_FLEX_ATTENTION = False
    _flex_attention_fn = None


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _use_fa3():
    """Determine whether to use FA3 based on availability and override."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    return HAS_FA3  # auto


# =============================================================================
# SDPA helpers
# =============================================================================
_flex_block_mask_cache = {}
_flex_debug_enabled = bool(int(os.environ.get("NANOCHAT_FLEX_DEBUG", "0")))
_flex_debug_printed = False
_flex_debug_reasons = set()

def _flex_debug_log(reason):
    if not _flex_debug_enabled or reason in _flex_debug_reasons:
        return
    print(f"[flex_attention] skipped: {reason}")
    _flex_debug_reasons.add(reason)

def _get_flex_block_mask(q_len, kv_len, window, device, batch_size, num_heads):
    key = (q_len, kv_len, window, device.type, device.index, batch_size, num_heads)
    block_mask = _flex_block_mask_cache.get(key)
    if block_mask is not None:
        return block_mask

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window)

    try:
        block_mask = _create_block_mask(
            mask_mod=mask_mod,
            B=batch_size,
            H=num_heads,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )
    except TypeError:
        # Fallback for older/newer signatures
        block_mask = _create_block_mask(mask_mod, batch_size, num_heads, q_len, kv_len, device=device)

    _flex_block_mask_cache[key] = block_mask
    return block_mask


def _get_flex_block_mask_split(q_len, kv_len, window, device, batch_size, num_heads, n_global_head):
    key = (q_len, kv_len, window, device.type, device.index, batch_size, num_heads, n_global_head)
    block_mask = _flex_block_mask_cache.get(key)
    if block_mask is not None:
        return block_mask

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        if h < n_global_head:
            return causal
        return causal & ((q_idx - kv_idx) <= window)

    try:
        block_mask = _create_block_mask(
            mask_mod=mask_mod,
            B=batch_size,
            H=num_heads,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )
    except TypeError:
        block_mask = _create_block_mask(mask_mod, batch_size, num_heads, q_len, kv_len, device=device)

    _flex_block_mask_cache[key] = block_mask
    return block_mask


def _flex_attention_sliding_window(q, k, v, window_size, enable_gqa):
    global _flex_debug_printed
    if not HAS_FLEX_ATTENTION:
        _flex_debug_log("HAS_FLEX_ATTENTION is False (import failed)")
        return None
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]
    if Tq != Tk:
        _flex_debug_log(f"Tq != Tk (Tq={Tq}, Tk={Tk})")
        return None
    if window < 0 or window >= Tq:
        _flex_debug_log(f"window not in [0, Tq) (window={window}, Tq={Tq})")
        return None
    block_mask = _get_flex_block_mask(Tq, Tk, window, q.device, q.size(0), q.size(1))
    try:
        y = _flex_attention_fn(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)
    except TypeError:
        try:
            y = _flex_attention_fn(q, k, v, block_mask)
        except Exception:
            if _flex_debug_enabled:
                import traceback
                traceback.print_exc()
            _flex_debug_log("flex_attention call failed (exception)")
            return None
    if _flex_debug_enabled and not _flex_debug_printed:
        print(f"[flex_attention] enabled for sliding window: T={Tq}, window={window}")
        _flex_debug_printed = True
    return y


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Sliding window (training / full-length) via FlexAttention if available
    if window >= 0 and window < Tq and Tq == Tk:
        y_flex = _flex_attention_sliding_window(q, k, v, window_size, enable_gqa)
        if y_flex is not None:
            return y_flex

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# FlexAttention split-head helper (single-call, per-head mask)
# =============================================================================
def _flex_attention_split_heads(q, k, v, window_size, n_global_head):
    if not HAS_FLEX_ATTENTION:
        _flex_debug_log("HAS_FLEX_ATTENTION is False (import failed)")
        return None
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]
    if Tq != Tk:
        _flex_debug_log(f"split: Tq != Tk (Tq={Tq}, Tk={Tk})")
        return None
    if window < 0 or window >= Tq:
        _flex_debug_log(f"split: window not in [0, Tq) (window={window}, Tq={Tq})")
        return None
    if not (0 < n_global_head < q.size(1)):
        _flex_debug_log("split: n_global_head not in (0, num_heads)")
        return None
    block_mask = _get_flex_block_mask_split(Tq, Tk, window, q.device, q.size(0), q.size(1), n_global_head)
    try:
        return _flex_attention_fn(q, k, v, block_mask=block_mask, enable_gqa=False)
    except TypeError:
        try:
            return _flex_attention_fn(q, k, v, block_mask)
        except Exception:
            if _flex_debug_enabled:
                import traceback
                traceback.print_exc()
            _flex_debug_log("split: flex_attention call failed (exception)")
            return None

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if _use_fa3():
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_func_split_heads(q, k, v, n_global_head, window_size=(-1, -1)):
    """
    FlexAttention single-call path for split-head sliding window (training only).
    Returns None if FlexAttention is unavailable or conditions aren't met.

    Args:
        q, k, v: (B, T, H, D) with H == H_kv (no GQA)
        n_global_head: number of heads with full causal context
        window_size: (left, right) sliding window. right ignored (causal only)
    """
    if not HAS_FLEX_ATTENTION:
        return None
    # Transpose to (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    y_sdpa = _flex_attention_split_heads(q_sdpa, k_sdpa, v_sdpa, window_size, n_global_head)
    if y_sdpa is None:
        return None
    return y_sdpa.transpose(1, 2)

def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if _use_fa3():
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
    flash_attn_func_split_heads=flash_attn_func_split_heads,
)
