"""
TieredKV — 三层 Training-Free KV 压缩

  Tier 1  Attention Sink      强制保留序列前 k_sink 个 token
  Tier 2  Depth-Adaptive      逐层差异化 TreeKV 几何分配（浅层偏均匀、深层偏近端）
  Tier 3  TreeKV              块内 attention top-k + 几何预算分配

注意：各层 KV 序列长度必须一致才能 decode。因此层间自适应体现在
「同一总预算下，不同层采用不同的块预算曲线」，而非各层保留不同 token 数。

创新：Unified Hierarchical KV Budget (UHB)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

# Pythia-70M: 6 layers — 浅层 ratio 高（分配更均匀），深层 ratio 低（更偏近端）
DEFAULT_LAYER_RATIOS = [0.70, 0.65, 0.55, 0.50, 0.40, 0.35]


@dataclass
class TieredKVConfig:
    """三层 KV 压缩配置。"""

    k_sink: int = 4
    k_ratio: float = 0.50
    layer_ratios: Optional[list[float]] = None
    obs_window: int = 32
    local_window: int = 32
    n_levels: int = 4
    query_pool: str = "mean"
    exp_lambda: float = 0.9
    topk_q: int = 8

    use_sink: bool = True
    use_layer_adaptive: bool = True
    use_cross_layer_fusion: bool = False
    use_treekv: bool = True

    def layer_ratio(self, layer_idx: int, n_layers: int) -> float:
        """返回该层的深度自适应系数（高=浅层，低=深层）。"""
        if not self.use_layer_adaptive:
            return self.k_ratio
        ratios = self.layer_ratios
        if ratios is None:
            ratios = _default_ratios_for_layers(n_layers)
        if layer_idx < len(ratios):
            return ratios[layer_idx]
        return ratios[-1]

    def layer_geometry_skew(self, layer_idx: int, n_layers: int) -> float:
        """
        将 layer_ratio 映射为 TreeKV 块预算曲线的 skew。
        浅层 skew 低 → 块预算更均匀 → 保留更多远端 token；
        深层 skew 高 → 块预算更集中近端。
        """
        if not self.use_layer_adaptive:
            return 1.0
        ratio = self.layer_ratio(layer_idx, n_layers)
        r_min, r_max = 0.35, 0.70
        skew_lo, skew_hi = 0.55, 1.75
        t = (r_max - ratio) / max(r_max - r_min, 1e-6)
        return skew_lo + t * (skew_hi - skew_lo)

    def effective_k_sink(self) -> int:
        return self.k_sink if self.use_sink else 0


def _default_ratios_for_layers(n_layers: int) -> list[float]:
    if n_layers == len(DEFAULT_LAYER_RATIOS):
        return list(DEFAULT_LAYER_RATIOS)
    if n_layers == 1:
        return [DEFAULT_LAYER_RATIOS[-1]]
    lo, hi = DEFAULT_LAYER_RATIOS[0], DEFAULT_LAYER_RATIOS[-1]
    return [lo + (hi - lo) * i / (n_layers - 1) for i in range(n_layers)]


def _pool_query_attention(
    obs_attn: torch.Tensor,
    query_pool: str = "mean",
    exp_lambda: float = 0.9,
    topk_q: int = 8,
) -> torch.Tensor:
    """
    obs_attn: [B, H, obs, S_k] -> importance [S_k]
    """
    query_key = obs_attn.mean(dim=1)[0]  # [obs, S_k]
    obs_len = query_key.size(0)
    if query_pool == "mean":
        return query_key.mean(dim=0)
    if query_pool == "exp":
        lam = min(max(exp_lambda, 1e-4), 0.9999)
        exps = torch.arange(obs_len - 1, -1, -1, device=query_key.device)
        weights = lam ** exps
        weights = weights / (weights.sum() + 1e-9)
        return (query_key * weights.unsqueeze(-1)).sum(dim=0)
    if query_pool == "max":
        return query_key.max(dim=0).values
    if query_pool == "topk_mean":
        k = max(1, min(int(topk_q), obs_len))
        return query_key.topk(k, dim=0).values.mean(dim=0)
    raise ValueError(f"Unsupported query_pool: {query_pool}")


def _importance_from_attention(
    attn_w: torch.Tensor,
    obs_window: int,
    query_pool: str = "mean",
    exp_lambda: float = 0.9,
    topk_q: int = 8,
) -> torch.Tensor:
    obs_start = max(0, attn_w.size(2) - obs_window)
    obs_attn = attn_w[:, :, obs_start:, :]
    return _pool_query_attention(
        obs_attn,
        query_pool=query_pool,
        exp_lambda=exp_lambda,
        topk_q=topk_q,
    )


def _block_budgets(n_levels: int, total_budget: int, geometry_skew: float) -> list[int]:
    """几何块预算（未考虑块容量）。"""
    if total_budget <= 0:
        return [0] * n_levels
    weights = [2 ** (i * geometry_skew) for i in range(n_levels)]
    denom = sum(weights)
    raw = [total_budget * w / denom for w in weights]
    budgets = [int(x) for x in raw]
    for i in range(n_levels):
        if budgets[i] < 1 and total_budget >= n_levels:
            budgets[i] = 1
    diff = total_budget - sum(budgets)
    order = sorted(range(n_levels), key=lambda i: raw[i] - budgets[i], reverse=True)
    idx = 0
    while diff > 0:
        budgets[order[idx % n_levels]] += 1
        diff -= 1
        idx += 1
    while diff < 0:
        j = order[idx % n_levels]
        if budgets[j] > 1:
            budgets[j] -= 1
            diff += 1
        idx += 1
    return budgets


def _cap_block_budgets(
    block_budgets: list[int],
    block_caps: list[int],
    total_budget: int,
) -> list[int]:
    """
    将几何块预算重分配至不超过各块容量，且总和 == total_budget。
    保证各层最终选中 token 数一致。
    """
    n = len(block_budgets)
    selected = [min(b, c) for b, c in zip(block_budgets, block_caps)]
    deficit = total_budget - sum(selected)
    if deficit <= 0:
        return selected
    order = sorted(range(n), key=lambda i: block_caps[i] - selected[i], reverse=True)
    idx = 0
    while deficit > 0:
        i = order[idx % n]
        room = block_caps[i] - selected[i]
        if room > 0:
            selected[i] += 1
            deficit -= 1
        idx += 1
        if idx > n * total_budget * 2:
            break
    return selected


def _compress_layer_treekv(
    layer,
    importance: torch.Tensor,
    *,
    k_sink: int,
    k_ratio: float,
    geometry_skew: float,
    local_window: int,
    n_levels: int,
):
    """对单层 KV 做 Sink + TreeKV 压缩（in-place）。"""
    S = layer.keys.size(2)
    if S <= local_window + 1:
        return

    hist_len = S - local_window
    block_start = min(k_sink, hist_len)
    block_len = hist_len - block_start

    selected_parts = []
    if k_sink > 0 and block_start > 0:
        selected_parts.append(torch.arange(0, block_start, device=layer.keys.device))

    if block_len > 0:
        chunk = block_len // n_levels
        if chunk == 0:
            mid_idx = torch.arange(block_start, hist_len, device=layer.keys.device)
            selected_parts.append(mid_idx)
        else:
            total_budget = max(1, int(block_len * k_ratio))
            raw_budgets = _block_budgets(n_levels, total_budget, geometry_skew)
            block_caps = []
            block_ranges = []
            for i in range(n_levels):
                start = block_start + i * chunk
                end = (
                    block_start + (i + 1) * chunk
                    if i < n_levels - 1
                    else hist_len
                )
                block_ranges.append((start, end))
                block_caps.append(end - start)
            block_budgets = _cap_block_budgets(raw_budgets, block_caps, total_budget)
            for i, (start, end) in enumerate(block_ranges):
                k_select = block_budgets[i]
                if k_select <= 0:
                    continue
                block_imp = importance[start:end]
                topk_local = block_imp.topk(k_select).indices + start
                selected_parts.append(topk_local.sort().values)

    local_idx = torch.arange(S - local_window, S, device=layer.keys.device)
    all_idx = torch.cat(selected_parts + [local_idx])
    all_idx = torch.unique(all_idx, sorted=True)

    layer.keys = layer.keys[:, :, all_idx, :]
    layer.values = layer.values[:, :, all_idx, :]


def _fused_importance(attentions, config: TieredKVConfig, n_layers: int) -> torch.Tensor:
    """跨层聚合 attention importance（Tier 2：层间信号融合）。"""
    per_layer = [
        _importance_from_attention(
            attentions[i],
            config.obs_window,
            query_pool=config.query_pool,
            exp_lambda=config.exp_lambda,
            topk_q=config.topk_q,
        )
        for i in range(n_layers)
    ]
    stacked = torch.stack(per_layer, dim=0)          # [L, S]
    # 浅层权重略高：低层更偏局部/句法，对 token 选取更敏感
    weights = torch.linspace(1.2, 0.8, n_layers, device=stacked.device)
    weights = weights / weights.sum()
    return (stacked * weights.unsqueeze(-1)).sum(dim=0)


def compress_kv_tiered(past_kv, attentions, config: TieredKVConfig):
    """三层 KV 压缩主入口（in-place）。"""
    if not config.use_treekv:
        return

    n_layers = len(past_kv.layers)
    k_sink = config.effective_k_sink()

    if config.use_cross_layer_fusion:
        importance = _fused_importance(attentions, config, n_layers)
        for layer in past_kv.layers:
            _compress_layer_treekv(
                layer,
                importance,
                k_sink=k_sink,
                k_ratio=config.k_ratio,
                geometry_skew=1.0,
                local_window=config.local_window,
                n_levels=config.n_levels,
            )
        return

    for layer_idx, layer in enumerate(past_kv.layers):
        importance = _importance_from_attention(
            attentions[layer_idx],
            config.obs_window,
            query_pool=config.query_pool,
            exp_lambda=config.exp_lambda,
            topk_q=config.topk_q,
        )
        _compress_layer_treekv(
            layer,
            importance,
            k_sink=k_sink,
            k_ratio=config.k_ratio,
            geometry_skew=config.layer_geometry_skew(layer_idx, n_layers),
            local_window=config.local_window,
            n_levels=config.n_levels,
        )


PRESET_CONFIGS: dict[str, TieredKVConfig] = {
    "treekv_only": TieredKVConfig(
        use_sink=False, use_layer_adaptive=False, use_treekv=True,
    ),
    "sink_treekv": TieredKVConfig(
        use_sink=True, use_layer_adaptive=False, use_treekv=True,
    ),
    "layer_treekv": TieredKVConfig(
        use_sink=False, use_layer_adaptive=True, use_treekv=True,
    ),
    "fusion_treekv": TieredKVConfig(
        use_sink=False, use_cross_layer_fusion=True, use_treekv=True,
    ),
    "tiered_full": TieredKVConfig(
        use_sink=True, use_cross_layer_fusion=True, use_treekv=True,
    ),
}


def make_compress_fn(config: TieredKVConfig) -> Callable:
    def _fn(past_kv, attentions):
        compress_kv_tiered(past_kv, attentions, config)
    return _fn


def estimate_attention_flops(
    n_layers: int, n_heads: int, head_dim: int,
    seq_len: int, kv_len: int, *, is_prefill: bool = False,
) -> int:
    per_layer = 4 * n_heads * head_dim
    if is_prefill:
        return n_layers * per_layer * seq_len * seq_len
    return n_layers * per_layer * kv_len


def estimate_total_flops(
    n_layers: int, n_heads: int, head_dim: int,
    prompt_tokens: int, gen_tokens: int, kv_len_after_compress: int,
) -> dict:
    prefill = estimate_attention_flops(
        n_layers, n_heads, head_dim, prompt_tokens, kv_len_after_compress,
        is_prefill=True,
    )
    decode = sum(
        estimate_attention_flops(
            n_layers, n_heads, head_dim, 1,
            kv_len_after_compress + t, is_prefill=False,
        )
        for t in range(gen_tokens - 1)
    )
    total = prefill + decode
    n_tokens = prompt_tokens + gen_tokens
    return {
        "prefill_flops": prefill,
        "decode_flops": decode,
        "total_flops": total,
        "avg_flops_per_token": total // max(n_tokens, 1),
    }
