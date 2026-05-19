"""
TieredKV 集成评测 — Sink + Depth-Adaptive + TreeKV

运行 ablation 实验，输出 PPL / TTFT / TPOT / Throughput / FLOPs / 显存。

用法:
  python integrated.py --mode all --skip_ppl          # 快速 benchmark
  python integrated.py --mode tiered_full             # 完整方案
  python integrated.py --mode all --long_bench        # 长上下文 benchmark
"""
import os
import json
import time
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from tiered_kv import (
    TieredKVConfig,
    PRESET_CONFIGS,
    make_compress_fn,
    compress_kv_tiered,
    estimate_total_flops,
)

CACHE_DIR  = os.path.join(os.path.dirname(__file__), "..", "cache")
MODEL_NAME = "EleutherAI/pythia-70m"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Pythia-70M architecture
N_LAYERS = 6
N_HEADS  = 8
HEAD_DIM = 64


# ── 模型加载 ───────────────────────────────────────────────────────────────────

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(DEVICE).eval()
    return tokenizer, model


# ── PPL（滑动窗口 + 压缩）────────────────────────────────────────────────────

def compute_ppl_with_compress(
    model,
    input_ids: torch.Tensor,
    compress_fn,
    stride: int = 512,
    max_length: int = 2048,
):
    seq_len = input_ids.size(1)
    nlls = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    ctx_len = max_length - stride

    for tgt_start in range(stride, seq_len, stride):
        tgt_end   = min(tgt_start + stride, seq_len)
        ctx_start = max(0, tgt_start - ctx_len)
        context = input_ids[:, ctx_start:tgt_start]
        target  = input_ids[:, tgt_start:tgt_end]
        if context.size(1) == 0:
            continue

        with torch.no_grad():
            out = model(context, use_cache=True, output_attentions=True)
        past_kv    = out.past_key_values
        attentions = out.attentions
        compress_fn(past_kv, attentions)

        orig_ctx_len = context.size(1)
        tgt_len      = target.size(1)
        pos_ids = torch.arange(
            orig_ctx_len, orig_ctx_len + tgt_len, device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            out2 = model(
                target, past_key_values=past_kv, use_cache=False,
                position_ids=pos_ids,
            )

        ctx_last_logit = out.logits[:, -1:, :]
        full_logits    = torch.cat([ctx_last_logit, out2.logits[:, :-1, :]], dim=1)
        nll = loss_fn(
            full_logits.view(-1, full_logits.size(-1)),
            target.view(-1),
        )
        nlls.append(nll.item())
        if tgt_end == seq_len:
            break

    n_tokens = sum(
        min(s + stride, seq_len) - s
        for s in range(stride, seq_len, stride) if s < seq_len
    )
    nll_sum = sum(nlls)
    ppl     = np.exp(nll_sum / n_tokens)
    return ppl, nll_sum / n_tokens, n_tokens


def eval_wikitext(model, tokenizer, compress_fn, max_length=2048, stride=512):
    from datasets import load_dataset
    dataset = load_dataset(
        "wikitext", "wikitext-2-raw-v1", cache_dir=CACHE_DIR, split="test",
    )
    text      = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    return compute_ppl_with_compress(
        model, input_ids, compress_fn, stride=stride, max_length=max_length,
    )


def _pg19_samples_path():
    for name in ("pg19_test_20samples.json", "pg19_test_5samples.json"):
        path = os.path.join(CACHE_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No pg-19 cache found. Run utils/prepare_data.py first.")


def eval_pg19(
    model, tokenizer, compress_fn,
    max_length=2048, stride=512, pg19_tokens=8192,
    num_samples=None, sample_idx=0,
):
    pg19_path = _pg19_samples_path()
    with open(pg19_path, encoding="utf-8") as f:
        samples = json.load(f)
    if num_samples is not None:
        indices = list(range(min(num_samples, len(samples))))
    else:
        indices = [sample_idx]

    ppls, nlls, n_toks = [], [], []
    per_sample = []
    for idx in indices:
        title = samples[idx].get("short_book_title", f"sample_{idx}")
        text  = samples[idx]["text"]
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
        input_ids = input_ids[:, :pg19_tokens]
        ppl, nll, n = compute_ppl_with_compress(
            model, input_ids, compress_fn,
            stride=stride, max_length=max_length,
        )
        ppls.append(ppl)
        nlls.append(nll)
        n_toks.append(n)
        per_sample.append({"sample_idx": idx, "title": title, "ppl": round(ppl, 4), "tokens": n})
        print(f"    sample {idx} ({title}): PPL={ppl:.4f}")

    # token-weighted average across samples
    total_n = sum(n_toks)
    avg_nll = sum(n * nll for n, nll in zip(n_toks, nlls)) / total_n
    avg_ppl = float(np.exp(avg_nll))
    return avg_ppl, avg_nll, total_n, per_sample


# ── Benchmark ──────────────────────────────────────────────────────────────────

def benchmark_generation(
    model,
    tokenizer,
    prompt: str,
    config: TieredKVConfig,
    gen_len: int = 200,
):
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = input_ids.size(1)
    compress_fn = make_compress_fn(config)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_attentions=True)
    past_kv    = out.past_key_values
    attentions = out.attentions
    compress_fn(past_kv, attentions)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    kv_len = past_kv.get_seq_length()
    next_token = out.logits[:, -1:, :].argmax(-1)

    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen_len - 1):
            out        = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv    = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(-1)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_decode = time.perf_counter() - t1

    total_time = t_prefill + t_decode
    peak_mem = (
        torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024
        if DEVICE == "cuda" else 0.0
    )
    flops = estimate_total_flops(
        N_LAYERS, N_HEADS, HEAD_DIM,
        prompt_len, gen_len, kv_len,
    )

    decode_steps = max(gen_len - 1, 1)
    return {
        "method"               : _mode_label(config),
        "k_sink"               : config.effective_k_sink(),
        "k_ratio"              : config.k_ratio,
        "layer_ratios"         : _default_ratios_display(config),
        "n_levels"             : config.n_levels,
        "obs_window"           : config.obs_window,
        "local_window"         : config.local_window,
        "prompt_tokens"        : prompt_len,
        "generated_tokens"     : gen_len,
        "kv_len_after_compress": kv_len,
        "ttft_sec"             : round(t_prefill, 4),
        "tpot_sec"             : round(t_decode / decode_steps, 6),
        "throughput_tps"       : round((prompt_len + gen_len) / total_time, 2),
        "prefill_sec"          : round(t_prefill, 4),
        "decode_sec"           : round(t_decode, 4),
        "decode_tps"           : round(gen_len / t_decode, 2),
        "peak_mem_mb"          : round(peak_mem, 1),
        "flops"                : flops,
    }


def _mode_label(config: TieredKVConfig) -> str:
    for name, preset in PRESET_CONFIGS.items():
        if (
            preset.use_sink == config.use_sink
            and preset.use_layer_adaptive == config.use_layer_adaptive
            and preset.use_cross_layer_fusion == config.use_cross_layer_fusion
            and preset.use_treekv == config.use_treekv
            and preset.k_sink == config.k_sink
            and preset.k_ratio == config.k_ratio
        ):
            return name
    parts = []
    if config.use_sink:
        parts.append("sink")
    if config.use_cross_layer_fusion:
        parts.append("fusion")
    elif config.use_layer_adaptive:
        parts.append("layer")
    if config.use_treekv:
        parts.append("treekv")
    return "+".join(parts) if parts else "none"


def load_long_prompt(tokenizer, prefill_tokens: int = 8192, sample_idx: int = 0) -> str:
    with open(_pg19_samples_path(), encoding="utf-8") as f:
        samples = json.load(f)
    text = samples[sample_idx]["text"]
    ids  = tokenizer(text, return_tensors="pt").input_ids[0, :prefill_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


# ── 单模式评测 ─────────────────────────────────────────────────────────────────

def run_mode(
    model,
    tokenizer,
    mode: str,
    args,
) -> dict:
    if mode not in PRESET_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(PRESET_CONFIGS)}")

    config = PRESET_CONFIGS[mode]
    config = TieredKVConfig(
        k_sink=args.k_sink,
        k_ratio=args.k_ratio,
        obs_window=args.obs_window,
        local_window=args.local_window,
        n_levels=args.n_levels,
        use_sink=config.use_sink,
        use_layer_adaptive=config.use_layer_adaptive,
        use_cross_layer_fusion=config.use_cross_layer_fusion,
        use_treekv=config.use_treekv,
    )
    compress_fn = make_compress_fn(config)
    result = {"mode": mode, "config": {
        "use_sink": config.use_sink,
        "use_layer_adaptive": config.use_layer_adaptive,
        "use_cross_layer_fusion": config.use_cross_layer_fusion,
        "use_treekv": config.use_treekv,
        "k_sink": config.effective_k_sink(),
        "k_ratio": config.k_ratio,
        "layer_ratios": _default_ratios_display(config),
        "n_levels": config.n_levels,
    }}

    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Sink={config.use_sink}  LayerAdaptive={config.use_layer_adaptive}  "
          f"Fusion={config.use_cross_layer_fusion}  TreeKV={config.use_treekv}")
    print(f"{'='*60}")

    if not args.skip_ppl and not args.skip_wikitext:
        print("\n[wikitext-2] 计算 PPL...")
        wt_ppl, wt_nll, wt_n = eval_wikitext(
            model, tokenizer, compress_fn,
            max_length=args.max_length, stride=args.stride,
        )
        print(f"  PPL = {wt_ppl:.4f}  (tokens={wt_n:,})")
        result["wikitext2_ppl"] = round(wt_ppl, 4)

    if not args.skip_ppl:
        print("\n[pg-19] 计算 PPL...")
        pg_ppl, pg_nll, pg_n, pg_detail = eval_pg19(
            model, tokenizer, compress_fn,
            max_length=args.pg19_max_length, stride=args.stride,
            pg19_tokens=args.pg19_tokens,
            num_samples=args.pg19_num_samples,
        )
        print(f"  avg PPL = {pg_ppl:.4f}  (tokens={pg_n:,}, samples={len(pg_detail)})")
        result["pg19_ppl"] = round(pg_ppl, 4)
        result["pg19_per_sample"] = pg_detail

    if not args.skip_bench:
        if args.long_bench:
            prompt = load_long_prompt(tokenizer, args.prefill_tokens)
            print(f"\n[Long Benchmark] prefill={args.prefill_tokens} tokens, "
                  f"gen={args.gen_len}")
        else:
            prompt = (
                "In the beginning of the long story, the protagonist found himself "
                "standing at the crossroads of fate. "
            )
            print(f"\n[Benchmark] prompt={len(tokenizer(prompt).input_ids)} tokens, "
                  f"gen={args.gen_len}")

        bench = benchmark_generation(
            model, tokenizer, prompt, config, gen_len=args.gen_len,
        )
        for k, v in bench.items():
            if k != "flops":
                print(f"  {k}: {v}")
        print(f"  total_flops: {bench['flops']['total_flops']:,}")
        print(f"  avg_flops_per_token: {bench['flops']['avg_flops_per_token']:,}")
        result["benchmark"] = bench

    return result


def _default_ratios_display(config: TieredKVConfig) -> list[float] | str:
    if not config.use_layer_adaptive:
        return "uniform"
    from tiered_kv import _default_ratios_for_layers
    return [round(r, 2) for r in _default_ratios_for_layers(N_LAYERS)]


# ── 主程序 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TieredKV integrated evaluation")
    parser.add_argument(
        "--mode", default="all",
        choices=list(PRESET_CONFIGS) + ["all"],
        help="Ablation 模式；all = 跑全部 preset",
    )
    parser.add_argument("--max_length",   type=int,   default=2048)
    parser.add_argument("--stride",       type=int,   default=512)
    parser.add_argument("--k_sink",       type=int,   default=4)
    parser.add_argument("--k_ratio",      type=float, default=0.5)
    parser.add_argument("--obs_window",   type=int,   default=32)
    parser.add_argument("--local_window", type=int,   default=32)
    parser.add_argument("--n_levels",     type=int,   default=4)
    parser.add_argument("--pg19_tokens",  type=int,   default=8192)
    parser.add_argument("--pg19_max_length", type=int, default=4096,
                        help="pg-19 滑动窗口大小（更长以体现 KV 压缩）")
    parser.add_argument("--pg19_num_samples", type=int, default=None,
                        help="pg-19 评测样本数；None=仅 sample 0")
    parser.add_argument("--gen_len",      type=int,   default=200)
    parser.add_argument("--prefill_tokens", type=int, default=8192,
                        help="长上下文 benchmark 的 prefill token 数")
    parser.add_argument("--long_bench", action="store_true",
                        help="使用 pg-19 长文本做 benchmark")
    parser.add_argument("--skip_ppl",   action="store_true")
    parser.add_argument("--skip_wikitext", action="store_true")
    parser.add_argument("--skip_bench", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("加载模型（eager + fp32）...")
    tokenizer, model = load_model()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    modes = list(PRESET_CONFIGS) if args.mode == "all" else [args.mode]
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "results_integrated.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for mode in modes:
        prev = all_results.get(mode, {})
        new = run_mode(model, tokenizer, mode, args)
        prev.update(new)
        all_results[mode] = prev

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
