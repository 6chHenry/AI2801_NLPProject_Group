"""
PG-19 大规模评测：20 samples PPL + baseline/compressed 长上下文 benchmark。

统一使用 fp32 + eager（与 integrated.py 一致），保证 PPL / 速度对比公平。

用法:
  python run_pg19_suite.py --num_samples 20 --skip_bench   # 仅 PPL
  python run_pg19_suite.py --num_samples 20 --skip_ppl     # 仅 benchmark
  python run_pg19_suite.py --num_samples 20                # 全部
"""
import os
import json
import time
import argparse
import torch
import numpy as np

from baseline import DEVICE
from integrated import (
    load_model,
    eval_pg19,
    eval_wikitext,
    load_long_prompt,
    benchmark_generation,
)
from tiered_kv import PRESET_CONFIGS, TieredKVConfig, make_compress_fn, estimate_total_flops

N_LAYERS, N_HEADS, HEAD_DIM = 6, 8, 64


def _out_path(k_ratio: float) -> str:
    tag = f"r{int(round(k_ratio * 100)):03d}"
    return os.path.join(
        os.path.dirname(__file__), "..", "results", f"results_pg19_suite_{tag}.json",
    )


def benchmark_baseline_long(model, tokenizer, prompt: str, gen_len: int = 100):
    """Full-KV long-context benchmark，指标格式与 integrated 对齐。"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = input_ids.size(1)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    past_kv = out.past_key_values
    kv_len = past_kv.get_seq_length()
    next_token = out.logits[:, -1:, :].argmax(-1)

    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen_len - 1):
            out = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(-1)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_decode = time.perf_counter() - t1

    total_time = t_prefill + t_decode
    peak_mem = (
        torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024
        if DEVICE == "cuda" else 0.0
    )
    flops = estimate_total_flops(N_LAYERS, N_HEADS, HEAD_DIM, prompt_len, gen_len, kv_len)
    decode_steps = max(gen_len - 1, 1)

    return {
        "method": "baseline_full_kv",
        "prompt_tokens": prompt_len,
        "generated_tokens": gen_len,
        "kv_len_after_prefill": kv_len,
        "kv_len_after_decode": past_kv.get_seq_length(),
        "ttft_sec": round(t_prefill, 4),
        "tpot_sec": round(t_decode / decode_steps, 6),
        "throughput_tps": round((prompt_len + gen_len) / total_time, 2),
        "prefill_sec": round(t_prefill, 4),
        "decode_sec": round(t_decode, 4),
        "decode_tps": round(gen_len / t_decode, 2),
        "peak_mem_mb": round(peak_mem, 1),
        "flops": flops,
    }


def _noop_compress(past_kv, attentions):
    """Full-KV baseline under the same prefill→(no compress)→target protocol."""
    del past_kv, attentions


def eval_baseline_pg19(model, tokenizer, args):
    ppl, _, _, details = eval_pg19(
        model, tokenizer, _noop_compress,
        max_length=args.pg19_max_length, stride=args.stride,
        pg19_tokens=args.pg19_tokens, num_samples=args.num_samples,
    )
    std_ppl = float(np.std([d["ppl"] for d in details]))
    out = {
        "mode": "baseline",
        "pg19_ppl_avg": round(ppl, 4),
        "pg19_ppl_std": round(std_ppl, 4),
        "pg19_per_sample": details,
        "num_samples": len(details),
    }
    if not args.skip_wikitext:
        wt_ppl, _, _ = eval_wikitext(model, tokenizer, _noop_compress)
        out["wikitext2_ppl"] = round(wt_ppl, 4)
    return out


def eval_compressed_pg19(model, tokenizer, mode, args):
    preset = PRESET_CONFIGS[mode]
    config = TieredKVConfig(
        k_sink=args.k_sink,
        k_ratio=args.k_ratio,
        obs_window=args.obs_window,
        local_window=args.local_window,
        n_levels=args.n_levels,
        use_sink=preset.use_sink,
        use_layer_adaptive=preset.use_layer_adaptive,
        use_cross_layer_fusion=preset.use_cross_layer_fusion,
        use_treekv=preset.use_treekv,
    )
    compress_fn = make_compress_fn(config)
    ppl, nll, n, details = eval_pg19(
        model, tokenizer, compress_fn,
        max_length=args.pg19_max_length, stride=args.stride,
        pg19_tokens=args.pg19_tokens, num_samples=args.num_samples,
    )
    std_ppl = float(np.std([d["ppl"] for d in details]))
    out = {
        "mode": mode,
        "k_ratio": args.k_ratio,
        "pg19_ppl_avg": round(ppl, 4),
        "pg19_ppl_std": round(std_ppl, 4),
        "pg19_per_sample": details,
        "num_samples": len(details),
    }
    if not args.skip_wikitext:
        wt_ppl, _, _ = eval_wikitext(model, tokenizer, compress_fn)
        out["wikitext2_ppl"] = round(wt_ppl, 4)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--pg19_tokens", type=int, default=8192)
    parser.add_argument("--pg19_max_length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--prefill_tokens", type=int, default=4096)
    parser.add_argument("--gen_len", type=int, default=100)
    parser.add_argument("--k_ratio", type=float, default=0.5)
    parser.add_argument("--k_sink", type=int, default=4)
    parser.add_argument("--obs_window", type=int, default=32)
    parser.add_argument("--local_window", type=int, default=32)
    parser.add_argument("--n_levels", type=int, default=4)
    parser.add_argument(
        "--modes", nargs="+",
        default=["baseline", "treekv_only", "sink_treekv", "tiered_full"],
    )
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_wikitext", action="store_true")
    parser.add_argument("--skip_bench", action="store_true")
    args = parser.parse_args()

    out_path = _out_path(args.k_ratio)
    print(f"Device: {DEVICE}  |  samples={args.num_samples}  |  "
          f"L={args.pg19_max_length}  |  k_ratio={args.k_ratio}")
    tokenizer, model = load_model()

    results = {}
    results["config"] = {
        "num_samples": args.num_samples,
        "pg19_tokens": args.pg19_tokens,
        "pg19_max_length": args.pg19_max_length,
        "stride": args.stride,
        "k_ratio": args.k_ratio,
        "prefill_tokens": args.prefill_tokens,
        "dtype": "fp32",
        "attn": "eager",
        "ppl_protocol": "integrated_sliding",
        "context_max_tokens": args.pg19_max_length - args.stride,
    }

    if not args.skip_ppl:
        if "baseline" in args.modes:
            print("\n[baseline] pg-19 PPL (full KV)...")
            results["baseline"] = eval_baseline_pg19(model, tokenizer, args)
            print(f"  avg={results['baseline']['pg19_ppl_avg']:.4f}  "
                  f"std={results['baseline']['pg19_ppl_std']:.4f}")

        for mode in args.modes:
            if mode == "baseline":
                continue
            print(f"\n[{mode}] pg-19 PPL (k_ratio={args.k_ratio})...")
            results[mode] = eval_compressed_pg19(model, tokenizer, mode, args)
            print(f"  avg={results[mode]['pg19_ppl_avg']:.4f}  "
                  f"std={results[mode]['pg19_ppl_std']:.4f}")

    if not args.skip_bench:
        prompt = load_long_prompt(tokenizer, args.prefill_tokens, sample_idx=0)
        print(f"\n[baseline] long benchmark (prefill={args.prefill_tokens})...")
        results["baseline_benchmark"] = benchmark_baseline_long(
            model, tokenizer, prompt, gen_len=args.gen_len,
        )
        for k, v in results["baseline_benchmark"].items():
            if k != "flops":
                print(f"  {k}: {v}")

        for mode in ("sink_treekv", "treekv_only"):
            if mode not in args.modes:
                continue
            preset = PRESET_CONFIGS[mode]
            config = TieredKVConfig(
                k_sink=args.k_sink, k_ratio=args.k_ratio,
                obs_window=args.obs_window, local_window=args.local_window,
                n_levels=args.n_levels,
                use_sink=preset.use_sink,
                use_layer_adaptive=preset.use_layer_adaptive,
                use_cross_layer_fusion=preset.use_cross_layer_fusion,
                use_treekv=preset.use_treekv,
            )
            print(f"\n[{mode}] long benchmark...")
            bench = benchmark_generation(model, tokenizer, prompt, config, gen_len=args.gen_len)
            results[f"{mode}_benchmark"] = bench
            print(f"  kv_len={bench['kv_len_after_compress']}  ttft={bench['ttft_sec']}s  "
                  f"throughput={bench['throughput_tps']}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if args.skip_bench and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            prev = json.load(f)
        for key in prev:
            if key.endswith("_benchmark"):
                results[key] = prev[key]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
