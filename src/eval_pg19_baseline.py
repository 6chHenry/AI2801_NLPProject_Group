"""Multi-sample pg-19 PPL for full-KV baseline (no compression)."""
import json
import os
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from baseline import compute_ppl, CACHE_DIR, DEVICE, load_model

def pg19_path():
    for name in ("pg19_test_20samples.json", "pg19_test_5samples.json"):
        p = os.path.join(CACHE_DIR, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("pg-19 cache missing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pg19_tokens", type=int, default=8192)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    tokenizer, model = load_model()
    with open(pg19_path(), encoding="utf-8") as f:
        samples = json.load(f)[: args.num_samples]

    details, n_toks, nlls = [], [], []
    for idx, sample in enumerate(samples):
        title = sample.get("short_book_title", f"s{idx}")
        ids = tokenizer(sample["text"], return_tensors="pt").input_ids.to(DEVICE)
        ids = ids[:, : args.pg19_tokens]
        ppl, nll, n = compute_ppl(model, ids, stride=args.stride, max_length=args.max_length)
        details.append({"sample_idx": idx, "title": title, "ppl": round(ppl, 4), "tokens": n})
        n_toks.append(n)
        nlls.append(nll)
        print(f"  sample {idx} ({title}): PPL={ppl:.4f}")

    avg_nll = sum(n * nl for n, nl in zip(n_toks, nlls)) / sum(n_toks)
    avg_ppl = float(np.exp(avg_nll))
    out = {
        "pg19_ppl_avg": round(avg_ppl, 4),
        "pg19_per_sample": details,
        "max_length": args.max_length,
        "num_samples": len(samples),
    }
    path = os.path.join(os.path.dirname(__file__), "..", "results", "results_baseline_pg19_multi.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nBaseline pg-19 avg PPL = {avg_ppl:.4f}  -> {path}")


if __name__ == "__main__":
    main()
