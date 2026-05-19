# TieredKV: Group Project (AI2801 NLP)

Training-free hierarchical KV cache compression for **Pythia-70M**, combining:

| Tier | Mechanism | Module |
|------|-----------|--------|
| T1 | Attention sink protection | `tiered_kv.py` |
| T2 | Cross-layer importance fusion (optional) | `tiered_kv.py` |
| T3 | TreeKV intra-layer block eviction | `tiered_kv.py` |

Paper: `group_work/tieredkv_paper.tex` (NeurIPS 2025 template, ≤4 pages).

Individual baseline implementations (StreamingLLM, SnapKV, TreeKV, SnapKV improvements) live in the **personal repo**: [KV_Cache_Compression](https://github.com/6chHenry/KV_Cache_Compression).

---

## Setup

```bash
conda create -n tieredkv python=3.10
conda activate tieredkv
pip install -r requirements.txt
cd utils && python prepare_data.py
```

Model weights and datasets are cached under `./cache/` (not tracked in git).

---

## Reproduce main results

```bash
cd src

# Integrated ablation + benchmarks (saves results/results_integrated.json)
python integrated.py --mode all

# PG-19 suite: 20 books, rho in {0.5, 0.8}
python run_pg19_suite.py --k_ratio 0.5 --num_samples 20
python run_pg19_suite.py --k_ratio 0.8 --num_samples 20
```

Committed JSON under `results/` matches the numbers in the paper tables.

---

## Layout

```
├── group_work/          # paper, plan, experiment script
├── src/
│   ├── tiered_kv.py     # core UHB compressor
│   ├── integrated.py    # ablation + PPL/benchmark harness
│   ├── run_pg19_suite.py
│   └── baseline.py, snapkv.py, treekv.py, streaming_llm.py  # baselines
├── results/             # experiment outputs
└── utils/
    ├── prepare_data.py
    └── download_pg19.py
```
