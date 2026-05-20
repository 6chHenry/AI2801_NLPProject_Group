# TieredKV: Group Project (AI2801 NLP)

Training-free hierarchical KV cache compression for **Pythia-70M**, combining:

| Tier | Mechanism | Module |
|------|-----------|--------|
| T1 | Attention sink protection | `tiered_kv.py` |
| T2 | Cross-layer importance fusion (optional) | `tiered_kv.py` |
| T3 | TreeKV intra-layer block eviction | `tiered_kv.py` |

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

# Ablation + PPL/benchmark (writes results/results_integrated.json)
python integrated.py --mode all

# Single configuration
python integrated.py --mode sink_treekv
python integrated.py --mode tiered_full --long_bench --prefill_tokens 4096
```

Committed JSON under `results/` matches the reported experiment tables.

---

## Layout

```
├── src/
│   ├── tiered_kv.py     # core UHB compressor
│   ├── integrated.py    # ablation + PPL/benchmark harness
│   └── baseline.py, snapkv.py, treekv.py, streaming_llm.py
├── results/             # experiment outputs
└── utils/
    ├── prepare_data.py
    └── download_pg19.py
```
