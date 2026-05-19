# TieredKV：三层 Training-Free KV 压缩 — 完整实施计划

> 小组大作业 · Pythia-70M · 无训练优化 · NeurIPS 4-page paper

---

## 1. 项目目标

在 **Pythia-70M** 上集成三种 **training-free** 加速机制，在 wikitext-2 / pg-19 上评测 **PPL** 与 **推理性能**，并撰写可复现实验的英文论文。

| 层面 | 方法 | 职责 |
|------|------|------|
| **Attention 机制** | Attention Sink 保护 | 强制保留序列前 `k_sink` 个 token，防止 attention 失稳 |
| **层间 KV 压缩** | Depth-Adaptive Layer Budget (DALB) | 各层统一 token 总数，但块预算曲线随深度变化（浅层更均匀、深层更偏近端） |
| **层内 KV 压缩** | TreeKV | 历史区分块 + 几何预算 + 块内 attention top-k |

**创新点**：Unified Hierarchical KV Budget (UHB) — 在总预算约束下联合分配 sink / 层间 / 层内预算，而非简单串联三个独立方法。

---

## 2. 系统架构

```
Prefill (eager + fp32, output_attentions=True)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Tier 1 — Attention Sink                            │
│  每层强制保留 index [0, k_sink)，不参与后续淘汰       │
├─────────────────────────────────────────────────────┤
│  Tier 2 — Depth-Adaptive Layer Budget               │
│  各层 KV 长度一致；layer_ratio 控制 TreeKV 几何曲线   │
│  浅层更均匀保留远端，深层更集中近端历史块              │
├─────────────────────────────────────────────────────┤
│  Tier 3 — TreeKV                                    │
│  对 [k_sink, S-local_window) 分 n_levels 块          │
│  块预算 ∝ 2^i，块内按 attention importance top-k     │
│  local_window 末尾 token 完全保留                      │
└─────────────────────────────────────────────────────┘
    │
    ▼
Decode (greedy, 可记录 TTFT / TPOT / Throughput)
```

### Pythia-70M 默认超参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k_sink` | 4 | StreamingLLM 推荐值 |
| `k_ratio` (base) | 0.50 | TreeKV 仅模式下的统一压缩率 |
| `layer_ratios` | `[0.70, 0.65, 0.55, 0.50, 0.40, 0.35]` | 6 层逐层压缩率 |
| `obs_window` | 32 | SnapKV/TreeKV 观察窗口 |
| `local_window` | 32 | 末尾保留窗口 |
| `n_levels` | 4 | TreeKV 几何块数（预算比 1:2:4:8） |

---

## 3. 代码结构

```
llm_acceleration/
├── group_work/
│   ├── Instruction.md          # 作业要求
│   └── PLAN.md                 # 本文件
├── src/
│   ├── baseline.py             # Full KV baseline
│   ├── streaming_llm.py        # 参考：Attention Sink
│   ├── snapkv.py               # 参考：层内 top-k
│   ├── treekv.py               # 参考：TreeKV 单独评测
│   ├── tiered_kv.py            # ★ 核心：三层压缩逻辑
│   └── integrated.py           # ★ 集成评测 + ablation
├── results/
│   └── results_integrated.json # 集成实验输出
└── utils/
    └── prepare_data.py
```

---

## 4. 实施阶段

### Phase 1 — 核心压缩模块 `tiered_kv.py` ✅

- [x] `compress_kv_tiered()` — 完整三层压缩
- [x] `TieredKVConfig` — 配置 dataclass，支持开关各 tier
- [x] 预设 ablation 配置：`treekv_only` / `sink_treekv` / `layer_treekv` / `tiered_full`
- [x] 块预算容量重分配（保证各层 KV 长度一致，decode 可用）

### Phase 2 — 集成评测 `integrated.py` ✅

- [x] 滑动窗口 PPL（wikitext-2 + pg-19）
- [x] 短 prompt benchmark（与现有方法对齐）
- [x] 长 prompt benchmark（pg-19 4096 tokens prefill）
- [x] 指标：TTFT / TPOT / Throughput / peak_mem / kv_len / FLOPs_est
- [x] 一键跑全部 ablation 并保存 JSON
- [x] `group_work/run_experiments.ps1` 实验脚本

### Phase 3 — 实验跑通

- [x] 短 prompt ablation benchmark
- [x] 长上下文 ablation benchmark（4096 prefill）
- [x] 全部 ablation PPL（已完成，见 `results/results_integrated.json`）
- [x] 整理结果表格供论文使用

### Phase 4 — 论文撰写

- [x] `group_work/tieredkv_paper.tex` — NeurIPS 格式英文论文（≤4 页正文）
- [ ] 填写作者信息、编译 PDF、组内审阅

---

## 5. Ablation 实验设计

| 编号 | 配置名 | Sink | Layer-Adaptive | TreeKV | 目的 |
|:----:|--------|:----:|:--------------:|:------:|------|
| A0 | baseline | — | — | — | 上限 |
| A1 | treekv_only | ✗ | ✗ | ✓ | 层内基线（已有最好 PPL） |
| A2 | sink_treekv | ✓ | ✗ | ✓ | 验证 Sink 贡献 |
| A3 | layer_treekv | ✗ | ✓ | ✓ | 验证层间预算贡献 |
| A4 | tiered_full | ✓ | ✓ | ✓ | 完整方案 |

### 运行命令

```bash
cd src/

# 快速 benchmark 验证（跳过 PPL）
python integrated.py --mode all --skip_ppl

# 完整实验
python integrated.py --mode all

# 单独跑完整方案
python integrated.py --mode tiered_full

# 长上下文 benchmark
python integrated.py --mode tiered_full --long_bench --prefill_tokens 8192
```

---

## 6. 评测指标定义

| 指标 | 公式 | 说明 |
|------|------|------|
| **TTFT** | `t_prefill` | Prefill 阶段耗时（秒） |
| **TPOT** | `t_decode / (gen_len - 1)` | 每输出 token 平均 decode 时间 |
| **Throughput** | `(prompt_tokens + gen_len) / (t_prefill + t_decode)` | 端到端 tokens/sec |
| **Peak Mem** | `max_memory_allocated` | 峰值 GPU 显存 (MB) |
| **KV Len** | `past_kv.get_seq_length()` | 压缩后 KV 序列长度 |
| **FLOPs (est)** | 见 `estimate_attention_flops()` | Attention 浮点运算估算 |
| **PPL** | 滑动窗口 NLL 指数 | wikitext-2 全量 + pg-19 单 sample |

---

## 7. 预期结果与风险

### 预期

- **PPL**：tiered_full ≤ treekv_only（~41.9），Sink + 层间预算在长文本上可能更优
- **显存/速度**：长 prompt（8K+）下 tiered_full 因深层 KV 更短而优于统一 k_ratio
- **创新分**：UHB 统一预算 + 完整 ablation 链条

### 风险与应对

| 风险 | 应对 |
|------|------|
| 短 prompt 测不出加速 | 增加 `--long_bench` 长上下文 benchmark |
| eager+fp32 显存虚高 | 论文中注明；decode 阶段可另测 sdpa+fp16 |
| 三层叠加 PPL 恶化 | ablation 逐步验证，可调低深层 k_ratio 降幅 |
| PPL 评测耗时长 | `--skip_ppl` 先跑 benchmark；单独跑 PPL |

---

## 8. 分工建议

| 成员 | 任务 |
|------|------|
| A | Tier 1 Sink 机制验证 + pg-19 长文本实验 |
| B | Tier 2 层间预算调参 + FLOPs 分析 |
| C (你) | Tier 3 TreeKV 集成 + integrated.py 维护 |
| 共同 | ablation 表格、论文 Method/Experiments 章节 |

---

## 9. 里程碑

| 日期 | 里程碑 |
|------|--------|
| Week 1 | Phase 1–2 代码完成，smoke test 通过 |
| Week 2 | 全部 ablation 实验跑完，结果入库 |
| Week 3 | 论文初稿 + 代码链接 |
| Week 4 | 定稿 + 复现包整理 |

---

## 10. 参考文献

1. Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, 2023.
2. Lian et al., *TreeKV: Smooth Key-Value Cache Compression with Tree Structures*, IJCAI 2025.
3. Li et al., *SnapKV*, 2024.
4. Biderman et al., *Pythia*, 2023.
