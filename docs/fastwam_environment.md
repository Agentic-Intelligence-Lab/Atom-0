# FastWAM 编译 / 运行环境配置（Cotrain RLDS）

本文说明在 Atom-0 上跑 **FastWAM + cotrain RLDS** 所需环境。  
**只写配置说明，不在本机实际安装。**

主数据路径与 [`cotrain_技术文档.md`](./cotrain_技术文档.md) 一致：`/mnt/data/RLDS/` + `uv sync --group rlds`。

---

## 1. 基础环境（Atom-0）

| 项目 | 要求 |
|------|------|
| OS | Linux x86_64 |
| Python | ≥ 3.11 |
| 包管理 | `uv` |
| 主依赖 | 见根目录 `pyproject.toml`（含 `torch==2.7.1` 等） |

```bash
cd Atom-0
uv sync
uv sync --group rlds   # FastWAM cotrain 必需：dlimp / tensorflow-cpu / tfds
```

验证：

```bash
uv run --group rlds python -c "import dlimp, tensorflow_datasets, torch, openpi; print('ok')"
```

---

## 2. FastWAM 额外依赖

| 依赖 | 用途 |
|------|------|
| `huggingface-hub` / `transformers` | Wan / UMT5 权重与 tokenizer |
| `einops` / `safetensors` / `rich` | 已在 Atom-0 主依赖中 |
| FlashAttention（可选） | MoT mixed attn 加速 |

上游曾钉 `torch==2.7.1+cu128`；Atom-0 用 `torch==2.7.1`。若需强制 cu128 wheel，见历史 FastWAM README（注意与 JAX/CUDA 共存冲突）。

---

## 3. 预训练权重

```bash
export DIFFSYNTH_MODEL_BASE_PATH="/绝对路径/Atom-0/checkpoints/fastwam"
export HF_HOME="${DIFFSYNTH_MODEL_BASE_PATH}/hf_cache"
```

建议目录：

```text
checkpoints/fastwam/
  Wan-AI/Wan2.2-TI2V-5B/
  DiffSynth-Studio/Wan-Series-Converted-Safetensors/   # VAE / T5
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt  # ActionDiT backbone（默认加载）
```

默认 `FastWAMConfig`：`Wan-AI/Wan2.2-TI2V-5B`，VAE/T5 可重定向 DiffSynth 转换权重；ActionDiT 默认从上述 `.pt` 加载 backbone（`action_encoder` / `head` 仍随机），与上游 FastWAM 一致。

生成 ActionDiT backbone（需本地已有 Wan2.2 DiT 权重）：

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints/fastwam"
uv run python scripts/preprocess_action_dit_backbone.py \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cpu \
  --dtype bfloat16
```

---

## 4. RLDS 数据（与 cotrain 相同）

- 根路径示例：`/mnt/data/RLDS/`
- 各集 `builder_dir` 写在 `openpi.cotrain.config`（如 piper30）
- 规范：`train` / `seen_test` / `unseen_test` splits；`uid` 唯一

归一化（per-dataset，native 维）：

```bash
uv run --group rlds scripts/compute_cotrain_norm_stats.py \
  --config-name fastwam_cotrain_piper30
```

（与 π0.5 cotrain 共用 `compute_cotrain_norm_stats.py`；stats 目录按 `uid`。）

---

## 5. 训练 / 调试命令

```bash
cd Atom-0
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints/fastwam"

# 正式：piper30 RLDS + FastWAM
uv run --group rlds scripts/train_fastwam.py fastwam_cotrain_piper30 \
  --exp_name=fw_piper30 --overwrite

# 联调（跳过 DiT 预训练加载；仍可能下 VAE）
uv run --group rlds scripts/train_fastwam.py fastwam_cotrain_piper30_debug
```

多卡：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_fastwam.py fastwam_cotrain_piper30 --exp_name=fw_piper30
```

> 配置名来自 **`openpi.cotrain.config`** registry，不是 `openpi.training.config`。

---

## 6. 硬件建议

| 场景 | 建议 |
|------|------|
| 全量 5B Video + Action | 多卡 A100/H100；全局 `batch_size` 从 1–4 试起 |
| 显存 | `mot_checkpoint_mixed_attn=True`；`shuffle_buffer_size` 默认已调小（视频窗更重） |
| 精度 | `bfloat16` |
| OOM | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

---

## 7. 验收清单

- [ ] `uv sync` + `uv sync --group rlds` 成功  
- [ ] `DIFFSYNTH_MODEL_BASE_PATH` 可加载 Wan DiT / VAE（/ T5）  
- [ ] `cotrain.get_config("fastwam_cotrain_piper30")` 正常  
- [ ] `compute_cotrain_norm_stats` 对 piper30（或目标集）跑通  
- [ ] `train_fastwam.py` 至少一个 step：`loss_video` / `loss_action` 有限  
- [ ] 产出 `model.safetensors`  

---

## 8. 相关文档

- 算法：[`fastwam_algorithm.md`](./fastwam_algorithm.md)  
- Cotrain RLDS：[`cotrain_技术文档.md`](./cotrain_技术文档.md)  
- 上游 FastWAM：`../../FastWAM/README_zh.md`
