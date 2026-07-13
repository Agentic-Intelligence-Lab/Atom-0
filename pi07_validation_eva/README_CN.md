# Piper JAX validation trajectory 离线评测工具

评测工程位于 `/mnt/workspace/zhengdongchen/Atom-0/pi07_validation_eva`，训练仓库 `/mnt/workspace/xule/pi07_reproduction` 只读导入。评测结果默认写入工程内的 `results/`，不会提交到 GitHub。

## 默认运行

```bash
bash /mnt/workspace/zhengdongchen/Atom-0/pi07_validation_eva/scripts/run_validation.sh
```

默认评测 `CHECKPOINT_STEP=20000`、`SPLIT=seen_test`、`ANCHORS_PER_EPISODE=20`、`ACTIONS_PER_INFERENCE=8`、`FLOW_LOSS_SAMPLES=4`、`DEVICE=cuda`。

## 后续 checkpoint

```bash
CHECKPOINT_STEP=25000 bash /mnt/workspace/zhengdongchen/Atom-0/pi07_validation_eva/scripts/run_validation.sh
CHECKPOINT_STEP=30000 bash /mnt/workspace/zhengdongchen/Atom-0/pi07_validation_eva/scripts/run_validation.sh
```

## 只做元数据或校验

```bash
bash /mnt/workspace/zhengdongchen/Atom-0/pi07_validation_eva/scripts/run_validation.sh --metadata-only
bash /mnt/workspace/zhengdongchen/Atom-0/pi07_validation_eva/scripts/run_validation.sh --validate-only
```

当前 SSH 环境如果 JAX 看不到 CUDA GPU，`--validate-only` 会失败并记录原因；脚本不会在无 GPU 情况下假装完成 H800 评测。

## 关键产物

- `config/eval_defaults.env`: 默认路径和参数。
- `config/training_metadata_0629.json`: 可选的本地训练元数据；如存在，报告生成器会自动读取。
- `manifests/seen_open_loop_h8_seed42.jsonl`: open-loop 8-step manifest。
- `manifests/seen_flow_loss_MODEL_HORIZON_seed42.jsonl`: full model horizon flow-loss manifest。
- `results/step_020000/report.md` / `report.html`: 中文报告。
