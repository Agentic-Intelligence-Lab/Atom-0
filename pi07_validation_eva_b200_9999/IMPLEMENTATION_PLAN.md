# B200 9999 Validation Version

本目录目标是保存当前 Piper 使用的 B200 real-only 9999 离线测评代码，便于和 GitHub 标准目录结构对齐。

已包含：

- `config/eval_defaults.env`: B200/9999 默认路径与参数。
- `scripts/evaluate_validation.py`: Piper 当前 80D 适配版离线测评脚本。
- `scripts/run_validation.sh`: 标准入口脚本。
- `README_CN.md`: 使用说明。

主要实现要求：

- 支持 80D unified action space。
- 支持 14D Piper action/state 和 80D model action 的双向映射。
- 支持 `cotrain_real_only` / `piper30` / B200 9999 checkpoint。
- 不导入或调用真机硬件、CAN、相机控制代码。
