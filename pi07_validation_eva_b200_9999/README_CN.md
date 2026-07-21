# B200 real-only 9999 Piper 离线测评脚本

这个目录保存 Piper 当前使用的 B200 `cotrain_real_only_b200_v1/9999` 离线测评版本。

## 关键点

- 模型动作空间是 80D unified action space。
- Piper 数据集原始 action/state 是 14D。
- 脚本不会把 `80D[:14]` 当成 Piper action。
- Piper 14D 映射到 80D 槽位为：

```text
left_joint_1..6   -> 80D[0:6]
left_gripper      -> 80D[16]
right_joint_1..6  -> 80D[29:35]
right_gripper     -> 80D[45]
```

## 默认路径

默认路径写在：

```bash
config/eval_defaults.env
```

这些默认路径面向 Piper 电脑：

```bash
/home/ps/Documents/zhengdongchen/b200_cotrain_real_only_9999
```

## 运行

在 Piper 上：

```bash
bash /home/ps/Documents/zhengdongchen/pi07_validation_eva_b200_9999/scripts/run_validation.sh --validate-only
```

最小 smoke：

```bash
EPISODES=1 \
ANCHORS_PER_EPISODE=1 \
ACTIONS_PER_INFERENCE=8 \
NATIVE_VAL_LOSS_SAMPLES=0 \
SPLIT=seen_test \
DEVICE=cuda \
bash /home/ps/Documents/zhengdongchen/pi07_validation_eva_b200_9999/scripts/run_validation.sh
```

全量 seen/unseen 需要注意内存占用，建议使用 GPU，并且不要和真机 server 同时占用显存。

## 说明

这个目录不是原始 `pi07_validation_eva` 的通用 H800 多 checkpoint 工程，而是当前 B200/9999 的专用版本。
