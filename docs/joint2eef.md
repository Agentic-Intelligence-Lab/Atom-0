1.urdf对齐
Agibot:
    agibot_G2.urdf
DROID:
    panda.urdf
RoboCOIN:
    Airbot_MMK2_s36_a36_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_10532:mmk2_s_g2.urdf
    Galaxea_R1_Lite_s14_a14_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_3650:r1_v2_1_0.urdf
    Unitree_G1_Dex3_phecda_s28_a28_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_1411:g1_body29_hand14.urdf
    discover_robotics_aitbot_mmk2_s36_a36_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_5747:mmk2_s_g2.urdf
    galaxea_r1_lite_s14_a14_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_5167:r1_v2_1_0.urdf
    galaxea_r1_lite_s16_a18_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_970:r1_v2_1_0.urdf
    leju_robot_s118_a54_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_17897:leju_s54.urdf
    leju_robot_s54_a54_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_394:leju_s54.urdf
    yinhe_s49_a16_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_5452:galbot_one_golf.urdf
RoboMIND:
    franka_fr3_dual_master_puppet_joint_position_h5_franka_fr3_dual_real_s16_a16_fps30_cam_high_cam_left_cam_right_cam_top__episodes_1774:dual_fr3.urdf
    franka_panda_master_puppet_joint_position_h5_franka_3rgb_real_s8_a8_fps30_cam_left_cam_right_cam_top__episodes_17219：panda.urdf
    franka_sim_simulation_franka_joint_position_h5_sim_franka_3rgb_sim_s8_a8_fps30_cam_front_external_cam_handeye_cam_left_external_cam_right_external__episodes_14488：panda.urdf
    franka_sim_simulation_franka_joint_position_h5_simulation_sim_s8_a8_fps30_cam_front_external_cam_handeye_cam_left_external_cam_right_external__episodes_11422：panda.urdf
    franka_sim_simulation_franka_joint_position_h5_simulation_sim_s8_a8_fps30_cam_handeye_cam_left_external_cam_right_external__episodes_158：panda.urdf
    franka_sim_simulation_franka_joint_position_none_sim_s8_a8_fps30_cam_front_external_cam_handeye_cam_left_external_cam_right_external__episodes_222：panda.urdf
    tienkung_humanoid_master_puppet_joint_position_h5_tienkung_gello_1rgb_real_s16_a16_fps30_cam_top__episodes_6626：tienkung2_lite.urdf
    tienkung_humanoid_master_puppet_joint_position_h5_tienkung_prod1_gello_1rgb_real_s16_a16_fps30_cam_top__episodes_2959：tienkung2_lite.urdf
    ur5e_master_puppet_joint_position_h5_ur_1rgb_real_s7_a7_fps30_cam_top__episodes_26380:ur5e.urdf


## 实现说明（代码侧）

- 注册表：`src/openpi/cotrain/fk_eef.py`（dataset_id → URDF / 臂关节名 / EE link）
- 校验脚本：`scripts/validate_joint2eef_fk.py`（检查 URDF 存在、关节名、**joint 数量是否与 80D arm 映射一致**、零位 FK）
- 仅 **校验通过** 的数据集会在 80D 中用 FK 填充绝对 EEF（`xyz + yaw/pitch/roll`），并打开对应 `action_mask`
- 校验失败的（如 tienkung URDF 仅 4 臂关节 vs 映射 7、galaxea s16 映射 7 vs URDF 6）**不做修改**
- 训练 / light-norm 管线在 delta 之前调用 `DispatchFillEefFromFk`

### Norm 两种变体

```bash
export RLDS_DATA_DIR=/mnt/workspace/RLDS
export PYTHONPATH=src

# 1) 先校验 FK
uv run --group rlds python scripts/validate_joint2eef_fk.py

# 2a) anchor：FK 成功的数据集 + piper2 + piper30 + egoverse
uv run --group rlds python scripts/compute_fk_eef_norm_stats.py \
  --variant anchor \
  --output-assets-dir assets/cotrain_fk_eef_plus_piper_ego \
  --rlds-data-dir /mnt/workspace/RLDS

# 2b) full：全量 mix（有 URDF 的填 EEF，没有的保持原样）
uv run --group rlds python scripts/compute_fk_eef_norm_stats.py \
  --variant full \
  --output-assets-dir assets/cotrain_full_all_full_norm \
  --rlds-data-dir /mnt/workspace/RLDS
```

对应训练 config：`cotrain_fk_eef_plus_piper_ego` 与 `cotrain_full_all_full_norm`。
