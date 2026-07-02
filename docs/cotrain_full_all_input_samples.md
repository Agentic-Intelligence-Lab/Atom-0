# cotrain_full_all 训练输入采样验证

生成时间：2026-07-01

本文件记录对 `cotrain_full_all` 中 43 个数据集逐一做单数据集模拟采样后的训练输入形态。采样方式：

- 使用 `openpi.cotrain.config.get_config("cotrain_full_all")` 的完整 dataset 列表。
- 对每个 `CotrainRLDSDataset` 临时替换为 `weight=1.0`，从 `train` split 抽取 `batch_size=1`。
- 使用训练配置的 `action_horizon=50`、`model.action_dim=64`、`image_resize_hw=(224, 224)`。
- 执行路径为 `CotrainRldsDataset -> StandardizedInputs`，用于验证 RLDS restructure 后进入训练 transform 的输入形态。
- 未在文档中展开 token id；pi0.5 的 tokenizer 会把文本渲染为 `... Task: <prompt>, State: <64 discretized bins>;\nAction:`。
- 归一化和 delta action 变换不改变本文关注的 prompt meta、shape 和 camera mask。

## 统一输入结构

每条样本在进入模型 tokenization 前的核心字段如下：

```python
{
    "image": {
        "base_0_rgb": uint8[224, 224, 3],
        "left_wrist_0_rgb": uint8[224, 224, 3],
        "right_wrist_0_rgb": uint8[224, 224, 3],
    },
    "image_mask": {
        "base_0_rgb": bool,
        "left_wrist_0_rgb": bool,
        "right_wrist_0_rgb": bool,
    },
    "state": float32[64],
    "actions": float32[50, 64],
    "prompt": str,
    "prompt_prefix": str,
}
```

文本最终进入 pi0.5 tokenizer 的形态：

```text
{prompt_prefix}Task: {prompt}, State: <64 discretized bins>;
Action:
```

其中：

- joint action 数据：`prompt_prefix = "Action Mode: joint. "`
- eef action 数据：`prompt_prefix = "Action Mode: eef. EEF Frame: {cartesian_frame}. "`

## 采样总览

所有 43 个数据集均采样成功。

| # | dataset_id | adapter | native dim | image mask B/L/R | prompt_prefix | sampled prompt |
|---:|---|---|---:|---|---|---|
| 1 | `agibot` | `agibot` | 20 | T/T/T | `Action Mode: joint.` | Pickup items in the supermarket |
| 2 | `droid` | `three_cam_task` | 8 | T/T/T | `Action Mode: joint.` | Hang the black t-shirt on the black frame |
| 3 | `egoverse_aria` | `egoverse_full` | 12 | T/F/F | `Action Mode: eef. EEF Frame: obs_head_pose.` | interview episodes |
| 4 | `egoverse_eva` | `egoverse_full` | 12 | T/T/T | `Action Mode: eef. EEF Frame: source_pose_frame.` | fold the t-shirt |
| 5 | `egoverse_human` | `egoverse_full` | 12 | T/F/F | `Action Mode: eef. EEF Frame: obs_head_pose.` | folding clothes |
| 6 | `egoverse_mecka` | `egoverse_full` | 12 | T/F/F | `Action Mode: eef. EEF Frame: obs_head_pose.` | The person is carefully constructing a paper item by aligning two sheets of paper... |
| 7 | `egoverse_scale` | `egoverse_full` | 12 | T/F/F | `Action Mode: eef. EEF Frame: obs_head_pose.` | Pack a suitcase for a trip |
| 8 | `piper30` | `three_cam_task` | 14 | T/T/T | `Action Mode: joint.` | hang the purple cup on the mug rack |
| 9 | `robocoin_agilex_cobot_magic_s26_a26` | `robocoin` | 26 | T/T/T | `Action Mode: joint.` | the gripper move the object. |
| 10 | `robocoin_airbot_mmk2_s36_a36` | `robocoin` | 36 | T/T/T | `Action Mode: joint.` | pick up the lid with both hands and cover the box. |
| 11 | `robocoin_galaxea_r1_lite_upper_s14_a14` | `robocoin` | 14 | T/T/T | `Action Mode: joint.` | use a gripper to pick up the cup and pour the powder into a bowl or tray. |
| 12 | `robocoin_realman_rmc_aida_l_s28_a28` | `robocoin` | 28 | T/T/T | `Action Mode: joint.` | the left gripper grasp the basket on the table, the right grippe pick up the towel... |
| 13 | `robocoin_unitree_g1_dex3_s28_a28` | `robocoin` | 28 | T/T/T | `Action Mode: joint.` | Pick up the red cup on the table. |
| 14 | `robocoin_agilex_decoupled_magic_s14_a14_fps30` | `robocoin` | 14 | T/T/T | `Action Mode: joint.` | put the steamed buns into the plate on the white-black tablecloth. |
| 15 | `robocoin_agilex_decoupled_magic_s14_a14_fps50` | `robocoin` | 14 | T/T/T | `Action Mode: joint.` | move the plate that can hold fruits to the right side of the table. |
| 16 | `robocoin_agilex_decoupled_magic_s26_a26` | `robocoin` | 26 | T/T/T | `Action Mode: joint.` | pick up the peach and put it into the basket. |
| 17 | `robocoin_aloha_s26_a26` | `robocoin` | 26 | T/T/T | `Action Mode: joint.` | Make sandwiches |
| 18 | `robocoin_alpha_bot_2_s28_a28` | `robocoin` | 28 | T/T/T | `Action Mode: joint.` | after avoiding the water bottle on the table, press the button. |
| 19 | `robocoin_discover_aitbot_mmk2_s36_a36` | `robocoin` | 36 | T/T/T | `Action Mode: joint.` | pick up the calculator box from the left side of the table with your left hand and pass it... |
| 20 | `robocoin_galaxea_r1_lite_s14_a14` | `robocoin` | 14 | T/T/T | `Action Mode: joint.` | move pillow from bedside to head then foot of bed. |
| 21 | `robocoin_galaxea_r1_lite_s16_a18` | `robocoin` | 16 | T/T/T | `Action Mode: joint.` | put the apple in the designated position. |
| 22 | `robocoin_leju_robot_s118_a54` | `robocoin` | 54 | T/T/T | `Action Mode: joint.` | take the parts from the cabinet and place them on the table. |
| 23 | `robocoin_leju_robot_s54_a54` | `robocoin` | 54 | T/T/T | `Action Mode: joint.` | take out the card place it on the sensor for recognition and then transfer it. |
| 24 | `robocoin_realman_rmc_aidal_s28_a28` | `robocoin` | 28 | T/T/T | `Action Mode: joint.` | pick up the peach from the table and place it in the brown handbag. |
| 25 | `robocoin_ruantong_a2d_s17_a17` | `robocoin` | 17 | T/T/T | `Action Mode: joint.` | take the parts from the box containing them and put them into an empty box. |
| 26 | `robocoin_ruantong_a2d_s41_a34` | `robocoin` | 34 | T/T/T | `Action Mode: joint.` | using a robotic arm to pick up a cardboard box and palace it the carton. |
| 27 | `robocoin_unitree_g1_s28_a28_high` | `robocoin` | 28 | T/F/F | `Action Mode: joint.` | pick up the rabbit doll from the table and place it on the plate. |
| 28 | `robocoin_unitree_g1_s28_a28` | `robocoin` | 28 | T/T/T | `Action Mode: joint.` | the left hand pick up the doll on the left side of the plate and place it into the plate. |
| 29 | `robocoin_unknown_s30_a30_high` | `robocoin` | 28 | T/F/F | `Action Mode: joint.` | Place the lemon |
| 30 | `robocoin_yinhe_s49_a16` | `robocoin` | 16 | T/T/T | `Action Mode: joint.` | take steamed baozi put them in steamer then cover it. |
| 31 | `robomind_agilex_cobot_magic_s14_a14` | `robomind_full` | 14 | T/T/T | `Action Mode: joint.` | take the egg out from the steamer,put the egg on the plate,close the lid |
| 32 | `robomind_franka_fr3_dual_s16_a16` | `robomind_full` | 16 | T/T/T | `Action Mode: joint.` | pick gray plate from plate low rack and place it on table with left arm |
| 33 | `robomind_franka_panda_s8_a8` | `robomind_full` | 8 | T/T/T | `Action Mode: joint.` | Close the drawer by pulling from the side |
| 34 | `robomind_franka_sim_franka_s8_a8` | `robomind_full` | 8 | T/T/T | `Action Mode: joint.` | Take strawberry from the bowl |
| 35 | `robomind_franka_sim_simulation_s8_a8` | `robomind_full` | 8 | T/T/T | `Action Mode: joint.` | Grab the oven door handle with one arm, and open the oven door with one arm |
| 36 | `robomind_franka_sim_simulation_no_front_s8_a8` | `robomind_full` | 8 | T/T/T | `Action Mode: joint.` | Pick up everything from the table with one arm, and put everything in the basket with one arm |
| 37 | `robomind_franka_sim_none_s8_a8` | `robomind_full` | 8 | T/T/T | `Action Mode: joint.` | put_apple_from_plate_into_bowl_with_five_others |
| 38 | `robomind_tienkung_gello_s16_a16` | `robomind_full` | 16 | T/F/F | `Action Mode: joint.` | Rotate green pot handle |
| 39 | `robomind_tienkung_prod1_gello_s16_a16` | `robomind_full` | 16 | T/F/F | `Action Mode: joint.` | clean table; pick rubbish on the table and place in the dustbin |
| 40 | `robomind_tienkung_xsens_s14_a14` | `robomind_full` | 14 | T/F/F | `Action Mode: joint.` | pick up the cylinder from the cube,place it in the box,flap close box |
| 41 | `robomind_tienkung_sim_s38_a38` | `robomind_full` | 38 | T/T/F | `Action Mode: joint.` | pick and place 02 |
| 42 | `robomind_tienkung_real_s38_a38` | `robomind_full` | 38 | T/T/F | `Action Mode: joint.` | place_cup_on_blue_bowl_with_right_hand_and_five_others |
| 43 | `robomind_ur5e_s7_a7` | `robomind_full` | 7 | T/F/F | `Action Mode: joint.` | insert the flowers into the vase |

## 代表性样例

### Joint Action

Dataset: `piper30`

```text
Action Mode: joint. Task: hang the purple cup on the mug rack, State: <64 discretized bins>;
Action:
```

Tensor/mask:

```text
state:   float32[64]
actions: float32[50, 64]
image:   three uint8[224,224,3] slots
mask:    base=true, left_wrist=true, right_wrist=true
```

### EEF Action

Dataset: `egoverse_eva`

```text
Action Mode: eef. EEF Frame: source_pose_frame. Task: fold the t-shirt, State: <64 discretized bins>;
Action:
```

Tensor/mask:

```text
state:   float32[64]
actions: float32[50, 64]
image:   three uint8[224,224,3] slots
mask:    base=true, left_wrist=true, right_wrist=true
```

Dataset: `egoverse_aria`

```text
Action Mode: eef. EEF Frame: obs_head_pose. Task: interview episodes, State: <64 discretized bins>;
Action:
```

Tensor/mask:

```text
state:   float32[64]
actions: float32[50, 64]
image:   three uint8[224,224,3] slots
mask:    base=true, left_wrist=false, right_wrist=false
```

## 结论

- `cotrain_full_all` 当前 43 个 dataset 都可以被单数据集模拟采样。
- 训练文本输入已包含上一轮加入的 action meta。
- 所有非 EgoVerse 数据在当前训练 action 选择下都是 `Action Mode: joint`。
- EgoVerse 全部为 `Action Mode: eef`，并正确带入 `cartesian_frame`，本次采样出现的 frame 为 `obs_head_pose` 或 `source_pose_frame`。
- 缺失 wrist 相机的数据仍会提供占位图像，但对应 `image_mask` 为 `false`，训练时模型可以区分真实相机和占位相机。
