# FLOWER_INSERT_DEXHAND

三階段插花任務（s0 抓取 → s1 對齊 → s2 插入）的 manager-based RL 專案，靈巧手版本。
與 `project/FLOWER_INSERT`（Franka 平行夾爪版本）完全獨立：獨立 Python 套件、獨立 gym 任務 ID、
獨立訓練腳本、獨立 USD 資產副本。場景（花朵、瓶子、桌面）、階段狀態機、獎勵設計皆複製自
`FLOWER_INSERT`，之後兩邊會各自演化。

機器人：**KUKA iiwa7（7 軸手臂）+ Allegro Hand（4 指 16 關節）**，取代原本的 Franka Panda + 平行夾爪。

## 目錄結構

```
project/FLOWER_INSERT_DEXHAND/
├── run_train.ps1                 # 一鍵訓練
├── run_play.ps1                  # 一鍵播放 checkpoint
├── scripts/rsl_rl/                # train.py / play.py / cli_args.py
└── source/FLOWER_INSERT_DEXHAND/
    ├── setup.py
    └── FLOWER_INSERT_DEXHAND/tasks/manager_based/cabinet/
        ├── cabinet_env_cfg.py    # 場景 + MDP 設定（機器人相關的欄位在 config/ 覆寫）
        ├── mdp/                  # actions / observations / rewards
        ├── config/kuka_allegro/  # gym 註冊、機器人替換、PPO cfg
        └── assets/                # BOTTLE.usd / Flower_*.usd / Sence_01.usd / cube.usd
```

沒有 `config/franka/`、沒有 `franka.usd`——這個專案只有 KUKA+Allegro 一種機器人。

## 任務 ID

| ID | 說明 |
| --- | --- |
| `Isaac-Flower-Insert-KukaAllegro-v0` | 關節位置控制（訓練） |
| `Isaac-Flower-Insert-KukaAllegro-Play-v0` | 關節位置控制（播放） |

## 跟 Franka 版本的差異

| | Franka | KUKA + Allegro |
| --- | --- | --- |
| 機器人 | `FRANKA_PANDA_LOCAL_CFG`（本地 USD） | `KUKA_ALLEGRO_CFG`（來自 Nucleus，需要網路） |
| `arm_action` | `panda_joint.*`（7 軸） | `iiwa7_joint_.*`（7 軸） |
| `gripper_action` | `panda_finger.*` 二指開合 | 16 個指關節，沿用二元開合語意（張開 / 握拳） |
| `ee_frame` 三個追蹤點 | `panda_hand` / 左指 / 右指 | `palm_link` / `index_biotac_tip` / `thumb_biotac_tip` |
| `contact_forces` | `panda_.*finger` | `(index\|thumb)_biotac_tip` |
| policy 觀察維度 | 67 | 73（多了 6 維指尖接觸力，見下） |

### 觸覺觀察值

`mdp.fingertip_contact_forces` 把 index / thumb 兩根指尖的接觸力（world frame，clip ±20N）餵進
policy 觀察值。Franka 版本沒有這一項——原本的設計是 policy 只能透過關節位置等間接線索推斷夾持狀態，
接觸力只用在 `is_catch()` 的獎勵判定，不進 observation。這裡是刻意加上的新行為，如果要跟 Franka
版本行為對齊，把 `joint_pos_env_cfg.py` 裡 `self.observations.policy.fingertip_contact` 那段拿掉即可。

### 已知需要重新校準的地方

`s0_grasp_flower`、`s0_lift_when_grasped` 兩個獎勵函式的判定邏輯（`open_joint_pos - joint_pos`、
`sum(joint_pos) < 0.03`）是照 Franka 平行夾爪的線性位移設計的。Allegro 是旋轉關節（弧度），
數值意義和量級都不同，`asset_cfg.joint_names` 雖然已經指向 Allegro 的 16 個指關節，但判定邏輯
本身還沒有針對多指靈巧手重寫，訓練前建議先檢視這兩項的實際輸出是否合理。

`KUKA_ALLEGRO_CFG` 預設 `disable_gravity=True`（沿用自 IsaacLab 官方的 in-hand manipulation 範例），
這個任務需要手臂抓花並舉起，可能需要改成 `False`。

握拳姿態（`close_command_expr`）的關節角度是初步估計，沒有針對花莖粗細校正過。

## 執行

需要能連上 Isaac Sim Nucleus 下載 `KukaAllegro/kuka.usd`（第一次執行會快取到本機）。

```powershell
.\project\FLOWER_INSERT_DEXHAND\run_train.ps1 --num_envs 1024 --max_iterations 2000
```

```powershell
.\project\FLOWER_INSERT_DEXHAND\run_play.ps1
```

不需要 `pip install`：`scripts/rsl_rl/*.py` 會自行把 `source/FLOWER_INSERT_DEXHAND` 加進 `sys.path`。
想要正式安裝的話：`pip install -e project/FLOWER_INSERT_DEXHAND/source/FLOWER_INSERT_DEXHAND`。
