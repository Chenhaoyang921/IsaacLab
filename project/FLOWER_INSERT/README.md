# FLOWER_INSERT（無階段對照版 / none-stage）

插花任務（抓取 → 對齊 → 插入）的 manager-based RL 專案。

這條 branch 是**無階段版本**，作為多階段獎勵塑形框架的 ablation baseline：

- 移除事件驅動狀態機（`_update_stage_state` / `_compute_stage_tag` / `_STAGE_BUDGETS`）
- 所有密集獎勵移除階段閘控，每步一起計算
- `complete_bonus` 改為各自條件首次成立時觸發一次，以全域剩餘步數（600 步基準）計分
- 瓶子固定在 `(0.4, 0.4, 0.0)`，沒有瞬移邏輯
- `dead_penalty` 系列恆為 0、階段逾時終止恆為 False（只保留 `time_out` 與 `task_success`）
- 動作平滑懲罰全程一致，沒有第二階段加權
- 移除 `stage_tag` 觀察值；夾爪由 RL 自由控制（無冷卻、無切換限額）

有階段的版本在 `manager-base` branch。獎勵項名稱刻意沿用 `s0_` / `s1_` / `s2_` 前綴，方便兩邊對照曲線。

## 目錄結構

```
project/FLOWER_INSERT/
├── run_train.ps1                 # 一鍵訓練
├── run_play.ps1                  # 一鍵播放 checkpoint
├── scripts/rsl_rl/               # train.py / play.py / cli_args.py
└── source/FLOWER_INSERT/
    ├── setup.py
    └── FLOWER_INSERT/tasks/manager_based/cabinet/
        ├── cabinet_env_cfg.py    # 場景 + MDP 設定
        ├── mdp/                  # actions / observations / rewards
        ├── config/franka/        # gym 註冊、joint_pos & IK-abs 版本、PPO cfg
        └── assets/               # franka.usd / BOTTLE.usd / Flower_*.usd（git-lfs）
```

## 任務 ID

| ID | 說明 |
| --- | --- |
| `Isaac-Flower-Insert-Franka-Custom-v0` | 關節位置控制（訓練） |
| `Isaac-Flower-Insert-Franka-Custom-Play-v0` | 關節位置控制（播放） |
| `Isaac-Flower-Insert-Franka-IK-Abs-Custom-v0` | IK 絕對位姿控制 |

任務 ID 與 `manager-base` 相同（同一時間只會有一條 branch 被 checkout），所以訓練 log 目錄請自行區分。

## 執行

```powershell
.\project\FLOWER_INSERT\run_train.ps1 --num_envs 2048 --max_iterations 1500
```

```powershell
.\project\FLOWER_INSERT\run_play.ps1
```

不需要 `pip install`：`scripts/rsl_rl/*.py` 會自行把 `source/FLOWER_INSERT` 加進 `sys.path`。
想要正式安裝的話：`pip install -e project/FLOWER_INSERT/source/FLOWER_INSERT`。
