# FLOWER_INSERT

三階段插花任務（s0 抓取 → s1 對齊 → s2 插入）的 manager-based RL 專案，
與 `project/FRANKA_POWER` 完全獨立：獨立 Python 套件、獨立 gym 任務 ID、獨立訓練腳本。

程式碼來源：`myfork/flower-insert-updates` 的 commit `f1f3a85367`
（*tune: S1 對齊角度門檻收緊至 10 度、瓶內基準點調整為 0.4m*）。

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
        └── assets/               # franka.usd / BOTTLE.usd / Flower_*.usd
```

## 任務 ID

| ID | 說明 |
| --- | --- |
| `Isaac-Flower-Insert-Franka-Custom-v0` | 關節位置控制（訓練） |
| `Isaac-Flower-Insert-Franka-Custom-Play-v0` | 關節位置控制（播放） |
| `Isaac-Flower-Insert-Franka-IK-Abs-Custom-v0` | IK 絕對位姿控制 |

## 執行

```powershell
.\project\FLOWER_INSERT\run_train.ps1 --num_envs 2048 --max_iterations 1500
```

```powershell
.\project\FLOWER_INSERT\run_play.ps1
```

不需要 `pip install`：`scripts/rsl_rl/*.py` 會自行把 `source/FLOWER_INSERT` 加進 `sys.path`。
想要正式安裝的話：`pip install -e project/FLOWER_INSERT/source/FLOWER_INSERT`。

## USD 資產

`assets/` 內的 `franka.usd`、`BOTTLE.usd`、`Flower_01.usd` 由 `project/FRANKA_DRAWER` 複製而來。
原 commit 引用的是 **`Flower_02.usd`**，該檔案未進 git，本 repo 目前沒有；
`cabinet_env_cfg.py` 找不到時會印訊息並退回 `Flower_01.usd`。
把 `Flower_02.usd` 放進 `assets/` 後就會自動改用它。
