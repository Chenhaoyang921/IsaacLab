# FRANKA_POWER — 機械手臂最小功率（電力）最佳化訓練框架

用 RL 找出「**每個關節最少需要多大的輸出功率（力矩上限）**，才能扛著指定重量的物品，
沿著使用者定義的多個路徑點走完全程」。

## 核心概念

| 元件 | 角色 |
|---|---|
| 差分 IK (`DifferentialIKController`) | 負責軌跡：自動帶著末端依序走過所有路徑點，**Agent 不控制軌跡** |
| Agent（PPO） | 每一步輸出 7 個關節的**力矩上限**（正規化 [-1,1] → [MIN, MAX] Nm） |
| `IdealPDActuator`（顯式致動器） | PD 律算出需求力矩後，被 Agent 給的上限**截斷**——上限太小手臂就撐不住、跟不上軌跡 |
| 獎勵函數 | 任務進度（到點加分）−「電能消耗」（機械功率 + 銅損發熱）− 功率上限開太大的懲罰 |
| 時間機制 | 每個路徑點有各自的時間預算：超時扣 `pen_timeout`；走完全程時依剩餘時間比例再加碼 `rew_speed_bonus_weight`，越快完成加越多 |

電力模型：`P = Σ|τ·ω|（機械功率） + heat_coeff·Στ²（銅損：靜態持力也要耗電）`，
每步扣 `P × dt` 焦耳對應的獎勵。因此 Agent 被逼著在「給的力不夠 → 掉點/失敗」與
「給的力太多 → 電力浪費」之間找出最佳平衡，這就是**最佳輸出功率**。

## 使用者要填的參數

全部集中在
[franka_power_env_cfg.py](source/FRANKA_POWER/FRANKA_POWER/tasks/direct/franka_power/franka_power_env_cfg.py)
檔案最上方：

```python
# 1. 路徑點（機器人基座座標系，單位公尺）
WAYPOINTS = [
    [0.50,  0.00, 0.35],                       # 只給位置（姿態預設夾爪朝下）
    [0.50,  0.30, 0.55],
    [0.45, -0.30, 0.55, 0.0, 1.0, 0.0, 0.0],   # 也可加四元數 (qw,qx,qy,qz) 指定姿態
    [0.55,  0.00, 0.70],
]

# 每個路徑點各自的時間預算（秒），長度必須跟 WAYPOINTS 一樣。
# 超過對應的時間還沒到達該點 -> 判定失敗，扣 pen_timeout 分並結束該 episode。
# 距離遠/需要繞路的點可以給多一點時間。
WAYPOINT_TIMEOUT_S = [3.0, 3.0, 3.0, 3.0]

# 2. 末端模擬物品的重量 (kg)，慣量會依比例增加（Franka 額定荷重 3 kg）
PAYLOAD_MASS = 1.0

# 3. 各關節力矩上限的搜尋範圍（Agent 在此範圍內選擇）
EFFORT_LIMIT_MIN = [0.0]*7
EFFORT_LIMIT_MAX = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]  # 官方額定值
```

其他可調參數（獎勵權重、到點容忍距離、episode 長度、`heat_coeff` 等）都在同檔案的
`FrankaPowerEnvCfg` 內，每項都有註解。

## 訓練（本機一鍵腳本，已驗證可用）

這台機器用 `C:\Users\James\Desktop\VENV\env_isaaclab` 這個 pip 安裝好的完整
Isaac Sim + IsaacLab venv。啟動腳本會把它的 `Scripts` 加到 PATH 最前面，
`isaaclab.bat` 就會自動偵測並使用這個 python：

```powershell
# 在 IsaacLab 根目錄
.\project\FRANKA_POWER\run_train.ps1 --num_envs 2048            # 訓練（headless）
.\project\FRANKA_POWER\run_play.ps1  --num_envs 16              # 觀看結果（GUI）
```

已用 3 個迭代 × 32 環境實測跑通全流程（IK 追蹤、力矩上限動態改寫、
TensorBoard 記錄 `EffortLimit_Nm/*`、checkpoint 存檔）。

也可以自己手動啟動 venv 後直接下官方指令：

```powershell
& C:\Users\James\Desktop\VENV\env_isaaclab\Scripts\Activate.ps1
cd C:\Users\James\IsaacLab
.\isaaclab.bat -p project\FRANKA_POWER\scripts\rsl_rl\train.py --task Franka-Power-Opt-v0 --num_envs 2048 --headless
.\isaaclab.bat -p project\FRANKA_POWER\scripts\rsl_rl\play.py --task Franka-Power-Opt-v0 --num_envs 16
```

`play.py` 每秒會印出 Agent 當下選擇的各關節力矩上限與平均消耗功率：

```
[力矩上限 Nm] joint1=23.4, joint2=41.2, ... | 平均功率 18.3 W | 平均路徑點進度 2.1
```

## 怎麼讀出「最佳輸出功率」？

```bat
.\isaaclab.bat -p -m tensorboard.main --logdir logs\rsl_rl\franka_power_opt
```

TensorBoard 中重點曲線：

- **`EffortLimit_Nm/panda_joint1 ~ 7`** — 每個關節在 episode 內的平均力矩上限。
  訓練收斂後，這 7 條曲線的穩定值就是「完成任務所需的最小輸出功率」答案。
- `Metrics/energy_J` — 每趟任務消耗的總電能（焦耳），應隨訓練逐漸下降。
- `Metrics/success_rate`、`Metrics/waypoints_reached` — 任務完成度，應維持高值。
- `Metrics/mean_power_W` — 平均瞬時功率。

## 專案結構

```
project/FRANKA_POWER/
├── README.md
├── scripts/rsl_rl/
│   ├── train.py          # 訓練
│   ├── play.py           # 推論 + 即時印出功率設定
│   └── cli_args.py
└── source/FRANKA_POWER/
    ├── setup.py
    └── FRANKA_POWER/tasks/direct/franka_power/
        ├── __init__.py               # gym 註冊 (Franka-Power-Opt-v0)
        ├── franka_power_env_cfg.py   # ★ 使用者填路徑點 / 重量 / 功率範圍
        ├── franka_power_env.py       # 環境本體（IK + 動態力矩上限 + 功率獎勵）
        └── agents/rsl_rl_ppo_cfg.py  # PPO 超參數
```

## 常見調整

- **手臂一直失敗（撐不住）**：`PAYLOAD_MASS` 太重或 `pen_energy_weight` 太大，
  先把 `pen_energy_weight` 降到 0.01 訓練成功後再逐步加大。
- **Agent 總是開滿功率**：加大 `pen_energy_weight` 或 `pen_effort_limit_weight`。
- **路徑點太遠走不完**：加大 `episode_length_s`。
- **Agent 為了省電把速度壓到過慢**：`WAYPOINT_TIMEOUT_S`（每點各自預設 3 秒）就是為此設計的——
  超過對應路徑點的時間還沒到，視為失敗、扣 `pen_timeout`（預設 100 分）並結束該 episode，
  逼 Agent 在「省電」跟「不能太慢」之間取捨。TensorBoard 上可看 `Metrics/timeout_rate`
  確認超時發生的頻率；卡在某個點一直超時，可以把那個點的時間預算調寬鬆一點，
  或代表該路徑點附近的關節功率被壓太低，需要調高 `EFFORT_LIMIT_MIN`。
- **想讓 Agent 更積極求快、不只是「不要超時」**：`rew_speed_bonus_weight`（預設 100）
  在走完全程那一刻，依「剩餘時間比例」給一次性加分——`bonus = weight × (1 - 已用步數/最大步數)`，
  越快走完全程加越多，壓線完成加越少，跟 `pen_timeout` 形成對稱的獎懲。TensorBoard 上看
  `Episode_Reward/speed_bonus`；想更強調速度就調高這個權重，想更強調省電就調低。
- **離線環境（無法連 Nucleus 下載 Franka USD）**：參考 `FRANKA_DRAWER` 專案的做法，
  把 `FRANKA_POWER_ROBOT_CFG.spawn.usd_path` 改成本地 `franka.usd` 路徑。
