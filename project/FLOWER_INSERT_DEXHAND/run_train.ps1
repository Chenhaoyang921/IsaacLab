# FLOWER_INSERT_DEXHAND 一鍵訓練腳本
# 用法:  .\project\FLOWER_INSERT_DEXHAND\run_train.ps1 [--num_envs 2048] [--max_iterations 1500] ...
#
# 需要一個已經 pip 安裝好 isaacsim + isaaclab 全套的 Python venv/conda 環境
# （官方安裝流程：https://isaac-sim.github.io/IsaacLab）。isaaclab.bat 會偵測
# PATH 上的 python 是否有 isaacsim-rl，自動採用它，不需要另外指定。
#
# venv 選擇順序（三選一，不用改這支腳本）：
#   1. 執行前先設定環境變數 FLOWER_INSERT_VENV 指向你的 venv 根目錄
#   2. 執行前自己先 Activate.ps1 啟動好 venv/conda（本腳本偵測到 VIRTUAL_ENV 或
#      CONDA_PREFIX 已設定就不會覆蓋）
#   3. 都沒有的話，退回這台機器上開發時使用的預設路徑（其他機器請用上面 1 或 2）

# 清掉同一個 shell session 裡可能殘留的舊環境變數
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:\CARB_APP_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\ISAAC_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\EXP_PATH -ErrorAction SilentlyContinue

if ($env:FLOWER_INSERT_VENV) {
    $venv = $env:FLOWER_INSERT_VENV
    $env:Path = "$venv\Scripts;" + $env:Path
    $env:VIRTUAL_ENV = $venv
} elseif ($env:VIRTUAL_ENV -or $env:CONDA_PREFIX) {
    # 已經啟動好環境了，直接使用，不覆蓋
} else {
    $venv = "C:\Users\James\Desktop\VENV\env_isaaclab"
    if (Test-Path $venv) {
        Write-Warning "使用預設 venv 路徑 ($venv)。在別的機器上請先設定環境變數 FLOWER_INSERT_VENV，或執行前自行啟動你的 isaaclab venv/conda。"
        $env:Path = "$venv\Scripts;" + $env:Path
        $env:VIRTUAL_ENV = $venv
    } else {
        Write-Error "找不到 IsaacLab 的 Python 環境。請先啟動你的 venv/conda，或設定環境變數 FLOWER_INSERT_VENV 指向已安裝 isaacsim+isaaclab 的 venv 根目錄。"
        exit 1
    }
}

$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $repoRoot
& ".\isaaclab.bat" -p "project\FLOWER_INSERT_DEXHAND\scripts\rsl_rl\train.py" `
    --task Isaac-Flower-Insert-KukaAllegro-v0 --headless `
    @args
