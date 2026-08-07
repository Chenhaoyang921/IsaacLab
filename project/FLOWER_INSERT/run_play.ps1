# FLOWER_INSERT 一鍵播放（載入訓練好的 checkpoint）腳本
# 用法:  .\project\FLOWER_INSERT\run_play.ps1 [--num_envs 16] [--checkpoint <path>] ...
#
# venv 選擇順序同 run_train.ps1（FLOWER_INSERT_VENV → 已啟動的 venv/conda → 預設路徑）

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
& ".\isaaclab.bat" -p "project\FLOWER_INSERT\scripts\rsl_rl\play.py" `
    --task Isaac-Flower-Insert-Franka-Custom-Play-v0 --num_envs 16 `
    @args
