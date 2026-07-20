# FRANKA_POWER 一鍵推論/觀看腳本（開 GUI）
# 用法:  .\project\FRANKA_POWER\run_play.ps1 [--num_envs 16] [--checkpoint <path>] ...
#
# venv 選擇順序：1. 環境變數 FRANKA_POWER_VENV  2. 已啟動好的 venv/conda  3. 預設路徑

Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:\CARB_APP_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\ISAAC_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\EXP_PATH -ErrorAction SilentlyContinue

if ($env:FRANKA_POWER_VENV) {
    $venv = $env:FRANKA_POWER_VENV
    $env:Path = "$venv\Scripts;" + $env:Path
    $env:VIRTUAL_ENV = $venv
} elseif ($env:VIRTUAL_ENV -or $env:CONDA_PREFIX) {
    # 已經啟動好環境了，直接使用，不覆蓋
} else {
    $venv = "C:\Users\James\Desktop\VENV\env_isaaclab"
    if (Test-Path $venv) {
        Write-Warning "使用預設 venv 路徑 ($venv)。在別的機器上請先設定環境變數 FRANKA_POWER_VENV，或執行前自行啟動你的 isaaclab venv/conda。"
        $env:Path = "$venv\Scripts;" + $env:Path
        $env:VIRTUAL_ENV = $venv
    } else {
        Write-Error "找不到 IsaacLab 的 Python 環境。請先啟動你的 venv/conda，或設定環境變數 FRANKA_POWER_VENV 指向已安裝 isaacsim+isaaclab 的 venv 根目錄。"
        exit 1
    }
}

$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $repoRoot
& ".\isaaclab.bat" -p "project\FRANKA_POWER\scripts\rsl_rl\play.py" `
    --task Franka-Power-Opt-v0 `
    @args
