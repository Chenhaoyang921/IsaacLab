# FRANKA_POWER 一鍵推論/觀看腳本（開 GUI）
# 用法:  .\project\FRANKA_POWER\run_play.ps1 [--num_envs 16] [--checkpoint <path>] ...

# 清掉同一個 shell session 裡可能殘留的舊環境變數
# （例如之前跑過舊版腳本，指向 Program Files 的 Isaac Sim standalone）
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:\CARB_APP_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\ISAAC_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\EXP_PATH -ErrorAction SilentlyContinue

$venv = "C:\Users\James\Desktop\VENV\env_isaaclab"
$env:Path = "$venv\Scripts;" + $env:Path
$env:VIRTUAL_ENV = $venv

Set-Location "C:\Users\James\IsaacLab"
& ".\isaaclab.bat" -p "project\FRANKA_POWER\scripts\rsl_rl\play.py" `
    --task Franka-Power-Opt-v0 `
    @args
