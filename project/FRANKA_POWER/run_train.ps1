# FRANKA_POWER 一鍵訓練腳本
# 用法:  .\project\FRANKA_POWER\run_train.ps1 [--num_envs 2048] [--max_iterations 1500] ...
#
# 使用 C:\Users\James\Desktop\VENV\env_isaaclab（已 pip 安裝好 isaacsim + isaaclab 全套）。
# isaaclab.bat 會偵測到 PATH 上的這個 python 有 isaacsim-rl，自動採用它。

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
& ".\isaaclab.bat" -p "project\FRANKA_POWER\scripts\rsl_rl\train.py" `
    --task Franka-Power-Opt-v0 --headless `
    @args
