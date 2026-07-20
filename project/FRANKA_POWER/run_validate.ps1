# FRANKA_POWER 機械手臂設定驗證（開 GUI，只 spawn 機器人本身，不含任務/RL）
# 用法:  .\project\FRANKA_POWER\run_validate.ps1 [--payload_mass 3.0]

Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:\CARB_APP_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\ISAAC_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\EXP_PATH -ErrorAction SilentlyContinue

$venv = "C:\Users\James\Desktop\VENV\env_isaaclab"
$env:Path = "$venv\Scripts;" + $env:Path
$env:VIRTUAL_ENV = $venv

Set-Location "C:\Users\James\IsaacLab"
& ".\isaaclab.bat" -p "project\FRANKA_POWER\scripts\validate_robot.py" @args
