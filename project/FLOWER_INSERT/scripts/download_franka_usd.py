"""
下載 Franka Panda USD 到本地 assets 資料夾。

使用方式（在 IsaacLab 根目錄）：
    isaaclab.bat -p project/FLOWER_INSERT/scripts/download_franka_usd.py
"""

import os
import omni.client
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# ── 目標路徑 ──────────────────────────────────────────────────────────────
NUCLEUS_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
LOCAL_DIR    = os.path.join(
    os.path.dirname(__file__), "..",
    "source", "FLOWER_INSERT", "FLOWER_INSERT",
    "tasks", "manager_based", "cabinet", "assets"
)
LOCAL_PATH = os.path.normpath(os.path.join(LOCAL_DIR, "panda_instanceable.usd"))

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print(f"[下載] 來源：{NUCLEUS_PATH}")
    print(f"[下載] 目標：{LOCAL_PATH}")

    # 讀取 Nucleus 上的 USD bytes
    result, version, content = omni.client.read_file(NUCLEUS_PATH)

    if result != omni.client.Result.OK:
        raise RuntimeError(f"無法讀取 Nucleus 上的檔案，錯誤碼：{result}")

    # 寫入本地檔案
    with open(LOCAL_PATH, "wb") as f:
        f.write(memoryview(content))

    print(f"[完成] 已儲存至：{LOCAL_PATH}  ({os.path.getsize(LOCAL_PATH):,} bytes)")


if __name__ == "__main__":
    main()
