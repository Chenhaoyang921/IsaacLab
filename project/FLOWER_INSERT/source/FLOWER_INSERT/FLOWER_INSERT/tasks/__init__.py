# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""註冊所有任務（import 時自動向 gymnasium 註冊）。"""

import os

# Conveniences to other module directories via relative paths
FLOWER_INSERT_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

from .manager_based import *  # noqa: F401, F403
