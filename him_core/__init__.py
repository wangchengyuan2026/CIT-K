"""him_core —— HIM（猫智能论）核心模型

模块对应论文给出的项目结构：

    grid_env.py          100×100 离散网格环境（论文 §3.1 实验环境）
    homeostasis.py       舒适度函数 L_t、感官经验 W_t、预测期望 X_t、希望机制
    motivation_engine.py ΔL 内生奖励 + 动作选择（HIM 完整模型）
    state_machine.py     【A 版 2018】离散性质体状态机（猫智能论状态机逻辑）
"""

from .grid_env import ComfortGrid, ACTION_NAMES, N_ACTIONS
from .homeostasis import Homeostasis
from .motivation_engine import HIMAgent
from .state_machine import CI2018Agent

__all__ = [
    "ComfortGrid", "ACTION_NAMES", "N_ACTIONS",
    "Homeostasis", "HIMAgent", "CI2018Agent",
]
