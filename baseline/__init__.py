"""baseline —— 对照基线

    fep_model.py   标准自由能原理主动推理（最小化惊奇度）
    rl_baseline.py 表格型 Q 学习（外部稀疏奖励）+ 均匀随机游走参考
"""

from .fep_model import FEPAgent
from .rl_baseline import RLAgent, RandomAgent

__all__ = ["FEPAgent", "RLAgent", "RandomAgent"]
