import math
from functools import lru_cache

class Sigmoid:
    """计算 Sigmoid 函数的值，并使用缓存加速计算。"""
    @staticmethod
    @lru_cache(maxsize=1000)
    def f(
        x: float, 
        lower: float, 
        upper: float, 
        midpoint: float, 
        steepness: float
    ) -> float:
        """
        计算 Sigmoid 变换，用于动态调整增长和衰减速率。

        Args:
            x (float): 输入值。
            lower (float): 最小值。
            upper (float): 最大值。
            midpoint (float): 中间点。
            steepness (float): 陡峭度。

        Returns:
            float: Sigmoid 计算结果。
        """
        return lower + (upper - lower) / (1 + math.exp(-steepness * (x - midpoint)))
    
class Tanh:
    """计算 tanh 函数的值，并使用缓存加速计算。"""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def f(
        x: float, 
        lower: float, 
        upper: float, 
        midpoint: float, 
        steepness: float
    ) -> float:
        """
        计算 tanh 变换，用于动态调整增长和衰减速率。

        Args:
            x (float): 输入值。
            lower (float): 最小值。
            upper (float): 最大值。
            midpoint (float): 中间点。
            steepness (float): 陡峭度。

        Returns:
            float: tanh 计算结果。
        """
        z = steepness * (x - midpoint)
        return lower + (upper - lower) * (1 + math.tanh(z)) / 2