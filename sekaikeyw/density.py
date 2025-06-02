import asyncio
import math
import numpy as np
from collections import defaultdict, deque
from functools import lru_cache
from enum import Enum
import time
from typing import Deque, Optional, Dict
from ._math import Sigmoid

# --------------------- 基础工具类（无需异步） ---------------------


class DensityStrategy(Enum):
    AUTO = "auto"
    EXPONENTIAL_MOVING_AVERAGE = "ema"
    SIMPLE_MOVING_AVERAGE = "sma"
    WEIGHTED_MOVING_AVERAGE = "wma"
    EXPONENTIALLY_WEIGHTED_MOVING_AVERAGE = "ewma"

# --------------------- 数据结构优化：时间轮替代deque ---------------------
class TimeWheel:
    """时间轮数据结构，用于高效管理时间窗口"""
    def __init__(self, window_size: int, time_window: float):
        self.slots: Dict[float, Deque[float]] = defaultdict(deque)
        self.window_size = window_size
        self.time_window = time_window
    
    def add(self, timestamp: float) -> None:
        slot_key = math.floor(timestamp / self.time_window) if self.time_window else 0
        self.slots[slot_key].append(timestamp)
        if len(self.slots[slot_key]) > self.window_size:
            self.slots[slot_key].popleft()
    
    def prune(self, current_time: float) -> None:
        expired = [k for k in self.slots if (current_time - k * self.time_window) > self.time_window]
        for k in expired:
            del self.slots[k]
    
    def get_all(self) -> Deque[float]:
        all_ts = deque()
        for slot in sorted(self.slots.keys()):
            all_ts.extend(self.slots[slot])
        return all_ts

# --------------------- 异步计算模块 ---------------------
class WeightedAverageCalculator:
    """
    计算指数加权移动平均（EWMA），用于平滑时间戳数据，减少短期波动影响。
    """
    _cached_weights = {}

    def __init__(self, adaptive_strength: float = 0.5):
        """
        初始化加权平均计算器。
        
        Args:
            adaptive_strength (float): 自适应强度，范围 0~1，值越大，越强调密度适应性。
        """
        self.adaptive_strength = adaptive_strength

    def compute(
        self,
        timestamps: deque[float],
        smoothing_factor: float = 0.9
    ) -> float:
        """
        计算指数加权平均时间间隔，并加入自适应调整。
        
        Args:
            timestamps (deque[float]): 事件发生的时间戳。
            smoothing_factor (float): 平滑因子，决定历史数据的衰减速度。

        Returns:
            float: 计算得到的加权平均时间间隔，具有自适应特性。
        """
        num_intervals = len(timestamps) - 1
        if num_intervals <= 0:
            return 100.0

        # 计算指数加权平均
        if num_intervals not in self._cached_weights:
            weights = np.exp(-smoothing_factor * np.linspace(0, num_intervals - 1, num_intervals)[::-1])
            self._cached_weights[num_intervals] = weights
        weights = self._cached_weights[num_intervals]
        weighted_avg = np.dot(weights, np.diff(timestamps)) / np.sum(weights)
        weighted_avg = max(weighted_avg, 1e-9)

        return weighted_avg


class DynamicDecayCalculator:
    """
    计算基于滑动窗口趋势的动态衰减因子。
    """
    @staticmethod
    def compute(avg_interval: float, decay_factor: float) -> float:
        """
        计算衰减因子，用于动态调整密度衰减速率。

        Args:
            avg_interval (float): 平均时间间隔。
            decay_factor (float): 衰减因子。

        Returns:
            float: 计算得到的衰减值。
        """
        dynamic_decay = decay_factor * Sigmoid.f(
            avg_interval, lower=0.5, upper=1.5, midpoint=6, steepness=1.5
        )
        return math.exp(-0.7 * dynamic_decay * avg_interval)

# --------------------- 策略选择与密度计算 ---------------------
class DensityCalculator:
    """
    计算密度值，支持多种计算策略，并附带 `AUTO` 模式。
    """
    def __init__(
        self,
        *,
        strategy: DensityStrategy,
        ema_alpha: float,
        decay_factor: float,
        auto_threshold_low: float = 4,
        auto_threshold_high: float = 10
    ):
        self.strategy = strategy
        self.ema_alpha = ema_alpha
        self.decay_factor = decay_factor
        self.auto_threshold_low = auto_threshold_low
        self.auto_threshold_high = auto_threshold_high
        self.recent_avg_intervals = deque(maxlen=5)
    
    def _auto_select_strategy(self, avg_interval: float) -> DensityStrategy:
        """
        更智能的 `AUTO` 选择策略，结合 `avg_interval` 的变化趋势。
        
        Args:
            avg_interval (float): 计算得到的加权平均时间间隔。
        
        Returns:
            DensityCalculationStrategy: 选择的最佳密度计算策略。
        """
        self.recent_avg_intervals.append(avg_interval)
        if len(self.recent_avg_intervals) < 2:
            trend = avg_interval
        else:
            trend = np.mean(self.recent_avg_intervals)
            # 增强：计算标准差辅助决策
            std_dev = np.std(self.recent_avg_intervals)
            if std_dev > 2:
                return DensityStrategy.EXPONENTIALLY_WEIGHTED_MOVING_AVERAGE
        
        if trend < self.auto_threshold_low:
            return DensityStrategy.EXPONENTIALLY_WEIGHTED_MOVING_AVERAGE
        elif trend < self.auto_threshold_high:
            return DensityStrategy.WEIGHTED_MOVING_AVERAGE
        else:
            return DensityStrategy.SIMPLE_MOVING_AVERAGE
    
    def calculate(
            self, 
            *,
            prev_weight: float, 
            density_increment: float, 
            decay_factor: float, 
            avg_interval: float, 
        ) -> float:
        """
        根据选定的策略计算密度。
        
        Args:
            prev_weight (float): 之前的密度权重。
            density_increment (float): 计算出的增长量。
            decay_factor (float): 计算出的衰减因子。
            avg_interval (float): 计算得到的加权平均时间间隔。
        
        Returns:
            float: 计算得到的密度权重。
        """
        strategy = self._auto_select_strategy(avg_interval) if self.strategy == DensityStrategy.AUTO else self.strategy
        
        if strategy == DensityStrategy.EXPONENTIAL_MOVING_AVERAGE:
            result = self.ema_alpha * prev_weight + (1 - self.ema_alpha) * density_increment * decay_factor
        elif strategy == DensityStrategy.SIMPLE_MOVING_AVERAGE:
            result = np.mean([prev_weight, density_increment * decay_factor])
        elif strategy == DensityStrategy.WEIGHTED_MOVING_AVERAGE:
            result = 0.6 * prev_weight + 0.4 * density_increment * decay_factor
        elif strategy == DensityStrategy.EXPONENTIALLY_WEIGHTED_MOVING_AVERAGE:
            result = self.ema_alpha * prev_weight + (1 - self.ema_alpha) * density_increment
        else:
            raise ValueError("Unsupported density calculation strategy")
        
        return result * DynamicDecayCalculator.compute(avg_interval, self.decay_factor)

# --------------------- 异步主管理类 ---------------------
class AsyncConditionDensityManager:
    """
    管理不同条件的密度权重，基于时间窗口计算，并支持不同的密度计算策略。
    
    主要功能：
    - 维护各个 `condition_key` 的时间戳记录。
    - 计算加权平均间隔 (`avg_interval`)。
    - 根据 `DensityCalculationStrategy` 选择最优密度计算方式。
    - 适用于流量控制、异常检测等应用场景。
    - 注意，本项目计算的结果值与密度实际值呈非线性关系，旨在对密度突变敏感。
    """
    def __init__(
        self,
        *,
        window_size: int = 20,
        decay_factor: float = 0.1,
        scaling_factor: int = 10,
        avg_smoothing_factor: float = 0.9,
        ema_alpha: float = 0.8,
        time_window: Optional[float] = None,
        density_strategy: DensityStrategy = DensityStrategy.AUTO,
        adaptive_strength: float = 0.9,
        auto_threshold_low: float = 4,
        auto_threshold_high: float = 10
    ):
        """
        Args:
            window_size (int): 维护时间窗口内的最大事件数。
            decay_factor (float): 衰减因子，控制密度变化速率。
            scaling_factor (int): 计算 `get_density_weight()` 时的缩放因子。
            avg_smoothing_factor (float): 平滑因子，用于计算 `avg_interval`。
            ema_alpha (float): 指数加权移动平均的平滑因子。
            time_window (float, optional): 事件的时间窗口，超出此窗口的数据将被清理。
            density_strategy (DensityCalculationStrategy): 密度计算策略，默认 `AUTO` 自动选择。
            adaptive_strength (float): 自适应强度，范围 0~1，值越大，越强调密度适应性。
        """
        self.condition_weights: Dict[str, float] = defaultdict(float)
        self.condition_timestamps: Dict[str, TimeWheel] = defaultdict(
            lambda: TimeWheel(window_size, time_window if time_window else 0)
        )
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.scaling_factor = scaling_factor * 100
        self.avg_smoothing_factor = avg_smoothing_factor
        self.time_window = time_window
        self.lock = asyncio.Lock()  # 异步锁
        
        self.density_calculator = DensityCalculator(
            strategy=density_strategy, 
            ema_alpha=ema_alpha, 
            decay_factor=decay_factor,
            auto_threshold_high=auto_threshold_low, 
            auto_threshold_low=auto_threshold_high
        )
        self.weighted_average_calculator = WeightedAverageCalculator(adaptive_strength)
        
        # 启动异步清理任务
        if time_window:
            asyncio.create_task(self.auto_prune(interval=60.0))
    
    async def auto_prune(self, interval: float = 60.0):
        """异步定时清理"""
        while True:
            await asyncio.sleep(interval)
            async with self.lock:
                current_time = time.time()
                for key in list(self.condition_timestamps.keys()):
                    self.condition_timestamps[key].prune(current_time)
                    if not self.condition_timestamps[key].get_all():
                        del self.condition_timestamps[key]
                        del self.condition_weights[key]
    
    async def update_condition_density(
        self, 
        condition_key: str, 
        current_time: Optional[float] = None
    ) -> None:
        """
        更新指定 `condition_key` 的密度。
        
        该方法会：
        - 记录 `condition_key` 的时间戳。
        - 计算 `avg_interval` 作为时间间隔的加权平均值。
        - 计算增长因子 (`growth_rate`) 和衰减因子 (`decay_rate`)。
        - 根据选定的密度计算策略更新密度值。
        
        Args:
            condition_key (str): 需要更新的条件标识符。
            current_time (float, optional): 事件发生时间戳，默认使用 `time.time()`。
        """

        async with self.lock:
            current_time = current_time or time.time()
            time_wheel = self.condition_timestamps[condition_key]
            time_wheel.add(current_time)
            
            if self.time_window:
                time_wheel.prune(current_time)
            
            timestamps = time_wheel.get_all()
            if len(timestamps) < 2:
                avg_interval = 100.0
            else:
                avg_interval = self.weighted_average_calculator.compute(timestamps, self.avg_smoothing_factor)

            # 计算增长率和衰减率
            growth_rate = Sigmoid.f(avg_interval, 0.275, 0.61, 8.6, 0.3)
            decay_rate = Sigmoid.f(avg_interval, 0.045, 0.155, 10, 0.3)
            
            # 结合历史权重平滑
            growth_rate = 0.9 * growth_rate + 0.1 * self.condition_weights[condition_key]
            decay_rate = max(0.05, 0.9 * decay_rate + 0.1 * (1 / (avg_interval + 1)))
            
            density_increment = growth_rate / (avg_interval + 1)
            decay_factor = math.exp(-decay_rate * avg_interval)
            
            # 计算新权重
            new_weight = self.density_calculator.calculate(
                prev_weight=self.condition_weights[condition_key],
                density_increment=density_increment,
                decay_factor=decay_factor,
                avg_interval=avg_interval
            )
            self.condition_weights[condition_key] = new_weight
    
    def get_density_weight(self, condition_key: str) -> float:
        """
        获取指定 `condition_key` 的密度权重。
        
        该值用于控制触发条件的动态调整，例如：
        - 在 API 速率限制中，权重越大，限制越严格。
        - 在异常检测中，权重越大，表示该事件发生频率较高。
        
        Args:
            condition_key (str): 需要查询的条件标识符。
        
        Returns:
            float: 计算得到的密度权重，范围通常在 `[0, 0.15]` 之间，经过 `log` 变换后映射到 [-0.2, 0.5] 之间。
        """

        weight = max(1e-9, self.condition_weights.get(condition_key, 1e-9) * 10)
        return 0.05 * max(-4, math.log(self.scaling_factor * weight))