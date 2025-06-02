# 修复 `np.mean(recent_intervals) if recent_intervals else 1.0` 的问题
# 直接使用 `len(recent_intervals) > 0` 来判断
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from itertools import islice
from .density import AsyncConditionDensityManager as ConditionTracker
from .density import DensityStrategy
import asyncio
import random

plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于 Windows
plt.rcParams['axes.unicode_minus'] = False

condition_key = "test"
n = 2
s = [10] * 5 * n
k = 2
rand = []
for i in range(1,40):
    r = random.random()
    rand.extend([r * 1000] * int(1/r))
scenarios = {
    "均匀触发 → 突然高频": s + ([10] * n * 14 + [2] * n * 30 + [10] * n * 14 + [2] * n * 30) * k,
    "均匀触发 → 突然低频": s + ([10] * n * 10 + [50] * n * 3 + [10] * n * 15) * k,
    "长期低频 → 突然高频": s + ([5] * n * 10 + [50] * n * 2 + [5] * n * 10) * k,
    "长期高频 → 突然低频": s + ([2] * n * 80 + [20] * n * 12) * k,
    #"随机": s + rand
}

results = {}
async def main():
    start_time = 0
    for scenario, intervals in scenarios.items():
        tracker = ConditionTracker(
            density_strategy=DensityStrategy.EXPONENTIAL_MOVING_AVERAGE
        )  # 重新初始化
        times = [start_time]
        weights = []

        for interval in intervals:
            start_time += interval
            await tracker.update_condition_density(condition_key, start_time)
            weights.append(tracker.get_density_weight(condition_key))

        results[scenario] = (intervals, weights)

    # 绘制结果
    plt.figure(figsize=(12, 8))
    for scenario, (intervals, weights) in results.items():
        x_values = np.cumsum(intervals)
        plt.plot(x_values, weights, label=scenario)
    #plt.plot(x_values, tracker.density_calculator.a, label='average')
    #plt.plot(x_values, b, label='average_r')
    plt.xlabel("时间 (s)")
    plt.ylabel("密度权重")
    plt.title("不同触发模式下的密度权重变化111")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())