import jieba
import math
import numpy as np
from collections import deque, defaultdict
from typing import List, Union

class EntropyCalculator:

    def __init__(self, alpha: float = 0.8):
        """
        Args:
            alpha: float, 平滑因子
        """
        self.alpha = alpha  # 平滑因子
        self._base_entropy: deque = deque(maxlen=10)  # 动态初始化历史熵

    def calculate_entropy(self, message: Union[str, List[str]]) -> float:
        """计算信息熵（基于 jieba 词频 + message 统计）"""

        words = message if isinstance(message, list) else jieba.lcut(message)

        # 计算 message 内部词频
        freq = defaultdict(int)
        total_words = len(words) or 1  # 避免除以 0
        for w in words:
            freq[w] += 1

        # 计算 word_probs，优先用 jieba 词频，否则用 message 词频
        total_freq = sum(jieba.get_FREQ(w) or freq[w] for w in words) or 1  # 计算总频率
        word_probs = [(jieba.get_FREQ(w) or freq[w]) / total_freq for w in words]  # 归一化
        
        # 计算信息熵
        entropy = -sum(p * math.log(max(p, 1e-9)) for p in word_probs)  # 避免 log(0)

        # 平滑历史熵
        base_entropy = np.mean(list(self._base_entropy) or [entropy])
        entropy = self.alpha * entropy + (1 - self.alpha) * base_entropy
        
        self._base_entropy.append(entropy)

        return entropy