import asyncio
import random
import math
from typing import Union, Set, List, Dict, Optional, Any
import time
import jieba
from collections import Counter
from .density import AsyncConditionDensityManager, DensityStrategy
from ._math import Sigmoid
from .matcher import RegexMatcher
from .entropy import EntropyCalculator
from .detrep import DetectionRepetition

class AsyncKeywordSystem:
    """统一触发系统（整合冷却、密度和信息熵调控）"""
    def __init__(
        self,
        verbose: bool = False,
        trigger_density: float = 0.0,
        activation_base: float = 70,
        level_base: int = 2,

        density_window: int = 20,
        decay_factor: float = 0.9,
        scaling_factor: int = 10, 
        ema_alpha: float = 0.8, 

        entropy_alpha: float = 0.8,
    ):
        """
        Args：
            verbose: bool, 控制是否输出调试信息, 默认为False
            trigger_density: float, 调控事件激活的整体概率, 范围为-1.0~1.0, 值越大激活概率越高
            activation_base: float, 权重和校准微调值
            level_base: int, 等级指数变换函数的底数
            density_window: int, 密度计算窗口大小，值越高密度调控越缓和
            decay_factor: float, 密度衰减因子
            alpha: float, 信息熵平滑因子
        """
        self.verbose = verbose
        self.trigger_density = - 0.1 * max(-1, min(1, trigger_density)) + 1
        self.activation_base = activation_base
        self.level_base = level_base

        # 组件初始化
        self.regex_matcher = RegexMatcher()
        self.condition_tracker = AsyncConditionDensityManager(
            window_size=density_window, 
            decay_factor=decay_factor, 
            scaling_factor=scaling_factor, 
            ema_alpha=ema_alpha,
            density_strategy=DensityStrategy.EXPONENTIAL_MOVING_AVERAGE
        )
        self.entropy_calculator = EntropyCalculator(entropy_alpha)
        self.detection_repetition = DetectionRepetition()

        # 状态存储
        self.condition_configs = {}
        self._lock = asyncio.Lock()


    async def add_conditions(self, conditions: List[Union[str, Set[str]]], level: int, tag: str = None):
        """注册关键词组"""
        await self.regex_matcher.register_keywords(conditions)
        tasks = [self.add_condition(condition, level, if_add=False, tag=tag) for condition in conditions]
        await asyncio.gather(*tasks)
                

    async def add_condition(self, condition: Union[str, Set[str]], level: int, if_add: bool = True, tag: str = None):
        """注册独立条件"""
        async with self._lock:
            # 生成唯一条件标识
            if isinstance(condition, (tuple, list, set)):
                condition_key = "&".join(sorted(condition))
            else:
                condition_key = str(condition)
            
            # 存储配置
            level = max(-5, min(5, level))
            level = int(math.copysign(self.level_base ** abs(level), level))
            self.condition_configs[condition_key] = {
                "level": level,
            }

            jieba.add_word(condition_key, freq = abs(level) * 500, tag = tag)

            if if_add: 
                await self.regex_matcher.register_keyword(condition_key)
                

    async def process_message(self, message: str) -> Union[bool, dict]:
        """统一触发处理逻辑"""
        msg_lower = message.lower().strip()
        if msg_lower == "":
            return self._format_output(None)
        async with self._lock:
            # 复读检测
            if await self.detection_repetition.is_repeat(msg_lower):
                return self._format_output(None)
            
            # 处理加权触发
            result = await self._process_message(msg_lower)
            
            return self._format_output(result)


    async def _process_message(self, message: str) -> Optional[dict]:
        """统一触发核心逻辑"""
        # 信息熵计算
        

        # 遍历所有条件
        triggered_conditions = []
        total_weight = 0.0
        total_weight, triggered_conditions = await self._calculate_trigger(message, self.condition_configs)

        # 概率计算
        if total_weight > 0:
            adjusted_weight = total_weight/ (self.activation_base) * 10
            activation_prob = Sigmoid.f(adjusted_weight, lower=0, upper=1, midpoint=2.7 * self.trigger_density, steepness=2.95)

            if random.random() < activation_prob:
                return {
                    "weight": total_weight,
                    "prob": activation_prob,
                    "triggers": triggered_conditions,
                }
        return None
    
    async def _calculate_trigger(self, message: str, config: Dict[str, Any]):
        now = time.time()

        words = jieba.lcut(message)
        keywords = set(config.keys())
        counter = Counter(words)
        hit_words = {item: [0, 1, 1.5, 1.75][min(count,3)] for item in keywords if (count := counter.get(item, 0)) > 0}
        
        if hit_words != {}:
            tasks = [
                self.condition_tracker.update_condition_density(keyword, now)
                for keyword in hit_words.keys()
            ]
            await asyncio.gather(*tasks)
            total_weight = sum(config[keyword]["level"] * item * (1 - self.condition_tracker.get_density_weight(keyword)/1.5) for keyword, item in hit_words.items())
            triggered_conditions = list(hit_words.keys())

            entropy = self.entropy_calculator.calculate_entropy(words)
            entropy = Sigmoid.f(entropy, lower=0.8, upper=1.2, midpoint=0.33, steepness=3)

            total_weight *= entropy

            return total_weight, triggered_conditions
        return 0.0, None

    def _format_output(self, result) -> Union[bool, dict]:
        """统一输出格式"""
        if not self.verbose:
            return bool(result)
        return {
            "trigger": bool(result),
            "detail": result,
        }
    
async def main():

    system = AsyncKeywordSystem(
        verbose=True,
        trigger_density=0,
    )

    await system.add_conditions(["可不"], level=4)
    await system.add_conditions(["bot"], level=-4)
    await system.add_conditions(["里命", "星界"], level=3)
    await system.add_conditions(["羽累", "狐子"], level=2)
    await system.add_conditions(["喜欢", "可爱", "亲亲", "抱抱"], level=1)
    # 测试消息
    messages = [
        "可不可爱",
        "好喜欢可不",
        "可不抱抱",
        "最喜欢星界了",  # 复读
        "这是可不bot",
        "可不和里命超好磕",  # 复读
        "亲亲可不",
        "常规系统日志信息",
        "重要通知：紧急维护通知",
        "【紧急通知】系统发生严重故障！",
        "【紧急通知】系统发生严重故障！",
        "（害羞地捂脸，脸上露出微笑）咦——！真的好开心哦！谢谢你喜欢我！我也觉得你超级棒的！我们可以一起玩耍、一起学习哦！",
        "星界可不两个宝宝",
        "服务器出现故障需要处理",
        "抱抱你，你好可爱！",
        "常规系统日志信息",
        "喜欢星界和里命",
        "重要通知：紧急维护通知",
        "折扣促销最后一天"
    ]
    test_messages = messages#random.choices(messages,k=20)

    start = time.time()
    results: List[bool | dict] = []
    i = 0
    for msg in test_messages:
        time.sleep(random.random())
        res = await system.process_message(msg)
        if res["trigger"]:
            if i != 0: print(f"有{i}条未激活")
            i = 0
            print(f"\033[32m{msg} => {res["trigger"]}: {res["detail"]}\033[0m")
        else: i += 1
        results.append(res)
    elapsed = time.time() - start

    print(f"\n处理 {len(test_messages)} 条消息, 用时: {elapsed:.2f} 秒")
    print(sum(item.get("trigger", item) for item in results))
    #for i, (msg, r) in enumerate(zip(test_messages, results), 1):
        #print(f"{i}. [{msg}] => {r}")


if __name__ == "__main__":
    asyncio.run(main())
