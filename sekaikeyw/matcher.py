import asyncio
import re
from collections import Counter
from typing import List, Tuple
from flashtext import KeywordProcessor
import asyncio
import re
from collections import Counter
from typing import List, Tuple, Dict, Literal, Deque
import jieba
from abc import ABC, abstractmethod
from datatypes import cacheDict


class Matcher(ABC):

    @abstractmethod
    def register_keywords(self, keywords: List[str]) -> None:
        """
        注册关键词
        """

    @abstractmethod
    def _match(self, text: str, keywords: List[str]) -> Counter:
        """
        高效统计文本中完全匹配的关键词数量
        
        Args:
            text (str): 需要匹配的文本
            keywords (List[str]): 关键词列表
        
        Returns:
            Counter: 匹配的关键词及其出现次数
        """

    async def aregister_keywords(self, keywords: List[str]) -> None:
        return await asyncio.to_thread(self.register_keywords, keywords)

    async def _amatch(self, text: str, keywords: List[str]) -> Counter:
        return await asyncio.to_thread(self._match, text, keywords)
    
    def match(self, text: str, keywords: List[str]) -> Counter:
        """
        高效统计文本中完全匹配的关键词数量（FlashText实现）
        
        Args:
            text (str): 需要匹配的文本
            keywords (List[str]): 关键词列表
        
        Returns:
            Counter: 匹配的关键词及其出现次数
        """
        self.register_keywords(keywords)
        return self._match(text, keywords)
    
    async def amatch(self, text: str, keywords: List[str]) -> Counter:
        """
        高效统计文本中完全匹配的关键词数量（FlashText实现）
        
        Args:
            text (str): 需要匹配的文本
            keywords (List[str]): 关键词列表
        
        Returns:
            Counter: 匹配的关键词及其出现次数
        """
        await self.aregister_keywords(keywords)
        return await self._amatch(text, keywords)
    
    async def lmatch(self, texts: List[str], keywords: List[str]) -> Dict[str, Counter]:
        """
        统计文本列表中关键词数量
        
        Args:
            texts (List[str]): 文本列表
            keywords (List[str]): 关键词列表
        
        Returns:
            Dict[str, Counter]: 分句匹配的关键词及其出现次数
        """
        await self.aregister_keywords(keywords)
        results = await asyncio.gather(*[self._amatch(s, keywords) for s in texts])
        return {s: result for s, result in zip(texts, results)}

    async def slmatch(self, texts: List[str], keywords: List[str]) -> Counter:
        """
        统计文本列表中关键词数量之和
        
        Args:
            texts (List[str]): 文本列表
            keywords (List[str]): 关键词列表
        
        Returns:
            Counter: 匹配的关键词及其出现次数
        """
        await self.aregister_keywords(keywords)
        return sum(await asyncio.gather(*[self._amatch(s, keywords) for s in texts]), Counter())
    
    async def smatch(self, text: str, keywords: List[str], mode: Literal["sentence", "paragraph"] = "paragraph") -> Dict[str, Counter]:
        """
        统计文本中每个句子的关键词数量（FlashText 实现）
        
        Args:
            text (str): 需要匹配的文本
            keywords (List[str]): 关键词列表
        
        Returns:
            Dict[str, Counter]: 分句匹配的关键词及其出现次数
        """
        if mode == "sentence": 
            parts = re.split(r'[。！？.!?]', text)
            return await self.lmatch(parts, keywords)
        elif mode == "paragraph":
            parts = re.split(r'[\n\r]', text)
            return await self.lmatch(parts, keywords)
        else:
            raise ValueError(f"Invalid mode, must be `sentence` or `paragraph`, not `{mode}`")
    
    
    async def ssmatch(self, text: str, keywords: List[str], mode: Literal["sentence", "paragraph"] = "paragraph") -> Counter:
        """
        统计文本中每个句子的关键词数量之和（FlashText 实现）
        
        Args:
            text (str): 需要匹配的文本
            keywords (List[str]): 关键词列表
        
        Returns:
            Dict[str, Counter]: 分句匹配的关键词及其出现次数
        """
        
        if mode == "sentence": 
            parts = re.split(r'[。！？.!?]', text)
            return await self.slmatch(parts, keywords)
        elif mode == "paragraph":
            parts = re.split(r'[\n\r]', text)
            return await self.slmatch(parts, keywords)
        else:
            raise ValueError(f"Invalid mode, must be `sentence` or `paragraph`, not `{mode}`")


class RegexMatcher(Matcher):
    """高效正则匹配器"""

    def __init__(self):
        self._pattern_cache: cacheDict[tuple, re.Pattern[str]] = cacheDict(max_size=20)

    def register_keywords(self, keywords: List[str]) -> None:
        """获取或编译关键词的正则模式"""

        keywords = tuple(sorted(keywords))
        if keywords in self._pattern_cache:
            return
        pattern_str = "|".join(map(re.escape, keywords))  
        compiled_pattern = re.compile(pattern_str, flags=re.IGNORECASE)
        self._pattern_cache[keywords] = compiled_pattern

    def _match(self, text: str, keywords: List[str]) -> Counter:
        """
        统计文本中完全匹配的关键词数量
        
        Args:
            text (str): 需要匹配的文本
            keywords (List[str]): 关键词列表
        
        Returns:
            Counter: 匹配的关键词及其出现次数
        """
        if not keywords:
            return Counter()
        pattern = self._pattern_cache[tuple(sorted(keywords))]
        result = Counter(match.group() for match in pattern.finditer(text))
        return result


class FlashTextMatcher(Matcher):
    """高效关键词匹配器（基于 FlashText ）"""

    def __init__(self):
        self._pattern_cache: cacheDict[tuple, KeywordProcessor] = cacheDict(max_size=20)

    def register_keywords(self, keywords: List[str]) -> None:
        """注册关键词（避免重复注册）"""
        kw = tuple(sorted(keywords))
        if kw in self._pattern_cache:
            return
        self._pattern_cache[kw] = KeywordProcessor()
        self._pattern_cache[kw].add_keywords_from_list(keywords)

    def _match(self, text: str, keywords: List[str]) -> Counter:
        """
        高效统计文本中完全匹配的关键词数量（FlashText 实现）
        
        Args:
            text (str): 需要匹配的文本
            keywords (List[str]): 关键词列表
        
        Returns:
            Counter: 匹配的关键词及其出现次数
        """
        if not keywords:
            return Counter()
        pattern = self._pattern_cache[tuple(sorted(keywords))]
        matched_keywords = pattern.extract_keywords(text)
        return Counter(matched_keywords)
    

class JiebaMatcher(Matcher):
    """基于 Jieba 的精准关键词匹配器"""

    def __init__(self, custom_dict: str = None):
        """
        初始化 Jieba 分词匹配器（独立实例）
        Args:
            custom_dict (str): 自定义词典路径（可选）
        """
        self.jieba = jieba.Tokenizer()
        
        if custom_dict:
            self.jieba.load_userdict(custom_dict)

        self.keywords_set = set()

    def register_keywords(self, keywords: List[str]) -> None:
        """注册关键词，添加到独立 Jieba 词典"""
        for kw in keywords:
            if kw not in self.keywords_set:
                self.jieba.add_word(kw)
                self.keywords_set.add(kw)

    def _match(self, text: str, keywords: List[str]) -> Counter:
        """
        使用 Jieba 精确匹配关键词并统计出现次数（独立实例，不影响全局）
        Args:
            text (str): 需要匹配的文本
            keywords (List[str]): 关键词列表
        
        Returns:
            Counter: 匹配的关键词及其出现次数
        """
        if not keywords:
            return Counter()
        words = self.jieba.lcut_for_search(text.lower())
        return Counter(word for word in words if word in self.keywords_set)


class AutoMatcher(Matcher):
    """基于多种匹配器的自动切换器"""

    def __init__(
        self,
        jieba_custom_dict: str = None
    ):
        self.regex_matcher = RegexMatcher()
        self.flash_text_matcher = FlashTextMatcher()
        self.jieba_matcher = JiebaMatcher(jieba_custom_dict)
        self.lang: cacheDict[tuple, callable] = cacheDict(max_size=20)

    @staticmethod
    def detect_main_language(text: str):
        """快速判断文本主要是中文、英文还是其他"""
        chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_count = sum(1 for char in text if char.isascii() and char.isalpha())
        total_count = len(text)

        if total_count == 0:
            return "unknown"

        chinese_ratio = chinese_count / total_count
        english_ratio = english_count / total_count

        if chinese_ratio > 0.5:
            return "ch"
        elif english_ratio > 0.5:
            return "en"
        else:
            return "other"
        
    def register_keywords(self, keywords: List[str]) -> None:
        """
        注册关键词
        """
        self.regex_matcher.register_keywords(keywords)
        self.flash_text_matcher.register_keywords(keywords)
        self.jieba_matcher.register_keywords(keywords)

    def _match(self, text: str, keywords: List[str]):
        if self.detect_main_language(text) == "ch":
            return self.jieba_matcher._match(text, keywords)
        elif len(text) > 10**6:
            return self.flash_text_matcher._match(text, keywords)
        return self.regex_matcher._match(text, keywords)
        
    
async def main():
    matcher = JiebaMatcher()
    text = ((
        "可不可爱。"
        "好喜欢可不。"
        "可不抱抱。"
        "最喜欢星界了。"  # 复读
        "这是可不bot。"
        "可不和里命超好磕。"  # 复读
        "亲亲可不。"
        "常规系统日志信息。"
        "重要通知：紧急维护通知。"
        "【紧急通知】系统发生严重故障！"
        "【紧急通知】系统发生严重故障！"
        "（害羞地捂脸，脸上露出微笑）咦——！真的好开心哦！谢谢你喜欢我！我也觉得你超级棒的！我们可以一起玩耍、一起学习哦！"
        "星界可不两个宝宝。"
        "服务器出现故障需要处理。"
        "抱抱你，你好可爱！"
        "常规系统日志信息。"
        "喜欢星界和里命。"
        "重要通知：紧急维护通知。"
        "折扣促销最后一天。"
    ) * 100 + '\n') * 20
    
    keywords = ["可不", "里命", "星界", "羽累", "狐子", "喜欢", "可爱", "亲亲", "抱抱"]
    import time
    start_time = time.time()
    result = await matcher.ssmatch(text, keywords)
    end_time = time.time()
    print(result, end_time-start_time)

    start_time = time.time()
    result = await matcher.amatch(text, keywords)
    end_time = time.time()
    print(result, end_time-start_time)

if __name__ == "__main__":
    asyncio.run(main())