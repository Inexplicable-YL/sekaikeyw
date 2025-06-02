import asyncio
import re
from collections import Counter
from typing import List, Tuple
from flashtext import KeywordProcessor
import asyncio
import re
from collections import Counter
from typing import List, Tuple

class RegexMatcher:
    """高效正则匹配器"""

    def __init__(self):
        self._pattern_cache = {}  # 缓存已编译的正则表达式

    def _get_combined_pattern(self, keywords: Tuple[str]) -> re.Pattern:
        """获取或编译关键词的正则模式"""
        if keywords in self._pattern_cache:
            return self._pattern_cache[keywords]
        
        pattern_str = "|".join(map(re.escape, keywords))  
        compiled_pattern = re.compile(pattern_str, flags=re.IGNORECASE)

        self._pattern_cache[keywords] = compiled_pattern
        return compiled_pattern

    async def exact_match(self, text: str, keywords: List[str]) -> Counter:
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

        pattern = self._get_combined_pattern(tuple(sorted(keywords)))

        result = Counter(match.group() for match in pattern.finditer(text))

        return result


class FlashTextMatcher:
    """超高效关键词匹配器（基于 FlashText + Counter）"""

    def __init__(self):
        self.keyword_processor = KeywordProcessor()
        self.keywords_set = set()

    async def register_keywords(self, keywords: List[str]) -> None:
        """注册关键词（避免重复注册）"""
        new_keywords = [kw for kw in keywords if kw not in self.keywords_set]
        if new_keywords:
            self.keyword_processor.add_keywords_from_list(new_keywords)
            self.keywords_set.update(new_keywords)

    async def exact_match(self, text: str, keywords: List[str]) -> Counter:
        """
        高效统计文本中完全匹配的关键词数量（FlashText 实现）
        
        Args:
            text (str): 需要匹配的文本
            keywords (List[str]): 关键词列表
        
        Returns:
            Counter: 匹配的关键词及其出现次数
        """
        await self.register_keywords(keywords)

        matched_keywords = self.keyword_processor.extract_keywords(text)

        return Counter(matched_keywords)

# **示例**

async def main():
    matcher = FlashTextMatcher()
    text = (
        "Python is a widely used high-level programming language. It was created by Guido van Rossum and first released in 1991."
        "Python is known for its simplicity and readability, making it a popular choice for beginners and developers."
        "Python is also known for its extensive standard library and large community support."
        "Python is also widely used in data analysis, web development, and AI development."
        "Python is also used in scientific computing, machine learning, and artificial intelligence."
        "Python is also used in blockchain development and cryptocurrency technology."
        "Python is also used in quantum computing and quantum cryptography."
        "Python is also used in game development and virtual reality."
        "Python is also used in software development and automation."
        "Python is also used in education and research."
        "Python is also used in natural language processing and speech recognition."
        "Python is also used in artificial intelligence and machine learning."
        "Python is also used in data analysis and visualization."
    ) * 1000 + "\n"
    
    keywords = [
        "python", 
        "programming", 
        "language", 
        "guido van rossum",
        "guido rossum",
        "high-level",
        "high level programming",
        "high level programming language",
        "widely used",
        "widely used high-level",
        "widely used high level programming",
        "widely used high level programming language",
        "created by",
    ]
    import time
    start_time = time.time()
    result1 = await matcher.exact_match(text * 1000, keywords)
    end_time = time.time()
    print(result1, end_time-start_time)

if __name__ == "__main__":
    asyncio.run(main())