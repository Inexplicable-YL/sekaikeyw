import time
import re
from flashtext import KeywordProcessor
from collections import Counter

# **生成测试文本**
text_size = 10**6  # 1MB 文字
text = "苹果 香蕉 橙子 " * (text_size)

# **生成关键词列表**
num_keywords = 1000  # 关键词数量
keywords = [f"关键词{i}" for i in range(num_keywords)]

# **正则匹配**
pattern = re.compile("|".join(map(re.escape, keywords)))

start = time.time()
re_count = Counter(pattern.findall(text))
re_time = time.time() - start

# **FlashText 匹配**
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(keywords)

start = time.time()
flashtext_count = Counter(keyword_processor.extract_keywords(text))
flashtext_time = time.time() - start

# **打印结果**
print(f"🔹 `re` 耗时: {re_time:.6f} 秒")
print(f"🔹 `FlashText` 耗时: {flashtext_time:.6f} 秒")
