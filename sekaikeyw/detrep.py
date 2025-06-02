import hashlib
from collections import deque

class DetectionRepetition:

    def __init__(
        self, 
        maxlen: int = 3,
    ):
        self.history = deque(maxlen=maxlen)

    async def is_repeat(self, message: str) -> bool:
        """复读检测（使用哈希优化）"""
        msg_hash = hashlib.md5(message.encode()).hexdigest()
        if msg_hash in self.history:
            return True
        self.history.append(msg_hash)
        return False