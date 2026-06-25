from collections import deque, Counter
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class QueryHistoryManager:
    def __init__(self, max_history_size: int = 10):
        self.max_history_size = max_history_size
        self._session_histories: Dict[str, deque] = {}
    
    def add_query(self, session_id: str, query: str) -> None:
        if session_id not in self._session_histories:
            self._session_histories[session_id] = deque(maxlen=self.max_history_size)
        
        if query and query.strip():
            self._session_histories[session_id].append(query.strip())
    
    def get_recent_queries(self, session_id: str, n: int = 3) -> List[str]:
        if session_id not in self._session_histories:
            return []
        
        history = list(self._session_histories[session_id])
        return history[-n:] if len(history) > n else history
    
    def get_term_frequency(self, session_id: str) -> Counter:
        queries = self.get_recent_queries(session_id, self.max_history_size)
        
        term_freq = Counter()
        for query in queries:
            for word in query.lower().split():
                if word.isalpha() and len(word) > 2:
                    term_freq[word] += 1
        
        return term_freq
    
    def clear_session(self, session_id: str) -> None:
        if session_id in self._session_histories:
            del self._session_histories[session_id]
