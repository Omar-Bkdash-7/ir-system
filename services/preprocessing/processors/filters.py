from typing import Optional
from ..core.resources import get_stop_words


class TokenFilter:
    def __init__(
        self,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        keep_numbers: bool = False,
        min_token_length: int = 2,
        max_token_length: Optional[int] = None,
        custom_stopwords: Optional[set[str]] = None,
    ):
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.keep_numbers = keep_numbers
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        
        if self.remove_stopwords:
            self.stop_words = get_stop_words()
            if custom_stopwords:
                self.stop_words = self.stop_words | custom_stopwords
        else:
            self.stop_words = custom_stopwords or set()
    
    def filter_tokens(self, tokens: list[str]) -> list[str]:
        filtered = []
        
        for token in tokens:
            if not token:
                continue
            
            if self.remove_punctuation and not token.isalnum():
                continue
            
            if not self.keep_numbers and token.isdigit():
                continue
            
            if len(token) < self.min_token_length:
                continue
            
            if self.max_token_length and len(token) > self.max_token_length:
                continue
            
            if self.stop_words and token.lower() in self.stop_words:
                continue
            
            filtered.append(token)
        
        return filtered
