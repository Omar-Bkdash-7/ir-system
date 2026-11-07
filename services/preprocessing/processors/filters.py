from typing import Optional
from ..core.resources import get_stop_words


class TokenFilter:
    """
    Length/character/stopword filtering applied after tokenization.

    Stopword policy:
    - the NLTK english set is applied only when `remove_stopwords=True`;
    - `custom_stopwords` (e.g. domain junk like "doi", "et al") are applied
      whenever provided, regardless of that toggle, and are matched
      case-insensitively (entries are normalized to lowercase);
    - tokens with no alphanumeric character at all are dropped when
      `remove_punctuation=True`, while tokens containing hyphens, plus signs
      etc. ("state-of-the-art", "c++", "f-16") are kept.
    """

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

        self.stop_words = set()
        if remove_stopwords:
            self.stop_words |= get_stop_words()
        if custom_stopwords:
            self.stop_words |= {w.lower() for w in custom_stopwords}
    
    def filter_tokens(self, tokens: list[str]) -> list[str]:
        filtered = []
        
        for token in tokens:
            if not token:
                continue
            
            # Check if token is purely punctuation (more flexible than isalnum)
            if self.remove_punctuation:
                # Allow tokens with alphanumeric chars and hyphens (e.g., "state-of-the-art")
                # Only reject if token has no alphanumeric characters at all
                if not any(c.isalnum() for c in token):
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
