import unicodedata


class TextNormalizer:
    def __init__(self, normalize_unicode: bool = True, lowercase: bool = True):
        self.normalize_unicode = normalize_unicode
        self.lowercase = lowercase
    
    def normalize(self, text: str) -> str:
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        
        if self.lowercase:
            text = text.lower()
        
        return text
