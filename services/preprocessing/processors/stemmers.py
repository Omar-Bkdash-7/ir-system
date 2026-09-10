from ..core.resources import get_stemmer


class PorterStemmerWrapper:
    def __init__(self):
        self.stemmer = get_stemmer()
    
    def stem(self, tokens: list[str]) -> list[str]:
        return [self.stemmer.stem(token) for token in tokens]
