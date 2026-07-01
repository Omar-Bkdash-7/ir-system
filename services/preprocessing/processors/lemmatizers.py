from ..core.resources import get_lemmatizer


class WordNetLemmatizerWrapper:
    def __init__(self):
        self.lemmatizer = get_lemmatizer()
    
    def lemmatize(self, tokens: list[str]) -> list[str]:
        return [self.lemmatizer.lemmatize(token) for token in tokens]
