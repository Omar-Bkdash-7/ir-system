import dataclasses
from dataclasses import dataclass, field
from typing import Optional, get_args, get_origin, Union
import logging

logger = logging.getLogger(__name__)


def _type_matches(expected, value) -> bool:
    """Check a configuration value against its declared dataclass annotation."""
    if get_origin(expected) is Union:
        # Optional[X] / Union[..]: None is always allowed, otherwise any member must match
        if value is None:
            return True
        return any(_type_matches(arg, value) for arg in get_args(expected))
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is str:
        return isinstance(value, str)
    # Fallback for generic annotations not covered above
    return isinstance(value, expected)


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline.

    Value semantics (see also TokenFilter):
    - custom_stopwords are matched case-insensitively (normalized to lowercase
      here), and apply whenever provided, regardless of remove_stopwords;
    - stemming wins over lemmatization when both are enabled;
    - max_token_length is ignored when it is not None and <= min_token_length.
    """

    normalize_unicode: bool = True
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    min_token_length: int = 2
    max_token_length: Optional[int] = None
    use_lemmatization: bool = True
    use_stemming: bool = False
    keep_numbers: bool = False
    use_spacy_tokenizer: bool = True
    custom_stopwords: set[str] = field(default_factory=set)
    
    def to_dict(self) -> dict:
        return {
            "normalize_unicode": self.normalize_unicode,
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
            "remove_stopwords": self.remove_stopwords,
            "min_token_length": self.min_token_length,
            "max_token_length": self.max_token_length,
            "use_lemmatization": self.use_lemmatization,
            "use_stemming": self.use_stemming,
            "keep_numbers": self.keep_numbers,
            "use_spacy_tokenizer": self.use_spacy_tokenizer,
            "custom_stopwords": list(self.custom_stopwords),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PreprocessingConfig":
        """
        Create PreprocessingConfig from dictionary with validation.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            PreprocessingConfig instance
            
        Raises:
            ValueError: If invalid configuration keys or values provided
        """
        data_copy = data.copy()
        custom_stopwords = data_copy.pop("custom_stopwords", [])
        
        # Convert to set if it's a list
        if isinstance(custom_stopwords, list):
            custom_stopwords = set(custom_stopwords)
        elif not isinstance(custom_stopwords, set):
            custom_stopwords = set()
        
        # Validate known keys
        valid_keys = {
            'normalize_unicode', 'lowercase', 'remove_punctuation', 
            'remove_stopwords', 'min_token_length', 'max_token_length',
            'use_lemmatization', 'use_stemming', 'keep_numbers', 'use_spacy_tokenizer'
        }
        invalid_keys = set(data_copy.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(f"Invalid configuration keys: {invalid_keys}. Valid keys: {valid_keys}")
        
        # Validate value types (dataclass annotations are not enforced at runtime)
        field_types = {f.name: f.type for f in dataclasses.fields(cls)}
        for key, value in data_copy.items():
            expected = field_types.get(key)
            if expected is not None and not _type_matches(expected, value):
                type_name = getattr(expected, "__name__", None) or str(expected)
                type_name = type_name.replace("typing.", "")
                raise ValueError(
                    f"Invalid configuration values: '{key}' must be of type {type_name}, "
                    f"got {type(value).__name__}"
                )
        
        try:
            return cls(**data_copy, custom_stopwords=custom_stopwords)
        except TypeError as e:
            raise ValueError(f"Invalid configuration values: {e}") from e
    
    def __post_init__(self):
        """Ensure custom_stopwords is always a lowercase set and validate config."""
        # Convert custom_stopwords to a lowercase set (matched case-insensitively)
        if isinstance(self.custom_stopwords, list):
            object.__setattr__(self, 'custom_stopwords', {w.lower() for w in self.custom_stopwords})
        elif not isinstance(self.custom_stopwords, set):
            object.__setattr__(self, 'custom_stopwords', set())
        else:
            object.__setattr__(self, 'custom_stopwords', {w.lower() for w in self.custom_stopwords})
        
        # Validate config values
        if self.min_token_length < 0:
            logger.warning(f"min_token_length cannot be negative ({self.min_token_length}), setting to 0")
            object.__setattr__(self, 'min_token_length', 0)
        
        if self.max_token_length is not None and self.max_token_length < self.min_token_length:
            logger.warning(f"max_token_length ({self.max_token_length}) < min_token_length ({self.min_token_length}), ignoring max_token_length")
            object.__setattr__(self, 'max_token_length', None)
        
        if self.use_stemming and self.use_lemmatization:
            logger.warning("Both stemming and lemmatization enabled, stemming will take precedence")
            object.__setattr__(self, 'use_lemmatization', False)
