"""Tests for PreprocessingConfig validation and conversion."""

import unittest
from services.preprocessing.core.config import PreprocessingConfig


class TestCustomStopwords(unittest.TestCase):
    """Test custom_stopwords type handling."""
    
    def test_from_list(self):
        config = PreprocessingConfig(custom_stopwords=["test", "stop"])
        self.assertIsInstance(config.custom_stopwords, set)
        self.assertEqual(config.custom_stopwords, {"test", "stop"})
    
    def test_from_set(self):
        config = PreprocessingConfig(custom_stopwords={"test", "stop"})
        self.assertIsInstance(config.custom_stopwords, set)
        self.assertEqual(config.custom_stopwords, {"test", "stop"})


class TestFromDict(unittest.TestCase):
    """Test from_dict validation."""
    
    def test_with_list_stopwords(self):
        data = {"lowercase": True, "custom_stopwords": ["custom", "words"]}
        config = PreprocessingConfig.from_dict(data)
        self.assertIsInstance(config.custom_stopwords, set)
        self.assertEqual(config.custom_stopwords, {"custom", "words"})
    
    def test_invalid_key_raises_error(self):
        data = {"lowercase": True, "invalid_key": "value"}
        with self.assertRaises(ValueError) as ctx:
            PreprocessingConfig.from_dict(data)
        self.assertIn("Invalid configuration keys", str(ctx.exception))
    
    def test_invalid_value_type_raises_error(self):
        data = {"lowercase": "not_a_bool"}
        with self.assertRaises(ValueError) as ctx:
            PreprocessingConfig.from_dict(data)
        self.assertIn("Invalid configuration values", str(ctx.exception))
    
    def test_empty_config_uses_defaults(self):
        config = PreprocessingConfig.from_dict({})
        self.assertTrue(config.lowercase)
        self.assertEqual(config.min_token_length, 2)
    
    def test_custom_stopwords_lowercased(self):
        config = PreprocessingConfig(custom_stopwords={"Brains", "Study"})
        self.assertEqual(config.custom_stopwords, {"brains", "study"})
        # normalization must survive the roundtrip too
        restored = PreprocessingConfig.from_dict(config.to_dict())
        self.assertEqual(restored.custom_stopwords, {"brains", "study"})


class TestConfigValidation(unittest.TestCase):
    """Test config validation in __post_init__."""
    
    def test_negative_min_token_length_corrected(self):
        config = PreprocessingConfig(min_token_length=-5)
        self.assertEqual(config.min_token_length, 0)
    
    def test_max_less_than_min_ignored(self):
        config = PreprocessingConfig(min_token_length=5, max_token_length=3)
        self.assertIsNone(config.max_token_length)
    
    def test_stemming_takes_precedence(self):
        config = PreprocessingConfig(use_stemming=True, use_lemmatization=True)
        self.assertTrue(config.use_stemming)
        self.assertFalse(config.use_lemmatization)


class TestRoundtrip(unittest.TestCase):
    """Test to_dict and from_dict roundtrip."""
    
    def test_roundtrip_conversion(self):
        original = PreprocessingConfig(
            lowercase=False,
            use_stemming=True,
            custom_stopwords={"test", "words"}
        )
        data = original.to_dict()
        restored = PreprocessingConfig.from_dict(data)
        
        self.assertEqual(restored.lowercase, original.lowercase)
        self.assertEqual(restored.use_stemming, original.use_stemming)
        self.assertEqual(restored.custom_stopwords, original.custom_stopwords)


if __name__ == "__main__":
    unittest.main()
