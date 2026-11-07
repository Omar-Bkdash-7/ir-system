"""Tests for end-to-end preprocessing pipeline."""

import unittest
from services.preprocessing.core.config import PreprocessingConfig
from services.preprocessing.core.pipeline import TextPreprocessor


class TestBasicPreprocessing(unittest.TestCase):
    """Test basic preprocessing flow."""
    
    def test_lowercase_and_stopwords(self):
        config = PreprocessingConfig(
            lowercase=True,
            remove_stopwords=True,
            use_lemmatization=False,
            use_stemming=False
        )
        preprocessor = TextPreprocessor(config)
        
        text = "The Quick Brown Fox"
        tokens = preprocessor.preprocess(text)
        
        # Should be lowercase (or non-alpha)
        for token in tokens:
            self.assertTrue(token.islower() or not token.isalpha())
        # 'the' is stopword
        self.assertNotIn("the", tokens)
    
    def test_hyphenated_words_in_pipeline(self):
        config = PreprocessingConfig(
            lowercase=True,
            remove_punctuation=True,
            remove_stopwords=False,
            use_lemmatization=False,
            use_stemming=False
        )
        preprocessor = TextPreprocessor(config)
        
        text = "state-of-the-art machine-learning"
        tokens = preprocessor.preprocess(text)
        
        self.assertIn("state-of-the-art", tokens)
        self.assertIn("machine-learning", tokens)
    
    def test_empty_text(self):
        preprocessor = TextPreprocessor()
        
        self.assertEqual(preprocessor.preprocess(""), [])
        self.assertEqual(preprocessor.preprocess("   "), [])
    
    def test_custom_stopwords_in_pipeline(self):
        config = PreprocessingConfig(
            remove_stopwords=True,
            custom_stopwords={"machine", "learning"}
        )
        preprocessor = TextPreprocessor(config)
        
        text = "machine learning is powerful"
        tokens = preprocessor.preprocess(text)
        
        self.assertNotIn("machine", tokens)
        self.assertNotIn("learning", tokens)
        self.assertIn("powerful", tokens)


if __name__ == "__main__":
    unittest.main()
