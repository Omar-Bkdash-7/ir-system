"""Tests for TokenFilter improvements (hyphenated words fix)."""

import unittest
from services.preprocessing.processors.filters import TokenFilter


class TestHyphenatedWords(unittest.TestCase):
    """Test that hyphenated words are preserved."""
    
    def test_hyphenated_words_preserved(self):
        filter_obj = TokenFilter(
            remove_punctuation=True,
            remove_stopwords=False,
            min_token_length=2
        )
        
        tokens = ["state-of-the-art", "machine-learning", "well-known"]
        filtered = filter_obj.filter_tokens(tokens)
        
        self.assertIn("state-of-the-art", filtered)
        self.assertIn("machine-learning", filtered)
        self.assertIn("well-known", filtered)
    
    def test_mixed_punctuation_preserved(self):
        filter_obj = TokenFilter(
            remove_punctuation=True,
            remove_stopwords=False,
            min_token_length=1
        )
        
        tokens = ["c++", "c#", "f-16", "covid-19"]
        filtered = filter_obj.filter_tokens(tokens)
        
        self.assertIn("c++", filtered)
        self.assertIn("c#", filtered)
        self.assertIn("f-16", filtered)
        self.assertIn("covid-19", filtered)


class TestPunctuation(unittest.TestCase):
    """Test punctuation filtering."""
    
    def test_pure_punctuation_removed(self):
        filter_obj = TokenFilter(
            remove_punctuation=True,
            remove_stopwords=False,
            min_token_length=1
        )
        
        tokens = ["word", "---", "!!!", "...", "test-word", ",,,"]
        filtered = filter_obj.filter_tokens(tokens)
        
        self.assertIn("word", filtered)
        self.assertIn("test-word", filtered)
        self.assertNotIn("---", filtered)
        self.assertNotIn("!!!", filtered)


class TestLengthFiltering(unittest.TestCase):
    """Test min/max token length filtering."""
    
    def test_min_length(self):
        filter_obj = TokenFilter(
            remove_punctuation=False,
            remove_stopwords=False,
            min_token_length=3
        )
        
        tokens = ["a", "ab", "abc", "abcd"]
        filtered = filter_obj.filter_tokens(tokens)
        
        self.assertNotIn("a", filtered)
        self.assertNotIn("ab", filtered)
        self.assertIn("abc", filtered)
        self.assertIn("abcd", filtered)
    
    def test_max_length(self):
        filter_obj = TokenFilter(
            remove_punctuation=False,
            remove_stopwords=False,
            min_token_length=1,
            max_token_length=5
        )
        
        tokens = ["ab", "abcde", "abcdef"]
        filtered = filter_obj.filter_tokens(tokens)
        
        self.assertIn("ab", filtered)
        self.assertIn("abcde", filtered)
        self.assertNotIn("abcdef", filtered)


class TestStopwords(unittest.TestCase):
    """Test stopwords filtering."""
    
    def test_custom_stopwords(self):
        filter_obj = TokenFilter(
            remove_punctuation=False,
            remove_stopwords=True,
            custom_stopwords={"custom", "stop"}
        )
        
        tokens = ["word", "custom", "test", "stop"]
        filtered = filter_obj.filter_tokens(tokens)
        
        self.assertIn("word", filtered)
        self.assertIn("test", filtered)
        self.assertNotIn("custom", filtered)
        self.assertNotIn("stop", filtered)
    
    def test_custom_stopwords_apply_even_without_default_stopwords(self):
        """Custom stopwords are domain terms: they apply regardless of remove_stopwords."""
        filter_obj = TokenFilter(
            remove_punctuation=False,
            remove_stopwords=False,
            custom_stopwords={"custom", "stop"}
        )
        
        tokens = ["word", "custom", "stop", "the"]
        filtered = filter_obj.filter_tokens(tokens)
        
        self.assertNotIn("custom", filtered)
        self.assertNotIn("stop", filtered)
        self.assertIn("word", filtered)
        self.assertIn("the", filtered)  # default NLTK stopwords NOT applied when toggle is off
    
    def test_custom_stopwords_case_insensitive(self):
        filter_obj = TokenFilter(
            remove_punctuation=False,
            remove_stopwords=False,
            custom_stopwords={"Custom"}
        )
        
        tokens = ["custom", "CUSTOM", "word"]
        filtered = filter_obj.filter_tokens(tokens)
        
        self.assertEqual(filtered, ["word"])
        # custom stopwords are normalized to lowercase
        self.assertEqual(filter_obj.stop_words, {"custom"})


if __name__ == "__main__":
    unittest.main()
