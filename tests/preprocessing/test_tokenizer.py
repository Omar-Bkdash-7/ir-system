"""Tests for tokenizers: SimpleTokenizer (regex) and SpacyTokenizer (hyphen merging, batch)."""

import unittest

try:
    from services.preprocessing.processors.tokenizer import SimpleTokenizer, SpacyTokenizer
    TOKENIZERS_AVAILABLE = True
except Exception:  # spacy / model may be unavailable outside the container
    TOKENIZERS_AVAILABLE = False

SPACY_AVAILABLE = False
if TOKENIZERS_AVAILABLE:
    try:
        import spacy  # noqa: F401
        SPACY_AVAILABLE = TOKENIZERS_AVAILABLE
    except ImportError:
        pass


@unittest.skipUnless(TOKENIZERS_AVAILABLE, "services.preprocessing.processors.tokenizer not importable (spacy missing)")
class TestSimpleTokenizer(unittest.TestCase):
    """Word-aware regex tokenizer (the dependency-free / use_spacy_tokenizer=False path)."""

    def setUp(self):
        self.tokenizer = SimpleTokenizer()

    def test_hyphenated_words_kept(self):
        tokens = self.tokenizer.tokenize("state-of-the-art machine-learning covid-19")
        self.assertIn("state-of-the-art", tokens)
        self.assertIn("machine-learning", tokens)
        self.assertIn("covid-19", tokens)

    def test_attached_punctuation_separated(self):
        tokens = self.tokenizer.tokenize("Hello, (world)!")
        self.assertEqual(tokens, ["Hello", "world"])

    def test_programming_style_tokens(self):
        tokens = self.tokenizer.tokenize("Python, c++ and f-16: c#.")
        self.assertIn("c++", tokens)
        self.assertIn("c#", tokens)
        self.assertIn("f-16", tokens)

    def test_apostrophes_kept(self):
        tokens = self.tokenizer.tokenize("They don't know. It's fine")
        self.assertIn("don't", tokens)
        self.assertIn("It's", tokens)

    def test_numbers(self):
        tokens = self.tokenizer.tokenize("In 2022 there were 1,234 cases")
        self.assertIn("2022", tokens)
        self.assertIn("1", tokens)
        self.assertIn("234", tokens)  # thousands separator splits

    def test_batch(self):
        results = self.tokenizer.tokenize_many(["a b", "", "c-d"])
        self.assertEqual(results, [["a", "b"], [], ["c-d"]])


@unittest.skipUnless(SPACY_AVAILABLE, "spaCy not available")
class TestSpacyTokenizer(unittest.TestCase):
    """Hyphen re-merging and streaming batch in the spaCy path."""

    def test_hyphenated_compounds_merged(self):
        tokenizer = SpacyTokenizer(use_lemmatization=False)
        tokens = tokenizer.tokenize("state-of-the-art machine-learning")
        self.assertIn("state-of-the-art", tokens)
        self.assertIn("machine-learning", tokens)

    def test_spaced_hyphen_not_merged(self):
        tokenizer = SpacyTokenizer(use_lemmatization=False)
        tokens = tokenizer.tokenize("value - level check")
        self.assertNotIn("value-level", tokens)
        self.assertIn("value", tokens)
        self.assertIn("level", tokens)

    def test_batch_preserves_order_and_blanks(self):
        tokenizer = SpacyTokenizer(use_lemmatization=False)
        texts = ["first text", "", "third", "machine-learning rocks"]
        results = tokenizer.tokenize_many(texts)
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0], ["first", "text"])
        self.assertEqual(results[1], [])
        self.assertEqual(results[2], ["third"])
        self.assertIn("machine-learning", results[3])

    def test_batch_empty_list(self):
        self.assertEqual(SpacyTokenizer().tokenize_many([]), [])


if __name__ == "__main__":
    unittest.main()
