"""Spelling corrections must keep the query's shape and honour the configured thresholds."""
import unittest

from services.query_refinement.engines import SpellingCorrectionEngine
from tests.query_refinement.fakes import STOP_WORDS, FakeSpellChecker


def engine(corrections=None, frequencies=None, **kwargs) -> SpellingCorrectionEngine:
    return SpellingCorrectionEngine(
        spell_checker=FakeSpellChecker(corrections, vocabulary={"control", "data", "receive"}, frequencies=frequencies),
        stop_words=STOP_WORDS,
        **kwargs,
    )


class CorrectionTest(unittest.TestCase):
    def test_a_misspelled_word_is_corrected_in_place(self):
        corrected, corrections = engine({"recieve": "receive"}).correct("recieve the data")
        self.assertEqual(corrected, "receive the data")
        self.assertEqual(corrections, {"recieve": "receive"})

    def test_case_is_preserved(self):
        corrected, corrections = engine({"recieve": "receive"}).correct("Recieve recieve")
        self.assertEqual(corrected, "Receive receive")
        self.assertEqual(corrections, {"Recieve": "Receive", "recieve": "receive"})

    def test_punctuation_and_spacing_survive(self):
        corrected, _ = engine({"recieve": "receive"}).correct("  recieve, then  data?  ")
        self.assertEqual(corrected, "  receive, then  data?  ")

    def test_known_words_are_never_corrected(self):
        corrected, corrections = engine({"control": "contrl"}).correct("control")
        self.assertEqual((corrected, corrections), ("control", {}))

    def test_short_words_stopwords_and_protected_terms_are_untouched(self):
        checker = engine({"api": "app", "the": "then", "cat": "car"})
        corrected, corrections = checker.correct("api the cat")
        self.assertEqual(corrected, "api the cat")
        self.assertEqual(corrections, {})

    def test_domain_terms_can_be_added_at_runtime(self):
        checker = engine({"frobnicate": "fabricate"})
        checker.add_domain_terms({"frobnicate"})
        self.assertEqual(checker.correct("frobnicate")[0], "frobnicate")

    def test_capitalised_names_after_the_first_word_are_left_alone(self):
        # The dictionary tried to rewrite "Quora" to "Quota" and "NASA" to "Nasa".
        corrected, corrections = engine({"quora": "quota", "nasa": "nasa"}).correct("Which question about Quora NASA")
        self.assertEqual(corrected, "Which question about Quora NASA")
        self.assertEqual(corrections, {})

    def test_sentence_initial_capitalisation_is_still_corrected(self):
        corrected, corrections = engine({"recieve": "receive"}).correct("Recieve the data")
        self.assertEqual(corrected, "Receive the data")
        self.assertEqual(corrections, {"Recieve": "Receive"})

    def test_a_word_without_a_suggestion_is_left_alone(self):
        self.assertEqual(engine().correct("unmatched")[0], "unmatched")

    def test_an_empty_query_is_returned_unchanged(self):
        self.assertEqual(engine().correct("   "), ("   ", {}))


class DomainVocabularyTest(unittest.TestCase):
    def test_a_rare_candidate_is_refused(self):
        # The dictionary really did rewrite "aeroelastic" to "ceroplastic" (frequency 50).
        checker = engine({"aeroelastic": "ceroplastic"}, frequencies={"ceroplastic": 50})
        self.assertEqual(checker.correct("aeroelastic"), ("aeroelastic", {}))

    def test_a_common_candidate_is_accepted(self):
        checker = engine({"boundray": "boundary"}, frequencies={"boundary": 3252})
        self.assertEqual(checker.correct("boundray")[0], "boundary")

    def test_regional_spelling_variants_are_left_alone(self):
        # behaviour -> behavior is not a typo, and rewriting it breaks matching against a British-spelled corpus.
        checker = engine({"behaviour": "behavior"}, frequencies={"behavior": 24294})
        self.assertEqual(checker.correct("behaviour"), ("behaviour", {}))

    def test_a_word_the_corpus_knows_is_never_corrected(self):
        # "pitot" is an ordinary word in cranfield and a misspelling nowhere else.
        checker = engine({"pitot": "pilot"})
        self.assertEqual(checker.correct("pitot tube", known_words={"pitot"}), ("pitot tube", {}))
        self.assertEqual(checker.correct("pitot tube")[0], "pilot tube")

    def test_the_minimum_frequency_is_configurable(self):
        checker = engine({"boundray": "boundary"}, frequencies={"boundary": 3252}, min_correction_frequency=5000)
        self.assertEqual(checker.correct("boundray"), ("boundray", {}))


class ThresholdTest(unittest.TestCase):
    def test_a_correction_farther_than_max_edit_distance_is_refused(self):
        corrected, corrections = engine({"layer": "laser"}, max_edit_distance=1).correct("layer")
        self.assertEqual(corrected, "laser")  # distance 1 on a 5-letter word: accepted
        self.assertEqual(corrections, {"layer": "laser"})

    def test_a_low_confidence_correction_is_refused(self):
        # "gulf" -> "gala" is two edits on a four-letter word: confidence 0.5, below the 0.7 default.
        corrected, corrections = engine({"gulf": "gala"}, max_edit_distance=2).correct("gulf")
        self.assertEqual(corrected, "gulf")
        self.assertEqual(corrections, {})

    def test_max_edit_distance_zero_only_accepts_exact_matches(self):
        corrected, corrections = engine({"gulf": "golf"}, max_edit_distance=0).correct("gulf")
        self.assertEqual((corrected, corrections), ("gulf", {}))


if __name__ == "__main__":
    unittest.main()
