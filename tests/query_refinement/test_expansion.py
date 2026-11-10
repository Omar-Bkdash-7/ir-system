"""Expansion stays conservative and never depends on a corpus being present."""
import unittest

from services.query_refinement.engines import QueryExpansionEngine
from tests.query_refinement.fakes import STOP_WORDS, FakeWordNet


def engine(wordnet, **kwargs) -> QueryExpansionEngine:
    return QueryExpansionEngine(
        max_synonyms_per_word=kwargs.pop("max_synonyms_per_word", 2),
        max_synsets=kwargs.pop("max_synsets", 2),
        min_term_length=kwargs.pop("min_term_length", 4),
        wordnet_reader=wordnet,
        stop_words=STOP_WORDS,
        max_query_words=kwargs.pop("max_query_words", 4),
        min_synonym_count=kwargs.pop("min_synonym_count", 1),
    )


WORDNET = FakeWordNet({
    "control": [["control", "restraint", "dominance"]],
    "boundary": [["boundary", "edge"]],
})


class ExpansionTest(unittest.TestCase):
    def test_synonyms_are_appended_and_the_query_is_kept(self):
        expanded, expansions = engine(WORDNET).expand("boundary layer control")
        self.assertTrue(expanded.startswith("boundary layer control "))
        self.assertEqual(expansions["boundary"], ["edge"])
        self.assertEqual(expansions["control"], ["restraint", "dominance"])
        self.assertIn("edge", expanded)

    def test_a_long_query_is_not_expanded(self):
        expanded, expansions = engine(WORDNET).expand("what is the boundary layer control theory")
        self.assertEqual((expanded, expansions), ("what is the boundary layer control theory", {}))

    def test_stopwords_short_words_and_unknown_words_are_skipped(self):
        expanded, expansions = engine(WORDNET).expand("the boundary of ox")
        self.assertEqual(expansions, {"boundary": ["edge"]})
        self.assertEqual(expanded, "the boundary of ox edge")

    def test_words_already_in_the_query_are_not_added_again(self):
        net = FakeWordNet({"control": [["control", "boundary"]]})
        expanded, expansions = engine(net).expand("control boundary")
        self.assertEqual(expansions, {})
        self.assertEqual(expanded, "control boundary")

    def test_at_most_max_synonyms_per_word(self):
        net = FakeWordNet({"control": [["control", "restraint", "dominance", "mastery"]]})
        _, expansions = engine(net, max_synonyms_per_word=2).expand("control")
        self.assertEqual(expansions["control"], ["restraint", "dominance"])

    def test_synonyms_are_ranked_by_frequency(self):
        net = FakeWordNet({"control": [["control", "restraint"], ["control", "dominance"]]})
        _, expansions = engine(net, max_synonyms_per_word=2).expand("control")
        self.assertEqual(expansions["control"], ["restraint", "dominance"])

    def test_synonyms_without_corpus_evidence_are_skipped(self):
        # WordNet gives "bound" for boundary and "bed" for layer with a count of 0; appending them is noise.
        net = FakeWordNet({"control": [{"restraint": 5, "dominance": 0}]})
        _, expansions = engine(net).expand("control")
        self.assertEqual(expansions["control"], ["restraint"])

    def test_synonyms_below_the_configured_count_are_skipped(self):
        net = FakeWordNet({"control": [{"restraint": 5, "dominance": 2}]})
        _, expansions = engine(net, min_synonym_count=3).expand("control")
        self.assertEqual(expansions["control"], ["restraint"])

    def test_a_wordnet_error_only_costs_that_word(self):
        net = FakeWordNet({"control": RuntimeError("bad corpus"), "boundary": [["boundary", "edge"]]})
        expanded, expansions = engine(net).expand("boundary control")
        self.assertEqual(expansions, {"boundary": ["edge"]})
        self.assertEqual(expanded, "boundary control edge")

    def test_without_wordnet_the_query_is_returned_unchanged(self):
        expanded, expansions = engine(None).expand("boundary control")
        self.assertEqual((expanded, expansions), ("boundary control", {}))
        self.assertFalse(engine(None).available)

    def test_an_empty_query_is_returned_unchanged(self):
        self.assertEqual(engine(WORDNET).expand("  "), ("  ", {}))


if __name__ == "__main__":
    unittest.main()
