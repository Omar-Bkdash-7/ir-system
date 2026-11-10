"""The dataset vocabulary is cached, bounded, and never fails a request."""
import unittest

from services.query_refinement.vocabulary import DatasetVocabulary, load_tfidf_vocabulary


class DatasetVocabularyTest(unittest.TestCase):
    def test_a_dataset_is_loaded_once(self):
        calls = []

        def loader(dataset):
            calls.append(dataset)
            return {"pitot", "planform"}

        vocabulary = DatasetVocabulary(loader=loader)
        self.assertEqual(vocabulary.words("cranfield"), {"pitot", "planform"})
        self.assertEqual(vocabulary.words("cranfield"), {"pitot", "planform"})
        self.assertEqual(calls, ["cranfield"])

    def test_without_a_dataset_nothing_is_protected(self):
        vocabulary = DatasetVocabulary(loader=lambda dataset: {"pitot"})
        self.assertEqual(vocabulary.words(None), set())
        self.assertEqual(vocabulary.words(""), set())

    def test_a_failing_loader_protects_nothing(self):
        def loader(dataset):
            raise OSError("index is gone")

        self.assertEqual(DatasetVocabulary(loader=loader).words("cranfield"), set())

    def test_the_cache_is_bounded(self):
        calls = []

        def loader(dataset):
            calls.append(dataset)
            return {dataset}

        vocabulary = DatasetVocabulary(cache_size=1, loader=loader)
        vocabulary.words("one")
        vocabulary.words("two")
        vocabulary.words("one")
        self.assertEqual(calls, ["one", "two", "one"])


class LoaderTest(unittest.TestCase):
    def test_an_unknown_dataset_yields_no_words_instead_of_raising(self):
        self.assertEqual(load_tfidf_vocabulary("no-such-dataset-anywhere"), set())


if __name__ == "__main__":
    unittest.main()
