"""A missing or corrupt corpus must cost a feature, never the request.

The image once carried a truncated ``wordnet.zip``; ``nltk.data.find`` raised ``BadZipFile``, the singleton failed to
initialise and every ``/refine`` call answered 500. These tests pin the replacement behaviour.
"""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from services.query_refinement import corpora as corpora_module
from services.query_refinement.corpora import FALLBACK_STOPWORDS, NltkCorpora, _is_corrupt


def _write_zip(path: Path, entries=("data.noun",)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as bundle:
        for entry in entries:
            bundle.writestr(entry, "content")
    return path


def _broken_wordnet(error):
    """Replace the lazy WordNet loader while the test runs: it cannot be patched in place."""
    return mock.patch.object(corpora_module, "nltk_wordnet", mock.Mock(synsets=mock.Mock(side_effect=error)))


class CorruptCorpusTest(unittest.TestCase):
    def test_a_broken_wordnet_is_reported_instead_of_raised(self):
        with _broken_wordnet(zipfile.BadZipFile("File is not a zip file")):
            corpora = NltkCorpora()
            self.assertIsNone(corpora.wordnet())
        status = corpora.status()["wordnet"]
        self.assertFalse(status.available)
        self.assertIn("BadZipFile", status.detail)

    def test_a_broken_wordnet_does_not_take_the_stopwords_with_it(self):
        with _broken_wordnet(OSError("gone")):
            corpora = NltkCorpora()
            self.assertIsNone(corpora.wordnet())
        self.assertTrue(corpora.stopwords())

    def test_stopwords_fall_back_when_the_corpus_is_unusable(self):
        with mock.patch.object(corpora_module, "nltk_stopwords",
                               mock.Mock(words=mock.Mock(side_effect=LookupError("no stopwords")))):
            corpora = NltkCorpora()
            self.assertEqual(corpora.stopwords(), set(FALLBACK_STOPWORDS))
        self.assertFalse(corpora.status()["stopwords"].available)

    def test_downloading_is_not_attempted_by_default(self):
        with _broken_wordnet(OSError("gone")), mock.patch.object(corpora_module.nltk, "download") as download:
            NltkCorpora().wordnet()
        download.assert_not_called()


class RepairTest(unittest.TestCase):
    def test_a_corrupt_archive_is_deleted_and_the_corpus_downloaded_once(self):
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / "corpora" / "wordnet.zip"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"this is not a zip")
            self.assertTrue(_is_corrupt(archive))

            def fake_download(name, download_dir, quiet=True):
                _write_zip(Path(download_dir) / "corpora" / f"{name}.zip")
                return True

            synsets = mock.Mock(side_effect=[zipfile.BadZipFile("broken"), ["synset"]])
            with mock.patch.object(corpora_module, "nltk_wordnet", mock.Mock(synsets=synsets)), \
                    mock.patch.object(corpora_module.nltk.data, "path", [scratch]), \
                    mock.patch.object(corpora_module.nltk, "download", side_effect=fake_download) as download:
                corpora = NltkCorpora(allow_download=True, download_dir=str(Path(scratch) / "downloads"))
                self.assertIsNotNone(corpora.wordnet())

            download.assert_called_once()
            self.assertFalse(archive.exists(), "the corrupt archive was left in place")
            self.assertTrue(corpora.status()["wordnet"].available)


class CorruptArchiveDetectionTest(unittest.TestCase):
    def test_a_truncated_archive_is_corrupt_and_a_real_one_is_not(self):
        with tempfile.TemporaryDirectory() as scratch:
            good = _write_zip(Path(scratch) / "good.zip")
            bad = Path(scratch) / "bad.zip"
            bad.write_bytes(b"PK\x03\x04 truncated")
            self.assertFalse(_is_corrupt(good))
            self.assertTrue(_is_corrupt(bad))


if __name__ == "__main__":
    unittest.main()
