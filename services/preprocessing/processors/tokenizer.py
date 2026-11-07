import logging
from typing import Optional
import re

import spacy.tokens

from ..core.resources import get_spacy_model
from ..core.exceptions import TokenizationError

logger = logging.getLogger(__name__)

# Number of documents spaCy buffers per internal batch when streaming through
# nlp.pipe(). Sweet spot for en_core_web_sm on CPU.
SPACY_BATCH_SIZE = 64

# Word-aware pattern for the dependency-free (fallback) tokenizer:
# - words and numbers (incl. unicode letters/digits via \w)
# - intra-word punctuation: hyphens ("state-of-the-art"), apostrophes ("don't")
#   and programming-style suffixes ("c++", "c#", "f-16")
# - surrounding punctuation ("machine,", "(fox") separates cleanly
_WORD_PATTERN = re.compile(r"\w+(?:[-+#']\w+)*[-+#]{0,2}")


class SimpleTokenizer:
    """
    Regex-based, dependency-free tokenizer.

    Unlike the earlier whitespace-only split, punctuation attached to words is
    separated, while stylistic intra-word punctuation from the same run
    (hyphens, c++ / f-16 style compounds, apostrophes) is kept intact so the
    downstream filter can preserve hyphenated terms.
    """

    def tokenize(self, text: str) -> list[str]:
        return _WORD_PATTERN.findall(text)

    def tokenize_many(self, texts: list[str], batch_size: Optional[int] = None) -> list[list[str]]:
        """Tokenize a batch of texts. Order is preserved, blanks yield []."""
        return [self.tokenize(text) for text in texts]


class SpacyTokenizer:
    def __init__(self, use_lemmatization: bool = True):
        self.use_lemmatization = use_lemmatization
        self.nlp = get_spacy_model()

    def tokenize(self, text: str) -> list[str]:
        try:
            doc = self.nlp(text)
            return self._collect_tokens(doc)
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise TokenizationError(f"Failed to tokenize text: {e}") from e

    def tokenize_many(self, texts: list[str], batch_size: Optional[int] = None) -> list[list[str]]:
        """
        Tokenize a batch of texts efficiently using spaCy's streaming nlp.pipe.

        Order of results matches the order of `texts`; blanks yield [].
        """
        if not texts:
            return []
        try:
            docs = self.nlp.pipe(
                texts,
                batch_size=batch_size or SPACY_BATCH_SIZE,
                n_process=1,
            )
            return [self._collect_tokens(doc) for doc in docs]
        except Exception as e:
            logger.error(f"Batch tokenization failed: {e}")
            raise TokenizationError(f"Failed to tokenize texts: {e}") from e

    def _collect_tokens(self, doc: "spacy.tokens.Doc") -> list[str]:
        """
        Convert a spaCy doc to tokens, re-joining hyphenated words.

        spaCy splits hyphenated compounds into pieces (e.g. "state-of-the-art"
        becomes state, -, of, -, the, -, art). These pieces are merged back so
        compounds survive the punctuation filter as single searchable tokens.
        Merged compounds are kept verbatim (not lemmatized) so their surface
        form stays consistent regardless of the lemmatization mode.

        A hyphen only glues pieces when it is not surrounded by spaces, so
        spaced compounds like "state - of" (ranges/dashes) stay separate.
        """
        doc_tokens = list(doc)
        tokens: list[str] = []
        use_lemma = self.use_lemmatization
        n = len(doc_tokens)
        i = 0
        while i < n:
            tok = doc_tokens[i]
            # Start a hyphenated compound: alphanumeric piece, hyphen, alphanumeric piece
            if (
                i + 2 < n
                and tok.text.isalnum()
                and tok.whitespace_ == ""
                and doc_tokens[i + 1].text == "-"
                and doc_tokens[i + 1].whitespace_ == ""
                and doc_tokens[i + 2].text.isalnum()
            ):
                parts = [tok.text]
                i += 1
                # Extend while the pattern continues (e.g. state - of - the - art)
                while (
                    i + 1 < n
                    and doc_tokens[i].text == "-"
                    and doc_tokens[i].whitespace_ == ""
                    and doc_tokens[i + 1].text.isalnum()
                ):
                    parts.append("-")
                    parts.append(doc_tokens[i + 1].text)
                    i += 2
                tokens.append("".join(parts))
                continue
            tokens.append(tok.lemma_ if use_lemma else tok.text)
            i += 1
        return tokens
