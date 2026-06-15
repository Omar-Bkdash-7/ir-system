import re
import unicodedata
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import logging

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
stemmer = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

def preprocess(text: str, mode: str = "document") -> list[str]:
    if not text or not text.strip():
        logger.warning(f"Empty or whitespace-only text received in {mode} mode")
        return []
    
    # (a) Unicode normalization (NFKC)
    text = unicodedata.normalize("NFKC", text)
    
    # (b) Lowercasing
    text = text.lower()
    
    # (c) + (f) spaCy processing on full context (tokenization + lemmatization together)
    doc = nlp(text)
    
    # Extract lemmas and apply filters
    tokens = []
    for token in doc:
        lemma = token.lemma_
        
        # (c) Skip punctuation and special characters
        if not lemma.isalnum():
            continue
        
        # (d) Already tokenized by spaCy
        
        # (e) Stop-word removal + length filter
        if lemma in STOP_WORDS or len(lemma) <= 1:
            continue
        
        tokens.append(lemma)
    
    if not tokens:
        return []
    
    # (g) Porter stemming (NLTK PorterStemmer)
    tokens = [stemmer.stem(t) for t in tokens]
    
    return tokens
