"""
Utility functions for statement 2
"""
import json
from typing import List

import pandas as pd
import spacy
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords

STOPWORDS = stopwords.words("english")

NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
POS_TAGS=["NOUN", "ADJ", "VERB", "ADV"]


def save_json(data: dict, filename: str) -> None:
    """Save data as json."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filename: str) -> dict:
    """Load json."""
    with open(filename, "r") as f:
        return json.load(f)


def load_data():
    """Load data."""
    df = pd.read_excel(
        "DAE002/DS2-assessment-simulated-employee-text.xlsx",
        sheet_name="Sheet1",
    )
    return df


def preprocess(data: List[str]) -> List[str]:
    """Preprocess list of sentences."""
    # Split sentences to words
    data_words = list(sent_to_words(data))

    # Build bigram model
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    bigram_model = gensim.models.phrases.Phraser(bigram)

    # Remove stop words
    data_words_nostops = remove_stopwords(data_words)

    # Form bigrams
    data_words_bigrams = make_bigrams(bigram_model, data_words_nostops)

    # Do lemmatization
    data_lemmatized = lemmatize(data_words_bigrams)
    return data_lemmatized


def sent_to_words(sentences: List[str]):
    """
    Converts sentence to words. it also removes punctuation, lowercases the text,
    and removes words that are too short or too long.
    """
    for sentence in sentences:
        yield simple_preprocess(str(sentence), deacc=True)


def remove_stopwords(texts: List[List[str]]) -> List[List[str]]:
    """Given a bigram model and a list of texts, return a list of lists of bigrams. """
    return [[word for word in doc if word not in STOPWORDS] for doc in texts]


def make_bigrams(bigram_model, texts):
    """Given a bigram model and a list of texts, return a list of lists of bigrams."""
    return [bigram_model[doc] for doc in texts]


def lemmatize(texts: List[List[str]]) -> List[List[str]]:
    """Lemmatize a list of texts."""
    texts_out = []
    for sent in texts:
        doc = NLP(" ".join(sent))
        if POS_TAGS is not None:
            tokens_lemma = [token.lemma_ for token in doc if token.pos_ in POS_TAGS]
        else:
            tokens_lemma = [token.lemma_ for token in doc]
        texts_out.append(tokens_lemma)
    return texts_out


def evaluate(lda_model, corpus, data_lemmatized, id2word):
    """Computes perplexity and coherence score."""
    # Compute perplexity: the lower the better
    perplexity = lda_model.log_perplexity(corpus)
    print(f"  Perplexity = {perplexity:.4f}")

    # Compute coherence score: higher coherence means the topic is more human interpretable.
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence="c_v"
    )
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"  Coherence Score = {coherence_lda:.4f}")
    return perplexity, coherence_lda
