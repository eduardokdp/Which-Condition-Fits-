"""Representation phi: TF-IDF over word n-grams that never see a masked condition name.

Masking replaces ADHD / anxiety / depression with one token (src/dataprep/lexicon.py). If that token
stayed in the vocabulary, "how often a post names any condition" and bigrams such as "my <mask>" would
still be features, which leaks the label (and the self-report rule used to build it) back into the
model. The analyzer below therefore treats every mask as a hard boundary: the text is split at each
mask, n-grams are formed only inside the pieces, and the mask itself is never a feature.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, strip_accents_unicode

from src.dataprep.lexicon import MASK_TOKEN

# keeps the pronoun "i" and contractions such as "can't" / "i've" (first-person and negation cues)
_TOKEN = re.compile(r"(?u)\b\w+(?:'\w+)?\b")


class MaskAwareAnalyzer:
    """Callable analyzer for TfidfVectorizer (a class, so trained models can be pickled)."""

    def __init__(self, ngram_range=(1, 2)):
        self.ngram_range = tuple(ngram_range)

    def __call__(self, text: str) -> list[str]:
        lo, hi = self.ngram_range
        grams: list[str] = []
        for piece in strip_accents_unicode(str(text).lower()).split(MASK_TOKEN):
            toks = _TOKEN.findall(piece)
            for n in range(lo, hi + 1):
                grams.extend(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
        return grams


def make_vectorizer(fcfg: dict) -> TfidfVectorizer:
    """phi: l2-normalised, sublinear TF-IDF over word uni- and bigrams (settings from the config)."""
    return TfidfVectorizer(analyzer=MaskAwareAnalyzer(fcfg["ngram_range"]), min_df=fcfg["min_df"],
                           max_features=fcfg["max_features"], sublinear_tf=fcfg["sublinear_tf"],
                           norm="l2", dtype=np.float64)
