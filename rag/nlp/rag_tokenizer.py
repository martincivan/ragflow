#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import unicodedata

import infinity.rag_tokenizer

# Languages tokenized with diacritics folded to ASCII and no stemming. The
# tokenizer's SPLIT_CHAR only keeps ASCII letter runs whole, so an accented
# word is otherwise fragmented before it reaches the index ("škola" -> "š
# kola"), and neither language has a Snowball stemmer.
_DIACRITIC_FOLDING_LANGUAGES = {"slovak", "czech"}


def _fold_char(char: str) -> str:
    # Latin-1 Supplement and Latin Extended-A letters whose NFD decomposition
    # is one ASCII letter plus combining marks fold to that letter (Š -> S,
    # ď -> d); everything else (æ, ø, ł, ß, non-Latin scripts) is kept. Same
    # rule as RAGAnalyzer::FoldDiacritics in the C++ analyzer.
    if not (0xC0 <= ord(char) < 0x180):
        return char
    decomposed = unicodedata.normalize("NFD", char)
    base = decomposed[0]
    if base.isascii() and base.isalpha() and all(unicodedata.combining(c) for c in decomposed[1:]):
        return base
    return char


def fold_diacritics(text: str) -> str:
    """Fold Latin diacritics to ASCII: 'škola' -> 'skola'."""
    if text.isascii():
        return text
    return "".join(_fold_char(c) for c in text)


class RagTokenizer(infinity.rag_tokenizer.RagTokenizer):
    # infinity-sdk 0.7.3 knows nothing about Slovak or Czech: set_language()
    # finds no Snowball entry, logs "keeping defaults" and leaves the English
    # stemmer and lemmatizer in place, so a Slovak dataset is tokenized exactly
    # like an English one. Elasticsearch cannot make up for it either -- the
    # *_tks fields use the `whitespace` analyzer, so the token strings written
    # here are the only normalization there is. The folding below mirrors
    # infinity's own tokenizer (infiniflow/infinity#3436) so the behaviour is
    # already correct on the released SDK; it is idempotent and can be dropped
    # once a release carries it.
    _fold_diacritics = False

    def set_language(self, language: str):
        super().set_language(language)
        self._fold_diacritics = language.strip().lower() in _DIACRITIC_FOLDING_LANGUAGES

    def _normalize_token(self, t: str) -> str:
        # A folding language is not stemmed: super().set_language() kept the
        # English stemmer, which would mangle the folded Slovak.
        if self._fold_diacritics:
            return t
        return super()._normalize_token(t)

    def tokenize(self, line: str) -> str:
        from common import settings  # moved from the top of the file to avoid circular import

        if settings.DOC_ENGINE_INFINITY:
            return line
        else:
            if self._fold_diacritics:
                line = fold_diacritics(line)
            return super().tokenize(line)

    def fine_grained_tokenize(self, tks: str) -> str:
        from common import settings  # moved from the top of the file to avoid circular import

        if settings.DOC_ENGINE_INFINITY:
            return tks
        else:
            return super().fine_grained_tokenize(tks)


def is_chinese(s):
    return infinity.rag_tokenizer.is_chinese(s)


def is_number(s):
    return infinity.rag_tokenizer.is_number(s)


def is_alphabet(s):
    return infinity.rag_tokenizer.is_alphabet(s)


def naive_qie(txt):
    return infinity.rag_tokenizer.naive_qie(txt)


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B
