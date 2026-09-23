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

import re
import unicodedata

import infinity.rag_tokenizer
import snowballstemmer

# Languages whose tokens are stemmed while they still carry their diacritics
# and are folded to ASCII only afterwards. Folding is what lets a query typed
# without accents match the index; stemming is what lets "škôl" and "školy"
# match each other, and it has to run first because both stemmers match
# suffixes written with accented letters.
_STEM_AND_FOLD_LANGUAGES = {"slovak", "czech"}

# Letters a stem-and-fold token may be built from: ASCII plus the Latin-1
# Supplement and Latin Extended-A block that _fold_char covers.
_LATIN_WORD_PATTERN = re.compile(r"[a-zA-Z_À-ſ-]+$")


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


def _build_latin_lower_table() -> dict:
    # U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE is the only letter in the
    # range whose str.lower() is two characters ("i" + U+0307); map it to a
    # plain "i" so the table stays one-to-one and the C++ RAGAnalyzer can
    # mirror it exactly.
    table = {}
    for codepoint in range(0xC0, 0x180):
        lowered = chr(codepoint).lower()
        table[codepoint] = lowered if len(lowered) == 1 else "i"
    return table


_LATIN_LOWER_TABLE = _build_latin_lower_table()


def lower_latin(text: str) -> str:
    """Lowercase Latin-1 Supplement and Latin Extended-A letters: 'Škola' -> 'škola'."""
    if text.isascii():
        return text
    return text.translate(_LATIN_LOWER_TABLE)


class _SlovakStemmer:
    """Light stemmer for Slovak.

    Ported from the Slovak stemmer in wikimedia/search-extra
    (opensearch-extra-analysis-slovak), licensed by the Wikimedia Foundation
    under the Apache License, Version 2.0. That implementation follows the
    structure of the Apache Lucene "Light Stemmer for Czech"
    (org.apache.lucene.analysis.cz.CzechStemmer, Apache License 2.0), which
    implements Dolamic and Savoy, "Indexing and stemming approaches for the
    Czech language" (Information Processing & Management 45(6), 2009).

    The Slovak suffix inventory is adapted from stemm-sk, Copyright (c) 2015
    Marek Suppa, licensed under the MIT License:

      Permission is hereby granted, free of charge, to any person obtaining a
      copy of this software and associated documentation files (the
      "Software"), to deal in the Software without restriction, including
      without limitation the rights to use, copy, modify, merge, publish,
      distribute, sublicense, and/or sell copies of the Software, and to
      permit persons to whom the Software is furnished to do so, subject to
      the following conditions:

      The above copyright notice and this permission notice shall be included
      in all copies or substantial portions of the Software.

    Input is expected to be lowercase and to still carry its diacritics.
    """

    @staticmethod
    def _ends_with(chars: list, length: int, suffix: str) -> bool:
        return length >= len(suffix) and chars[length - len(suffix) : length] == list(suffix)

    @classmethod
    def _palatalise(cls, chars: list, length: int) -> int:
        ends_with = cls._ends_with
        if any(ends_with(chars, length, s) for s in ("ci", "ce", "či", "če")):
            chars[length - 2] = "k"  # [cč][ie] -> k
        elif any(ends_with(chars, length, s) for s in ("zi", "ze", "ži", "že")):
            chars[length - 2] = "h"  # [zž][ie] -> h
        elif any(ends_with(chars, length, s) for s in ("čte", "čti", "čtí")):
            chars[length - 3] = "c"  # čt[eií] -> ck
            chars[length - 2] = "k"
        elif any(ends_with(chars, length, s) for s in ("šte", "šti", "ští")):
            chars[length - 3] = "s"  # št[eií] -> sk
            chars[length - 2] = "k"
        return length - 1

    @classmethod
    def _remove_case(cls, chars: list, length: int) -> int:
        ends_with = cls._ends_with
        if length > 7 and ends_with(chars, length, "atoch"):
            return length - 5
        if length > 6 and ends_with(chars, length, "aťom"):
            return cls._palatalise(chars, length - 3)
        if length > 5:
            if any(ends_with(chars, length, s) for s in ("och", "ich", "ích", "ého", "ami", "emi", "ému", "ete", "eti", "iho", "ího", "ími", "imu", "aťa")):
                return cls._palatalise(chars, length - 2)
            if any(ends_with(chars, length, s) for s in ("ách", "ata", "aty", "ých", "ové", "ovi", "ými")):
                return length - 3
        if length > 4:
            if ends_with(chars, length, "om"):
                return cls._palatalise(chars, length - 1)
            if any(ends_with(chars, length, s) for s in ("es", "ém", "ím")):
                return cls._palatalise(chars, length - 2)
            if any(ends_with(chars, length, s) for s in ("úm", "at", "ám", "os", "us", "ým", "mi", "ou", "ej")):
                return length - 2
        if length > 3:
            if chars[length - 1] in ("e", "i", "í"):
                return cls._palatalise(chars, length)
            if chars[length - 1] in ("ú", "y", "a", "o", "á", "é", "ý"):
                return length - 1
        return length

    @classmethod
    def _remove_possessives(cls, chars: list, length: int) -> int:
        if length > 5:
            if cls._ends_with(chars, length, "ov"):
                return length - 2
            if cls._ends_with(chars, length, "in"):
                return cls._palatalise(chars, length - 1)
        return length

    @staticmethod
    def _remove_prefixes(chars: list, length: int) -> int:
        if length > 5 and chars[0:3] == ["n", "a", "j"]:
            del chars[0:3]
            return length - 3
        return length

    def stemWord(self, word: str) -> str:  # noqa: N802 - mirrors the snowballstemmer API
        chars = list(word)
        length = self._remove_case(chars, len(chars))
        length = self._remove_possessives(chars, length)
        length = self._remove_prefixes(chars, length)
        return "".join(chars[:length])


class RagTokenizer(infinity.rag_tokenizer.RagTokenizer):
    # infinity-sdk 0.7.3 knows nothing about Slovak or Czech: set_language()
    # finds no Snowball entry, logs "keeping defaults" and leaves the English
    # stemmer and lemmatizer in place, so a Slovak dataset is tokenized exactly
    # like an English one. Elasticsearch cannot make up for it either -- the
    # *_tks fields use the `whitespace` analyzer, so the token strings written
    # here are the only normalization there is. The overrides below mirror
    # infinity's own tokenizer so the behaviour is already correct on the
    # released SDK; they can be dropped once a release carries it.

    # The SDK's SPLIT_CHAR keeps only ASCII letter runs whole, so it cuts an
    # accented word at the accent ('škola' -> 'š', 'kola') and the stemmer
    # would never see a whole word. This variant also admits the block
    # fold_diacritics covers.
    SPLIT_CHAR_LATIN = r"([ ,\.<>/?;:'\[\]\\`!@#$%^&*\(\)\{\}\|_+=《》，。？、；‘’：“”【】~！￥%……（）——-]+|[a-zA-Z0-9À-ſ,\.-]+)"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._split_char_ascii = self.SPLIT_CHAR
        self._stem_and_fold = False
        self._latin_stemmer = None

    def set_language(self, language: str):
        super().set_language(language)
        lang_key = language.strip().lower()
        self._stem_and_fold = lang_key in _STEM_AND_FOLD_LANGUAGES
        self.SPLIT_CHAR = self.SPLIT_CHAR_LATIN if self._stem_and_fold else self._split_char_ascii
        if self._stem_and_fold:
            # Snowball has no Slovak algorithm, so Slovak uses the light
            # stemmer above; Czech uses Snowball's own, added in Snowball
            # 3.1.0 and absent from the nltk copy the SDK stems with. It goes
            # in its own slot rather than self.stemmer, which the SDK calls
            # with a different API and leaves in place for a language it does
            # not know.
            self._latin_stemmer = _SlovakStemmer() if lang_key == "slovak" else snowballstemmer.stemmer(lang_key)

    def _normalize_token(self, t: str) -> str:
        # Stem the accented token and fold its stem, so that 'škôl' and
        # 'školy' meet at 'skol'; folding first would strip the accents the
        # suffix rules match on.
        if self._stem_and_fold:
            if _LATIN_WORD_PATTERN.match(t):
                t = self._latin_stemmer.stemWord(t)
            return fold_diacritics(t)
        return super()._normalize_token(t)

    def tokenize(self, line: str) -> str:
        from common import settings  # moved from the top of the file to avoid circular import

        if settings.DOC_ENGINE_INFINITY:
            return line
        if self._stem_and_fold:
            # The SDK lowercases with str.lower(), which the ASCII-only
            # lowercase in the C++ analyzer cannot match on accented letters;
            # do it here, where both sides do it the same way.
            line = lower_latin(line)
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
