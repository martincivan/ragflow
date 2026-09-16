#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
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

from types import SimpleNamespace

import infinity.rag_tokenizer
import pytest

from common import settings
from rag.nlp import dataset_language, rag_tokenizer

# Slovak/Czech stemming and folding are implemented in infinity-sdk; skip on
# releases that predate them instead of failing on the pinned version.
_SDK_STEMS_AND_FOLDS = hasattr(infinity.rag_tokenizer, "lower_latin")
needs_stemming_sdk = pytest.mark.skipif(not _SDK_STEMS_AND_FOLDS, reason="infinity-sdk without Slovak/Czech stemming")


@pytest.fixture(autouse=True)
def non_infinity_engine(monkeypatch):
    # tokenize() is a passthrough under DOC_ENGINE=infinity (tokenization
    # happens server-side); force the local tokenizer path.
    monkeypatch.setattr(settings, "DOC_ENGINE_INFINITY", False, raising=False)
    yield
    rag_tokenizer.tokenizer.set_language("English")


@needs_stemming_sdk
@pytest.mark.p2
@pytest.mark.parametrize("language", ["Slovak", "slovak", "Czech", "czech"])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("škola", "skol"),
        ("daňové priznanie", "dan priznani"),
        ("požiarna bezpečnosť", "poziarn bezpecnost"),
        ("příliš žluťoučký kůň", "prilis zlutouck kun"),
    ],
)
def test_stem_and_fold_languages_keep_words_whole(language, text, expected):
    rag_tokenizer.tokenizer.set_language(language)

    assert rag_tokenizer.tokenize(text) == expected


@needs_stemming_sdk
@pytest.mark.p2
def test_stem_and_fold_languages_match_unaccented_queries():
    # Users routinely type Slovak without diacritics; index and query
    # tokens must agree either way.
    rag_tokenizer.tokenizer.set_language("Slovak")

    assert rag_tokenizer.tokenize("požiarna bezpečnosť") == rag_tokenizer.tokenize("poziarna bezpecnost")


@needs_stemming_sdk
@pytest.mark.p2
@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("Slovak", "školy škôl školám školách"),
        ("Czech", "školy škol školám ve školách"),
    ],
)
def test_stem_and_fold_languages_collapse_case_forms(language, text):
    # The point of stemming on top of folding: the case forms of "škola"
    # become one term, which folding alone cannot do.
    rag_tokenizer.tokenizer.set_language(language)
    tokens = [tk for tk in rag_tokenizer.tokenize(text).split() if tk != "ve"]

    assert set(tokens) == {"skol"}


@pytest.mark.p2
def test_slovak_strips_the_superlative_prefix():
    rag_tokenizer.tokenizer.set_language("Slovak")

    assert rag_tokenizer.tokenize("najlepšie lepšie") == "lepsi lepsi"


@pytest.mark.p2
def test_stem_and_fold_languages_leave_english_words_alone():
    # English lemmatization and Snowball stemming are off, and no Slovak rule
    # matches this word, so it survives unchanged.
    rag_tokenizer.tokenizer.set_language("Slovak")

    assert rag_tokenizer.tokenize("running") == "running"


@needs_stemming_sdk
@pytest.mark.p2
def test_fine_grained_tokenize_preserves_stemmed_tokens():
    # fine_grained_tokenize runs over tokens tokenize() already stemmed and
    # folded, so it must not stem them a second time.
    rag_tokenizer.tokenizer.set_language("Slovak")
    tks = rag_tokenizer.tokenize("daňové priznanie k dani z nehnuteľností")

    assert rag_tokenizer.fine_grained_tokenize(tks).split() == tks.split()


@needs_stemming_sdk
@pytest.mark.p2
def test_switching_back_to_english_restores_stemming():
    rag_tokenizer.tokenizer.set_language("Slovak")
    rag_tokenizer.tokenizer.set_language("English")

    assert rag_tokenizer.tokenize("running") == "run"
    # Accented words keep the legacy fragmentation outside stem-and-fold
    # languages.
    assert rag_tokenizer.tokenize("škola") == "š kola"


@pytest.mark.p2
@pytest.mark.parametrize(
    ("languages", "expected"),
    [
        ([], None),
        (["Slovak"], "Slovak"),
        (["Slovak", " slovak "], "Slovak"),
        # No single answer for a mixed set: the query is tokenized once, but the
        # datasets were indexed under different rules. Fall back to English.
        (["Slovak", "English"], None),
        (["English", "Slovak"], None),
        (["Slovak", None], None),
        ([None, None], None),
    ],
)
def test_dataset_language_requires_agreement(languages, expected):
    kbs = [SimpleNamespace(id=f"kb-{i}", language=lang) for i, lang in enumerate(languages)]
    assert dataset_language(kbs) == expected
