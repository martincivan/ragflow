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
"""Duplicate files yield chunks with identical text. ``Dealer.retrieval`` must
return each distinct text once, so a file uploaded five times cannot fill the
whole page with the same paragraph, while still reporting where else the text
was found."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from common import settings
from rag.nlp.search import Dealer

pytestmark = pytest.mark.p2


def _sres(chunks):
    """chunks: list of (chunk_id, doc_id, text). The order is the rank order."""
    field = {cid: {"doc_id": did, "docnm_kwd": f"{did}.pdf", "kb_id": "kb", "content_with_weight": text, "content_ltks": text.lower()} for cid, did, text in chunks}
    return Dealer.SearchResult(total=len(chunks), ids=[c[0] for c in chunks], query_vector=[0.1, 0.2], field=field, highlight={})


def test_collapse_keeps_first_copy_and_records_the_rest():
    sres = _sres(
        [
            ("c1", "d1", "The permit is valid for two years."),
            ("c2", "d2", "The permit is  valid for\ntwo years. "),  # same text, different whitespace
            ("c3", "d3", "Fees are due on submission."),
            ("c4", "d4", "The permit is valid for two years."),
        ]
    )

    kept, duplicates_of = Dealer._collapse_duplicate_chunks(sres, [0, 1, 2, 3])

    assert kept == [0, 2]
    assert duplicates_of == {0: [1, 3]}


def test_collapse_never_merges_empty_texts():
    sres = _sres([("c1", "d1", ""), ("c2", "d2", "   "), ("c3", "d3", "x")])

    kept, duplicates_of = Dealer._collapse_duplicate_chunks(sres, [0, 1, 2])

    assert kept == [0, 1, 2]
    assert duplicates_of == {}


def _dealer(monkeypatch, sres, sims):
    """Dealer with search internals stubbed: the candidates are ``sres`` and
    their fused similarities are ``sims`` (aligned with sres.ids)."""
    for flag in ("DOC_ENGINE_INFINITY", "DOC_ENGINE_OCEANBASE", "DOC_ENGINE_SERENEDB", "DOC_ENGINE_GAUSSDB"):
        monkeypatch.setattr(settings, flag, False, raising=False)
    dealer = Dealer.__new__(Dealer)
    dealer.qryr = MagicMock()
    dealer.search = AsyncMock(return_value=sres)
    dealer._prune_deleted_chunks = AsyncMock(side_effect=lambda s: s)
    dealer._knn_scores = AsyncMock(return_value={})
    dealer.rerank_with_knn = MagicMock(return_value=(sims, sims, sims))
    return dealer


async def test_retrieval_returns_distinct_texts_with_their_duplicates(monkeypatch):
    sres = _sres(
        [
            ("c1", "d1", "The permit is valid for two years."),
            ("c2", "d2", "The permit is valid for two years."),
            ("c3", "d3", "Fees are due on submission."),
            ("c4", "d4", "The permit is valid for two years."),
        ]
    )
    dealer = _dealer(monkeypatch, sres, [0.9, 0.8, 0.7, 0.6])

    ranks = await dealer.retrieval("permit", None, ["t"], ["kb"], 1, 2, similarity_threshold=0.1, rerank_candidates_count=64)

    # Two distinct texts, so total and the page are 2 even though four chunks matched.
    assert ranks["total"] == 2
    assert [c["chunk_id"] for c in ranks["chunks"]] == ["c1", "c3"]
    assert ranks["chunks"][0]["duplicates"] == [
        {"chunk_id": "c2", "document_id": "d2", "document_name": "d2.pdf", "dataset_id": "kb", "similarity": 0.8},
        {"chunk_id": "c4", "document_id": "d4", "document_name": "d4.pdf", "dataset_id": "kb", "similarity": 0.6},
    ]
    assert ranks["chunks"][1]["duplicates"] == []
    # Documents that only contributed duplicates are not counted as distinct hits.
    assert [(a["doc_id"], a["count"]) for a in ranks["doc_aggs"]] == [("d1", 1), ("d3", 1)]


async def test_retrieval_can_keep_duplicates(monkeypatch):
    sres = _sres([("c1", "d1", "same"), ("c2", "d2", "same")])
    dealer = _dealer(monkeypatch, sres, [0.9, 0.8])

    ranks = await dealer.retrieval("q", None, ["t"], ["kb"], 1, 5, similarity_threshold=0.1, collapse_duplicates=False)

    assert ranks["total"] == 2
    assert [c["chunk_id"] for c in ranks["chunks"]] == ["c1", "c2"]
    assert "duplicates" not in ranks["chunks"][0]


def test_chunks_format_keeps_duplicates_for_chat_and_agent_references():
    from rag.prompts.generator import chunks_format

    dup = {"chunk_id": "c2", "document_id": "d2", "document_name": "d2.pdf", "dataset_id": "kb", "similarity": 0.8}
    formatted = chunks_format({"chunks": [{"chunk_id": "c1", "doc_id": "d1", "docnm_kwd": "d1.pdf", "kb_id": "kb", "duplicates": [dup]}, {"chunk_id": "c3"}]})

    assert formatted[0]["duplicates"] == [dup]
    assert formatted[1]["duplicates"] == []
