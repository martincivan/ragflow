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
"""Chunk-metadata filter/boost plumbing in ``rag.nlp.search.Dealer``."""

import numpy as np
import pytest

import common.settings  # noqa: F401  (import order, see test_search_rerank.py)
from common.chunk_metadata import BoostCondition
from rag.nlp.search import Dealer


def _dealer():
    return Dealer.__new__(Dealer)


def test_get_filters_passes_chunk_metadata_scope():
    d = _dealer()
    req = {
        "kb_ids": ["kb"],
        "doc_ids": None,
        "meta_filter": {"conditions": [{"key": "project", "op": "=", "value": "nst"}], "logic": "and"},
        "meta_boost": [BoostCondition("flow", "=", "expedition", 0.2)],
    }
    cond = d.get_filters(req)
    assert cond["kb_id"] == ["kb"]
    assert "doc_id" not in cond
    assert cond["meta_filter"] == req["meta_filter"]
    assert cond["meta_boost"] == req["meta_boost"]
    # empty scopes are not forwarded
    assert "meta_filter" not in d.get_filters({"kb_ids": ["kb"], "meta_filter": {"conditions": []}})
    assert "meta_boost" not in d.get_filters({"kb_ids": ["kb"], "meta_boost": []})


def test_meta_boost_scores_follow_candidate_order():
    sres = Dealer.SearchResult(
        total=3,
        ids=["c1", "c2", "c3"],
        field={
            "c1": {"meta_flow_kwd": "in"},
            "c2": {"meta_flow_kwd": "expedition", "meta_expedition_date_dt": "2026-02-12"},
            "c3": {"meta_flow_kwd": "expedition", "meta_expedition_date_dt": "2025-12-18"},
        },
    )
    boosts = [BoostCondition("flow", "=", "expedition", 0.2), BoostCondition("expedition_date", "max", None, 0.1)]
    scores = Dealer._meta_boost_scores(boosts, 0.5, sres)
    assert scores.tolist() == pytest.approx([0.0, 0.3, 0.2])
    assert Dealer._meta_boost_scores([], 0.5, sres).tolist() == [0.0, 0.0, 0.0]
    assert isinstance(scores, np.ndarray)
