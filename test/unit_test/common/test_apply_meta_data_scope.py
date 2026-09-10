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
"""``apply_meta_data_scope``: chunk-level filters and boosts next to the
classic doc-id resolution, and the wrapper keeping the old contract."""

import pytest

from common import chunk_metadata as cm
from common import metadata_utils
from rag.prompts import generator

ACTIVE = cm.ChunkMetadataConfig(enabled=True, fields=["project", "phase", "flow", "expedition_date"], ready=True)
NOT_READY = cm.ChunkMetadataConfig(enabled=True, fields=["project", "phase"], ready=False)
METAS = {"project": {"nst": ["d1", "d2"], "bystricka": ["d3"]}, "phase": {"sp": ["d1"], "rp": ["d2", "d3"]}}


def _llm(conditions, logic="and"):
    """Fake gen_meta_filter that records what it was asked for."""
    calls = []

    async def fake(chat_mdl, meta_data, query, constraints=None, allow_soft=False):
        calls.append({"keys": sorted(meta_data.keys()), "constraints": constraints, "allow_soft": allow_soft})
        return {"logic": logic, "conditions": conditions}

    return fake, calls


@pytest.mark.asyncio
async def test_manual_hard_condition_becomes_chunk_filter_when_active(monkeypatch):
    monkeypatch.setattr(metadata_utils, "_try_meta_pushdown", lambda *a: pytest.fail("doc-id path must not run"))
    scope = await metadata_utils.apply_meta_data_scope(
        {"method": "manual", "manual": [{"key": "project", "op": "=", "value": "nst"}]},
        METAS,
        "q",
        None,
        None,
        kb_ids=["kb"],
        chunk_meta=ACTIVE,
    )
    assert scope.doc_ids == []
    assert scope.chunk_filter == {"conditions": [{"key": "project", "op": "=", "value": "nst"}], "logic": "and"}
    assert scope.boosts == []


@pytest.mark.asyncio
async def test_manual_falls_back_to_doc_ids_when_not_ready_or_key_unknown(monkeypatch):
    monkeypatch.setattr(metadata_utils, "_try_meta_pushdown", lambda kb_ids, conds, logic: ["d1", "d2"])
    for chunk_meta in (None, NOT_READY):
        scope = await metadata_utils.apply_meta_data_scope(
            {"method": "manual", "manual": [{"key": "project", "op": "=", "value": "nst"}]},
            METAS,
            "q",
            None,
            None,
            kb_ids=["kb"],
            chunk_meta=chunk_meta,
        )
        assert scope.doc_ids == ["d1", "d2"] and scope.chunk_filter is None
    # whitelisted keys only: "client" is not on chunks
    scope = await metadata_utils.apply_meta_data_scope(
        {"method": "manual", "manual": [{"key": "client", "op": "=", "value": "x"}]},
        METAS,
        "q",
        None,
        None,
        kb_ids=["kb"],
        chunk_meta=ACTIVE,
    )
    assert scope.doc_ids == ["d1", "d2"] and scope.chunk_filter is None
    # negatives stay on the doc-id path (multi-value semantics)
    scope = await metadata_utils.apply_meta_data_scope(
        {"method": "manual", "manual": [{"key": "phase", "op": "≠", "value": "sp"}]},
        METAS,
        "q",
        None,
        None,
        kb_ids=["kb"],
        chunk_meta=ACTIVE,
    )
    assert scope.chunk_filter is None


@pytest.mark.asyncio
async def test_manual_no_match_keeps_the_minus_999_contract(monkeypatch):
    monkeypatch.setattr(metadata_utils, "_try_meta_pushdown", lambda *a: [])
    scope = await metadata_utils.apply_meta_data_scope(
        {"method": "manual", "manual": [{"key": "client", "op": "=", "value": "nobody"}]},
        METAS,
        "q",
        None,
        None,
        kb_ids=["kb"],
        chunk_meta=ACTIVE,
    )
    assert scope.doc_ids == ["-999"]


@pytest.mark.asyncio
async def test_manual_boosts_are_carried_when_active():
    filt = {
        "method": "manual",
        "manual": [],
        "boost": {"manual": [{"key": "flow", "op": "=", "value": "expedition", "weight": 0.2}, {"key": "expedition_date", "op": "max", "weight": 0.1}], "max_total": 0.4},
    }
    scope = await metadata_utils.apply_meta_data_scope(filt, METAS, "q", None, None, kb_ids=["kb"], chunk_meta=ACTIVE)
    assert [(b.key, b.op, b.weight) for b in scope.boosts] == [("flow", "=", 0.2), ("expedition_date", "max", 0.1)]
    assert scope.boost_max_total == 0.4
    inactive = await metadata_utils.apply_meta_data_scope(filt, METAS, "q", None, None, kb_ids=["kb"], chunk_meta=None)
    assert inactive.boosts == [] and inactive.doc_ids == []


@pytest.mark.asyncio
async def test_auto_splits_hard_and_soft_in_one_llm_call(monkeypatch):
    fake, calls = _llm(
        [
            {"key": "project", "op": "=", "value": "nst", "strength": "hard"},
            {"key": "phase", "op": "in", "value": "sp,dsp", "strength": "soft"},
        ]
    )
    monkeypatch.setattr(generator, "gen_meta_filter", fake)
    monkeypatch.setattr(metadata_utils, "_try_meta_pushdown", lambda *a: pytest.fail("doc-id path must not run"))
    scope = await metadata_utils.apply_meta_data_scope(
        {"method": "auto", "boost": {"method": "auto", "auto_weight": 0.12}},
        METAS,
        "which phase?",
        object(),
        None,
        kb_ids=["kb"],
        chunk_meta=ACTIVE,
    )
    assert len(calls) == 1 and calls[0]["allow_soft"] is True
    assert scope.chunk_filter["conditions"] == [{"key": "project", "op": "=", "value": "nst"}]
    assert [(b.key, b.op, b.value, b.weight) for b in scope.boosts] == [("phase", "in", "sp,dsp", 0.12)]


@pytest.mark.asyncio
async def test_auto_without_boost_config_never_asks_for_soft(monkeypatch):
    fake, calls = _llm([{"key": "project", "op": "=", "value": "nst"}])
    monkeypatch.setattr(generator, "gen_meta_filter", fake)
    scope = await metadata_utils.apply_meta_data_scope({"method": "auto"}, METAS, "q", object(), None, kb_ids=["kb"], chunk_meta=ACTIVE)
    assert calls[0]["allow_soft"] is False
    assert scope.chunk_filter is not None and scope.boosts == []


@pytest.mark.asyncio
async def test_auto_only_soft_conditions_means_no_filter(monkeypatch):
    fake, _ = _llm([{"key": "phase", "op": "=", "value": "sp", "strength": "soft"}])
    monkeypatch.setattr(generator, "gen_meta_filter", fake)
    scope = await metadata_utils.apply_meta_data_scope(
        {"method": "auto", "boost": {"method": "auto"}},
        METAS,
        "q",
        object(),
        None,
        kb_ids=["kb"],
        chunk_meta=ACTIVE,
    )
    # nothing hard → auto's "no filter" contract, but the preference is kept
    assert scope.doc_ids is None
    assert [b.key for b in scope.boosts] == ["phase"]


@pytest.mark.asyncio
async def test_semi_auto_offers_boost_keys_in_the_same_call(monkeypatch):
    fake, calls = _llm([{"key": "project", "op": "=", "value": "nst", "strength": "hard"}, {"key": "phase", "op": "=", "value": "rp", "strength": "soft"}])
    monkeypatch.setattr(generator, "gen_meta_filter", fake)
    scope = await metadata_utils.apply_meta_data_scope(
        {"method": "semi_auto", "semi_auto": [{"key": "project", "op": "="}], "boost": {"method": "semi_auto", "semi_auto": ["phase"]}},
        METAS,
        "q",
        object(),
        None,
        kb_ids=["kb"],
        chunk_meta=ACTIVE,
    )
    assert calls[0]["keys"] == ["phase", "project"] and calls[0]["constraints"] == {"project": "="}
    assert scope.chunk_filter["conditions"][0]["key"] == "project"
    assert [b.key for b in scope.boosts] == ["phase"]


@pytest.mark.asyncio
async def test_soft_conditions_dropped_when_chunk_metadata_inactive(monkeypatch):
    fake, calls = _llm([{"key": "project", "op": "=", "value": "nst"}, {"key": "phase", "op": "=", "value": "sp", "strength": "soft"}])
    monkeypatch.setattr(generator, "gen_meta_filter", fake)
    monkeypatch.setattr(metadata_utils, "_try_meta_pushdown", lambda *a: ["d1"])
    scope = await metadata_utils.apply_meta_data_scope({"method": "auto", "boost": {"method": "auto"}}, METAS, "q", object(), None, kb_ids=["kb"], chunk_meta=None)
    assert calls[0]["allow_soft"] is False
    assert scope.doc_ids == ["d1"] and scope.boosts == [] and scope.chunk_filter is None


@pytest.mark.asyncio
async def test_apply_meta_data_filter_wrapper_keeps_contract(monkeypatch):
    monkeypatch.setattr(metadata_utils, "_try_meta_pushdown", lambda *a: ["d3"])
    assert await metadata_utils.apply_meta_data_filter({"method": "manual", "manual": [{"key": "project", "op": "=", "value": "bystricka"}]}, METAS, "q", None, None, kb_ids=["kb"]) == ["d3"]
    assert await metadata_utils.apply_meta_data_filter(None, METAS, "q", None, ["base"]) == ["base"]
