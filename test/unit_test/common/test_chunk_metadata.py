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

import pytest

from common import chunk_metadata as cm
from common.metadata_es_filter import UnsupportedMetaFilter

# --- configuration ------------------------------------------------------------


def test_parse_config_reads_whitelist_and_flags():
    cfg = cm.parse_config({"chunk_metadata": {"enabled": True, "fields": ["project", "phase", "project"], "ready": False}})
    assert cfg.enabled and not cfg.ready and not cfg.active
    assert cfg.fields == ["project", "phase"]


def test_parse_config_ignores_invalid_config():
    assert cm.parse_config(None) is None
    assert cm.parse_config({"chunk_metadata": "yes"}) is None
    # a reserved or unsafe key disables the feature instead of raising at query time
    assert cm.parse_config({"chunk_metadata": {"enabled": True, "fields": ["doc_id"], "ready": True}}) is None
    assert cm.parse_config({"chunk_metadata": {"enabled": True, "fields": ["has space"], "ready": True}}) is None


def test_validate_fields_caps_count():
    with pytest.raises(ValueError):
        cm.validate_fields([f"k{i}" for i in range(cm.MAX_FIELDS + 1)])


def test_config_for_datasets_requires_all_ready_and_intersects_keys():
    a = {"chunk_metadata": {"enabled": True, "fields": ["project", "phase"], "ready": True}}
    b = {"chunk_metadata": {"enabled": True, "fields": ["phase", "doc_type"], "ready": True}}
    c = {"chunk_metadata": {"enabled": True, "fields": ["phase"], "ready": False}}
    assert cm.config_for_datasets([a, b]).fields == ["phase"]
    assert cm.config_for_datasets([a, c]) is None
    assert cm.config_for_datasets([a, {}]) is None
    assert cm.config_for_datasets([]) is None


# --- chunk fields -------------------------------------------------------------


def test_chunk_fields_types_and_multivalues():
    meta = {
        "project": "2015_062 bytovy dom",
        "project_id": "2015_062",  # underscore digits stay a string
        "phase": ["sp", "dsp"],
        "project_year": 2015,
        "doc_date": "2026-02-12",
        "ratio": "0.35",
        "ignored": "x",
        "empty": "",
    }
    out = cm.chunk_fields(meta, ["project", "project_id", "phase", "project_year", "doc_date", "ratio", "empty", "missing"])
    assert out["meta_project_kwd"] == "2015_062 bytovy dom"
    assert out["meta_project_id_kwd"] == "2015_062" and "meta_project_id_int" not in out
    assert out["meta_phase_kwd"] == ["sp", "dsp"]
    assert out["meta_project_year_kwd"] == "2015" and out["meta_project_year_int"] == 2015
    assert out["meta_doc_date_kwd"] == "2026-02-12" and out["meta_doc_date_dt"] == "2026-02-12"
    assert out["meta_ratio_flt"] == pytest.approx(0.35)
    assert "meta_ignored_kwd" not in out and "meta_empty_kwd" not in out and "meta_missing_kwd" not in out


def test_removal_fields_lists_stale_twins():
    rm = cm.removal_fields({"phase": "sp"}, ["phase", "gone"])
    assert "meta_phase_kwd" not in rm
    assert {"meta_phase_int", "meta_phase_flt", "meta_phase_dt", "meta_gone_kwd", "meta_gone_int", "meta_gone_flt", "meta_gone_dt"} <= set(rm)


# --- filter translation -------------------------------------------------------


def test_is_chunk_filterable():
    keys = ["project", "phase"]
    assert cm.is_chunk_filterable([{"key": "project", "op": "=", "value": "nst"}], keys)
    assert not cm.is_chunk_filterable([{"key": "client", "op": "=", "value": "x"}], keys)  # not whitelisted
    assert not cm.is_chunk_filterable([{"key": "phase", "op": "≠", "value": "sp"}], keys)  # multi-value unsafe
    assert not cm.is_chunk_filterable([], keys)


def test_equal_and_in_are_case_insensitive_terms_on_the_keyword_twin():
    eq = cm.condition_clause({"key": "project", "op": "=", "value": "2018_055 NST"})
    assert eq == {"term": {"meta_project_kwd": {"value": "2018_055 nst", "case_insensitive": True}}}
    num = cm.condition_clause({"key": "year", "op": "=", "value": "2015"})
    assert num["term"]["meta_year_kwd"]["value"] == "2015"
    inn = cm.condition_clause({"key": "phase", "op": "in", "value": "SP, DSP"})
    assert inn["bool"]["minimum_should_match"] == 1
    assert [c["term"]["meta_phase_kwd"]["value"] for c in inn["bool"]["should"]] == ["sp", "dsp"]


def test_ranges_pick_the_typed_twin():
    d = cm.condition_clause({"key": "doc_date", "op": "≥", "value": "2025-04-01"})
    assert d == {"range": {"meta_doc_date_dt": {"gte": "2025-04-01"}}}
    n = cm.condition_clause({"key": "year", "op": "<", "value": "2020"})
    assert n == {"range": {"meta_year_int": {"lt": 2020}}}
    f = cm.condition_clause({"key": "ratio", "op": ">", "value": "0.5"})
    assert f == {"range": {"meta_ratio_flt": {"gt": 0.5}}}


def test_string_operators_and_existence():
    c = cm.condition_clause({"key": "project", "op": "contains", "value": "Bystric"})
    assert c["wildcard"]["meta_project_kwd"] == {"value": "*Bystric*", "case_insensitive": True}
    assert cm.condition_clause({"key": "project", "op": "start with", "value": "2015"})["prefix"]["meta_project_kwd"]["value"] == "2015"
    assert cm.condition_clause({"key": "hop", "op": "empty"}) == {"bool": {"must_not": [{"exists": {"field": "meta_hop_kwd"}}]}}
    assert cm.condition_clause({"key": "hop", "op": "not empty"}) == {"exists": {"field": "meta_hop_kwd"}}


def test_unsupported_operator_raises():
    with pytest.raises(UnsupportedMetaFilter):
        cm.condition_clause({"key": "phase", "op": "not in", "value": "sp"})
    with pytest.raises(UnsupportedMetaFilter):
        cm.condition_clause({"key": "phase", "op": "=", "value": None})
    with pytest.raises(UnsupportedMetaFilter):
        cm.build_chunk_filter([{"key": "phase", "op": "=", "value": "sp"}], logic="xor")


def test_build_chunk_filter_logic():
    conds = [{"key": "project", "op": "=", "value": "nst"}, {"key": "phase", "op": "in", "value": "sp,dsp"}]
    both = cm.build_chunk_filter(conds, "and")
    assert list(both["bool"].keys()) == ["filter"] and len(both["bool"]["filter"]) == 2
    either = cm.build_chunk_filter(conds, "or")
    assert either["bool"]["minimum_should_match"] == 1 and len(either["bool"]["should"]) == 2


# --- boost --------------------------------------------------------------------


def test_parse_boost_defaults_and_validation():
    cfg = cm.parse_boost(
        {
            "method": "semi_auto",
            "manual": [
                {"key": "flow", "op": "=", "value": "expedition", "weight": 0.2},
                {"key": "expedition_date", "op": "max", "weight": 2},  # clamped to 1
                {"key": "flow", "op": "≠", "value": "in"},  # unsupported op dropped
                {"key": "secret", "op": "=", "value": "x"},  # not whitelisted
                {"key": "flow", "op": "="},  # no value
            ],
            "semi_auto": ["phase", "discipline", "secret"],
            "auto_weight": 0.1,
            "max_total": "0.25",
        },
        keys=["flow", "expedition_date", "phase", "discipline"],
    )
    assert cfg.method == "semi_auto" and cfg.uses_llm
    assert [(b.key, b.op, b.weight) for b in cfg.manual] == [("flow", "=", 0.2), ("expedition_date", "max", 1.0)]
    assert cfg.semi_auto == ["phase", "discipline"]
    assert cfg.auto_weight == 0.1 and cfg.max_total == 0.25
    assert cm.parse_boost(None).method == "off"
    assert cm.parse_boost({"manual": [{"key": "a", "op": "=", "value": "b"}]}).method == "manual"


def test_boost_should_clauses_carry_es_boost():
    boosts = [
        cm.BoostCondition("flow", "=", "expedition", 0.2),
        cm.BoostCondition("phase", "in", "sp,dsp", 0.1),
        cm.BoostCondition("doc_date", "≥", "2025-01-01", 0.1),
        cm.BoostCondition("expedition_date", "max", None, 0.1),  # scored in Python only
        cm.BoostCondition("project", "contains", "nst", 0.1),  # not pushed to ES
    ]
    clauses = cm.boost_should_clauses(boosts)
    assert len(clauses) == 3
    assert clauses[0]["term"]["meta_flow_kwd"]["boost"] == pytest.approx(2.0)
    assert clauses[1]["bool"]["boost"] == pytest.approx(1.5)
    assert clauses[2]["range"]["meta_doc_date_dt"] == {"gte": "2025-01-01", "boost": pytest.approx(1.5)}


def test_fields_for_boosts():
    fields = cm.fields_for_boosts([cm.BoostCondition("flow", "=", "x"), cm.BoostCondition("flow", "max")])
    assert fields == ["meta_flow_kwd", "meta_flow_int", "meta_flow_flt", "meta_flow_dt"]


def test_boost_scores_match_max_and_cap():
    chunks = [
        {"meta_flow_kwd": "expedition", "meta_expedition_date_dt": "2026-02-12", "meta_phase_kwd": ["sp", "dsp"]},
        {"meta_flow_kwd": "in", "meta_expedition_date_dt": "2025-12-18"},
        {"meta_flow_kwd": "Expedition"},  # no date → no recency score
    ]
    boosts = [
        cm.BoostCondition("flow", "=", "EXPEDITION", 0.2),
        cm.BoostCondition("expedition_date", "max", None, 0.1),
        cm.BoostCondition("phase", "in", "dsp", 0.05),
        cm.BoostCondition("expedition_date", "≥", "2026-01-01", 0.05),
    ]
    s = cm.boost_scores(boosts, chunks, max_total=0.5)
    assert s[0] == pytest.approx(0.2 + 0.1 + 0.05 + 0.05)  # newest date → full recency weight
    assert s[1] == pytest.approx(0.0)  # oldest date → zero recency, no match
    assert s[2] == pytest.approx(0.2)  # case-insensitive equality
    capped = cm.boost_scores([cm.BoostCondition("flow", "=", "expedition", 1.0)] * 3, chunks, max_total=0.3)
    assert capped[0] == pytest.approx(0.3)


def test_boost_scores_min_and_numeric_range():
    chunks = [{"meta_year_int": 2015}, {"meta_year_int": 2020}, {}]
    s = cm.boost_scores([cm.BoostCondition("year", "min", None, 0.2), cm.BoostCondition("year", ">", "2016", 0.1)], chunks)
    assert s == [pytest.approx(0.2), pytest.approx(0.1), pytest.approx(0.0)]


def test_boost_scores_empty_inputs():
    assert cm.boost_scores([], [{"a": 1}]) == [0.0]
    assert cm.boost_scores([cm.BoostCondition("k", "=", "v")], []) == []


# --- LLM conditions: hard / soft ---------------------------------------------


def test_split_conditions_and_boosts_from_soft():
    conds = [
        {"key": "project", "op": "=", "value": "nst", "strength": "hard"},
        {"key": "phase", "op": "in", "value": "sp,dsp", "strength": "soft"},
        {"key": "discipline", "op": "≠", "value": "fire", "strength": "soft"},  # negative preference dropped
        {"key": "doc_type", "op": "=", "value": "report"},  # default hard
    ]
    hard, soft = cm.split_conditions(conds)
    assert [c["key"] for c in hard] == ["project", "doc_type"] and all("strength" not in c for c in hard)
    boosts = cm.boosts_from_conditions(soft, 0.15, keys=["phase", "discipline"])
    assert [(b.key, b.op, b.weight) for b in boosts] == [("phase", "in", 0.15)]
