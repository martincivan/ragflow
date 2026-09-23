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
"""What the agentic path hands the answer model, and what it publishes back.

Two halves of one citation: the answer model must not see an identifier it would
cite (nothing but an evidence-block number resolves), and the pool it cites has to
carry the document list the client opens the citation through.
"""

from types import SimpleNamespace

from rag.advanced_rag.agentic_rag_graph import _render_slot_draft, _strip_internal_pointers
from rag.advanced_rag.harness.chunk_utils import admit_chunk


def _slot(**kwargs):
    base = {
        "id": 0,
        "type": "number",
        "candidate": "25226",
        "candidate_strength": 0.94,
        "discovered_clues": [],
        "question_clues": [],
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _table(*slots):
    return SimpleNamespace(state=list(slots))


class TestSlotDraftIdentifiers:
    def test_summary_rendering_carries_no_evidence_ids(self):
        """Rendered without slot_evidence — the form the answer prompt gets — the
        draft must not name a chunk: the answer model cites whatever id it reads."""
        draft = _render_slot_draft(_table(_slot()))
        assert "evidence_ids" not in draft
        assert "terminal=" not in draft
        assert "25226" in draft

    def test_sca_rendering_keeps_evidence_ids(self):
        """The SCA verifies a candidate against the passage that produced it, so
        its rendering keeps the ids the summary drops."""
        draft = _render_slot_draft(
            _table(_slot()),
            slot_evidence={"0": {"evidence_ids": ["a1b2c3d4e5f60718"], "terminal_type": "state"}},
        )
        assert "a1b2c3d4e5f60718" in draft

    def test_clue_pointers_are_stripped_from_the_draft(self):
        """The session model quotes chunk ids and ordinals back into its clues;
        the answer model then cites "chunk 1" as [ID:1] over a one-chunk pool."""
        draft = _render_slot_draft(_table(_slot(discovered_clues=["doc 9f8e7d6c5b4a39281706f5e4d3c2b1a0, chunk 1: 'price excluding VAT 25226'"])))
        assert "9f8e7d6c5b4a39281706f5e4d3c2b1a0" not in draft
        assert "chunk 1" not in draft
        assert "price excluding VAT 25226" in draft

    def test_unresolved_slots_still_listed(self):
        draft = _render_slot_draft(_table(_slot(candidate=None, question_clues=["price of the offer"])))
        assert "NOT RESOLVED" in draft
        assert "price of the offer" in draft


class TestStripInternalPointers:
    def test_keeps_text_that_names_no_chunk(self):
        assert _strip_internal_pointers("price excluding VAT 25226") == "price excluding VAT 25226"

    def test_strips_document_and_chunk_pointers(self):
        assert _strip_internal_pointers("doc 9f8e7d6c5b4a39281706f5e4d3c2b1a0, chunk 12: 'x'") == "'x'"

    def test_handles_empty_input(self):
        assert _strip_internal_pointers(None) == ""


class TestAdmitChunk:
    def test_pool_growth_records_the_document(self):
        """doc_aggs is how the client turns a citation into an openable document;
        a pool that grows without it yields citations that open nothing."""
        kbinfos = {"chunks": [], "doc_aggs": []}
        admit_chunk(kbinfos, {"chunk_id": "c1", "doc_id": "d1", "docnm_kwd": "quarterly-report.pdf"})
        assert kbinfos["chunks"] == [{"chunk_id": "c1", "doc_id": "d1", "docnm_kwd": "quarterly-report.pdf"}]
        assert kbinfos["doc_aggs"] == [{"doc_id": "d1", "doc_name": "quarterly-report.pdf", "count": 1}]

    def test_second_chunk_of_the_same_document_counts_once(self):
        kbinfos = {"chunks": [], "doc_aggs": []}
        admit_chunk(kbinfos, {"chunk_id": "c1", "doc_id": "d1", "docnm_kwd": "quarterly-report.pdf"})
        admit_chunk(kbinfos, {"chunk_id": "c2", "doc_id": "d1", "docnm_kwd": "quarterly-report.pdf"})
        assert len(kbinfos["chunks"]) == 2
        assert kbinfos["doc_aggs"] == [{"doc_id": "d1", "doc_name": "quarterly-report.pdf", "count": 2}]

    def test_a_named_passage_fills_in_a_pseudo_chunks_missing_title(self):
        """Claim rows enter the pool with a doc_id and no title; the first real
        passage of that document supplies the name the UI puts on the link."""
        kbinfos = {"chunks": [], "doc_aggs": []}
        admit_chunk(kbinfos, {"chunk_id": "claim_x", "doc_id": "d1"})
        admit_chunk(kbinfos, {"chunk_id": "c1", "doc_id": "d1", "docnm_kwd": "quarterly-report.pdf"})
        assert kbinfos["doc_aggs"] == [{"doc_id": "d1", "doc_name": "quarterly-report.pdf", "count": 2}]

    def test_chunk_without_a_document_is_still_pooled(self):
        kbinfos = {"chunks": [], "doc_aggs": []}
        admit_chunk(kbinfos, {"chunk_id": "c1"})
        assert len(kbinfos["chunks"]) == 1
        assert kbinfos["doc_aggs"] == []
