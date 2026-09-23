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
"""Every citation that reaches the reader must name a chunk the client can open.

The client resolves a marker by indexing the published chunk list with the number
it carries, so a marker carrying anything else renders as literal text. Models
produce plenty of "anything else": full-width brackets, chunk ids copied out of
their own context, indexes past the end of the evidence.
"""

import pytest

from api.db.services.dialog_service import has_citation_markers, repair_bad_citation_formats


def _kbinfos(*chunk_ids):
    return {"chunks": [{"chunk_id": cid, "doc_id": f"doc-{cid}"} for cid in chunk_ids]}


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("The cable run is 50 m【ID:1】.", "The cable run is 50 m[ID:1]."),
        ("The cable run is 50 m【ID: 1】.", "The cable run is 50 m[ID:1]."),
        ("The cable run is 50 m【ID：1】.", "The cable run is 50 m[ID:1]."),
        ("The cable run is 50 m(ID:1).", "The cable run is 50 m[ID:1]."),
        ("The cable run is 50 m[ID: 1].", "The cable run is 50 m[ID:1]."),
        ("The cable run is 50 m[1].", "The cable run is 50 m[ID:1]."),
    ],
)
def test_malformed_brackets_become_canonical_markers(answer, expected):
    repaired, idx = repair_bad_citation_formats(answer, _kbinfos("a", "b"), set())
    assert repaired == expected
    assert idx == {1}


def test_chunk_id_marker_resolves_to_its_position():
    """gpt-oss reads an id out of the research summary and cites it verbatim.

    The id names a real passage, so it is rewritten to that passage's index
    rather than thrown away with the citation.
    """
    kbinfos = _kbinfos("aaa", "a1b2c3d4e5f60718", "ccc")
    repaired, idx = repair_bad_citation_formats("The quoted price is 25226【ID:a1b2c3d4e5f60718】.", kbinfos, set())
    assert repaired == "The quoted price is 25226[ID:1]."
    assert idx == {1}


def test_unknown_chunk_id_marker_is_dropped():
    repaired, idx = repair_bad_citation_formats("The quoted price is 25226[ID:deadbeefdeadbeef].", _kbinfos("aaa"), set())
    assert repaired == "The quoted price is 25226."
    assert idx == set()


def test_out_of_range_index_is_dropped():
    """The evidence block numbering starts at 0, so "[ID:1]" over one chunk is a
    number the model invented; a chip built from it opens nothing."""
    repaired, idx = repair_bad_citation_formats("The quoted price is 25226【ID:1】.", _kbinfos("aaa"), set())
    assert repaired == "The quoted price is 25226."
    assert idx == set()


def test_mixed_markers_keep_only_what_resolves():
    kbinfos = _kbinfos("aaa", "a1b2c3d4e5f60718")
    repaired, idx = repair_bad_citation_formats("25226【ID:9】【ID:a1b2c3d4e5f60718】.", kbinfos, set())
    assert repaired == "25226[ID:1]."
    assert idx == {1}


def test_arabic_digit_marker_is_normalised():
    repaired, idx = repair_bad_citation_formats("value [ID:١].", _kbinfos("a", "b"), set())
    assert repaired == "value [ID:1]."
    assert idx == {1}


def test_markdown_link_text_is_not_a_citation():
    """A bare bracket is a citation only when it holds digits — otherwise every
    markdown link in the answer would be eaten by the resolve pass."""
    answer = "See [Project Brief](https://example.com/brief) for details."
    repaired, idx = repair_bad_citation_formats(answer, _kbinfos("a", "b"), set())
    assert repaired == answer
    assert idx == set()


def test_ref_prose_survives_when_it_names_no_chunk():
    repaired, _ = repair_bad_citation_formats("According to ref 9 it is so.", _kbinfos("a", "b"), set())
    assert repaired == "According to ref 9 it is so."


def test_ref_shorthand_becomes_a_citation_when_it_resolves():
    repaired, idx = repair_bad_citation_formats("According to ref1 it is so.", _kbinfos("a", "b"), set())
    assert repaired == "According to [ID:1] it is so."
    assert idx == {1}


def test_empty_pool_drops_every_marker():
    repaired, idx = repair_bad_citation_formats("Nothing [ID:0].", {"chunks": []}, set())
    assert repaired == "Nothing."
    assert idx == set()


@pytest.mark.parametrize("answer", ["cited [ID:1]", "cited 【ID:1】", "cited (ID:1)", "cited [1]"])
def test_has_citation_markers_sees_every_shape(answer):
    """Similarity-based insertion must not tag an answer the model already cited,
    whichever bracket style it reached for."""
    assert has_citation_markers(answer) is True


@pytest.mark.parametrize("answer", ["", "no citations here", "see [Project Brief](https://x)"])
def test_has_citation_markers_ignores_uncited_text(answer):
    assert has_citation_markers(answer) is False
