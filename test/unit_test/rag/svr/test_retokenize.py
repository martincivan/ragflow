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

from common import settings
from rag.nlp import rag_tokenizer
from rag.svr import retokenize

pytestmark = pytest.mark.p2


@pytest.fixture(autouse=True)
def english_local_tokenizer(monkeypatch):
    monkeypatch.setattr(settings, "DOC_ENGINE_INFINITY", False, raising=False)
    rag_tokenizer.tokenizer.set_language("English")
    yield
    rag_tokenizer.tokenizer.set_language("English")


def test_retokenize_chunk_recomputes_every_token_field_like_indexing():
    source = {
        "content_with_weight": "<table><tr><td>Running dogs</td></tr></table> chased cats",
        "docnm_kwd": "Annual report.pdf",
        "important_kwd": ["running", "dogs"],
        "question_kwd": ["Who chased the cats?"],
    }

    fields = retokenize.retokenize_chunk(source)

    assert set(fields) == set(retokenize.TOKEN_FIELDS)
    # Table markup is dropped before tokenizing, exactly as rag.nlp.tokenize does.
    assert "table" not in fields["content_ltks"]
    assert fields["content_ltks"] == rag_tokenizer.tokenize("Running dogs chased cats")
    assert fields["content_sm_ltks"] == rag_tokenizer.fine_grained_tokenize(fields["content_ltks"])
    # The title is tokenized without its file extension, as the parsers do.
    assert fields["title_tks"] == rag_tokenizer.tokenize("Annual report")
    assert fields["important_tks"] == rag_tokenizer.tokenize("running dogs")
    assert fields["question_tks"] == rag_tokenizer.tokenize("Who chased the cats?")


def test_retokenize_chunk_leaves_absent_inputs_alone():
    fields = retokenize.retokenize_chunk({"content_with_weight": "plain text", "important_kwd": [], "docnm_kwd": ""})

    assert set(fields) == {"content_ltks", "content_sm_ltks"}


def test_update_actions_only_touch_chunks_whose_tokens_changed():
    up_to_date = {"content_with_weight": "plain text"}
    up_to_date.update(retokenize.retokenize_chunk(up_to_date))
    stale = {"content_with_weight": "plain text", "content_ltks": "p lain t ext", "content_sm_ltks": "p lain t ext"}

    actions = list(retokenize.update_actions("ragflow_t1", [("c1", up_to_date), ("c2", stale)]))

    assert [a["_id"] for a in actions] == ["c2"]
    assert actions[0]["_op_type"] == "update"
    assert actions[0]["_index"] == "ragflow_t1"
    # Only the token fields are written, so vectors, text and metadata stay as they are.
    assert set(actions[0]["doc"]) == {"content_ltks", "content_sm_ltks"}


def test_read_progress_tolerates_missing_and_blank_lines(tmp_path):
    assert retokenize.read_progress(None) == set()
    assert retokenize.read_progress(tmp_path / "missing") == set()
    progress = tmp_path / "p"
    progress.write_text("d1\n\nd2\n")
    assert retokenize.read_progress(progress) == {"d1", "d2"}
