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

import pytest

from api.db.services.knowledgebase_service import KnowledgebaseService
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


def test_scan_body_selects_the_dataset_and_only_slices_when_asked():
    single = retokenize.scan_body("kb1", 0, 1)

    assert single["query"] == {"bool": {"filter": [{"term": {"kb_id": "kb1"}}]}}
    # A one-slice scroll is the plain scan; both backends reject {"id": 0, "max": 1}.
    assert "slice" not in single
    # The token fields come back with the chunk so a re-run can tell what changed,
    # and the vector is left out of every response.
    assert set(retokenize.TOKEN_FIELDS) <= set(single["_source"])
    assert not [f for f in single["_source"] if f.endswith("_vec")]

    assert retokenize.scan_body("kb1", 3, 8)["slice"] == {"id": 3, "max": 8}


def test_retokenize_dataset_scans_a_slice_and_writes_only_stale_chunks(monkeypatch):
    up_to_date = {"content_with_weight": "plain text"}
    up_to_date.update(retokenize.retokenize_chunk(up_to_date))
    stale = {"content_with_weight": "plain text", "content_ltks": "p lain t ext", "content_sm_ltks": "p lain t ext"}
    scans, bulks = [], []

    class FakeHelpers:
        @staticmethod
        def scan(es, index, query, size, **kwargs):
            scans.append((index, query, size))
            return iter([{"_id": "c1", "_source": up_to_date}, {"_id": "c2", "_source": stale}])

        @staticmethod
        def bulk(es, actions, **kwargs):
            bulks.append(list(actions))
            return len(bulks[-1]), []

    kb = SimpleNamespace(id="kb1", name="dataset", language="English", tenant_id="t1")
    monkeypatch.setattr(retokenize, "_helpers", lambda es: FakeHelpers)
    monkeypatch.setattr(settings, "docStoreConn", SimpleNamespace(es=object()), raising=False)
    monkeypatch.setattr(KnowledgebaseService, "get_by_id", classmethod(lambda cls, _id: (True, kb)))

    scanned, updated = retokenize.retokenize_dataset("kb1", None, dry_run=False, batch_size=500, slice_id=2, slices=4)

    assert (scanned, updated) == (2, 1)
    assert scans[0][0] == "ragflow_t1"
    assert scans[0][1]["slice"] == {"id": 2, "max": 4}
    assert [a["_id"] for a in bulks[0]] == ["c2"]


def test_retokenize_dataset_writes_nothing_on_a_dry_run(monkeypatch):
    stale = {"content_with_weight": "plain text", "content_ltks": "p lain t ext"}

    class FakeHelpers:
        @staticmethod
        def scan(es, index, query, size, **kwargs):
            return iter([{"_id": "c1", "_source": stale}])

        @staticmethod
        def bulk(es, actions, **kwargs):
            raise AssertionError("a dry run must not write")

    kb = SimpleNamespace(id="kb1", name="dataset", language="English", tenant_id="t1")
    monkeypatch.setattr(retokenize, "_helpers", lambda es: FakeHelpers)
    monkeypatch.setattr(settings, "docStoreConn", SimpleNamespace(es=object()), raising=False)
    monkeypatch.setattr(KnowledgebaseService, "get_by_id", classmethod(lambda cls, _id: (True, kb)))

    assert retokenize.retokenize_dataset("kb1", None, dry_run=True, batch_size=500) == (1, 1)
