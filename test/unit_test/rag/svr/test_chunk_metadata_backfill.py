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
"""The chunk-metadata backfill task: grouping and the write loop against a
fake doc store."""

from types import SimpleNamespace

import pytest

import common.settings  # noqa: F401
from rag.svr.task_executor_refactor import chunk_metadata_service as svc


def test_group_documents_groups_identical_metadata_and_keeps_empty_group():
    rows = [
        ("d1", {"project": "nst", "phase": "sp", "client": "x"}),
        ("d2", {"project": "nst", "phase": "sp"}),
        ("d3", {"project": "nst", "phase": "rp"}),
        ("d4", {}),
        (None, {"project": "ignored"}),
    ]
    groups = svc.group_documents(rows, ["project", "phase"])
    by_docs = {tuple(ids): fields for fields, ids in groups.values()}
    assert by_docs[("d1", "d2")] == {"meta_project_kwd": "nst", "meta_phase_kwd": "sp"}
    assert by_docs[("d3",)] == {"meta_project_kwd": "nst", "meta_phase_kwd": "rp"}
    assert by_docs[("d4",)] == {}


class _Store:
    def __init__(self):
        self.calls = []
        self.refreshed = []

    def update_chunk_metadata(self, index_name, kb_id, doc_ids, set_fields, remove_fields, refresh=True):
        self.calls.append((index_name, kb_id, list(doc_ids), dict(set_fields), list(remove_fields), refresh))
        return True

    def iter_doc_metadata(self, index_name, kb_id, page_size=1000):
        yield "d1", {"project": "nst", "phase": "sp"}
        yield "d2", {"project": "nst", "phase": "sp"}
        yield "d3", {"project": "bystricka"}

    def index_exist(self, index_name, kb_id):
        return True

    def refresh_idx(self, index_name):
        self.refreshed.append(index_name)
        return True


class _Ctx:
    def __init__(self):
        self.kb_id = "kb1"
        self.id = "task1"
        self.progress = []

    def progress_cb(self, prog=None, msg=""):
        self.progress.append((prog, msg))

    def has_canceled_func(self, task_id):
        return False


@pytest.mark.asyncio
async def test_backfill_writes_one_update_per_group_and_marks_ready(monkeypatch):
    store = _Store()
    kb = SimpleNamespace(id="kb1", tenant_id="t1", parser_config={"chunk_metadata": {"enabled": True, "fields": ["project", "phase"], "ready": False}, "chunk_token_num": 512})
    saved = {}
    monkeypatch.setattr(svc.KnowledgebaseService, "get_by_id", classmethod(lambda cls, _id: (True, kb)))
    monkeypatch.setattr(svc.KnowledgebaseService, "update_by_id", classmethod(lambda cls, _id, data: saved.update(data) or True))
    monkeypatch.setattr(svc.settings, "docStoreConn", store)

    ctx = _Ctx()
    await svc.run_chunk_metadata_backfill(ctx)

    assert len(store.calls) == 2
    idx, kb_id, ids, fields, removed, refresh = store.calls[0]
    assert (idx, kb_id, ids, refresh) == ("ragflow_t1", "kb1", ["d1", "d2"], False)
    assert fields == {"meta_project_kwd": "nst", "meta_phase_kwd": "sp"}
    assert "meta_project_int" in removed and "meta_project_kwd" not in removed
    assert store.calls[1][2] == ["d3"] and store.calls[1][3] == {"meta_project_kwd": "bystricka"} and "meta_phase_kwd" in store.calls[1][4]
    assert store.refreshed == ["ragflow_t1"]
    assert saved["parser_config"]["chunk_metadata"] == {"enabled": True, "fields": ["project", "phase"], "ready": True}
    assert saved["parser_config"]["chunk_token_num"] == 512
    assert ctx.progress[-1][0] == 1.0


@pytest.mark.asyncio
async def test_backfill_refuses_when_not_enabled(monkeypatch):
    kb = SimpleNamespace(id="kb1", tenant_id="t1", parser_config={})
    monkeypatch.setattr(svc.KnowledgebaseService, "get_by_id", classmethod(lambda cls, _id: (True, kb)))
    ctx = _Ctx()
    await svc.run_chunk_metadata_backfill(ctx)
    assert ctx.progress[-1][0] == -1.0
