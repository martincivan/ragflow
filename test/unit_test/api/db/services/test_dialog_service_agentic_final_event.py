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
"""The agentic stream's final event has to carry the answer, not just the reference.

decorate_answer() is where a citation marker is canonicalised and where one that
names no published chunk is dropped. The client keeps the streamed deltas unless
the final event replaces them, so an empty final answer means the reader sees the
model's raw markers — "【ID:1】" over a one-chunk pool — however well the server
repaired them.

The progress block travels with it: both the web client and structure_answer()
replace the whole message with this text, so an answer sent on its own would take
the ReAct trajectory the user just watched down with it.
"""

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest


def _install_cv2_stub_if_unavailable():
    try:
        import cv2  # noqa: F401

        return
    except Exception:
        pass
    stub = types.ModuleType("cv2")

    def _module_getattr(name):
        if name.isupper():
            return 0
        raise RuntimeError(f"cv2.{name} is unavailable in this test environment")

    stub.__getattr__ = _module_getattr
    sys.modules["cv2"] = stub


_install_cv2_stub_if_unavailable()

from api.db.services import dialog_service  # noqa: E402

_DIALOG = SimpleNamespace(
    id="dialog-1",
    tenant_id="tenant-1",
    kb_ids=["kb-1"],
    llm_id="gpt-oss-120b@OpenAI-API-Compatible",
    llm_setting={"temperature": 0.1},
    prompt_config={"reasoning": 1, "quote": True},
    meta_data_filter=None,
    similarity_threshold=0.2,
    vector_similarity_weight=0.3,
    top_n=6,
    rerank_candidates_count=64,
    top_k=1024,
)

_KB = SimpleNamespace(id="kb-1", tenant_id="tenant-1")

_CHUNKS = [
    {"chunk_id": "a1b2c3d4e5f60718", "doc_id": "doc-1", "content_with_weight": "price excluding VAT 25226"},
    {"chunk_id": "0f1e2d3c4b5a6978", "doc_id": "doc-2", "content_with_weight": "an unrelated page"},
]


class _StubRAGTools:
    """Stands in for the agentic tool set: holds the pool and the answer sink."""

    instances: list = []

    def __init__(self, *_args, **_kwargs):
        self.kbinfos = {
            "chunks": [dict(c) for c in _CHUNKS],
            "doc_aggs": [
                {"doc_id": "doc-1", "doc_name": "quarterly-report.pdf", "count": 1},
                {"doc_id": "doc-2", "doc_name": "iny.pdf", "count": 1},
            ],
        }
        self.tools = []
        self.answer_sink = None
        self.tool_started_sink = None
        _StubRAGTools.instances.append(self)

    def sys_prompt(self):
        return "You are a helpful assistant."


class _StreamingChatModel:
    """Streams an outer think span, then pushes the inner graph's answer deltas."""

    def __init__(self, answer_pieces):
        self.is_tools = True
        self.model_config = {"model_type": "chat", "llm_factory": "OpenAI"}
        self.mdl = SimpleNamespace(terminal_tools=set())
        self._answer_pieces = answer_pieces

    def bind_tools(self, _toolcall_session, _tools):
        pass

    async def async_chat_streamly_delta(self, _system, _messages, _gen_conf, **_kwargs):
        tools = _StubRAGTools.instances[-1]
        yield "<think>Looking up the quote.</think>"
        tools.tool_started_sink()
        for piece in self._answer_pieces:
            tools.answer_sink(piece)
        await asyncio.sleep(0)


def _run_agentic_stream(monkeypatch, answer_pieces):
    _StubRAGTools.instances.clear()
    chat_mdl = _StreamingChatModel(answer_pieces)
    monkeypatch.setattr(dialog_service, "get_models", lambda _dialog, **_kw: ([_KB], None, None, chat_mdl, None))
    monkeypatch.setattr(dialog_service, "RAGTools", _StubRAGTools)
    monkeypatch.setattr(dialog_service, "tts", lambda _mdl, _text: None)

    messages = [{"role": "user", "content": "What is the quoted price?"}]

    async def _run():
        return [ev async for ev in dialog_service.rag_agent(_DIALOG, messages, True, reasoning="2")]

    return asyncio.run(_run())


def _final_event(events):
    finals = [e for e in events if e.get("final")]
    assert len(finals) == 1, f"expected exactly one final event, got {len(finals)}"
    return finals[0]


@pytest.mark.p2
def test_final_event_carries_the_repaired_answer(monkeypatch):
    """The model cited with full-width brackets and with a chunk id, the two shapes
    gpt-oss produced against the slot-table research summary."""
    events = _run_agentic_stream(monkeypatch, ["The quoted price is 25226", "【ID:a1b2c3d4e5f60718】."])
    final = _final_event(events)

    assert "【" not in final["answer"]
    assert "[ID:0]" in final["answer"]
    assert "The quoted price is 25226" in final["answer"]


@pytest.mark.p2
def test_final_event_replays_the_progress_block(monkeypatch):
    events = _run_agentic_stream(monkeypatch, ["The quoted price is 25226【ID:0】."])
    final = _final_event(events)

    assert final["answer"].startswith("<think>")
    assert "</think>" in final["answer"]
    assert "Looking up the quote." in final["answer"]
    # The deliverable follows the block rather than being buried inside it.
    assert "The quoted price is 25226" in final["answer"].split("</think>", 1)[1]


@pytest.mark.p2
def test_final_event_publishes_only_the_cited_document(monkeypatch):
    events = _run_agentic_stream(monkeypatch, ["The quoted price is 25226【ID:0】."])
    reference = _final_event(events)["reference"]

    assert [d["doc_id"] for d in reference["doc_aggs"]] == ["doc-1"]
    assert reference["chunks"]


@pytest.mark.p2
def test_uncited_answer_keeps_its_text(monkeypatch):
    """No citation is not an error: the answer still has to reach the client."""
    events = _run_agentic_stream(monkeypatch, ["I found nothing in the documentation."])
    final = _final_event(events)

    assert "I found nothing in the documentation." in final["answer"]
