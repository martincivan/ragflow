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
"""The executor-side wiring a KB-level fan-out task needs to survive pickup.

``chunk_metadata`` queues its Task row under the ``GRAPH_RAPTOR_FAKE_DOC_ID``
sentinel, so nothing in the row itself resolves a tenant. ``collect()``
recovers one only for task types listed in
``PIPELINE_SPECIAL_PROGRESS_FREEZE_TASK_TYPES``, and only when the queue call
supplied real ``doc_ids`` for ``TaskService.get_task`` to join through. Miss
either and the task dies at pickup with "Task must contain 'tenant_id'" — far
from the backfill code the other tests in this directory cover, which is why
these assert on the wiring rather than on behaviour.
"""

import ast
from pathlib import Path

import pytest

from api.db import PIPELINE_SPECIAL_PROGRESS_FREEZE_TASK_TYPES

pytestmark = pytest.mark.p2

_ROOT = Path(__file__).resolve().parents[4]
_DOCUMENT_SERVICE = _ROOT / "api" / "db" / "services" / "document_service.py"
_DATASET_SERVICE = _ROOT / "api" / "apps" / "services" / "dataset_api_service.py"


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_every_fanout_task_type_is_registered_for_hydration():
    queue_fn = _function(_DOCUMENT_SERVICE, "queue_raptor_o_graphrag_tasks")
    guard = next(n for n in ast.walk(queue_fn) if isinstance(n, ast.Assert) and isinstance(n.test, ast.Compare))
    accepted = {e.value for e in guard.test.comparators[0].elts}

    assert accepted, "could not read the accepted task types out of queue_raptor_o_graphrag_tasks"
    assert accepted <= PIPELINE_SPECIAL_PROGRESS_FREEZE_TASK_TYPES


def test_backfill_queues_a_real_doc_id():
    backfill_fn = _function(_DATASET_SERVICE, "run_chunk_metadata_backfill")
    call = next(n for n in ast.walk(backfill_fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "queue_raptor_o_graphrag_tasks")
    kwargs = {kw.arg: kw.value for kw in call.keywords}

    assert kwargs["ty"].value == "chunk_metadata"
    doc_ids = kwargs.get("doc_ids")
    assert doc_ids is not None, "without doc_ids the sentinel doc_id joins to no document and tenant_id never hydrates"
    assert not (isinstance(doc_ids, ast.List) and not doc_ids.elts)
