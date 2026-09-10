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
"""``parser_config.chunk_metadata.ready`` is owned by the backfill task.

``update_dataset`` must not let a client flip it to true, and a changed field
whitelist must reset it. The guard is a pure function; it is loaded from the
service source with ``ast`` so the test does not need the service's DB imports.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.p2

_SERVICE = Path(__file__).resolve().parents[5] / "api" / "apps" / "services" / "dataset_api_service.py"


@pytest.fixture(scope="module")
def guard():
    tree = ast.parse(_SERVICE.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_guard_chunk_metadata_ready")
    namespace: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(_SERVICE), "exec"), namespace)  # noqa: S102
    return namespace["_guard_chunk_metadata_ready"]


def _cfg(**chunk_metadata):
    return {"chunk_metadata": chunk_metadata}


def test_client_cannot_set_ready(guard):
    new = _cfg(enabled=True, fields=["project"], ready=True)
    guard(_cfg(enabled=True, fields=["project"], ready=False), new)
    assert new["chunk_metadata"]["ready"] is False


def test_ready_survives_unrelated_update(guard):
    new = _cfg(enabled=True, fields=["phase", "project"], ready=True)
    guard(_cfg(enabled=True, fields=["project", "phase"], ready=True), new)
    assert new["chunk_metadata"]["ready"] is True


def test_changed_fields_reset_ready(guard):
    new = _cfg(enabled=True, fields=["project", "year"], ready=True)
    guard(_cfg(enabled=True, fields=["project"], ready=True), new)
    assert new["chunk_metadata"]["ready"] is False


def test_client_may_reset_ready(guard):
    new = _cfg(enabled=True, fields=["project"], ready=False)
    guard(_cfg(enabled=True, fields=["project"], ready=True), new)
    assert new["chunk_metadata"]["ready"] is False


def test_noop_without_chunk_metadata(guard):
    new = {"chunk_token_num": 256}
    guard(_cfg(enabled=True, fields=["project"], ready=True), new)
    assert "chunk_metadata" not in new
    guard(None, _cfg(enabled=True, fields=["project"], ready=True))
