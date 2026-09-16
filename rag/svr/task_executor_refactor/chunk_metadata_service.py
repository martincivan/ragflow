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
"""Dataset-wide backfill of document metadata onto chunks.

Task type ``chunk_metadata`` (queued by ``POST /datasets/<id>/chunk_metadata/backfill``).
Walks the doc-meta index of the dataset, groups documents that share the
same whitelisted metadata, and writes each group with one ``update_by_query``
over its documents' chunks. A dataset with 100 k documents and ~100 distinct
metadata combinations is covered in ~100 requests. When every group is
written, ``parser_config.chunk_metadata.ready`` is set, which is what lets
the read path start filtering and boosting on the chunk fields.
"""

import json
import logging
from collections import OrderedDict

from api.db.services.knowledgebase_service import KnowledgebaseService
from common import chunk_metadata, settings
from common.misc_utils import thread_pool_exec

# ES rejects a ``terms`` clause above ``index.max_terms_count`` (65 536).
DOC_IDS_PER_UPDATE = 20000


def group_documents(rows, keys: list[str]) -> "OrderedDict[str, tuple[dict, list[str]]]":
    """(doc_id, meta_fields) rows → {signature: (chunk fields, [doc_ids])}.

    Documents without any whitelisted key still get a group (empty fields)
    so stale ``meta_*`` from a previous configuration are removed from them.
    """
    groups: OrderedDict[str, tuple[dict, list[str]]] = OrderedDict()
    for doc_id, meta in rows:
        if not doc_id:
            continue
        fields = chunk_metadata.chunk_fields(meta, keys)
        sig = json.dumps(fields, sort_keys=True, ensure_ascii=False, default=str)
        if sig not in groups:
            groups[sig] = (fields, [])
        groups[sig][1].append(doc_id)
    return groups


async def run_chunk_metadata_backfill(ctx) -> None:
    ok, kb = KnowledgebaseService.get_by_id(ctx.kb_id)
    if not ok:
        ctx.progress_cb(prog=-1.0, msg="Cannot find the dataset for the chunk metadata backfill")
        return
    cfg = chunk_metadata.parse_config(kb.parser_config)
    if cfg is None or not cfg.enabled or not cfg.fields:
        ctx.progress_cb(prog=-1.0, msg="chunk_metadata is not enabled (or has no fields) on this dataset")
        return
    store = settings.docStoreConn
    if not chunk_metadata.store_supports(store):
        ctx.progress_cb(prog=-1.0, msg=f"{type(store).__name__} does not support chunk metadata fields")
        return

    index_name = f"ragflow_{kb.tenant_id}"
    meta_index = f"ragflow_doc_meta_{kb.tenant_id}"
    ctx.progress_cb(msg=f"Reading document metadata for keys {cfg.fields} …")
    rows = list(store.iter_doc_metadata(meta_index, kb.id)) if store.index_exist(meta_index, "") else []
    groups = group_documents(rows, cfg.fields)
    remove_all = chunk_metadata.all_field_names(cfg.fields)
    ctx.progress_cb(msg=f"{len(rows)} documents in {len(groups)} metadata group(s)")

    failed: list[str] = []
    for done_groups, (fields, doc_ids) in enumerate(groups.values(), start=1):
        if ctx.has_canceled_func(ctx.id):
            ctx.progress_cb(prog=-1.0, msg="Chunk metadata backfill cancelled")
            return
        remove = [f for f in remove_all if f not in fields]
        for i in range(0, len(doc_ids), DOC_IDS_PER_UPDATE):
            batch = doc_ids[i : i + DOC_IDS_PER_UPDATE]
            # refresh once at the end; per-request refresh on a large index is the slow part
            written = await thread_pool_exec(store.update_chunk_metadata, index_name, kb.id, batch, fields, remove, False)
            if not written:
                failed.extend(batch)
        if done_groups % 10 == 0 or done_groups == len(groups):
            ctx.progress_cb(prog=0.05 + 0.9 * done_groups / max(len(groups), 1), msg=f"{done_groups}/{len(groups)} groups written")

    try:
        store.refresh_idx(index_name)
    except Exception:
        logging.exception("refresh after chunk metadata backfill failed")

    if failed:
        ctx.progress_cb(prog=-1.0, msg=f"Chunk metadata backfill failed for {len(failed)} document(s); `ready` left unchanged")
        return

    parser_config: dict = dict(kb.parser_config or {})
    parser_config[chunk_metadata.CONFIG_KEY] = {**cfg.to_dict(), "ready": True}
    if not KnowledgebaseService.update_by_id(kb.id, {"parser_config": parser_config}):
        ctx.progress_cb(prog=-1.0, msg="Chunks written but saving chunk_metadata.ready failed")
        return
    ctx.progress_cb(prog=1.0, msg=f"Chunk metadata ready: {len(rows)} documents, {len(groups)} groups, fields {cfg.fields}")
