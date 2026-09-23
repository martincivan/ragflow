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
"""Recompute the full-text token fields of chunks that are already indexed.

Re-parsing a dataset after a tokenizer change (for example switching it to a
language the tokenizer now handles differently) redoes parsing, chunking and
embedding although only the token fields changed. This tool rewrites just those
fields in the document store: ``content_ltks``, ``content_sm_ltks``,
``title_tks``, ``title_sm_tks``, ``important_tks`` and ``question_tks``. Text,
vectors, positions and metadata are left untouched, and the dataset stays
searchable while it runs.

    python rag/svr/retokenize.py --kb-id <dataset id> [--kb-id ...]
        [--language Slovak] [--dry-run] [--slice 0 --slices 8]

The work is driven by a scan of the chunk index, not by the document table: a
chunk names its dataset in ``kb_id``, so the run covers every chunk of the
dataset whatever the document rows say and in whatever order they come back.
``--slices`` splits that scan across processes -- each slice is disjoint, so N
processes with ``--slice 0..N-1`` do the whole dataset N times faster.

Only chunks whose token fields differ are written, which makes the run
idempotent: an interrupted one is finished by starting it again, and a second
run over a converted dataset writes nothing. The language defaults to the
dataset's own. Elasticsearch and OpenSearch document stores only.
"""

import argparse
import logging
import re
import sys
import time
from collections.abc import Iterable, Iterator

TOKEN_FIELDS = ("content_ltks", "content_sm_ltks", "title_tks", "title_sm_tks", "important_tks", "question_tks")
SOURCE_FIELDS = ("content_with_weight", "docnm_kwd", "important_kwd", "question_kwd", *TOKEN_FIELDS)

_TABLE_TAGS = re.compile(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>")
_FILE_EXTENSION = re.compile(r"\.[a-zA-Z]+$")


def retokenize_chunk(source: dict) -> dict:
    """Token fields for one chunk, computed the way indexing computes them.

    The caller has already selected the tokenizer language. Only fields whose
    input exists on the chunk are returned; a chunk without keywords keeps
    whatever ``important_tks`` it had.
    """
    from rag.nlp import rag_tokenizer

    fields: dict = {}
    content = source.get("content_with_weight")
    if isinstance(content, str) and content.strip():
        fields["content_ltks"] = rag_tokenizer.tokenize(_TABLE_TAGS.sub(" ", content))
        fields["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(fields["content_ltks"])
    title = source.get("docnm_kwd")
    if isinstance(title, str) and title.strip():
        fields["title_tks"] = rag_tokenizer.tokenize(_FILE_EXTENSION.sub("", title))
        fields["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(fields["title_tks"])
    keywords = source.get("important_kwd")
    if isinstance(keywords, list) and keywords:
        fields["important_tks"] = rag_tokenizer.tokenize(" ".join(str(k) for k in keywords))
    questions = source.get("question_kwd")
    if isinstance(questions, list) and questions:
        fields["question_tks"] = rag_tokenizer.tokenize("\n".join(str(q) for q in questions))
    return fields


def changed_fields(source: dict, fields: dict) -> dict:
    """The subset of ``fields`` that differs from what the chunk holds now."""
    return {k: v for k, v in fields.items() if source.get(k) != v}


def update_actions(index: str, chunks: Iterable[tuple[str, dict]]) -> Iterator[dict]:
    """Bulk partial-update actions for the chunks whose token fields changed."""
    for chunk_id, source in chunks:
        fields = changed_fields(source, retokenize_chunk(source))
        if fields:
            yield {"_op_type": "update", "_index": index, "_id": chunk_id, "doc": fields}


def scan_body(kb_id: str, slice_id: int, slices: int) -> dict:
    """The scan request for one slice of a dataset's chunks.

    A slice is only asked for when there is more than one: a single-slice
    scroll is the plain scan, and asking for ``{"id": 0, "max": 1}`` is an
    error on both backends.
    """
    body: dict = {"query": {"bool": {"filter": [{"term": {"kb_id": kb_id}}]}}, "_source": list(SOURCE_FIELDS)}
    if slices > 1:
        body["slice"] = {"id": slice_id, "max": slices}
    return body


def _helpers(es):
    """The scan/bulk helpers matching the client (elasticsearch-py or opensearch-py)."""
    if type(es).__module__.startswith("opensearchpy"):
        from opensearchpy import helpers
    else:
        from elasticsearch import helpers
    return helpers


def iter_chunks(es, index: str, kb_id: str, slice_id: int = 0, slices: int = 1, scan_size: int = 500) -> Iterator[tuple[str, dict]]:
    """Every chunk of ``kb_id`` in this slice, as (chunk id, source)."""
    for hit in _helpers(es).scan(es, index=index, query=scan_body(kb_id, slice_id, slices), size=scan_size, request_timeout=180):
        yield hit["_id"], hit["_source"]


def _bulk(es, actions: list[dict]) -> int:
    if not actions:
        return 0
    ok, errors = _helpers(es).bulk(es, actions, raise_on_error=False, stats_only=False)
    for error in errors:
        logging.error("retokenize: bulk update failed: %s", error)
    return ok


def retokenize_dataset(kb_id: str, language: str | None, dry_run: bool, batch_size: int, slice_id: int = 0, slices: int = 1, scan_size: int = 500) -> tuple[int, int]:
    """Rewrite the token fields of one slice of a dataset's chunks.

    Returns (chunks scanned, chunks updated).
    """
    from api.db.services.knowledgebase_service import KnowledgebaseService
    from common import settings
    from rag.nlp import rag_tokenizer
    from rag.nlp.search import index_name

    es = getattr(settings.docStoreConn, "es", None)
    if es is None:
        raise SystemExit(f"retokenize: document store {type(settings.docStoreConn).__name__} is not Elasticsearch/OpenSearch")

    ok, kb = KnowledgebaseService.get_by_id(kb_id)
    if not ok or kb is None:
        raise SystemExit(f"retokenize: dataset {kb_id} not found")
    language = language or kb.language or "English"
    rag_tokenizer.tokenizer.set_language(language)
    index = index_name(kb.tenant_id)
    logging.info("retokenize: dataset %s (%s) language=%s index=%s slice=%d/%d", kb.name, kb_id, language, index, slice_id, slices)

    scanned = updated = 0
    actions: list[dict] = []
    started = time.time()

    def counted(chunks):
        # update_actions only yields the chunks that changed, so the scan is
        # counted on the way in.
        nonlocal scanned
        for chunk in chunks:
            scanned += 1
            yield chunk
            if scanned % 100000 == 0:
                logging.info("retokenize: %d chunks scanned, %d updated, %.0fs", scanned, updated, time.time() - started)

    for action in update_actions(index, counted(iter_chunks(es, index, kb_id, slice_id, slices, scan_size))):
        actions.append(action)
        if len(actions) >= batch_size:
            updated += len(actions) if dry_run else _bulk(es, actions)
            actions = []
    if actions:
        updated += len(actions) if dry_run else _bulk(es, actions)
    logging.info("retokenize: done: %d chunks scanned, %d %s in %.0fs", scanned, updated, "would be updated" if dry_run else "updated", time.time() - started)
    return scanned, updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kb-id", action="append", required=True, help="dataset id (repeatable)")
    parser.add_argument("--language", help="tokenizer language; defaults to the dataset's language")
    parser.add_argument("--dry-run", action="store_true", help="compute and report, write nothing")
    parser.add_argument("--slice", type=int, default=0, help="which slice of the scan this process takes (0-based)")
    parser.add_argument("--slices", type=int, default=1, help="how many processes share the scan; each takes one --slice")
    parser.add_argument("--batch-size", type=int, default=500, help="chunks per bulk request")
    parser.add_argument("--scan-size", type=int, default=500, help="chunks per scan request")
    args = parser.parse_args(argv)
    if not 0 <= args.slice < args.slices:
        parser.error(f"--slice must be in [0, {args.slices}) for --slices {args.slices}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from common import settings

    settings.init_settings()
    for kb_id in args.kb_id:
        retokenize_dataset(kb_id, args.language, args.dry_run, args.batch_size, args.slice, args.slices, args.scan_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
