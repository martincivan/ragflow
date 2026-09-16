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
        [--language Slovak] [--dry-run] [--progress-file retokenize.progress]

The language defaults to the dataset's own. Processed document ids are appended
to the progress file, so an interrupted run resumes where it stopped.
Elasticsearch and OpenSearch document stores only.
"""

import argparse
import logging
import re
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

TOKEN_FIELDS = ("content_ltks", "content_sm_ltks", "title_tks", "title_sm_tks", "important_tks", "question_tks")
SOURCE_FIELDS = ("doc_id", "kb_id", "content_with_weight", "docnm_kwd", "important_kwd", "question_kwd", *TOKEN_FIELDS)

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


def read_progress(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _helpers(es):
    """The scan/bulk helpers matching the client (elasticsearch-py or opensearch-py)."""
    if type(es).__module__.startswith("opensearchpy"):
        from opensearchpy import helpers
    else:
        from elasticsearch import helpers
    return helpers


def _iter_document_chunks(es, index: str, kb_id: str, doc_id: str) -> Iterator[tuple[str, dict]]:
    helpers = _helpers(es)
    query = {"bool": {"filter": [{"term": {"kb_id": kb_id}}, {"term": {"doc_id": doc_id}}]}}
    for hit in helpers.scan(es, index=index, query={"query": query}, _source=list(SOURCE_FIELDS), size=500):
        yield hit["_id"], hit["_source"]


def _bulk(es, actions: list[dict]) -> int:
    if not actions:
        return 0
    ok, errors = _helpers(es).bulk(es, actions, raise_on_error=False, stats_only=False)
    for error in errors:
        logging.error("retokenize: bulk update failed: %s", error)
    return ok


def retokenize_dataset(kb_id: str, language: str | None, dry_run: bool, progress_file: Path | None, batch_size: int) -> tuple[int, int]:
    """Rewrite the token fields of every chunk of one dataset.

    Returns (documents processed, chunks updated).
    """
    from api.db.services.document_service import DocumentService
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
    index = index_name(kb.tenant_id)
    done = read_progress(progress_file)
    docs = [d for d in DocumentService.query(kb_id=kb_id, order_by="id") if d.id not in done]
    logging.info("retokenize: dataset %s (%s) language=%s documents=%d (%d already done)", kb.name, kb_id, language, len(docs), len(done))

    processed = updated = 0
    started = time.time()
    for doc in docs:
        rag_tokenizer.tokenizer.set_language(language)
        actions = list(update_actions(index, _iter_document_chunks(es, index, kb_id, doc.id)))
        if dry_run:
            for action in actions[:3]:
                logging.info("retokenize: would update %s: %s", action["_id"], {k: v[:80] for k, v in action["doc"].items()})
            updated += len(actions)
        else:
            for start in range(0, len(actions), batch_size):
                updated += _bulk(es, actions[start : start + batch_size])
            if progress_file is not None:
                with progress_file.open("a") as f:
                    f.write(doc.id + "\n")
        processed += 1
        if processed % 100 == 0:
            logging.info("retokenize: %d/%d documents, %d chunks updated, %.0fs", processed, len(docs), updated, time.time() - started)
    logging.info("retokenize: done: %d documents, %d chunks %s", processed, updated, "would be updated" if dry_run else "updated")
    return processed, updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kb-id", action="append", required=True, help="dataset id (repeatable)")
    parser.add_argument("--language", help="tokenizer language; defaults to the dataset's language")
    parser.add_argument("--dry-run", action="store_true", help="compute and report, write nothing")
    parser.add_argument("--progress-file", type=Path, help="file recording processed document ids; lets an interrupted run resume")
    parser.add_argument("--batch-size", type=int, default=500, help="chunks per bulk request")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from common import settings

    settings.init_settings()
    for kb_id in args.kb_id:
        retokenize_dataset(kb_id, args.language, args.dry_run, args.progress_file, args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
