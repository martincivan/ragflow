"""Unit tests for ESConnection's deep-paging path.

search_after is chosen when a request reaches past the index's
max_result_window. It pages by the sort key of the last hit, so the choice has
to follow the sort the query actually carries — which is not the same as the
sort the caller asked for.
"""

import logging

import pytest

pytestmark = pytest.mark.p2

# Imported inside the fixture, not at module scope: rag.utils.es_conn and
# common.settings import each other, so pulling es_conn in first raises a
# circular ImportError, and importing common.settings first at collection time
# perturbs the import order the rest of the suite's patches rely on.
def _es_conn_module():
    from common import settings  # noqa: F401  breaks the es_conn import cycle

    import rag.utils.es_conn as es_conn

    return es_conn


@pytest.fixture
def es_conn():
    return _es_conn_module()


@pytest.fixture
def es_class(es_conn):
    """The class behind @singleton, which otherwise constructs (and connects)."""
    return next(cell.cell_contents for cell in es_conn.ESConnection.__closure__ if isinstance(cell.cell_contents, type))


@pytest.fixture
def conn(monkeypatch, es_class):
    """An ESConnection with no live cluster behind it."""
    c = object.__new__(es_class)
    c.logger = logging.getLogger("test-es-conn")
    monkeypatch.setattr(es_class, "es", None, raising=False)
    return c


def test_search_after_refuses_an_unsorted_query(conn, es_conn):
    """Regression: an unsorted query used to page to nothing, silently.

    The hits carry no sort key, so the skip loop broke on its first iteration
    and the caller got an empty page — indistinguishable from the end of the
    result set, which is how a metadata scan silently truncated at 10 000 rows.
    """
    with pytest.raises(ValueError, match="sorted query"):
        conn._search_with_search_after(["idx"], {"query": {"match_all": {}}}, es_conn.MAX_RESULT_WINDOW, 100)


def test_search_after_pages_a_sorted_query(conn, es_class, monkeypatch):
    pages = [
        {"hits": {"hits": [{"_source": {"id": str(i)}, "sort": [str(i)]} for i in range(100)]}},
        {"hits": {"hits": [{"_source": {"id": str(i)}, "sort": [str(i)]} for i in range(100, 150)]}},
    ]
    monkeypatch.setattr(es_class, "_es_search_once",
                        lambda self, idx, q, track_total_hits: pages.pop(0), raising=False)
    res = conn._search_with_search_after(
        ["idx"], {"query": {"match_all": {}}, "sort": [{"id": {"order": "asc"}}]}, 100, 50)
    assert [h["_source"]["id"] for h in res["hits"]["hits"]] == [str(i) for i in range(100, 150)]


def test_dropped_id_sort_does_not_select_search_after(conn, es_conn, es_class, monkeypatch):
    """`id` is stripped from the sort, so the query is unsorted and must not page.

    order_by.fields is non-empty here, which is what the gate used to read.
    """
    calls = {"search_after": 0, "once": 0}

    def _once(self, index_names, query, track_total_hits):
        calls["once"] += 1
        assert "sort" not in query or not query["sort"]
        return {"hits": {"hits": [], "total": {"value": 0}}}

    def _after(self, index_names, query, offset, limit):
        calls["search_after"] += 1
        return {"hits": {"hits": []}}

    monkeypatch.setattr(es_class, "_es_search_once", _once, raising=False)
    monkeypatch.setattr(es_class, "_search_with_search_after", _after, raising=False)

    from common.doc_store.doc_store_base import OrderByExpr

    order_by = OrderByExpr()
    order_by.asc("id")
    conn.search(["*"], [], {}, [], order_by, es_conn.MAX_RESULT_WINDOW, 1000, "idx", ["kb"])

    assert calls["search_after"] == 0, "an unsorted query must not take the search_after path"
    assert calls["once"] == 1


def test_sortable_field_still_selects_search_after(conn, es_conn, es_class, monkeypatch):
    calls = {"search_after": 0}
    monkeypatch.setattr(es_class, "_es_search_once",
                        lambda self, i, q, track_total_hits: {"hits": {"hits": [], "total": {"value": 0}}},
                        raising=False)

    def _after(self, index_names, query, offset, limit):
        calls["search_after"] += 1
        return {"hits": {"hits": []}}

    monkeypatch.setattr(es_class, "_search_with_search_after", _after, raising=False)

    from common.doc_store.doc_store_base import OrderByExpr

    order_by = OrderByExpr()
    order_by.asc("create_time")
    conn.search(["*"], [], {}, [], order_by, es_conn.MAX_RESULT_WINDOW, 1000, "idx", ["kb"])

    assert calls["search_after"] == 1
