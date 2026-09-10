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
"""Document metadata on chunks: fields, filter clauses and score boosts.

A metadata filter is normally resolved to a list of document ids and pushed
into the chunk query as a ``terms`` clause. That list is capped by the doc
store's result window (10 000 on Elasticsearch) and grows linearly with the
match set, so a filter over a large dataset is either silently truncated or
slow. When a dataset opts in (``parser_config.chunk_metadata``), the
whitelisted metadata keys are copied onto every chunk as ``meta_<key>_kwd``
(plus a typed twin for numbers and dates), and a condition becomes one term
or range clause on the chunk query — exact and independent of how many
documents match.

The same fields carry a *boost*: conditions with a weight that raise the
fused score of matching chunks instead of excluding the others ("prefer the
newest expedition", "prefer the phase the user named"). Boosts are applied
twice, in the doc store (``should`` clauses so a chunk that only wins on
metadata still enters the candidate pool) and in the fused score computed by
``rag.nlp.search.Dealer`` (so preferences survive reranking).

Field names ride on the dynamic templates of ``conf/mapping.json``: ``*_kwd``
is a keyword, ``*_int`` an integer, ``*_flt`` a float, ``*_dt`` a date. No
mapping change is needed on Elasticsearch or OpenSearch.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from common.metadata_es_filter import (
    MULTIVALUE_UNSAFE_NEGATIVE_OPS,
    SUPPORTED_OPERATORS,
    UnsupportedMetaFilter,
    _coerce_range_value,
    _coerce_scalar,
    _coerce_string,
    _csv_or_list,
    _escape_wildcard,
)

CONFIG_KEY = "chunk_metadata"
CHUNK_META_PREFIX = "meta_"
MAX_FIELDS = 32

# Keys that would collide with RAGflow's own chunk fields after prefixing, or
# that are not valid as ES field names.
_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
RESERVED_KEYS = frozenset({"kb_id", "doc_id", "id", "content", "vec", "tks", "ltks", "kwd", "int", "flt", "dt"})

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Operators a chunk-level filter can express. The two negatives are refused
# for the same reason the doc-meta push-down refuses them: on a multi-valued
# field ``must_not term`` excludes a chunk whose other value would satisfy
# the in-memory semantics, and nothing at query time tells us a field is
# single-valued.
CHUNK_FILTER_OPS: frozenset[str] = SUPPORTED_OPERATORS - MULTIVALUE_UNSAFE_NEGATIVE_OPS

BOOST_OPS: frozenset[str] = frozenset({"=", "in", ">", "<", "≥", "≤", "contains", "max", "min"})
DEFAULT_AUTO_WEIGHT = 0.15
DEFAULT_MAX_TOTAL = 0.3
# A boost weight in [0, 1] on a fused score in [0, 1] becomes an ES clause
# boost in [1, 6]: enough to pull a metadata-only winner into the candidate
# pool without letting it outrank a strong text match on its own.
_ES_BOOST_SCALE = 5.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ChunkMetadataConfig:
    enabled: bool = False
    fields: list[str] = field(default_factory=list)
    # Set by the backfill once every existing chunk of the dataset carries the
    # fields. Until then queries keep using the doc-id path: no half-filtered
    # result is ever served.
    ready: bool = False

    @property
    def active(self) -> bool:
        return self.enabled and self.ready and bool(self.fields)

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "fields": list(self.fields), "ready": self.ready}


def validate_fields(keys: Iterable[Any]) -> list[str]:
    """Whitelist keys as ES field-name-safe, deduplicated, capped."""
    out: list[str] = []
    for k in keys:
        if not isinstance(k, str):
            raise TypeError(f"chunk_metadata.fields entries must be strings, got {k!r}")
        k = k.strip()
        if not _KEY_RE.match(k):
            raise ValueError(f"chunk_metadata field {k!r} must match {_KEY_RE.pattern}")
        if k.lower() in RESERVED_KEYS:
            raise ValueError(f"chunk_metadata field {k!r} collides with a chunk field")
        if k not in out:
            out.append(k)
    if len(out) > MAX_FIELDS:
        raise ValueError(f"chunk_metadata allows at most {MAX_FIELDS} fields, got {len(out)}")
    return out


def parse_config(parser_config: dict | None) -> ChunkMetadataConfig | None:
    """``parser_config.chunk_metadata`` → config, or None when absent/invalid.

    Invalid entries are logged and ignored rather than raised: this runs on
    every retrieval and a bad dataset setting must degrade to the doc-id path,
    not break search.
    """
    if not isinstance(parser_config, dict):
        return None
    raw = parser_config.get(CONFIG_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        fields = validate_fields(raw.get("fields") or [])
    except (TypeError, ValueError) as e:
        logging.warning("Ignoring chunk_metadata config: %s", e)
        return None
    return ChunkMetadataConfig(enabled=bool(raw.get("enabled", False)), fields=fields, ready=bool(raw.get("ready", False)))


def store_supports(doc_store: Any) -> bool:
    """Only connectors that can write the fields can be trusted to have them.

    Elasticsearch/OpenSearch implement ``update_chunk_metadata``; Infinity,
    OceanBase, GaussDB and SereneDB have fixed chunk schemas and keep the
    doc-id path.
    """
    return callable(getattr(doc_store, "update_chunk_metadata", None))


_CURRENT_STORE = object()


def config_for_kbs(kbs: Sequence[Any], doc_store: Any = _CURRENT_STORE) -> ChunkMetadataConfig | None:
    """Config usable for a query over ``kbs`` on ``doc_store`` (default: the
    configured doc store), else None."""
    if doc_store is _CURRENT_STORE:
        from common import settings  # lazy: settings imports the connectors that import this module

        doc_store = getattr(settings, "docStoreConn", None)
    if not store_supports(doc_store) or not kbs:
        return None
    return config_for_datasets([getattr(kb, "parser_config", None) or {} for kb in kbs])


def config_for_datasets(parser_configs: Sequence[dict | None]) -> ChunkMetadataConfig | None:
    """The configuration a query over several datasets can rely on.

    Every dataset must be active, and only the keys they all carry are usable:
    a condition on a key that one dataset lacks would silently exclude that
    dataset's chunks.
    """
    if not parser_configs:
        return None
    common: set | None = None
    for pc in parser_configs:
        cfg = parse_config(pc)
        if cfg is None or not cfg.active:
            return None
        common = set(cfg.fields) if common is None else common & set(cfg.fields)
    if not common:
        return None
    return ChunkMetadataConfig(enabled=True, fields=sorted(common), ready=True)


# ---------------------------------------------------------------------------
# Chunk fields
# ---------------------------------------------------------------------------


def kwd_field(key: str) -> str:
    return f"{CHUNK_META_PREFIX}{key}_kwd"


def int_field(key: str) -> str:
    return f"{CHUNK_META_PREFIX}{key}_int"


def flt_field(key: str) -> str:
    return f"{CHUNK_META_PREFIX}{key}_flt"


def dt_field(key: str) -> str:
    return f"{CHUNK_META_PREFIX}{key}_dt"


def all_field_names(keys: Iterable[str]) -> list[str]:
    out: list[str] = []
    for k in keys:
        out.extend((kwd_field(k), int_field(k), flt_field(k), dt_field(k)))
    return out


def _typed(value: Any):
    """(kind, coerced) for one scalar metadata value; kind in int/flt/dt/None."""
    if isinstance(value, bool):
        return "int", int(value)
    if isinstance(value, int):
        return "int", value
    if isinstance(value, float):
        if math.isfinite(value):
            return "flt", value
        return None, None
    if isinstance(value, str):
        s = value.strip()
        if _DATE_RE.match(s):
            return "dt", s
        # "2015_062" must stay a string: only plain decimal literals are numbers.
        if re.fullmatch(r"[+-]?\d+", s):
            try:
                return "int", int(s)
            except ValueError:
                return None, None
        if re.fullmatch(r"[+-]?\d+\.\d+", s):
            try:
                return "flt", float(s)
            except ValueError:
                return None, None
    return None, None


def chunk_fields(meta_fields: dict | None, keys: Sequence[str]) -> dict[str, Any]:
    """Chunk-level copy of the whitelisted metadata.

    Every present key gets ``meta_<key>_kwd`` with the string form of each
    value (a list for multi-valued metadata). Values that are integers, floats
    or ``YYYY-MM-DD`` dates additionally get the typed twin so range operators
    and ``max``/``min`` boosts compare numerically.
    """
    out: dict[str, Any] = {}
    if not meta_fields:
        return out
    for key in keys:
        if key not in meta_fields:
            continue
        raw = meta_fields[key]
        values = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
        values = [v for v in values if v is not None and v != ""]
        if not values:
            continue
        kwds = [str(v).strip() for v in values]
        out[kwd_field(key)] = kwds if len(kwds) > 1 else kwds[0]
        typed: dict[str, list] = {}
        for v in values:
            kind, coerced = _typed(v)
            if kind:
                typed.setdefault(kind, []).append(coerced)
        for kind, vals in typed.items():
            name = {"int": int_field, "flt": flt_field, "dt": dt_field}[kind](key)
            out[name] = vals if len(vals) > 1 else vals[0]
    return out


def removal_fields(meta_fields: dict | None, keys: Sequence[str]) -> list[str]:
    """Chunk fields that must be dropped so a metadata edit removing a key or
    changing its type does not leave a stale twin behind."""
    keep = set(chunk_fields(meta_fields, keys).keys())
    return [f for f in all_field_names(keys) if f not in keep]


# ---------------------------------------------------------------------------
# Filter → chunk query clauses
# ---------------------------------------------------------------------------


def is_chunk_filterable(conditions: Sequence[dict], keys: Iterable[str]) -> bool:
    """Can every condition be expressed on chunk fields of these keys?"""
    allowed = set(keys)
    if not conditions:
        return False
    for flt in conditions:
        if not isinstance(flt, dict):
            return False
        if flt.get("key") not in allowed:
            return False
        if flt.get("op") not in CHUNK_FILTER_OPS:
            return False
    return True


def _range_field(key: str, coerced: Any) -> str:
    if isinstance(coerced, (bool, int)):
        return int_field(key)
    if isinstance(coerced, float):
        return flt_field(key)
    if isinstance(coerced, str) and _DATE_RE.match(coerced):
        return dt_field(key)
    return kwd_field(key)


def _term(key: str, coerced: Any) -> dict:
    if isinstance(coerced, str):
        return {"term": {kwd_field(key): {"value": coerced, "case_insensitive": True}}}
    # Numbers and booleans are stored as strings in the keyword twin too, so
    # equality on the keyword field works for every value type and does not
    # depend on which typed twin the indexer chose.
    return {"term": {kwd_field(key): {"value": str(coerced), "case_insensitive": True}}}


def condition_clause(flt: dict) -> dict:
    """One condition → one ES bool clause on chunk fields.

    Raises ``UnsupportedMetaFilter`` for operators or values that cannot be
    expressed, so callers fall back to the doc-id path unchanged.
    """
    op = flt.get("op")
    key = flt.get("key")
    value = flt.get("value")
    if not isinstance(key, str) or not key:
        raise UnsupportedMetaFilter("condition is missing a string key", flt)
    if op not in CHUNK_FILTER_OPS:
        raise UnsupportedMetaFilter(f"operator {op!r} is not chunk-filterable", flt)
    kwd = kwd_field(key)
    if op == "empty":
        return {"bool": {"must_not": [{"exists": {"field": kwd}}]}}
    if op == "not empty":
        return {"exists": {"field": kwd}}
    if op == "=":
        return _term(key, _coerce_scalar(value, flt))
    if op in ("in",):
        members = _csv_or_list(value, flt)
        return {"bool": {"should": [_term(key, m) for m in members], "minimum_should_match": 1}}
    if op in (">", "<", "≥", "≤"):
        coerced = _coerce_range_value(value, flt)
        es_op = {">": "gt", "<": "lt", "≥": "gte", "≤": "lte"}[op]
        return {"range": {_range_field(key, coerced): {es_op: coerced}}}
    text = _coerce_string(value, flt)
    if op == "contains":
        return {"wildcard": {kwd: {"value": f"*{_escape_wildcard(text)}*", "case_insensitive": True}}}
    if op == "not contains":
        return {"bool": {"must": [{"exists": {"field": kwd}}], "must_not": [{"wildcard": {kwd: {"value": f"*{_escape_wildcard(text)}*", "case_insensitive": True}}}]}}
    if op == "start with":
        return {"prefix": {kwd: {"value": text, "case_insensitive": True}}}
    if op == "end with":
        return {"wildcard": {kwd: {"value": f"*{_escape_wildcard(text)}", "case_insensitive": True}}}
    raise UnsupportedMetaFilter(f"no handler for operator {op!r}", flt)


def build_chunk_filter(conditions: Sequence[dict], logic: str = "and") -> dict:
    """Conditions → one ES query dict to attach to the chunk query's filter."""
    if logic not in ("and", "or"):
        raise UnsupportedMetaFilter(f"unknown logic {logic!r}")
    clauses = [condition_clause(flt) for flt in conditions]
    if not clauses:
        raise UnsupportedMetaFilter("no conditions")
    if logic == "and":
        return {"bool": {"filter": clauses}}
    return {"bool": {"should": clauses, "minimum_should_match": 1}}


# ---------------------------------------------------------------------------
# Boost
# ---------------------------------------------------------------------------


@dataclass
class BoostCondition:
    key: str
    op: str
    value: Any = None
    weight: float = DEFAULT_AUTO_WEIGHT

    def to_dict(self) -> dict:
        return {"key": self.key, "op": self.op, "value": self.value, "weight": self.weight}


@dataclass
class BoostKey:
    """A key offered to the LLM in ``semi_auto`` boost mode; ``op``/``weight``
    may be pinned per key the way the filter's ``semi_auto`` pins ``op``."""

    key: str
    op: str | None = None
    weight: float | None = None


@dataclass
class BoostConfig:
    method: str = "off"  # off | manual | auto | semi_auto
    manual: list[BoostCondition] = field(default_factory=list)
    semi_auto: list[BoostKey] = field(default_factory=list)
    auto_weight: float = DEFAULT_AUTO_WEIGHT
    max_total: float = DEFAULT_MAX_TOTAL

    @property
    def uses_llm(self) -> bool:
        return self.method in ("auto", "semi_auto")

    @property
    def semi_auto_keys(self) -> list[str]:
        return [b.key for b in self.semi_auto]


def _clamp(x: Any, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return min(1.0, max(0.0, v))


def parse_boost(raw: Any, keys: Iterable[str] | None = None) -> BoostConfig:
    """``meta_data_filter.boost`` → BoostConfig. Unknown ops/keys are dropped."""
    cfg = BoostConfig()
    if not isinstance(raw, dict):
        return cfg
    method = raw.get("method", "manual" if raw.get("manual") else "off")
    cfg.method = method if method in ("off", "manual", "auto", "semi_auto") else "off"
    cfg.auto_weight = _clamp(raw.get("auto_weight", DEFAULT_AUTO_WEIGHT), DEFAULT_AUTO_WEIGHT)
    cfg.max_total = _clamp(raw.get("max_total", DEFAULT_MAX_TOTAL), DEFAULT_MAX_TOTAL)
    allowed = set(keys) if keys is not None else None
    for item in raw.get("manual") or []:
        b = boost_from_dict(item, cfg.auto_weight, allowed)
        if b:
            cfg.manual.append(b)
    for item in raw.get("semi_auto") or []:
        if isinstance(item, str):
            item = {"key": item}
        if not isinstance(item, dict) or not isinstance(item.get("key"), str) or not item["key"]:
            continue
        if allowed is not None and item["key"] not in allowed:
            continue
        op = item.get("op") if item.get("op") in BOOST_OPS else None
        weight = _clamp(item["weight"], cfg.auto_weight) if item.get("weight") not in (None, "") else None
        cfg.semi_auto.append(BoostKey(key=item["key"], op=op, weight=weight))
    return cfg


def boost_from_dict(item: Any, default_weight: float, allowed: set | None = None) -> BoostCondition | None:
    if not isinstance(item, dict):
        return None
    key, op = item.get("key"), item.get("op", "=")
    if not isinstance(key, str) or not key or op not in BOOST_OPS:
        return None
    if allowed is not None and key not in allowed:
        return None
    if op not in ("max", "min") and item.get("value") in (None, ""):
        return None
    return BoostCondition(key=key, op=op, value=item.get("value"), weight=_clamp(item.get("weight", default_weight), default_weight))


def boost_should_clauses(boosts: Sequence[BoostCondition]) -> list[dict]:
    """ES ``should`` clauses that lift matching chunks into the candidate pool.

    ``max``/``min`` have no fixed value to match and are scored in Python
    only; ``contains`` is left out because a boosted wildcard over the whole
    pool is expensive for a small ranking gain.
    """
    clauses: list[dict] = []
    for b in boosts:
        es_boost = 1.0 + _ES_BOOST_SCALE * b.weight
        try:
            if b.op == "=":
                c = _term(b.key, _coerce_scalar(b.value, b.to_dict()))
                c["term"][kwd_field(b.key)]["boost"] = es_boost
            elif b.op == "in":
                members = _csv_or_list(b.value, b.to_dict())
                c = {"bool": {"should": [_term(b.key, m) for m in members], "minimum_should_match": 1, "boost": es_boost}}
            elif b.op in (">", "<", "≥", "≤"):
                coerced = _coerce_range_value(b.value, b.to_dict())
                es_op = {">": "gt", "<": "lt", "≥": "gte", "≤": "lte"}[b.op]
                c = {"range": {_range_field(b.key, coerced): {es_op: coerced, "boost": es_boost}}}
            else:
                continue
        except UnsupportedMetaFilter:
            continue
        clauses.append(c)
    return clauses


def fields_for_boosts(boosts: Sequence[BoostCondition]) -> list[str]:
    """Chunk fields the scorer needs returned with each hit."""
    return all_field_names(sorted({b.key for b in boosts}))


def _chunk_values(chunk: dict, key: str) -> list:
    v = chunk.get(kwd_field(key))
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def _chunk_typed_value(chunk: dict, key: str) -> float | None:
    """Numeric view of a chunk's value for ranges and max/min; dates become
    ordinal-like numbers so that "newest" works on ``_dt`` fields too."""
    for name in (int_field(key), flt_field(key)):
        v = chunk.get(name)
        if isinstance(v, list):
            v = v[0] if v else None
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    v = chunk.get(dt_field(key))
    if isinstance(v, list):
        v = v[0] if v else None
    if isinstance(v, str) and _DATE_RE.match(v[:10]):
        y, m, d = int(v[:4]), int(v[5:7]), int(v[8:10])
        return float(y * 10000 + m * 100 + d)
    return None


def _matches(b: BoostCondition, chunk: dict) -> bool:
    if b.op == "=":
        target = str(b.value).strip().lower()
        return any(str(v).strip().lower() == target for v in _chunk_values(chunk, b.key))
    if b.op == "in":
        try:
            members = {str(m).lower() for m in _csv_or_list(b.value, b.to_dict())}
        except UnsupportedMetaFilter:
            return False
        return any(str(v).strip().lower() in members for v in _chunk_values(chunk, b.key))
    if b.op == "contains":
        needle = str(b.value).lower()
        return any(needle in str(v).lower() for v in _chunk_values(chunk, b.key))
    if b.op in (">", "<", "≥", "≤"):
        have = _chunk_typed_value(chunk, b.key)
        if have is None:
            return False
        kind, want = _typed(b.value)
        if kind == "dt":
            want = _chunk_typed_value({dt_field(b.key): want}, b.key)
        if want is None:
            return False
        return {">": have > want, "<": have < want, "≥": have >= want, "≤": have <= want}[b.op]
    return False


def boost_scores(boosts: Sequence[BoostCondition], chunks: Sequence[dict], max_total: float = DEFAULT_MAX_TOTAL) -> list[float]:
    """Additive score per chunk: Σ weight·match, ``max``/``min`` normalised
    over the candidate pool (1.0 for the extreme value, 0.0 for the other
    end), capped at ``max_total`` so metadata never outranks relevance alone."""
    n = len(chunks)
    scores = [0.0] * n
    if not boosts or not n:
        return scores
    for b in boosts:
        if b.op in ("max", "min"):
            vals = [_chunk_typed_value(c, b.key) for c in chunks]
            present = [v for v in vals if v is not None]
            if not present:
                continue
            lo, hi = min(present), max(present)
            for i, v in enumerate(vals):
                if v is None:
                    continue
                if hi == lo:
                    frac = 1.0
                else:
                    frac = (v - lo) / (hi - lo)
                    if b.op == "min":
                        frac = 1.0 - frac
                scores[i] += b.weight * frac
            continue
        for i, c in enumerate(chunks):
            if _matches(b, c):
                scores[i] += b.weight
    cap = max(0.0, float(max_total))
    return [min(cap, s) for s in scores]


# ---------------------------------------------------------------------------
# LLM conditions: hard vs soft
# ---------------------------------------------------------------------------


def split_conditions(conditions: Sequence[dict]) -> tuple[list, list]:
    """``strength: soft`` conditions become boosts; everything else filters."""
    hard, soft = [], []
    for c in conditions or []:
        if not isinstance(c, dict):
            continue
        if str(c.get("strength", "hard")).lower() == "soft":
            soft.append({k: v for k, v in c.items() if k != "strength"})
        else:
            hard.append({k: v for k, v in c.items() if k != "strength"})
    return hard, soft


def boosts_from_conditions(conditions: Sequence[dict], weight: float, keys: Iterable[str] | None = None) -> list[BoostCondition]:
    allowed = set(keys) if keys is not None else None
    out: list[BoostCondition] = []
    for c in conditions:
        op = c.get("op")
        # A soft "≠"/"not in" cannot be a positive preference; "in" and "="
        # cover what the model means in practice.
        if op not in BOOST_OPS:
            continue
        b = boost_from_dict({**c, "weight": weight}, weight, allowed)
        if b:
            out.append(b)
    return out
