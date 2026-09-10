#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
import ast
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict

import json_repair

from common.metadata_es_filter import UnsupportedMetaFilter


def convert_conditions(metadata_condition):
    if metadata_condition is None:
        metadata_condition = {}
    op_mapping = {"is": "=", "not is": "≠", ">=": "≥", "<=": "≤", "!=": "≠"}
    return [{"op": op_mapping.get(cond["comparison_operator"], cond["comparison_operator"]), "key": cond["name"], "value": cond["value"]} for cond in metadata_condition.get("conditions", [])]


def meta_filter(metas: dict, filters: list[dict], logic: str = "and"):
    doc_ids = None

    def normalize_string_values(value):
        if isinstance(value, str):
            return value.lower()
        if isinstance(value, list):
            return [item.lower() if isinstance(item, str) else item for item in value]
        return value

    def filter_out(v2docs, operator, value):
        ids = []
        original_value = value
        for input, docids in v2docs.items():
            # Reset to the pristine filter value each iteration -- the comparison branch
            # below reassigns `value` in place (date normalization, literal_eval coercion),
            # and reusing that mutated value on the next dict entry compares it against
            # the wrong (already-coerced) type/content instead of the original filter value.
            value = original_value
            if operator in ["=", "≠", ">", "<", "≥", "≤"]:
                # Check if input is in YYYY-MM-DD date format
                input_str = str(input).strip()
                value_str = str(value).strip()

                # Strict date format detection: YYYY-MM-DD (must be 10 chars with correct format)
                is_input_date = len(input_str) == 10 and input_str[4] == "-" and input_str[7] == "-" and input_str[:4].isdigit() and input_str[5:7].isdigit() and input_str[8:10].isdigit()

                is_value_date = len(value_str) == 10 and value_str[4] == "-" and value_str[7] == "-" and value_str[:4].isdigit() and value_str[5:7].isdigit() and value_str[8:10].isdigit()

                if is_value_date:
                    # Query value is in date format
                    if is_input_date:
                        # Data is also in date format: perform date comparison
                        input = input_str
                        value = value_str
                    else:
                        # Data is not in date format: skip this record (no match)
                        continue
                else:
                    # Query value is not in date format: use original logic
                    try:
                        if isinstance(input, list):
                            input = input[0]
                    except Exception:
                        pass
                    # Commit both literal_eval results together, or neither -- assigning
                    # just one side (e.g. "None" parses but "none" doesn't) would compare
                    # mismatched types after lowercasing and silently break the
                    # case-insensitive match below.
                    try:
                        input, value = ast.literal_eval(input), ast.literal_eval(value)
                    except Exception:
                        pass

                    # Convert strings to lowercase
                    if isinstance(input, str):
                        input = input.lower()
                    if isinstance(value, str):
                        value = value.lower()
            else:
                # Non-comparison operators: maintain original logic
                input = normalize_string_values(input)
                value = normalize_string_values(value)

            matched = False
            try:
                if operator == "contains":
                    matched = str(input).find(value) >= 0 if not isinstance(input, list) else any(str(i).find(value) >= 0 for i in input)
                elif operator == "not contains":
                    matched = str(input).find(value) == -1 if not isinstance(input, list) else all(str(i).find(value) == -1 for i in input)
                elif operator == "in":
                    matched = input in value if not isinstance(input, list) else all(i in value for i in input)
                elif operator == "not in":
                    matched = input not in value if not isinstance(input, list) else all(i not in value for i in input)
                elif operator == "start with":
                    matched = str(input).lower().startswith(str(value).lower()) if not isinstance(input, list) else "".join([str(i).lower() for i in input]).startswith(str(value).lower())
                elif operator == "end with":
                    matched = str(input).lower().endswith(str(value).lower()) if not isinstance(input, list) else "".join([str(i).lower() for i in input]).endswith(str(value).lower())
                elif operator == "empty":
                    matched = not input
                elif operator == "not empty":
                    matched = bool(input)
                elif operator == "=":
                    matched = input == value
                elif operator == "≠":
                    matched = input != value
                elif operator == ">":
                    matched = input > value
                elif operator == "<":
                    matched = input < value
                elif operator == "≥":
                    matched = input >= value
                elif operator == "≤":
                    matched = input <= value
            except Exception:
                pass

            if matched:
                ids.extend(docids)
        return ids

    for f in filters:
        k = f["key"]
        if k not in metas:
            # Key not found in metas: treat as no match
            ids = []
        else:
            v2docs = metas[k]
            ids = filter_out(v2docs, f["op"], f["value"])

        if doc_ids is None:
            doc_ids = set(ids)
        else:
            if logic == "and":
                doc_ids = doc_ids & set(ids)
                if not doc_ids:
                    logging.debug(f"meta_filter filters={filters}, logic={logic}, early return []")
                    return []
            else:
                doc_ids = doc_ids | set(ids)
    return list(doc_ids or [])


async def apply_meta_data_filter(
    meta_data_filter: dict | None,
    metas: dict | None = None,
    question: str = "",
    chat_mdl: Any = None,
    base_doc_ids: list[str] | None = None,
    manual_value_resolver: Callable[[dict], dict] | None = None,
    kb_ids: list[str] | None = None,
    metas_loader: Callable[[], dict] | None = None,
) -> list[str] | None:
    """
    Apply metadata filtering rules and return the filtered doc_ids.

    meta_data_filter supports three modes:
    - auto: generate filter conditions via LLM (gen_meta_filter)
    - semi_auto: generate conditions using selected metadata keys only
    - manual: directly filter based on provided conditions

    When ``kb_ids`` is supplied, metadata filters are pushed down to the doc metadata
    index (ES/Infinity) via ``DocMetadataService.filter_doc_ids_by_metadata`` instead
    of being evaluated in Python over ``metas``. The in-memory ``meta_filter`` path
    remains the fallback so callers without a KB scope, or backends without push-down
    support, behave exactly as before.

    ``metas`` may be supplied eagerly or via ``metas_loader``. The loader is
    only invoked when the metadata dict is actually needed — i.e. for the LLM
    context in ``auto`` / ``semi_auto`` modes, or as the in-memory fallback
    when push-down can't service a request. ``manual`` mode that lands on the
    push-down path therefore skips the expensive
    ``get_flatted_meta_by_kbs`` round-trip entirely.

    Returns:
        list of doc_ids, ["-999"] when manual filters yield no result, or None
        when auto/semi_auto filters return empty.
    """
    scope = await apply_meta_data_scope(
        meta_data_filter,
        metas,
        question,
        chat_mdl,
        base_doc_ids,
        manual_value_resolver,
        kb_ids=kb_ids,
        metas_loader=metas_loader,
        chunk_meta=None,
    )
    return scope.doc_ids


@dataclass
class MetaScope:
    """What a metadata configuration resolved to for one query.

    ``doc_ids`` keeps the contract of ``apply_meta_data_filter`` (a list, or
    ``["-999"]`` for "manual matched nothing", or ``None`` for "auto matched
    nothing → no filter"). ``chunk_filter`` carries the conditions that the
    doc store applies directly on chunk metadata fields instead — see
    ``common.chunk_metadata`` — and ``boosts`` the weighted preferences that
    raise a chunk's score without excluding the others.
    """

    doc_ids: list[str] | None
    chunk_filter: dict | None = None
    boosts: list = None  # list[common.chunk_metadata.BoostCondition]
    boost_max_total: float = 0.3

    def __post_init__(self):
        if self.boosts is None:
            self.boosts = []

    @property
    def chunk_filter_conditions(self) -> list[dict]:
        return (self.chunk_filter or {}).get("conditions", [])


async def apply_meta_data_scope(
    meta_data_filter: dict | None,
    metas: dict | None = None,
    question: str = "",
    chat_mdl: Any = None,
    base_doc_ids: list[str] | None = None,
    manual_value_resolver: Callable[[dict], dict] | None = None,
    kb_ids: list[str] | None = None,
    metas_loader: Callable[[], dict] | None = None,
    chunk_meta=None,
) -> MetaScope:
    """``apply_meta_data_filter`` that can also scope on chunk metadata fields
    and produce score boosts.

    ``chunk_meta`` is the ``common.chunk_metadata.ChunkMetadataConfig`` the
    queried datasets have in common, or ``None`` when any of them has not
    opted in / been backfilled — then this is exactly the doc-id path.

    With an active ``chunk_meta``:
    - hard conditions whose keys are whitelisted and whose operators are
      chunk-safe are returned in ``chunk_filter`` and never resolved to
      document ids;
    - ``meta_data_filter["boost"]`` (manual preferences) and the conditions the
      LLM tags ``"strength": "soft"`` become ``boosts``;
    - the LLM is called once for both, with ``allow_soft`` so it can tell a
      requirement from a preference.
    """
    from common import chunk_metadata as cm
    from rag.prompts.generator import gen_meta_filter  # move from the top of the file to avoid circular import

    doc_ids = list(base_doc_ids) if base_doc_ids else []
    scope = MetaScope(doc_ids=doc_ids)

    if not meta_data_filter:
        return scope

    method = meta_data_filter.get("method")
    active_keys = list(chunk_meta.fields) if (chunk_meta is not None and getattr(chunk_meta, "active", False)) else None
    boost_cfg = cm.parse_boost(meta_data_filter.get("boost"), active_keys) if active_keys is not None else cm.BoostConfig()
    scope.boost_max_total = boost_cfg.max_total
    if active_keys is not None:
        scope.boosts.extend(boost_cfg.manual)
    allow_soft = active_keys is not None and boost_cfg.uses_llm

    # Memoised metadata loader. ``_get_metas`` materialises the dict at most
    # once per call; downstream branches that never reach an in-memory eval
    # leave the loader untouched.
    cached_metas: dict | None = metas

    def _get_metas() -> dict:
        nonlocal cached_metas
        if cached_metas is None:
            cached_metas = metas_loader() if metas_loader else {}
        return cached_metas

    def _run_metadata_filter(conditions: list[dict], logic: str) -> list[str]:
        """Run conditions through ES/Infinity push-down when possible, in-memory otherwise."""
        return filter_doc_ids_by_metadata(kb_ids or [], conditions, logic, _get_metas)

    def _take_soft(conditions: list[dict]) -> list[dict]:
        """Split LLM output: soft conditions become boosts, hard ones are returned."""
        hard, soft = cm.split_conditions(conditions)
        if soft and allow_soft:
            scope.boosts.extend(cm.boosts_from_conditions(soft, boost_cfg.auto_weight, active_keys))
        elif soft:
            # No chunk fields to score on: a preference we cannot honour is
            # dropped rather than turned into a hard filter the user did not ask for.
            logging.debug(f"Dropping {len(soft)} soft metadata condition(s): chunk metadata not active")
        return hard

    def _apply_hard(conditions: list[dict], logic: str, manual: bool) -> bool:
        """Chunk filter when possible, doc ids otherwise. Returns False when the
        filter matched nothing in the sense the caller must act on."""
        if active_keys is not None and cm.is_chunk_filterable(conditions, active_keys):
            try:
                cm.build_chunk_filter(conditions, logic)
            except UnsupportedMetaFilter as e:
                logging.debug(f"Chunk metadata filter not expressible ({e}); falling back to doc ids")
            else:
                scope.chunk_filter = {"conditions": conditions, "logic": logic}
                logging.debug(f"Metadata filter applied on chunk fields: {scope.chunk_filter}")
                return True
        found = _run_metadata_filter(conditions, logic)
        scope.doc_ids.extend(found)
        if conditions and not scope.doc_ids:
            if manual:
                scope.doc_ids = ["-999"]
                return True
            return False
        return True

    if method == "auto":
        filters: dict = await gen_meta_filter(chat_mdl, _get_metas(), question, allow_soft=allow_soft)
        logging.debug(f"Metadata filter(auto) generated: {filters}")
        hard = _take_soft(filters["conditions"])
        if hard and not _apply_hard(hard, filters.get("logic", "and"), manual=False):
            scope.doc_ids = None
            return scope
        if not hard and not scope.doc_ids and not scope.chunk_filter:
            scope.doc_ids = None
            return scope
    elif method == "semi_auto":
        selected_keys = []
        constraints = {}
        for item in meta_data_filter.get("semi_auto", []):
            if isinstance(item, str):
                selected_keys.append(item)
            elif isinstance(item, dict):
                key = item.get("key")
                op = item.get("op")
                selected_keys.append(key)
                if op:
                    constraints[key] = op
        # Preference keys share the single LLM call; they can only ever come
        # back as soft conditions because they are offered without constraints.
        for key in boost_cfg.semi_auto:
            if key not in selected_keys:
                selected_keys.append(key)

        if selected_keys:
            current_metas = _get_metas()
            filtered_metas = {key: current_metas[key] for key in selected_keys if key in current_metas}
            if filtered_metas:
                filters: dict = await gen_meta_filter(chat_mdl, filtered_metas, question, constraints=constraints, allow_soft=allow_soft)
                logging.debug(f"Metadata filter(semi_auto) generated: {filters}")
                hard = _take_soft(filters["conditions"])
                if hard and not _apply_hard(hard, filters.get("logic", "and"), manual=False):
                    scope.doc_ids = None
                    return scope
                if not hard and not scope.doc_ids and not scope.chunk_filter:
                    scope.doc_ids = None
                    return scope
    elif method == "manual":
        filters = meta_data_filter.get("manual", [])
        if manual_value_resolver:
            filters = [manual_value_resolver(flt) for flt in filters]
        logging.debug(f"Metadata filter(manual): {filters}")
        if filters:
            _apply_hard(filters, meta_data_filter.get("logic", "and"), manual=True)

    logging.debug(f"apply_meta_data_scope meta_filter={meta_data_filter}, returning doc_ids={scope.doc_ids}, chunk_filter={scope.chunk_filter}, boosts={len(scope.boosts)}")
    return scope


def _try_meta_pushdown(
    kb_ids: list[str],
    conditions: list[dict],
    logic: str,
) -> list[str] | None:
    """Attempt metadata-index push-down; return ``None`` to fall back in memory.

    Lazy-imports ``DocMetadataService`` so this module stays usable in
    environments where the API/db layer hasn't been wired up (e.g. unit tests
    that exercise ``meta_filter`` directly).
    """
    try:
        from api.db.services.doc_metadata_service import DocMetadataService
    except ImportError as e:
        logging.debug(f"Metadata filter push-down disabled because the service import failed: {e}")
        return None
    return DocMetadataService.filter_doc_ids_by_meta_pushdown(kb_ids, conditions, logic)


def filter_doc_ids_by_metadata(
    kb_ids: list[str],
    conditions: list[dict],
    logic: str,
    metas_loader: Callable[[], dict],
) -> list[str]:
    """Filter document IDs through the metadata index with a lazy exact fallback."""
    doc_ids = _try_meta_pushdown(kb_ids, conditions, logic) if conditions and kb_ids else None
    if doc_ids is not None:
        logging.debug(
            "Metadata filter used push-down: kb_count=%d condition_count=%d matched_doc_count=%d",
            len(kb_ids),
            len(conditions),
            len(doc_ids),
        )
        return doc_ids
    logging.debug(
        "Metadata filter uses in-memory fallback: kb_count=%d condition_count=%d",
        len(kb_ids),
        len(conditions),
    )
    return meta_filter(metas_loader(), conditions, logic)


def dedupe_list(values: list) -> list:
    seen = set()
    deduped = []
    for item in values:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def update_metadata_to(metadata, meta):
    """Merge ``meta`` into ``metadata``.

    String / list[str] values keep the previous multi-value merge + dedupe
    behavior (used by LLM-extracted metadata). Scalars (bool / int / float /
    None) and structured values (list[dict], dict, ...) are preserved so that
    document system fields such as ``_isCurrent`` / ``_version`` and PDF
    ``outline`` survive merges that later fully replace ``meta_fields``.
    """
    if not meta:
        return metadata
    if isinstance(meta, str):
        try:
            meta = json_repair.loads(meta)
        except Exception:
            logging.error("Meta data format error.")
            return metadata
    if not isinstance(meta, dict):
        return metadata

    for k, v in meta.items():
        if isinstance(v, list):
            if all(isinstance(vv, str) for vv in v):
                if not v:
                    continue
                v = dedupe_list(v)
            else:
                # Structured list (e.g. outline [{title, depth}, ...]).
                if k not in metadata:
                    logging.debug(
                        "update_metadata_to preserve structured list key=%s src=%s",
                        k,
                        type(v).__name__,
                    )
                    metadata[k] = v
                else:
                    logging.debug(
                        "update_metadata_to skip structured list key=%s src=%s dst=%s",
                        k,
                        type(v).__name__,
                        type(metadata[k]).__name__,
                    )
                continue
        elif isinstance(v, (str, bool, int, float)) or v is None:
            pass
        else:
            # dict / other structured values: keep if absent.
            if k not in metadata:
                logging.debug(
                    "update_metadata_to preserve structured value key=%s src=%s",
                    k,
                    type(v).__name__,
                )
                metadata[k] = v
            else:
                logging.debug(
                    "update_metadata_to skip structured value key=%s src=%s dst=%s",
                    k,
                    type(v).__name__,
                    type(metadata[k]).__name__,
                )
            continue

        if k not in metadata:
            if not isinstance(v, (list, str)):
                logging.debug(
                    "update_metadata_to preserve scalar key=%s src=%s",
                    k,
                    type(v).__name__,
                )
            metadata[k] = v
            continue
        if isinstance(metadata[k], list) and isinstance(v, (list, str)):
            if not all(isinstance(x, str) for x in metadata[k]):
                logging.debug(
                    "update_metadata_to skip merge into structured list key=%s src=%s dst=%s",
                    k,
                    type(v).__name__,
                    type(metadata[k]).__name__,
                )
                continue
            if isinstance(v, list):
                metadata[k].extend(v)
            else:
                metadata[k].append(v)
            metadata[k] = dedupe_list(metadata[k])
        else:
            if not isinstance(v, (list, str)):
                logging.debug(
                    "update_metadata_to replace scalar key=%s src=%s dst=%s",
                    k,
                    type(v).__name__,
                    type(metadata[k]).__name__,
                )
            metadata[k] = v

    return metadata


def metadata_schema(metadata: dict | list | None) -> Dict[str, Any]:
    if not metadata:
        return {}
    properties = {}

    for item in metadata:
        key = item.get("key")
        if not key:
            continue

        prop_schema = {"description": item.get("description", "")}
        if "enum" in item and item["enum"]:
            prop_schema["enum"] = item["enum"]
            prop_schema["type"] = "string"

        properties[key] = prop_schema

    json_schema = {
        "type": "object",
        "properties": properties,
    }

    json_schema["additionalProperties"] = False
    return json_schema


def _is_json_schema(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    if "$schema" in obj:
        return True
    return obj.get("type") == "object" and isinstance(obj.get("properties"), dict)


def _is_metadata_list(obj: list) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    for item in obj:
        if not isinstance(item, dict):
            return False
        key = item.get("key")
        if not isinstance(key, str) or not key:
            return False
        if "enum" in item and item["enum"] is not None and not isinstance(item["enum"], list):
            return False
        if "description" in item and item["description"] is not None and not isinstance(item["description"], str):
            return False
        if "descriptions" in item and item["descriptions"] is not None and not isinstance(item["descriptions"], str):
            return False
    return True


def turn2jsonschema(obj: dict | list) -> Dict[str, Any]:
    if isinstance(obj, dict) and _is_json_schema(obj):
        return obj
    if isinstance(obj, list) and _is_metadata_list(obj):
        normalized = []
        for item in obj:
            description = item.get("description") or item.get("descriptions") or ""
            normalized_item = {
                "key": item.get("key"),
                "description": description,
            }
            if "enum" in item and item["enum"] is not None:
                normalized_item["enum"] = item["enum"]
            normalized.append(normalized_item)
        return metadata_schema(normalized)
    return {}
