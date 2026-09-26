from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from common.metadata_utils import apply_meta_data_filter


@pytest.mark.asyncio
async def test_auto_reports_the_conditions_the_llm_generated():
    resolved: dict = {}
    metas = {"year": {"2026": ["doc1"]}}
    generated = {"conditions": [{"key": "year", "op": "=", "value": "2026"}], "logic": "and"}

    with patch("rag.prompts.generator.gen_meta_filter", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = generated
        doc_ids = await apply_meta_data_filter({"method": "auto"}, metas, "2026", MagicMock(), diagnostics=resolved)

    assert doc_ids == ["doc1"]
    assert resolved == {"method": "auto", "status": "applied", "logic": "and", "conditions": generated["conditions"], "matched_document_count": 1}


@pytest.mark.asyncio
async def test_semi_auto_reports_the_method_even_without_usable_keys():
    resolved: dict = {}
    meta_data_filter = {"method": "semi_auto", "semi_auto": ["missing"]}

    with patch("rag.prompts.generator.gen_meta_filter", new_callable=AsyncMock) as mock_gen:
        doc_ids = await apply_meta_data_filter(meta_data_filter, {"year": {"2026": ["doc1"]}}, "q", MagicMock(), diagnostics=resolved)
        mock_gen.assert_not_called()

    assert doc_ids == []
    assert resolved == {"method": "semi_auto", "status": "not_generated", "logic": "and", "conditions": [], "matched_document_count": 0}


@pytest.mark.asyncio
async def test_manual_reports_the_conditions_after_value_resolution():
    resolved: dict = {}
    meta_data_filter = {
        "method": "manual",
        "logic": "or",
        "manual": [{"key": "year", "op": "=", "value": "last"}],
    }
    metas = {"year": {"2026": ["doc1"]}}

    doc_ids = await apply_meta_data_filter(
        meta_data_filter,
        metas,
        manual_value_resolver=lambda flt: {**flt, "value": "2026"},
        diagnostics=resolved,
    )

    assert doc_ids == ["doc1"]
    assert resolved == {"method": "manual", "status": "applied", "logic": "or", "conditions": [{"key": "year", "op": "=", "value": "2026"}], "matched_document_count": 1}


@pytest.mark.asyncio
async def test_no_filter_reports_disabled():
    resolved: dict = {}

    assert await apply_meta_data_filter(None, {}, diagnostics=resolved) == []
    assert resolved == {"method": "disabled", "status": "disabled", "logic": "and", "conditions": [], "matched_document_count": 0}
