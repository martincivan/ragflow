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
        doc_ids = await apply_meta_data_filter({"method": "auto"}, metas, "2026", MagicMock(), resolved_out=resolved)

    assert doc_ids == ["doc1"]
    assert resolved == {"method": "auto", "logic": "and", "conditions": generated["conditions"]}


@pytest.mark.asyncio
async def test_semi_auto_reports_the_method_even_without_usable_keys():
    resolved: dict = {}
    meta_data_filter = {"method": "semi_auto", "semi_auto": ["missing"]}

    with patch("rag.prompts.generator.gen_meta_filter", new_callable=AsyncMock) as mock_gen:
        doc_ids = await apply_meta_data_filter(meta_data_filter, {"year": {"2026": ["doc1"]}}, "q", MagicMock(), resolved_out=resolved)
        mock_gen.assert_not_called()

    assert doc_ids == []
    assert resolved == {"method": "semi_auto", "logic": "and", "conditions": []}


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
        resolved_out=resolved,
    )

    assert doc_ids == ["doc1"]
    assert resolved == {"method": "manual", "logic": "or", "conditions": [{"key": "year", "op": "=", "value": "2026"}]}


@pytest.mark.asyncio
async def test_no_filter_leaves_the_sink_untouched():
    resolved: dict = {}

    assert await apply_meta_data_filter(None, {}, resolved_out=resolved) == []
    assert resolved == {}
