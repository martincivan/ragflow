import pytest
from common.metadata_utils import apply_meta_data_filter
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.mark.asyncio
async def test_apply_meta_data_filter_semi_auto_key():
    meta_data_filter = {"method": "semi_auto", "semi_auto": ["key1", "key2"]}
    metas = {"key1": {"val1": ["doc1"]}, "key2": {"val2": ["doc2"]}}
    question = "find val1"

    chat_mdl = MagicMock()

    with patch("rag.prompts.generator.gen_meta_filter", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {"conditions": [{"key": "key1", "op": "=", "value": "val1"}], "logic": "and"}

        doc_ids = await apply_meta_data_filter(meta_data_filter, metas, question, chat_mdl)
        assert doc_ids == ["doc1"]

        # Check that constraints is an empty dict by default for legacy
        mock_gen.assert_called_once()
        args, kwargs = mock_gen.call_args
        assert kwargs["constraints"] == {}


@pytest.mark.asyncio
async def test_apply_meta_data_filter_semi_auto_key_and_operator():
    meta_data_filter = {"method": "semi_auto", "semi_auto": [{"key": "key1", "op": ">"}, "key2"]}
    metas = {"key1": {"10": ["doc1"]}, "key2": {"val2": ["doc2"]}}
    question = "find key1 > 5"

    chat_mdl = MagicMock()

    with patch("rag.prompts.generator.gen_meta_filter", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {"conditions": [{"key": "key1", "op": ">", "value": "5"}], "logic": "and"}

        doc_ids = await apply_meta_data_filter(meta_data_filter, metas, question, chat_mdl)
        assert doc_ids == ["doc1"]

        # Check that constraints are correctly passed
        mock_gen.assert_called_once()
        args, kwargs = mock_gen.call_args
        assert kwargs["constraints"] == {"key1": ">"}


# ---------------------------------------------------------------------------
# Operator constraints are a contract, not a hint: the prompt says the model
# MUST use the pinned operator, so the answer has to be held to it.
# ---------------------------------------------------------------------------


def test_enforce_operator_constraints_rewrites_a_deviating_operator():
    from common.metadata_utils import enforce_operator_constraints

    conditions = [{"key": "phase", "op": "contains", "value": "AD"}]
    assert enforce_operator_constraints(conditions, {"phase": "="}) == [{"key": "phase", "op": "=", "value": "AD"}]


def test_enforce_operator_constraints_leaves_unconstrained_keys_alone():
    from common.metadata_utils import enforce_operator_constraints

    conditions = [{"key": "project", "op": "contains", "value": "x"}]
    assert enforce_operator_constraints(conditions, {"phase": "="}) == conditions


def test_enforce_operator_constraints_keeps_multi_valued_membership():
    """`in ["AD", "UR"]` cannot become `=` without changing what the filter means.

    Two `=` conditions on one key under the shared `and` logic match nothing, so
    the model's operator stands and the mismatch is logged instead.
    """
    from common.metadata_utils import enforce_operator_constraints

    conditions = [{"key": "phase", "op": "in", "value": "AD, UR"}]
    assert enforce_operator_constraints(conditions, {"phase": "="}) == conditions
    assert enforce_operator_constraints([{"key": "phase", "op": "in", "value": ["AD", "UR"]}], {"phase": "="}) == [
        {"key": "phase", "op": "in", "value": ["AD", "UR"]}
    ]


def test_enforce_operator_constraints_rewrites_single_valued_membership():
    from common.metadata_utils import enforce_operator_constraints

    conditions = [{"key": "phase", "op": "in", "value": "AD"}]
    assert enforce_operator_constraints(conditions, {"phase": "="}) == [{"key": "phase", "op": "=", "value": "AD"}]


@pytest.mark.asyncio
async def test_semi_auto_applies_the_pinned_operator():
    meta_data_filter = {"method": "semi_auto", "semi_auto": [{"key": "phase", "op": "="}]}
    metas = {"phase": {"AD": ["doc1"]}}
    chat_mdl = MagicMock()

    with patch("rag.prompts.generator.gen_meta_filter", new_callable=AsyncMock) as mock_gen:
        # The model ignores the constraint and answers with `contains`.
        mock_gen.return_value = {"conditions": [{"key": "phase", "op": "contains", "value": "AD"}], "logic": "and"}
        with patch("common.metadata_utils.filter_doc_ids_by_metadata") as mock_filter:
            mock_filter.return_value = ["doc1"]
            await apply_meta_data_filter(meta_data_filter, metas, "phase AD", chat_mdl)
            assert mock_filter.call_args[0][1] == [{"key": "phase", "op": "=", "value": "AD"}]
