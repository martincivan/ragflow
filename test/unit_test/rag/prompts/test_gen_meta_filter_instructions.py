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
"""``gen_meta_filter``: the chat/agent owner's filtering guidance in the prompt."""

import pytest

from rag.prompts import generator


class _ChatModel:
    max_length = 100_000

    def __init__(self):
        self.system_prompts = []

    async def async_chat(self, system, history, *args, **kwargs):
        self.system_prompts.append(system)
        return '{"logic": "and", "conditions": []}'


async def _prompt(**kwargs) -> str:
    mdl = _ChatModel()
    await generator.gen_meta_filter(mdl, {"phase": {"SP": ["d1"], "DRP": ["d2"]}}, "q", **kwargs)
    return mdl.system_prompts[0]


@pytest.mark.asyncio
async def test_guidance_is_rendered_when_given():
    prompt = await _prompt(instructions="  Use SP only for building permits.  ")
    assert '- Filtering guidance:\n"""\nUse SP only for building permits.\n"""' in prompt
    assert "Follow the filtering guidance below" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("instructions", [None, "", "   ", 42])
async def test_no_guidance_block_without_text(instructions):
    prompt = await _prompt(instructions=instructions)
    assert "Filtering guidance" not in prompt
    assert "filtering guidance" not in prompt


@pytest.mark.asyncio
async def test_strength_is_mentioned_only_with_soft_conditions():
    assert "operators and strength" not in await _prompt(instructions="x")
    assert "operators and strength" in await _prompt(instructions="x", allow_soft=True)


@pytest.mark.asyncio
async def test_long_guidance_is_capped():
    prompt = await _prompt(instructions="a" * (generator.META_FILTER_INSTRUCTIONS_LIMIT + 50))
    assert "a" * generator.META_FILTER_INSTRUCTIONS_LIMIT + "…" in prompt
    assert "a" * (generator.META_FILTER_INSTRUCTIONS_LIMIT + 1) not in prompt
