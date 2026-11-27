"""Ask the configured model for replacements. Tests inject their own function instead."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from improve.settings import GeneratorUnavailable, ImproveSettings

Generator = Callable[[str, str], list[dict]]


def model_generator(settings: ImproveSettings) -> Generator:
    """Build a generator that refuses to call the network when the key is missing."""

    def generate(prompt: str, relation: str) -> list[dict]:
        key = os.environ.get(settings.api_key_env, "").strip()
        if not key:
            raise GeneratorUnavailable(f"环境变量 {settings.api_key_env} 未设置，未调用生成模型")
        return _request_items(settings, key, prompt, relation)

    return generate


def _request_items(settings: ImproveSettings, key: str, prompt: str, relation: str) -> list[dict]:
    import urllib.request

    instruction = (
        "Return a JSON object with an items array. Each item has original, replacement, "
        "kind (degree or sentence), and relation (synonym or antonym). "
        f"relation must be {relation}. Change only one degree word or one complete sentence. "
        "Every other character must stay identical to the prompt.\n"
        f"Prompt:\n{prompt}"
    )
    body = {
        "model": settings.model_name,
        "messages": [{"role": "user", "content": instruction}],
        "temperature": 0,
    }
    request = urllib.request.Request(
        settings.api_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    items = parsed.get("items") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        raise GeneratorUnavailable("生成模型的返回不是结构化列表")
    return [item for item in items if isinstance(item, dict)]
