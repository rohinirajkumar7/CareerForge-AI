import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import llm  # noqa: E402


class FakeLLM:
    """Records prompts and returns canned JSON, so no test touches the network."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, system, prompt, temperature):
        self.calls.append({"system": system, "prompt": prompt, "temperature": temperature})
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)


@pytest.fixture
def fake_llm():
    def install(*responses):
        fake = FakeLLM(*responses)
        llm.set_transport(fake)
        return fake

    yield install
    llm.set_transport(None)
