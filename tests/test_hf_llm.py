"""Guards for the HuggingFace backend behind the overseer (issue #15).

The overseer's client seam is "anything with `.messages.create(**kwargs)`
returning `.content` and `.usage`"; `mind/llm.py` adapts the HF router to it
so `Overseer._call` cannot tell the vendors apart. Everything here runs
against a fake `fetch` -- a suite that needs a token and a network is a suite
that fails for reasons that are not about the code.
"""

import json

import pytest

from pluggybot.mind import llm, overseer
from pluggybot.mind.llm import FORMAT_NOTE, HFClient, is_hf_model
from pluggybot.mind.overseer import Menu, Overseer


def fake_fetch(script):
  """A fetch that replays `script` (list of (status, payload)) and records
  every request it was asked to make."""
  calls = []

  def fetch(url, body, headers, timeout):
    calls.append({"url": url, "body": body, "headers": headers,
                  "timeout": timeout})
    return script.pop(0)

  fetch.calls = calls
  return fetch


def ok_payload(text, prompt_tokens=1200, completion_tokens=40):
  return (200, {
    "choices": [{"message": {"content": text}}],
    "usage": {"prompt_tokens": prompt_tokens,
              "completion_tokens": completion_tokens},
  })


SYSTEM = [{"type": "text", "text": "STABLE PREFIX",
           "cache_control": {"type": "ephemeral"}}]
SCHEMA = {"type": "object", "properties": {"action": {"type": "string"}},
          "required": ["action"], "additionalProperties": False}


def create(fetch, **over):
  client = HFClient(token="hf_test", fetch=fetch)
  kw = dict(model="org/model-8b", max_tokens=64, system=SYSTEM,
            output_config={"format": {"type": "json_schema",
                                      "schema": SCHEMA}},
            messages=[{"role": "user", "content": "state"}])
  kw.update(over)
  return client.messages.create(**kw)


def test_model_ids_route_by_slash():
  assert is_hf_model("Qwen/Qwen3-8B")
  assert not is_hf_model("claude-haiku-4-5")
  assert not is_hf_model("")


def test_the_request_wears_the_router_shape():
  """System blocks flatten to ONE system message (cache_control dropped --
  there is nothing on the router for it to mark) and the schema rides as
  OpenAI-style response_format."""
  fetch = fake_fetch([ok_payload('{"action": "idle"}')])
  create(fetch)
  body = fetch.calls[0]["body"]
  assert body["messages"][0] == {"role": "system", "content": "STABLE PREFIX"}
  assert body["messages"][1] == {"role": "user", "content": "state"}
  assert "cache_control" not in json.dumps(body)
  assert body["response_format"]["json_schema"]["schema"] == SCHEMA
  assert fetch.calls[0]["headers"]["Authorization"] == "Bearer hf_test"


def test_the_response_wears_the_anthropic_shape():
  """What comes back must satisfy `_extract_json` and `_meter` untouched."""
  resp = create(fake_fetch([ok_payload('{"action": "idle"}',
                                       prompt_tokens=999,
                                       completion_tokens=7)]))
  assert resp.content[0].type == "text"
  assert json.loads(resp.content[0].text) == {"action": "idle"}
  assert resp.usage.input_tokens == 999
  assert resp.usage.output_tokens == 7
  # No prompt cache on the router: zero is the honest reading, and it must
  # be PRESENT so the overseer's metering does not need a vendor branch.
  assert resp.usage.cache_read_input_tokens == 0
  assert resp.usage.cache_creation_input_tokens == 0


def test_a_provider_that_rejects_the_schema_gets_one_retry_in_prose():
  """Structured outputs are provider-dependent. A 4xx naming the field is
  retried ONCE with the schema spelled into the system text; validate() is
  the enforcement either way."""
  fetch = fake_fetch([
    (400, {"error": {"message": "response_format is not supported"}}),
    ok_payload('{"action": "idle"}'),
  ])
  resp = create(fetch)
  assert json.loads(resp.content[0].text) == {"action": "idle"}
  retry = fetch.calls[1]["body"]
  assert "response_format" not in retry
  assert FORMAT_NOTE.strip().splitlines()[0] in retry["messages"][0]["content"]
  assert '"properties"' in retry["messages"][0]["content"]


def test_any_other_error_raises_and_does_not_retry():
  """A 401 or a 5xx is the overseer's fallback path, not a format problem --
  a second identical request would just double the bill for the same no."""
  fetch = fake_fetch([(401, {"error": {"message": "Invalid credentials"}})])
  with pytest.raises(RuntimeError, match="HF router 401"):
    create(fetch)
  assert len(fetch.calls) == 1


def test_a_reasoning_models_thinking_is_not_a_malformed_answer():
  """`<think>` before the JSON is the model reasoning, not the model failing
  to answer -- only the second should ever become fallback:garbled."""
  resp = create(fake_fetch([ok_payload(
    '<think>hmm, the battery is fine\nso...</think>\n{"action": "idle"}')]))
  assert json.loads(resp.content[0].text) == {"action": "idle"}


def test_a_missing_token_fails_at_construction(monkeypatch):
  """The opposite of `anthropic.Anthropic()`, whose late failure is why the
  cool-off exists: with no token there is nothing to back off FROM, so the
  overseer resolves it to `fallback:no-client` at the first decision."""
  monkeypatch.delenv(llm.TOKEN_ENV, raising=False)
  with pytest.raises(ValueError, match="HF_TOKEN"):
    HFClient()
  boss = Overseer(Menu(zones=("garden",)), model="org/model-8b")
  decision = boss.decide({"decisions": 0})
  assert decision.source == "fallback:no-client"


def test_pricing_takes_the_cheapest_live_provider():
  catalogue = (200, {"data": [
    {"id": "other/model", "providers": [
      {"status": "live", "pricing": {"input": 9.0, "output": 9.0}}]},
    {"id": "org/model-8b", "providers": [
      {"status": "live", "pricing": {"input": 0.04, "output": 0.2}},
      {"status": "offline", "pricing": {"input": 0.001, "output": 0.001}},
      {"status": "live", "pricing": {"input": 0.02, "output": 0.1}},
      {"status": "live"},                  # no published rate: not a candidate
    ]},
  ]})
  client = HFClient(token="hf_test", fetch=fake_fetch([catalogue]))
  assert client.pricing("org/model-8b") == (0.02, 0.1)


def test_an_unanswered_catalogue_prices_as_unknown_not_zero():
  client = HFClient(token="hf_test", fetch=fake_fetch([(500, {})]))
  assert client.pricing("org/model-8b") is None


def test_a_slash_model_reaches_the_llm_through_the_same_seam(monkeypatch):
  """End to end through Overseer: an HF answer becomes an ordinary `llm`
  decision, metered with the router's own rates."""
  fetch = fake_fetch([
    (200, {"data": [{"id": "org/model-8b", "providers": [
      {"status": "live", "pricing": {"input": 0.05, "output": 0.25}}]}]}),
    ok_payload(json.dumps({"action": "explore", "zone": "garden",
                           "reason": "mapping", "board": "", "program": "",
                           "note": "", "respond_to": "", "outcome": "",
                           "reply": "", "task": "", "answer": ""}),
               prompt_tokens=2000, completion_tokens=50),
  ])
  monkeypatch.setattr(llm, "_default_fetch", fetch)
  monkeypatch.setenv(llm.TOKEN_ENV, "hf_test")
  boss = Overseer(Menu(zones=("garden",)), model="org/model-8b")
  decision = boss.decide({"decisions": 0})
  assert decision.source == "llm"
  assert decision.action == "explore" and decision.zone == "garden"
  assert boss.usage.priced
  assert boss.usage.usd_per_mtok_in == 0.05
  assert boss.usage.usd == pytest.approx(
    (2000 * 0.05 + 50 * 0.25) / 1e6)


def test_build_reads_the_model_from_the_environment(monkeypatch):
  monkeypatch.setenv(overseer.MODEL_ENV, "Qwen/Qwen3-8B")
  boss, _ = overseer.build("room_hub", enabled=True, client=object())
  assert boss.model == "Qwen/Qwen3-8B"
  monkeypatch.delenv(overseer.MODEL_ENV)
  boss, _ = overseer.build("room_hub", enabled=True, client=object())
  assert boss.model == overseer.MODEL
