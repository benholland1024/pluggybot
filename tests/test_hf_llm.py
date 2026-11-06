"""Guards for the HuggingFace backend behind the overseer (issue #15).

The overseer's client seam is "anything with `.messages.create(**kwargs)`
returning `.content` and `.usage`"; `mind/llm.py` adapts the HF router to it
so `Overseer._call` cannot tell the vendors apart. Everything here runs
against a fake `fetch` -- a suite that needs a token and a network is a suite
that fails for reasons that are not about the code.
"""

import json
import time
from types import SimpleNamespace

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


def test_a_bare_string_error_body_is_still_the_providers_no():
  """Some providers' 4xx bodies are `{"error": "<text>"}`, not a dict:
  MEASURED (issue #264, ladder B) as `AttributeError: 'str' object has no
  attribute 'get'` out of the error path itself, filed as `fallback:
  offline` -- a crash in the handler, not the provider's answer."""
  assert not llm._blames_format({"error": "rate limit exceeded"})
  assert llm._blames_format({"error": "response_format is not supported"})
  fetch = fake_fetch([(429, {"error": "rate limit exceeded"})])
  with pytest.raises(RuntimeError, match="HF router 429"):
    create(fetch)
  assert len(fetch.calls) == 1


def test_a_reasoning_models_thinking_is_not_a_malformed_answer():
  """`<think>` before the JSON is the model reasoning, not the model failing
  to answer -- only the second should ever become fallback:garbled."""
  resp = create(fake_fetch([ok_payload(
    '<think>hmm, the battery is fine\nso...</think>\n{"action": "idle"}')]))
  assert json.loads(resp.content[0].text) == {"action": "idle"}


def test_a_streamed_answer_says_who_answered(monkeypatch):
  """`:cheapest` ties four providers at one price, from 12 to 97 tokens/s,
  and one answered a trivial ask with no reasoning at all (#481's smoke
  call): the router names who answered in a header, and the answer keeps
  it."""
  class Answered:
    headers = {"x-inference-provider": "novita"}

    def __enter__(self):
      return self

    def __exit__(self, *a):
      return False

    def __iter__(self):
      yield b'data: {"choices": [{"delta": {"reasoning_content": "hm"}}]}\n'
      yield (b'data: {"choices": [{"delta": {"content": "{\\"action\\": \\"idle\\"}"}}], '
             b'"usage": {"prompt_tokens": 9, "completion_tokens": 3}}\n')
      yield b"data: [DONE]\n"
  monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: Answered())
  status, payload = llm.stream_fetch(llm.ROUTER + "/chat/completions", {"model": "m"}, {}, 5.0)
  assert status == 200 and payload["provider"] == "novita"
  resp = create(fake_fetch([(status, payload)]))
  assert resp.provider == "novita" and json.loads(resp.content[0].text) == {"action": "idle"}
  assert resp.usage.output_tokens == 3
  # a router that says nothing names nobody
  assert create(fake_fetch([ok_payload('{"action": "idle"}')])).provider is None


def test_a_streamed_answer_its_tokens_ran_out_on_says_so(monkeypatch):
  """37 of #493's answers ran out of their tokens still thinking, and read
  as an answer that was not JSON: the stream's finish is kept, and the
  answer says it ran out in the SDK's word."""
  def stream(finish):
    class Answered:
      headers = {}

      def __enter__(self):
        return self

      def __exit__(self, *a):
        return False

      def __iter__(self):
        yield b'data: {"choices": [{"delta": {"reasoning_content": "hm, and hm"}}]}\n'
        yield ('data: {"choices": [{"delta": {}, "finish_reason": "%s"}]}\n' % finish).encode()
        yield b"data: [DONE]\n"
    return Answered()
  for finish, stop in (("length", "max_tokens"), ("stop", "end_turn")):
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: stream(finish))
    status, payload = llm.stream_fetch(llm.ROUTER + "/chat/completions", {"model": "m"}, {}, 5.0)
    assert payload["choices"][0]["finish_reason"] == finish
    assert create(fake_fetch([(status, payload)])).stop_reason == stop


def test_a_stream_the_far_side_closed_mid_answer_says_so(monkeypatch):
  """#560's table smoke: the router closes a stream at 20 minutes, and three
  answers still thinking at 1,200.3 s read as ended -- empty, "not JSON" --
  and were asked again as if nothing had happened. A stream that ends with
  no `[DONE]` and no `finish_reason` was closed mid-answer; one that says
  `[DONE]` with no reason ended."""
  def stream(lines):
    class Answered:
      headers = {}

      def __enter__(self):
        return self

      def __exit__(self, *a):
        return False

      def __iter__(self):
        yield from lines
    return Answered()
  thinking = b'data: {"choices": [{"delta": {"reasoning_content": "hm, and hm"}}]}\n'
  for lines, finish, stop in (([thinking], "closed", "closed"),
                              ([thinking, b"data: [DONE]\n"], None, "end_turn")):
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: stream(lines))
    status, payload = llm.stream_fetch(llm.ROUTER + "/chat/completions", {"model": "m"}, {}, 5.0)
    assert payload["choices"][0]["finish_reason"] == finish
    assert create(fake_fetch([(status, payload)])).stop_reason == stop


THINKING = b'data: {"choices": [{"delta": {"reasoning_content": "hm "}}]}\n'


def thinking_stream(monkeypatch, lines=1000, stall_at=None, clock=None):
  """A router answer thinking `lines` lines, under a clock ticking a second
  a read; `stall_at` makes a read past it time out, the clock jumped to
  `clock`. Returns what `urlopen` was asked, the socket's timeouts, and
  the order the timeouts and the lines came in (`order`)."""
  now = [0.0]

  def tick():
    now[0] += 1.0
    return now[0]
  monkeypatch.setattr(llm, "monotonic", tick)
  seen = {"socket": [], "order": []}

  class Socket:
    def settimeout(self, s):
      seen["socket"].append(s)
      seen["order"].append("timeout")

  class Thinking:
    headers = {"x-inference-provider": "baseten"}
    fp = SimpleNamespace(raw=SimpleNamespace(_sock=Socket()))

    def __enter__(self):
      return self

    def __exit__(self, *a):
      return False

    def __iter__(self):
      for i in range(lines):
        if stall_at is not None and i == stall_at:
          now[0] = clock
          raise TimeoutError("timed out")
        seen["order"].append("line")
        yield THINKING

  def urlopen(req, timeout):
    seen.update(timeout=timeout, body=json.loads(req.data))
    return Thinking()
  monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
  return seen


def test_a_robots_stream_is_held_to_its_deadline_in_all(monkeypatch):
  """#604's follow-up: the robots' calls stream, so nothing but our own
  deadline ends a long think -- not streamed, none outlived the router's
  gateway (a 504 at 120 s of silence). A stream still going at the deadline
  ends "deadline", what it reasoned so far and who was answering kept; no
  read waits past the wall (a worker still reading makes the next decision
  `busy`: the review's catch), and one the wall cut short is the deadline;
  a stream silent its whole `STREAM_SILENCE_S` before it failed after its
  first delta, and brought nothing back before one (a timeout). Shown to
  fail without the wall: the stream was read to its end."""
  seen = thinking_stream(monkeypatch)
  status, payload = llm.deadline_fetch(llm.ROUTER + "/chat/completions", {"model": "m"},
                                       {}, 10.0)
  choice = payload["choices"][0]
  assert status == 200 and choice["finish_reason"] == "deadline"
  assert payload["provider"] == "baseten"
  assert 0 < choice["message"]["content"].count("hm") <= 10
  assert seen["body"]["stream"] is True and seen["timeout"] == 10.0
  assert seen["socket"] == sorted(seen["socket"], reverse=True) and seen["socket"][-1] <= 1.0
  assert seen["order"][0] == "timeout", "the first line waits no longer than the wall either"
  assert create(fake_fetch([(status, payload)])).stop_reason == "deadline"
  seen = thinking_stream(monkeypatch, lines=3)
  llm.deadline_fetch(llm.ROUTER + "/chat/completions", {"model": "m"}, {}, 300.0)
  assert seen["timeout"] == llm.STREAM_SILENCE_S < 300.0
  thinking_stream(monkeypatch, stall_at=3, clock=400.0)
  status, payload = llm.deadline_fetch(llm.ROUTER + "/chat/completions", {"model": "m"},
                                       {}, 300.0)
  assert payload["choices"][0]["finish_reason"] == "deadline"
  # silent its whole read before the wall: after deltas a stream that
  # failed (the next test), before any a timeout that brought nothing back
  thinking_stream(monkeypatch, stall_at=3, clock=120.0)
  status, payload = llm.deadline_fetch(llm.ROUTER + "/chat/completions", {"model": "m"},
                                       {}, 300.0)
  assert payload["choices"][0]["finish_reason"] == "failed"
  thinking_stream(monkeypatch, stall_at=0, clock=120.0)
  with pytest.raises(TimeoutError):
    llm.deadline_fetch(llm.ROUTER + "/chat/completions", {"model": "m"}, {}, 300.0)


def test_a_stream_whose_count_never_came_is_booked_at_an_estimate(monkeypatch):
  """#604's follow-up review: a stream stopped at its deadline (or by a
  provider that sends no count) bills what it streamed, and its usage comes
  at its end: booked at none it would read free to the purse. So it is
  booked at an ESTIMATE, said as one -- its request's text and what it
  streamed at `CHARS_PER_TOKEN` (2.9-3.8 a token measured, the low end), a
  picture at `IMAGE_TOKENS`, never its base64."""
  thinking_stream(monkeypatch)
  body = {"model": "m", "messages": [
    {"role": "system", "content": "x" * 3000},
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:" + "A" * 9000}},
                                 {"type": "text", "text": "y" * 600}]}]}
  _, payload = llm.deadline_fetch(llm.ROUTER + "/chat/completions", body, {}, 10.0)
  streamed = len(payload["choices"][0]["message"]["content"]) - len("<think></think>")
  assert payload["usageEstimated"] is True
  assert payload["usage"] == {"prompt_tokens": round(3600 / llm.CHARS_PER_TOKEN) + llm.IMAGE_TOKENS,
                              "completion_tokens": round(streamed / llm.CHARS_PER_TOKEN)}
  resp = create(fake_fetch([(200, payload)]))
  assert resp.usage_estimated and resp.usage.output_tokens > 0
  # a provider's own count stands, and says nothing of an estimate
  assert not create(fake_fetch([ok_payload('{"action": "idle"}')])).usage_estimated


def test_the_robots_calls_stream(monkeypatch):
  """#604's follow-up, the one line of wiring: the router's client the
  overseer builds streams, held to the call's own deadline. Shown to fail
  before it: `_default_fetch`, a call not streamed."""
  monkeypatch.setenv(llm.TOKEN_ENV, "hf_test")
  monkeypatch.setattr(HFClient, "pricing", lambda self, model: (0.15, 0.5))
  client = llm.build_client("huggingface", "org/m", timeout=300.0)
  assert client.fetch is llm.deadline_fetch and client.timeout == 300.0
  boss = Overseer(Menu(zones=("garden",)), model="org/m")
  assert boss.client.fetch is llm.deadline_fetch
  assert boss.client.timeout == overseer.CALL_TIMEOUT_S


def test_how_hard_it_thinks_is_ours_to_say(monkeypatch):
  """#604's follow-up: unset, `reasoning_effort` is the PROVIDER's default,
  and on the live turns GLM-5.3-Flash thought 1.5k tokens a turn on
  deepinfra and 5.2k on baseten, DeepSeek-V4.1-Flash none on deepinfra and
  15-51k on baseten. `$PLUGGY_REASONING_EFFORT` names it: on every request
  where it is set and on none where not, and said in `stats()` and the
  header. Shown to fail before it: no request carried one."""
  from pluggybot.evaluation.identity import build_identity
  fetch = fake_fetch([ok_payload('{"action": "idle"}'), ok_payload('{"action": "idle"}')])
  create(fetch)
  assert "reasoning_effort" not in fetch.calls[0]["body"]
  HFClient(token="hf_test", fetch=fetch, reasoning_effort="low").messages.create(
    model="org/model-8b", max_tokens=64, system=SYSTEM,
    messages=[{"role": "user", "content": "state"}])
  assert fetch.calls[1]["body"]["reasoning_effort"] == "low"
  monkeypatch.setenv(llm.TOKEN_ENV, "hf_test")
  monkeypatch.setenv(overseer.EFFORT_ENV, "low")
  monkeypatch.setattr(HFClient, "pricing", lambda self, model: (0.15, 0.5))
  boss = overseer.build("home_humanoid", enabled=True, model="org/m")
  assert boss.reasoning_effort == boss.client.reasoning_effort == "low"
  assert boss.stats()["reasoningEffort"] == "low"
  header = dict(arm="autonomous", commit="x", hashes={})
  assert build_identity("home_humanoid", effort="low", **header)["reasoningEffort"] == "low"
  assert "reasoningEffort" not in build_identity("home_humanoid", **header)
  monkeypatch.delenv(overseer.EFFORT_ENV)
  assert overseer.build("home_humanoid", enabled=True, model="org/m").reasoning_effort is None


def test_an_answer_still_being_written_at_its_deadline_is_a_timeout(monkeypatch):
  """#604's follow-up: streamed, a long think can meet our own deadline
  mid-answer. It is `fallback:timeout`, our patience and a failure, never
  `garbled` (its text is half a think); it names who was answering, its
  row says no count came (`None`, never 0), it counts toward the cooloff
  as a timeout always did, and it is BOOKED at its estimate, as every
  billed call is (the review's catch: unbooked, the purse read it free).
  Its JSON whole before the stop, it is an answer; a stream the far side
  closed is `offline`. Shown to fail before the fix: `fallback:garbled`."""
  from pluggybot.mind.spend import SpendBook
  thinking_stream(monkeypatch)
  _, still = llm.deadline_fetch(llm.ROUTER + "/chat/completions",
                                {"model": "m", "messages": [{"role": "user", "content": "s"}]},
                                {}, 10.0)

  def payload(text, finish):
    choice = {"message": {"content": text}, "finish_reason": finish}
    return (200, {**still, "choices": [choice]})

  def decide(script):
    book = SpendBook(None)
    boss = Overseer(Menu(zones=("garden",)), spend=book, client=HFClient(
      token="hf_test", fetch=fake_fetch([script])))
    boss.usage.usd_per_mtok_in, boss.usage.usd_per_mtok_out = 0.15, 0.5
    return boss, book, boss.decide({"decisions": 0})
  boss, book, d = decide((200, still))
  assert d.source == "fallback:timeout" and overseer.fallback_class(d.source) == "failure"
  assert d.call["provider"] == "baseten" and not d.call["cut"]
  assert d.call["tokensIn"] is None and d.call["tokensOut"] is None
  assert boss.usage.output_tokens == still["usage"]["completion_tokens"] > 0
  assert len(book.entries) == 1 and book.spent > 0
  assert boss._errors_in_a_row == 1
  assert "? in, ? out" in overseer.call_said(d.call)
  # ...and the robot is told its think outlived its patience, as a cut-off
  # is: streamed and pinned, a timeout is mostly its own, not nobody's
  assert d.reason.endswith(f"your answer was still being written when its "
                           f"{overseer.CALL_TIMEOUT_S:.0f} s ran out")
  boss = Overseer(Menu(zones=("garden",)), client=HFClient(
    token="hf_test", fetch=fake_fetch([(200, still)])))
  boss.start_interrupt({}, "the drawer", "battery_below")
  while boss.interrupt_pending:
    time.sleep(0.005)
  late = boss.interrupt_result()
  assert (late["source"], late["continue"], late["call"]["provider"]) == (
    "fallback:timeout", False, "baseten")
  assert late["why"].startswith("the answer was still being written")
  boss, book, d = decide(payload("<think>hm</think>the stream went aw", "closed"))
  assert d.source == "fallback:offline" and len(book.entries) == 1
  assert "still being written" not in d.reason
  answered = json.dumps({"action": "explore", "zone": "garden", "reason": "mapping"})
  boss, book, d = decide(payload(answered, "deadline"))
  assert d.source == "llm" and d.action == "explore"


def scripted_stream(monkeypatch, items):
  """A router answer of `items`: each a line yielded, or an exception
  raised where the stream breaks."""
  class Answered:
    headers = {"x-inference-provider": "baseten"}

    def __enter__(self):
      return self

    def __exit__(self, *a):
      return False

    def __iter__(self):
      for item in items:
        if isinstance(item, BaseException):
          raise item
        yield item
  monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: Answered())


def test_a_stream_that_fails_mid_answer_is_booked_and_names_who_sent_it(monkeypatch):
  """#604's follow-up review: a stream that fails after its first delta --
  an error chunk ("Stream ended before generation completed", 1 in 12 of
  the probe's on baseten), a read silent for its whole `timeout`, a reset,
  a cut chunk -- was billed for what it streamed. With a wall, it ends
  "failed": what it streamed, who sent it and how it failed kept, booked at
  its estimate, `offline` to the overseer with its words in the error. One
  that fails before its first delta, and every stream without a wall
  (practice's), raise or answer 500 as they always did."""
  import http.client
  from pluggybot.mind.spend import SpendBook
  error = b'data: {"error": {"message": "Stream ended before generation completed"}}\n'
  url, body = llm.ROUTER + "/chat/completions", {"model": "m", "messages": []}
  for breaks, said in ((error, "Stream ended before generation completed"),
                       (TimeoutError("timed out"), "silent for"),
                       (http.client.IncompleteRead(b""), "IncompleteRead"),
                       (ConnectionResetError("reset"), "ConnectionResetError")):
    scripted_stream(monkeypatch, [THINKING, THINKING, breaks])
    status, payload = llm.deadline_fetch(url, body, {}, 300.0)
    choice = payload["choices"][0]
    assert (status, choice["finish_reason"], payload["provider"]) == (200, "failed", "baseten")
    assert said in payload["streamError"] and payload["usageEstimated"]
    assert choice["message"]["content"].count("hm") == 2
  # before its first delta, and without a wall, as it always was
  scripted_stream(monkeypatch, [error])
  assert llm.deadline_fetch(url, body, {}, 300.0)[0] == 500
  scripted_stream(monkeypatch, [TimeoutError("timed out")])
  with pytest.raises(TimeoutError):
    llm.deadline_fetch(url, body, {}, 300.0)
  scripted_stream(monkeypatch, [THINKING, error])
  assert llm.stream_fetch(url, body, {}, 300.0)[0] == 500
  scripted_stream(monkeypatch, [THINKING])
  unwalled = llm.stream_fetch(url, body, {}, 300.0)[1]
  assert unwalled["usage"] == {} and "usageEstimated" not in unwalled
  # through the overseer: offline, booked, and its words kept
  scripted_stream(monkeypatch, [THINKING, error])
  _, failed = llm.deadline_fetch(url, {"model": "m", "messages": [
    {"role": "user", "content": "s" * 30000}]}, {}, 300.0)
  book = SpendBook(None)
  boss = Overseer(Menu(zones=("garden",)), spend=book, client=HFClient(
    token="hf_test", fetch=fake_fetch([(200, failed)])))
  boss.usage.usd_per_mtok_in, boss.usage.usd_per_mtok_out = 0.15, 0.5
  d = boss.decide({"decisions": 0})
  assert d.source == "fallback:offline" and d.call["provider"] == "baseten"
  assert len(book.entries) == 1 and book.spent > 0
  assert any("Stream ended before generation completed" in e for e in boss.usage.errors)


def test_the_format_retry_waits_only_the_calls_time_left(monkeypatch):
  """#604's follow-up review: an endpoint that refuses the schema is asked
  again in prose -- in the time the call has LEFT, never a second whole
  `timeout`, which would outlive the overseer's deadline and make the next
  decision `busy`."""
  clock = iter([0.0, 5.0])
  monkeypatch.setattr(llm, "monotonic", lambda: next(clock))
  fetch = fake_fetch([(400, {"error": {"message": "response_format is not supported"}}),
                      ok_payload('{"action": "idle"}')])
  create(fetch)
  assert [c["timeout"] for c in fetch.calls] == [8.0, 3.0]


def test_the_anthropic_path_states_no_effort_it_never_sends():
  """#604's follow-up review: the Anthropic path sends no
  `reasoning_effort` (`effort` is a 400 on Haiku 4.5), so the mind names
  none -- the header, `stats()` and History say what RAN -- and the
  operator is told it was set and not sent."""
  boss = Overseer(Menu(zones=("garden",)), reasoning_effort="low")
  assert boss.backend == "anthropic" and boss.reasoning_effort is None
  assert any(e.startswith("effort:") for e in boss.usage.errors)
  assert Overseer(Menu(zones=("garden",)), model="org/m",
                  reasoning_effort="low").reasoning_effort == "low"


def test_the_probe_reports_a_call_whose_count_never_came(capsys):
  """#604's follow-up review: a call whose stream sent no count has null
  tokens (`_call_record`), and the probe's report of who answered says so
  with a `?` -- formatted as a number, it crashed after the paid calls,
  before the energy gate and the cost were printed."""
  import importlib.util
  from pathlib import Path
  from types import SimpleNamespace as Bag
  path = Path(__file__).resolve().parent.parent / "scripts" / "overseer_probe.py"
  spec = importlib.util.spec_from_file_location("overseer_probe", path)
  probe = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(probe)
  calls = [{"provider": "baseten", "tokensIn": None, "tokensOut": None, "cut": False, "wallS": 300.2},
           {"provider": "deepinfra", "tokensIn": 9, "tokensOut": 1500, "cut": False, "wallS": 12.0}]
  probe.report_calls(Bag(decisions=[Bag(call=c) for c in calls], max_tokens=32768))
  out = capsys.readouterr().out
  assert "?,300" in out and "1500" in out


def test_an_answer_not_streamed_says_who_answered(monkeypatch):
  """Issue #604: the deployed mind's answers are NOT streamed, and only the
  streamed path read the router's header, so a quarter of its answers
  coming back empty could not say who sent them. The header rides the
  answer either way; an endpoint that names nobody names nobody."""
  class Answered:
    status = 200

    def __init__(self, headers):
      self.headers = headers

    def __enter__(self):
      return self

    def __exit__(self, *a):
      return False

    def read(self):
      return json.dumps({"choices": [{"message": {"content": '{"action": "idle"}'}}],
                         "usage": {"prompt_tokens": 9, "completion_tokens": 3}}).encode()
  for headers, provider in (({"x-inference-provider": "baseten"}, "baseten"), ({}, None)):
    monkeypatch.setattr(llm.urllib.request, "urlopen",
                        lambda req, timeout, h=headers: Answered(h))
    status, payload = llm._default_fetch(llm.ROUTER + "/chat/completions", {"model": "m"},
                                         {}, 5.0)
    assert status == 200 and create(fake_fetch([(status, payload)])).provider == provider


def cut_payload(text="", finish="length"):
  """What the router answered the deployed mind a quarter of the time from
  04:00 UTC on 11-03 (issue #604): its reasoning took the budget, and the
  answer came back empty, billed in full."""
  return (200, {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 14000, "completion_tokens": 8192},
                "provider": "baseten"})


def test_an_answer_its_tokens_ran_out_on_is_a_cutoff_not_garbled():
  """Issue #604: a quarter of the deployed minds' answers came back EMPTY
  and read `fallback:garbled` ("decision was not JSON: ''"), while the
  adapter knew the tokens had run out. A cut-off is a FAILURE of its own,
  said to the robot as what it was -- both robots believed their JSON was
  being refused -- and the decision names who answered and what it billed.
  Shown to fail before the fix: `fallback:garbled`, and no `call`. The
  robot is told the tokens the provider says it spent, which one capping
  below the ask makes fewer than the budget (here 8,192 of 16,384)."""
  boss = Overseer(Menu(zones=("garden",)), client=HFClient(
    token="hf_test", fetch=fake_fetch([cut_payload()])))
  d = boss.decide({"decisions": 0})
  assert d.source == "fallback:cutoff" and d.scripted
  assert overseer.fallback_class(d.source) == "failure"
  assert overseer.MAX_TOKENS_AUTONOMOUS != 8192
  assert ("your answer was cut off: it ran out of its 8,192 tokens "
          "before it was written") in d.reason
  call = d.as_dict()["call"]
  assert (call["provider"], call["tokensIn"], call["tokensOut"], call["cut"]) == (
    "baseten", 14000, 8192, True)
  # ...and only a cut-off: an empty answer that says it FINISHED is
  # garbled, and one that ran out with its JSON whole is an answer
  answered = json.dumps({"action": "explore", "zone": "garden", "reason": "mapping"})
  for payload, source, cut in ((cut_payload(finish="stop"), "fallback:garbled", False),
                               (cut_payload(text=answered), "llm", True)):
    d = Overseer(Menu(zones=("garden",)), client=HFClient(
      token="hf_test", fetch=fake_fetch([payload]))).decide({"decisions": 0})
    assert d.source == source and "cut off" not in d.reason
    assert d.call["provider"] == "baseten" and d.call["cut"] is cut


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
  decision, metered with the router's own rates. ⚠ The robots' fetch is
  `deadline_fetch` since #604's follow-up, the catalogue's read included:
  a fake of `_default_fetch` alone let the decision out to the network."""
  fetch = fake_fetch([
    (200, {"data": [{"id": "org/model-8b", "providers": [
      {"status": "live", "pricing": {"input": 0.05, "output": 0.25}}]}]}),
    ok_payload(json.dumps({"action": "explore", "zone": "garden",
                           "reason": "mapping", "note": "", "respond_to": "",
                           "outcome": "", "reply": "", "task": ""}),
               prompt_tokens=2000, completion_tokens=50),
  ])
  monkeypatch.setattr(llm, "deadline_fetch", fetch)
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
  boss = overseer.build("home_humanoid", enabled=True, client=object())
  assert boss.model == "Qwen/Qwen3-8B"
  monkeypatch.delenv(overseer.MODEL_ENV)
  boss = overseer.build("home_humanoid", enabled=True, client=object())
  assert boss.model == overseer.MODEL


def test_every_request_names_itself_and_not_urllib():
  """A Cloudflare WAF in front of at least one provider answers urllib's
  default `Python-urllib/3.x` with a 403 page (issue #225): the same body
  under any other name goes through. So every request carries our own
  User-Agent, and one that dropped it would be a served robot one re-route
  away from `fallback:offline` on every decision."""
  fetch = fake_fetch([ok_payload('{"action": "idle"}')])
  create(fetch)
  ua = fetch.calls[0]["headers"].get("User-Agent", "")
  assert ua.startswith("pluggybot/"), ua
  assert "urllib" not in ua.lower()


CATALOGUE = (200, {"data": [
  {"id": "org/model-8b", "providers": [
    {"provider": "fast-co", "status": "live",
     "pricing": {"input": 0.35, "output": 0.75}},
    {"provider": "cheap-co", "status": "live",
     "pricing": {"input": 0.04, "output": 0.2}},
    {"provider": "gone-co", "status": "offline",
     "pricing": {"input": 0.001, "output": 0.001}},
  ]},
]})


def test_a_named_provider_is_priced_as_itself():
  """The router routes a bare id by its own preference order (measured: a
  bare gpt-oss-120b went to Cerebras at ten times the cheapest rate), so a
  deployment pins `org/name:<provider>` and the bill is that provider's --
  the cheapest live rate would be a lie about a call that never went there."""
  client = HFClient(token="hf_test", fetch=fake_fetch([CATALOGUE]))
  assert client.pricing("org/model-8b:fast-co") == (0.35, 0.75)
  client = HFClient(token="hf_test", fetch=fake_fetch([CATALOGUE]))
  assert client.pricing("org/model-8b:cheap-co") == (0.04, 0.2)


def test_cheapest_and_bare_price_as_the_cheapest_live_provider():
  client = HFClient(token="hf_test", fetch=fake_fetch([CATALOGUE]))
  assert client.pricing("org/model-8b:cheapest") == (0.04, 0.2)
  client = HFClient(token="hf_test", fetch=fake_fetch([CATALOGUE]))
  assert client.pricing("org/model-8b") == (0.04, 0.2)


def test_a_policy_nobody_can_price_is_unknown_not_a_guess():
  """`:fastest` -- and an offline or unknown provider -- means nothing here
  can say who answers, and "unknown" is the report, never a number."""
  for suffix in ("fastest", "gone-co", "nobody"):
    client = HFClient(token="hf_test", fetch=fake_fetch([CATALOGUE]))
    assert client.pricing(f"org/model-8b:{suffix}") is None, suffix


def test_the_suffix_rides_to_the_router_untouched():
  """The suffix is the ROUTER's syntax: the request carries the id as
  given, and it is only `pricing()` that takes it apart."""
  fetch = fake_fetch([ok_payload('{"action": "idle"}')])
  create(fetch, model="org/model-8b:cheapest")
  assert fetch.calls[0]["body"]["model"] == "org/model-8b:cheapest"
