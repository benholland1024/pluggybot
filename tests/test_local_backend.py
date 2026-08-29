"""Guards for the SELECTABLE backend behind the overseer (issue #19).

Issue #15 gave the overseer one client seam and two vendors; this is the
third and fourth -- a model on this machine (ollama, llama.cpp) and somebody
else's OpenAI-compatible endpoint -- and the claim under test is that the
decision loop cannot tell. Everything that made the Anthropic path safe (the
call budget, the cool-off, the timeout, the tagged `fallback:<why>` rotation)
is asserted here against a LOCAL endpoint, because a local backend that
stalls is the same failure as an API that times out and the guards are the
only reason either is survivable.

Two of these run against a REAL HTTP server on 127.0.0.1 rather than a fake
`fetch`. That is deliberate and it is the only way `_default_fetch`, urllib's
error shapes and the "no network" claim get exercised at all -- every other
test in this file stops one layer above them.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pluggybot.mind import llm, overseer
from pluggybot.mind.llm import ChatClient, build_client, resolve_backend
from pluggybot.mind.overseer import MODEL, Menu, Overseer

MENU = Menu(zones=("garden",), boards=("whiteboard_a",), programs=("circle",))


def answer(**over) -> str:
  """A complete, valid decision -- every schema key present, as a
  constrained decoder would emit it."""
  raw = {"action": "explore", "zone": "garden", "reason": "mapping",
         "board": "", "program": "", "note": "", "respond_to": "",
         "outcome": "", "reply": "", "task": "", "answer": "",
         "learn": "", "forget": ""}
  raw.update(over)
  return json.dumps(raw)


def fake_fetch(script):
  """A fetch replaying `script` (list of (status, payload)), recording every
  request. The last entry repeats, so a test that does not care how many
  calls it takes does not have to count them."""
  calls = []

  def fetch(url, body, headers, timeout):
    calls.append({"url": url, "body": body, "headers": headers,
                  "timeout": timeout})
    return script[0] if len(script) == 1 else script.pop(0)

  fetch.calls = calls
  return fetch


def ok(text, prompt_tokens=1200, completion_tokens=40):
  return (200, {"choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": completion_tokens}})


def local_overseer(fetch, **over):
  """An overseer wired to a local endpoint through the injected fetch."""
  kw = dict(model=llm.LOCAL_MODEL, backend="local")
  kw.update(over)
  boss = Overseer(MENU, **kw)
  boss._client = ChatClient(base_url=llm.LOCAL_URL, fetch=fetch,
                            label="local backend")
  boss._client_ready = True
  boss._meter_rates()
  return boss


# ---- which mind, and how it is chosen ----------------------------------------


def test_auto_still_routes_by_the_model_id():
  """Issue #15's rule, untouched: a deployment that only ever set
  $PLUGGY_MODEL must route exactly where it always did."""
  assert resolve_backend("auto", "Qwen/Qwen3-4B-Instruct-2507") == "huggingface"
  assert resolve_backend(None, MODEL) == "anthropic"
  assert resolve_backend("", "") == "anthropic"


def test_a_named_backend_beats_the_id_shape():
  """`qwen3:4b-instruct` has no slash and is not an Anthropic model -- an id
  cannot say which MACHINE you meant, so the flag has to win."""
  assert resolve_backend("local", "qwen3:4b-instruct") == "local"
  assert resolve_backend("local", "Qwen/Qwen3-4B-Instruct-2507") == "local"
  assert resolve_backend("anthropic", "org/whatever") == "anthropic"


def test_an_unknown_backend_is_refused_by_name():
  with pytest.raises(ValueError, match="unknown overseer backend"):
    resolve_backend("ollama-ish", "")


def test_an_endpoint_with_no_model_is_refused_rather_than_guessed():
  """A local runtime has a default worth having; a stranger's endpoint does
  not, and guessing produces a 404 from somebody else's server instead of a
  sentence an operator can act on."""
  with pytest.raises(ValueError, match="needs a model id"):
    build_client("openai-compatible", "")


def test_build_reads_the_backend_and_its_default_model(monkeypatch):
  monkeypatch.delenv(overseer.MODEL_ENV, raising=False)
  monkeypatch.setenv(overseer.BACKEND_ENV, "local")
  boss, _ = overseer.build("room_hub", enabled=True, client=object())
  assert boss.backend == "local"
  # ⚠ `claude-haiku-4-5` is not a thing ollama can serve: a backend chosen
  # without a model has to get THAT backend's default or the run 404s
  # against the robot's own machine.
  assert boss.model == llm.LOCAL_MODEL
  monkeypatch.setenv(overseer.MODEL_ENV, "qwen3:8b")
  boss, _ = overseer.build("room_hub", enabled=True, client=object())
  assert (boss.backend, boss.model) == ("local", "qwen3:8b")


def test_the_anthropic_default_is_unchanged(monkeypatch):
  """The regression criterion: nothing about a default run moves."""
  monkeypatch.delenv(overseer.BACKEND_ENV, raising=False)
  monkeypatch.delenv(overseer.MODEL_ENV, raising=False)
  boss, _ = overseer.build("room_hub", enabled=True, client=object())
  assert (boss.backend, boss.model) == ("anthropic", MODEL)
  assert boss.usage.usd_per_mtok_in == overseer.USD_PER_MTOK_IN
  assert boss.stats()["backend"] == "anthropic"


# ---- the request a local model actually gets ---------------------------------


def test_a_local_request_carries_no_key_and_the_menu_as_a_schema():
  """No Authorization header at all -- a local runtime does not want one, and
  an empty `Bearer` is a request some servers reject outright."""
  fetch = fake_fetch([ok(answer())])
  boss = local_overseer(fetch)
  boss.decide({"decisions": 0})
  call = fetch.calls[0]
  assert call["url"] == f"{llm.LOCAL_URL}/chat/completions"
  assert "Authorization" not in call["headers"]
  schema = call["body"]["response_format"]["json_schema"]["schema"]
  assert schema["properties"]["action"]["enum"] == list(MENU.available())


def test_the_local_path_cannot_return_an_action_outside_the_menu():
  """The grammar constraint, from both ends.

  Server-side, `action` is an ENUM of this world's menu, so a decoder
  honouring the schema has no token sequence that spells an action which does
  not exist -- that is what makes a 4B model safe in this seat. Sim-side, an
  endpoint that ignored the schema anyway is caught by `validate` and becomes
  a tagged fallback: the menu is enforced twice and an out-of-menu action
  reaches the mission loop by neither path.
  """
  enum = MENU.schema()["properties"]["action"]["enum"]
  assert set(enum) == set(MENU.available())
  assert "fly" not in enum
  boss = local_overseer(fake_fetch([ok(answer(action="fly"))]))
  decision = boss.decide({"decisions": 0})
  assert decision.scripted and decision.source.startswith("fallback:")
  assert decision.action in MENU.available()


def test_an_endpoint_that_refuses_the_schema_says_so_once():
  """A local runtime with no structured-output support still decides -- the
  schema goes into the prose and `validate` becomes the only enforcement.
  That is a weaker guarantee than the default one, so it is REPORTED: a
  silent downgrade would show up only as a mysteriously higher fallback
  rate."""
  fetch = fake_fetch([
    (400, {"error": {"message": "response_format is not supported"}}),
    ok(answer()),
  ])
  boss = local_overseer(fetch)
  assert boss.constrained
  for _ in range(3):
    assert boss.decide({"decisions": 0}).source == "llm"
  assert not boss.constrained
  notes = [e for e in boss.usage.errors if e.startswith("schema:")]
  assert len(notes) == 1 and "local" in notes[0]
  assert not boss.stats()["constrained"]


def test_a_local_answer_is_metered_but_never_priced_as_money():
  """Zero here is a MEASUREMENT -- a model on this machine is billed by
  nobody -- and the token counts are real, so the report can say "no API
  cost" rather than either "$0.00000" or "unknown"."""
  boss = local_overseer(fake_fetch([ok(answer(), 2000, 50)]))
  assert boss.decide({"decisions": 0}).source == "llm"
  stats = boss.stats()
  assert (stats["inputTokens"], stats["outputTokens"]) == (2000, 50)
  assert stats["usd"] == 0.0 and stats["priced"] is True
  assert stats["backend"] == "local" and stats["model"] == llm.LOCAL_MODEL


def test_a_third_party_endpoint_prices_as_unknown_not_free():
  """The other half of the same criterion: an endpoint whose rates this code
  cannot read must not report a confident number. Haiku's rates would be a
  fabricated invoice; zero would claim it was free."""
  boss = Overseer(MENU, model="some/model", backend="openai-compatible")
  boss._client = ChatClient(base_url="http://example.invalid/v1",
                            fetch=fake_fetch([ok(answer(), 2000, 50)]))
  boss._client_ready = True
  boss._meter_rates()
  boss.decide({"decisions": 0})
  stats = boss.stats()
  assert stats["priced"] is False and stats["usd"] == 0.0
  assert stats["inputTokens"] == 2000


def test_an_anthropic_model_the_rate_table_does_not_cover_is_unknown_too():
  """Same rule pointed at our own vendor: the rates at the top of
  mind/overseer.py are Haiku 4.5's, and charging a Sonnet run at them would be
  inventing a bill from the one direction it would be easiest to believe."""
  boss = Overseer(MENU, model="claude-sonnet-5", backend="anthropic")
  boss._client = object()
  boss._meter_rates()
  assert boss.usage.priced is False


# ---- every guard, on the local path ------------------------------------------


def test_a_stalled_local_endpoint_falls_back_like_a_slow_api():
  """A local backend that stalls is the same failure as an API that times
  out, and the caller must be released by the CLOCK rather than by the
  endpoint."""
  def slow(url, body, headers, timeout):
    # Longer than the deadline the caller is actually released by, which is
    # `timeout_s + POLL_GRACE_S` -- a sleep shorter than that measures
    # nothing, because the answer lands first.
    time.sleep(overseer.POLL_GRACE_S + 1.0)
    return ok(answer())

  boss = local_overseer(slow, timeout_s=0.05)
  decision = boss.decide({"decisions": 0})
  assert decision.source == "fallback:timeout"
  assert decision.action in MENU.available()


#: What a cold local model measured on the dev box (GTX 1660 Super, 6 GB,
#: qwen3:4b-instruct, the real ~11 kB prompt): 27.3 s to load the weights and
#: answer, against 3.4-5.5 s once they are resident.
COLD_LOAD_S = 27.3


def test_the_local_deadline_survives_a_cold_model_load():
  """The measured constant, pinned -- and the defect it fixes, pinned beside
  it so the premise cannot rot.

  A local backend's first decision includes loading the weights into VRAM.
  On the Anthropic path's 8 s deadline that is a certainty of failure rather
  than a risk of one: every mission's opening decision, and every one after
  an errand long enough for ollama to unload the model, arrives as
  `fallback:TimeoutError` with the model still working on an answer nobody
  will read. Measured three for three before this constant existed.
  """
  assert overseer.CALL_TIMEOUT_S < COLD_LOAD_S, \
      "the defect: the API deadline cannot cover a model load"
  assert llm.LOCAL_TIMEOUT_S > COLD_LOAD_S
  assert Overseer(MENU, model=llm.LOCAL_MODEL,
                  backend="local").timeout_s == llm.LOCAL_TIMEOUT_S
  # ...and the API path keeps ITS number, which is the regression half: a
  # slow local runtime must not buy a slow API eight times the patience.
  assert Overseer(MENU).timeout_s == overseer.CALL_TIMEOUT_S
  assert Overseer(MENU, model="org/model-8b").timeout_s == \
      overseer.CALL_TIMEOUT_S


def test_a_dead_local_endpoint_is_a_fallback_not_a_crash():
  """Nothing listening on the port: an ordinary failed call, tagged with what
  went wrong, backing the endpoint off after MAX_CONSECUTIVE_ERRORS."""
  boss = Overseer(MENU, model=llm.LOCAL_MODEL, backend="local",
                  base_url="http://127.0.0.1:1/v1", timeout_s=1.0)
  for _ in range(overseer.MAX_CONSECUTIVE_ERRORS):
    decision = boss.decide({"decisions": 0})
    assert decision.scripted and decision.source.startswith("fallback:")
  assert boss.decide({"decisions": 0}).source == "fallback:cooloff"


def test_a_spent_budget_is_scripted_on_the_local_backend_too():
  """The call budget is backend-independent -- and it still is when the calls
  are free, because the reason it exists (a loop bug that hammers an
  endpoint) does not care what the endpoint charges."""
  fetch = fake_fetch([ok(answer())])
  boss = local_overseer(fetch, calls_per_hour=2)
  assert [boss.decide({"decisions": 0}).source for _ in range(3)] == [
    "llm", "llm", "fallback:budget"]
  assert len(fetch.calls) == 2


# ---- the flag, on the script a visitor's world is served by ------------------


def test_serve_puts_a_named_backend_in_front_of_the_same_loop(monkeypatch):
  """`--overseer-backend local` on serve.py, checked through the whole
  wiring: the flag has to reach `overseer.build`, pick that backend's default
  MODEL (there is no ollama tag called claude-haiku-4-5) and with it that
  backend's deadline. A flag parsed and dropped looks identical to a flag
  that worked until the first decision times out."""
  from test_webserver import _serve_wiring

  life, _, _ = _serve_wiring(monkeypatch, [
    "--world", "room_hub", "--free-run", "--overseer",
    "--overseer-backend", "local", "--overseer-url", "http://127.0.0.1:9/v1"])
  boss = life.init_kwargs["overseer"]
  assert (boss.backend, boss.model) == ("local", llm.LOCAL_MODEL)
  assert boss.base_url == "http://127.0.0.1:9/v1"
  assert boss.timeout_s == llm.LOCAL_TIMEOUT_S
  assert boss.stats()["backend"] == "local"


def test_serve_without_the_flag_is_the_anthropic_run_it_always_was(monkeypatch):
  """The regression criterion at the top level: an unflagged served world
  builds exactly the overseer it built before issue #19."""
  from test_webserver import _serve_wiring

  monkeypatch.delenv(overseer.BACKEND_ENV, raising=False)
  monkeypatch.delenv(overseer.MODEL_ENV, raising=False)
  life, _, _ = _serve_wiring(monkeypatch,
                             ["--world", "room_hub", "--free-run",
                              "--overseer"])
  boss = life.init_kwargs["overseer"]
  assert (boss.backend, boss.model) == ("anthropic", MODEL)
  assert boss.timeout_s == overseer.CALL_TIMEOUT_S


# ---- the real socket ---------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
  """The smallest thing that answers like ollama's OpenAI endpoint."""

  def do_POST(self):                        # noqa: N802 -- BaseHTTPRequestHandler
    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
    self.server.requests.append(body)
    payload = json.dumps({
      "choices": [{"message": {"content": self.server.reply}}],
      "usage": {"prompt_tokens": 1234, "completion_tokens": 42}}).encode()
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(payload)))
    self.end_headers()
    self.wfile.write(payload)

  def log_message(self, *args):             # keep pytest -s readable
    pass


@pytest.fixture()
def local_endpoint():
  server = HTTPServer(("127.0.0.1", 0), _Handler)
  server.requests = []
  server.reply = answer()
  threading.Thread(target=server.serve_forever, daemon=True).start()
  yield server
  server.shutdown()
  server.server_close()


def test_a_whole_decision_flies_over_a_real_socket_to_localhost(local_endpoint):
  """The end-to-end claim, with `_default_fetch` and urllib in the path: an
  ordinary `llm` decision, metered off the endpoint's own usage block, from a
  model reachable without leaving the machine."""
  host, port = local_endpoint.server_address
  boss = Overseer(MENU, model=llm.LOCAL_MODEL, backend="local",
                  base_url=f"http://{host}:{port}/v1", timeout_s=5.0)
  decision = boss.decide({"decisions": 0})
  assert decision.source == "llm"
  assert (decision.action, decision.zone) == ("explore", "garden")
  assert boss.usage.input_tokens == 1234
  assert boss.usage.usd == 0.0 and boss.usage.priced
  sent = local_endpoint.requests[0]
  assert sent["model"] == llm.LOCAL_MODEL
  assert sent["messages"][0]["role"] == "system"
  # The prefix/volatile split survives the translation: the stable half is
  # the system message, the state is the user turn. Nothing local BILLS for
  # the prefix, but every local runtime reuses its KV cache for it, which is
  # seconds off a decision the robot is standing still for.
  assert boss.system[0]["text"] in sent["messages"][0]["content"]
  assert sent["messages"][1]["role"] == "user"


def test_a_garbled_local_answer_is_a_tagged_fallback(local_endpoint):
  """Prose instead of JSON -- the small-model failure the schema exists to
  prevent, and what happens when an endpoint ignores it anyway."""
  host, port = local_endpoint.server_address
  local_endpoint.reply = "Sure! I think the robot should probably explore."
  boss = Overseer(MENU, model=llm.LOCAL_MODEL, backend="local",
                  base_url=f"http://{host}:{port}/v1", timeout_s=5.0)
  decision = boss.decide({"decisions": 0})
  assert decision.source.startswith("fallback:ValueError")
  assert decision.action in MENU.available()
