"""OpenAI-compatible chat endpoints as the overseer's client (issues #15, #19).

The overseer's injection seam has always been "anything with
`.messages.create(**kwargs)` returning an object with `.content` and
`.usage`" -- the Anthropic client's shape, and the seam every fake in
tests/test_overseer.py already speaks. This module adapts an OpenAI-style
`/chat/completions` endpoint to THAT seam, so `Overseer._call`, its
validation, its metering and its fallback machinery do not know or care which
vendor -- or which machine -- answered.

Three endpoints wear the one adapter, because the protocol is the same one:

  huggingface        the HF Inference Providers router. Where candidate models
                     are MEASURED (`scripts/overseer_probe.py --model
                     Qwen/...`) and where per-provider pricing comes from, so
                     the cost report has honest numbers instead of a hardcoded
                     table. Needs `$HF_TOKEN`.
  local              a model on this machine -- ollama, llama.cpp's server,
                     anything speaking the same protocol on `$PLUGGY_OVERSEER_
                     URL` (default `http://localhost:11434/v1`, ollama's).
                     No token, no network, no bill.
  openai-compatible  the same shape pointed at somebody else's endpoint, with
                     `$PLUGGY_OVERSEER_KEY` if it wants one.

...and `anthropic` is the fourth backend, which is the SDK rather than this
module. `build_client` is the one place that knows all four, so nothing above
it has a vendor branch. `resolve_backend("auto", model)` keeps issue #15's
rule -- every HF id is `org/name` and no Anthropic id contains a slash -- so
a deployment that only ever set `$PLUGGY_MODEL` is routed exactly as it was.

Differences from the Anthropic path, and how they are handled:

  - No prompt caching. The router bills full input tokens every call; the
    stable/volatile prompt split is kept anyway (some providers, and every
    local runtime, reuse the KV cache for LATENCY), and the usage shim
    reports the cache fields as 0, so `cacheHitRate: 0` is the HONEST
    reading here rather than the 4096-token floor the Anthropic path
    documents.
  - Structured outputs are endpoint-dependent. The request carries the same
    JSON schema as OpenAI-style `response_format`, which is what makes a
    small model safe here: `action` is an ENUM of the menu, so a decoder
    honouring the schema cannot emit an action that does not exist. An
    endpoint that rejects the field (a 4xx naming it) gets ONE retry with
    the schema spelled out in the system text instead -- and `constrained`
    goes False, because "the menu is enforced at the decoder" and "the menu
    is enforced only by validate() after the fact" are different guarantees
    and the operator should be told which one is running. The overseer's
    `_extract_json` + `validate` remain the last word either way.
  - A missing token fails at CONSTRUCTION, not on the first request (the
    opposite of `anthropic.Anthropic()`, whose late failure is why the
    cool-off exists). The overseer's client property already catches this and
    resolves to `fallback:no-client`, which is the right story: there is
    nothing to back off from when nothing could ever have been dialled.
  - Reasoning models may prefix their answer with a `<think>` block. It is
    stripped before parsing, because "the model reasoned first" and "the
    model did not answer JSON" are different events and only the second
    should become `fallback:garbled`.

Stdlib `urllib` on purpose: the serving image installs six pinned packages
(deploy/requirements-serve.txt) and this backend should not grow that list to
talk to one HTTP endpoint sixty times an hour -- least of all to talk to one
on localhost.
"""

import json
import os
import urllib.error
import urllib.request
from types import SimpleNamespace

#: Which backend decides. `auto` is the default and infers from the model id,
#: which is what every deployment written before issue #19 relies on.
BACKENDS = ("auto", "anthropic", "huggingface", "local", "openai-compatible")
#: The backends this module serves; `anthropic` is the SDK, in `build_client`.
CHAT_BACKENDS = ("huggingface", "local", "openai-compatible")

TOKEN_ENV = "HF_TOKEN"
#: A key for a THIRD-PARTY OpenAI-compatible endpoint. Separate from
#: `$HF_TOKEN` on purpose: they are different secrets for different hosts, and
#: one variable holding "whichever one is in play" is how the wrong one gets
#: sent to the wrong place.
KEY_ENV = "PLUGGY_OVERSEER_KEY"
#: Where the local / third-party endpoint lives. Not a flag on the serving
#: image, like every other deploy knob.
URL_ENV = "PLUGGY_OVERSEER_URL"

ROUTER = "https://router.huggingface.co/v1"
#: ollama's OpenAI-compatible endpoint, which is what `--overseer-backend
#: local` means on a box with nothing else configured.
LOCAL_URL = "http://localhost:11434/v1"
#: The default local model, and the reason is the card rather than a
#: preference: an 8B at Q4 is ~4.5-5 GB, which fits a 6 GB GTX 1660 Super with
#: room for KV cache, and a 4B instruct leaves considerably more. Instruct
#: rather than thinking -- a thinking model spends its `max_tokens` budget on
#: `<think>` and truncates before the answer (docs/Overseer.md §6, measured).
LOCAL_MODEL = "qwen3:4b-instruct"
#: ⚠ A LOCAL DECISION IS NOT AN API DECISION, and the difference is the model
#: LOAD. Measured here (GTX 1660 Super, 6 GB, qwen3:4b-instruct, the real
#: 11 kB prompt): a warm decision is 3.4-5.5 s, and the first one after the
#: weights are out of VRAM is **27.3 s**. The Anthropic path's 8 s deadline
#: therefore turns every mission's opening decision into a guaranteed
#: `fallback:timeout` -- measured, three for three, before this constant
#: existed -- and ollama unloads an idle model after five minutes, so a robot
#: coming back from a long errand pays it again. The budget, the cool-off and
#: the fallback are untouched; only the number they are measured against
#: moves, and it moves because the endpoint is genuinely different.
LOCAL_TIMEOUT_S = 45.0

#: Answer-format instructions for the one-retry path when an endpoint rejects
#: `response_format`. Kept terse: the schema itself rides along, and the
#: overseer's validate() is what actually enforces it.
FORMAT_NOTE = ("\n\nANSWER FORMAT\n\nAnswer with a single JSON object and "
               "nothing else -- no prose around it, no code fences. It must "
               "match this JSON schema exactly:\n")


def is_hf_model(model: str) -> bool:
  """Every HF id is `org/name`; no Anthropic id contains a slash."""
  return "/" in (model or "")


def resolve_backend(backend: str | None, model: str = "") -> str:
  """A backend name (or `auto`, or nothing) -> one of `BACKENDS` minus auto.

  `auto` is issue #15's rule, kept verbatim so nothing that worked before
  issue #19 routes anywhere new: a model id with a slash is a HuggingFace id
  and goes to the router, and anything else is Anthropic's. A backend named
  explicitly always wins -- a local runtime is perfectly happy to serve a
  model called `qwen3:4b-instruct`, and an id shape cannot tell you which
  machine you meant.
  """
  name = (backend or "auto").strip().lower()
  if name not in BACKENDS:
    raise ValueError(f"unknown overseer backend {backend!r}; "
                     f"pick one of {', '.join(BACKENDS)}")
  if name != "auto":
    return name
  return "huggingface" if is_hf_model(model) else "anthropic"


def default_timeout(backend: str, anthropic_s: float) -> float:
  """Wall seconds one decision may take on this backend.

  `anthropic_s` is the caller's own default (`overseer.CALL_TIMEOUT_S`) --
  passed in rather than imported so this module stays free of the overseer,
  which imports it.
  """
  return LOCAL_TIMEOUT_S if backend in ("local",
                                       "openai-compatible") else anthropic_s


def default_url(backend: str) -> str:
  """Where a chat backend lives when nothing says otherwise."""
  if backend == "huggingface":
    return ROUTER
  return os.environ.get(URL_ENV, "").strip() or LOCAL_URL


def _default_fetch(url: str, body: dict | None, headers: dict,
                   timeout: float) -> tuple[int, dict]:
  """One HTTP exchange: (status, parsed JSON). The injection seam for tests.

  An HTTP error status is RETURNED rather than raised, because a 4xx body is
  data this module reads (the retry decision, the endpoint's error message);
  only transport-level failures raise.
  """
  data = json.dumps(body).encode() if body is not None else None
  req = urllib.request.Request(url, data=data, headers=headers,
                               method="POST" if body is not None else "GET")
  try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      return resp.status, json.loads(resp.read().decode())
  except urllib.error.HTTPError as e:
    try:
      payload = json.loads(e.read().decode())
    except Exception:                       # noqa: BLE001 -- body may be HTML
      payload = {"error": {"message": str(e)}}
    return e.code, payload


class ChatClient:
  """An OpenAI-compatible endpoint, wearing the Anthropic client's shape.

  `fetch` is the test seam: (url, body, headers, timeout) -> (status, json).
  `label` is what an error line calls this endpoint, because "HF router 401"
  and "local backend 401" send an operator to different places.
  """

  #: Set False the first time an endpoint makes us drop `response_format`.
  #: Read by the overseer, which says so once in its error list: the decoder
  #: is no longer refusing to name an action off the menu, and only
  #: `validate()` stands between a small model and a fallback per call.
  constrained = True

  def __init__(self, base_url: str = LOCAL_URL, token: str | None = None,
               timeout: float = 8.0, fetch=None,
               label: str = "chat backend") -> None:
    self.base_url = (base_url or LOCAL_URL).rstrip("/")
    self.token = (token or "").strip()
    self.timeout = timeout
    self.label = label
    self.fetch = fetch or _default_fetch
    # `.messages.create(...)` -- the seam, shaped like the SDK's.
    self.messages = SimpleNamespace(create=self._create)

  def _headers(self) -> dict:
    # No Authorization header at all when there is no token: a local runtime
    # does not want one, and sending `Bearer ` empty is a request some
    # servers reject outright.
    headers = {"Content-Type": "application/json"}
    if self.token:
      headers["Authorization"] = f"Bearer {self.token}"
    return headers

  def _create(self, model: str, max_tokens: int, system, output_config=None,
              messages=()) -> SimpleNamespace:
    """`messages.create`, translated: Anthropic call shape in, shim out."""
    # The system prompt arrives as Anthropic content blocks (the cached-prefix
    # shape); the endpoint wants one system message. cache_control is dropped
    # -- there is nothing here for it to mark.
    sys_text = "\n\n".join(b["text"] for b in (system or ())
                           if isinstance(b, dict) and b.get("text"))
    schema = ((output_config or {}).get("format") or {}).get("schema")
    body = {
      "model": model,
      "max_tokens": max_tokens,
      "messages": [{"role": "system", "content": sys_text},
                   *[dict(m) for m in messages]],
    }
    if schema is not None:
      body["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "decision", "schema": schema, "strict": True},
      }
    url = f"{self.base_url}/chat/completions"
    status, payload = self.fetch(url, body, self._headers(), self.timeout)
    if (400 <= status < 500 and schema is not None
        and _blames_format(payload)):
      # This endpoint does not do constrained decoding. Spell the schema out
      # in the system text instead and ask once more -- validate() remains
      # the enforcement either way, and `constrained` records that it is now
      # the ONLY enforcement.
      self.constrained = False
      retry = dict(body)
      retry.pop("response_format")
      retry["messages"] = [{"role": "system",
                            "content": sys_text + FORMAT_NOTE
                            + json.dumps(schema, sort_keys=True)},
                           *[dict(m) for m in messages]]
      status, payload = self.fetch(url, retry, self._headers(), self.timeout)
    if status != 200 or "error" in payload:
      raise RuntimeError(_error_line(self.label, status, payload))
    text = _answer_text(payload)
    usage = payload.get("usage") or {}
    return SimpleNamespace(
      content=[SimpleNamespace(type="text", text=text)],
      usage=SimpleNamespace(
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
      ))


class HFClient(ChatClient):
  """The HF router: the same adapter, with a required token and a price list.

  Why the router: Ben's hardware runs ~8B models at a decent speed, so the
  plan is to MEASURE model quality over the API first
  (`scripts/overseer_probe.py --model Qwen/...`) and move a proven one on to
  the local backend. The router speaks one protocol for all of them and
  publishes per-provider pricing in `/v1/models`, which is where `pricing()`
  gets honest numbers for the cost-per-sim-hour report.
  """

  def __init__(self, token: str | None = None, timeout: float = 8.0,
               fetch=None) -> None:
    token = token or os.environ.get(TOKEN_ENV, "").strip()
    if not token:
      # Failing HERE is deliberate -- see the module docstring. The overseer
      # catches this at client construction and runs scripted with
      # `fallback:no-client`, exactly as it does for a missing SDK.
      raise ValueError(f"${TOKEN_ENV} is not set")
    super().__init__(base_url=ROUTER, token=token, timeout=timeout,
                     fetch=fetch, label="HF router")

  def pricing(self, model: str) -> tuple[float, float] | None:
    """(USD per Mtok in, out) for a model, off the router's own catalogue.

    The router picks the provider per request, so this is the CHEAPEST live
    provider's rate -- an estimate, and an honest one: the catalogue is where
    HF publishes what it bills, and a hardcoded table here would be stale by
    the second model Ben tries. None when the catalogue does not answer, and
    the caller reports cost as unknown rather than as zero.
    """
    try:
      status, payload = self.fetch(f"{self.base_url}/models", None,
                                   self._headers(), self.timeout)
    except Exception:                       # noqa: BLE001 -- report, not gate
      return None
    if status != 200:
      return None
    for entry in payload.get("data", ()):
      if entry.get("id") != model:
        continue
      rates = [(p["pricing"]["input"], p["pricing"]["output"])
               for p in entry.get("providers", ())
               if p.get("status") == "live" and p.get("pricing")]
      if rates:
        return min(rates)
    return None


def build_client(backend: str, model: str, timeout: float = 8.0,
                 base_url: str | None = None, token: str | None = None):
  """The one place that knows which vendor is which (issue #19).

  Every caller above this is vendor-blind: the overseer holds a thing with
  `.messages.create`, and what kind of thing it is shows up in exactly two
  places -- the metering policy in `Overseer.client` (a local model has no
  bill; the router publishes one; a stranger's endpoint we cannot know) and
  the line in `stats()` that says which mind was deciding.

  Raises rather than returning None. The overseer catches everything here and
  degrades to `fallback:no-client` with the reason in `usage.errors`, which
  is how a missing key has always been reported and is more use than a silent
  scripted run.
  """
  if backend == "anthropic":
    # Lazy because `anthropic` is a runtime dependency of the SERVE path and
    # the import must not be a hard requirement of importing the mission
    # stack -- tests/test_deploy.py flies the robot with the image's package
    # set, and a module-level import would make the overseer's absence a
    # crash rather than a fallback.
    import anthropic
    return anthropic.Anthropic(timeout=timeout, max_retries=0)
  if backend == "huggingface":
    return HFClient(timeout=timeout, token=token)
  if backend not in CHAT_BACKENDS:
    raise ValueError(f"unknown overseer backend {backend!r}")
  if not model:
    # A local runtime has a sensible default (`LOCAL_MODEL`); somebody
    # else's endpoint does not, and guessing one produces a 404 from a
    # stranger's server rather than a sentence an operator can act on.
    raise ValueError(f"backend {backend!r} needs a model id "
                     "(--overseer-model / $PLUGGY_MODEL)")
  local = backend == "local"
  return ChatClient(
    base_url=base_url or default_url(backend), timeout=timeout,
    # A local endpoint takes no key and should not be sent one; a third-party
    # one usually wants its own, which is not $HF_TOKEN.
    token=None if local else (token or os.environ.get(KEY_ENV, "").strip()),
    label="local backend" if local else "openai-compatible backend")


def _blames_format(payload: dict) -> bool:
  msg = str((payload.get("error") or {}).get("message", payload))[:500].lower()
  return ("response_format" in msg or "json_schema" in msg
          or "structured" in msg or "grammar" in msg)


def _error_line(label: str, status: int, payload: dict) -> str:
  err = payload.get("error")
  msg = err.get("message") if isinstance(err, dict) else err
  return f"{label} {status}: {str(msg or payload)[:160]}"


def _answer_text(payload: dict) -> str:
  choices = payload.get("choices") or []
  message = (choices[0].get("message") or {}) if choices else {}
  text = str(message.get("content") or "")
  # A reasoning model thinks out loud before answering. Keep the answer.
  if "</think>" in text:
    text = text.rsplit("</think>", 1)[-1]
  return text.strip()
