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
#: therefore turned every mission's opening decision into a guaranteed
#: `fallback:timeout` -- measured, three for three, before this constant
#: existed -- and ollama unloads an idle model after five minutes, so a robot
#: coming back from a long errand pays it again. The budget, the cool-off and
#: the fallback are untouched; only the number they are measured against
#: moves, and it moves because the endpoint is genuinely different.
#:
#: ⚠ It is now a FLOOR rather than the local answer (see `default_timeout`).
#: Since issue #117 the API deadline is 90 s and covers the cold load on its
#: own, so this number stops being the one that bites -- but it is kept, and
#: kept measured, because it is the only thing in the tree that knows how
#: long a model takes to reach VRAM. If the API deadline ever comes back
#: down, the local path must not come down with it.
LOCAL_TIMEOUT_S = 45.0

#: What every request says it is. ⚠ NOT DECORATION (issue #225): a
#: Cloudflare WAF in front of at least one of the router's providers answers
#: urllib's default `Python-urllib/3.x` with a 403 HTML page and the same
#: body under any other name with a 200 -- measured on gpt-oss-120b and
#: Qwen3.5-9B, every call, and the 4B on nscale never met it. The router
#: picks the provider per request, so without this a served robot is one
#: re-route away from `fallback:offline` on every decision.
USER_AGENT = "pluggybot/0.1 (+https://github.com/benholland1024/pluggybot)"

#: Answer-format instructions for the one-retry path when an endpoint rejects
#: `response_format`. Kept terse: the schema itself rides along, and the
#: overseer's validate() is what actually enforces it.
FORMAT_NOTE = ("\n\nANSWER FORMAT\n\nAnswer with a single JSON object and "
               "nothing else -- no prose around it, no code fences. It must "
               "match this JSON schema exactly:\n")


#: The picture's media type, wherever a request carries one (issue #275).
IMAGE_MEDIA_TYPE = "image/jpeg"


def image_part(backend: str, b64: str) -> dict:
  """One picture as a content part of a user turn, in the backend's own
  shape (issue #275): the Anthropic SDK's `image` block with a base64
  source, and the OpenAI-style `image_url` data URL everywhere else --
  the router, a local runtime and a stranger's endpoint all speak the
  latter. MEASURED on the router (2026-09-21): the deployed model takes
  the data URL beside the structured-output schema on the providers
  `:cheapest` lands on. The one place a vendor's image shape is known,
  on `build_client`'s terms.
  """
  if backend == "anthropic":
    return {"type": "image", "source": {"type": "base64",
                                        "media_type": IMAGE_MEDIA_TYPE,
                                        "data": b64}}
  return {"type": "image_url",
          "image_url": {"url": f"data:{IMAGE_MEDIA_TYPE};base64,{b64}"}}


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

  ⚠ A FLOOR, NOT AN OVERRIDE (issue #117). This used to hand the local
  backend `LOCAL_TIMEOUT_S` outright, which was the same thing while that
  number was the larger one (45 against 8). At a 90 s API deadline it would
  invert the constant's own reason for existing -- the local path is the one
  with a MEASURED slow case, a 27.3 s cold model load, so it must never be
  given LESS patience than an endpoint on the far side of the internet.
  `max` says both facts at once and stays right whichever number moves.
  """
  if backend in ("local", "openai-compatible"):
    return max(LOCAL_TIMEOUT_S, anthropic_s)
  return anthropic_s


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
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
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

    The catalogue is where HF publishes what each provider bills, and a
    hardcoded table here would be stale by the second model Ben tries. None
    when the catalogue does not answer, and the caller reports cost as
    unknown rather than as zero.

    ⚠ WHICH PROVIDER IS THE MODEL ID'S TO SAY (issue #225). The router
    accepts `org/name:<provider>` and the policies `:cheapest` / `:fastest`,
    and WITHOUT a suffix it routes by its own preference order -- measured:
    a bare `openai/gpt-oss-120b` went to Cerebras at $0.35/$0.75 while the
    catalogue's cheapest was DeepInfra at $0.037/$0.17, ten times apart on
    the same call. So a named provider is priced as itself, `:cheapest` and
    a bare id as the cheapest live provider (the bare id's number is an
    ESTIMATE the router need not honour, and a deployment that cares what
    it pays pins the policy in `$PLUGGY_MODEL`), and `:fastest` or any
    other policy as unknown, because nothing here can say who will answer.
    """
    base, _, policy = model.partition(":")
    try:
      status, payload = self.fetch(f"{self.base_url}/models", None,
                                   self._headers(), self.timeout)
    except Exception:                       # noqa: BLE001 -- report, not gate
      return None
    if status != 200:
      return None
    for entry in payload.get("data", ()):
      if entry.get("id") != base:
        continue
      live = [p for p in entry.get("providers", ())
              if p.get("status") == "live" and p.get("pricing")]
      if policy and policy != "cheapest":
        named = [p for p in live if p.get("provider") == policy]
        if not named:
          return None
        return (named[0]["pricing"]["input"], named[0]["pricing"]["output"])
      rates = [(p["pricing"]["input"], p["pricing"]["output"]) for p in live]
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
  # `error` is a dict on most providers and a bare STRING on some (the
  # router's own 4xx bodies): MEASURED as `AttributeError: 'str' object
  # has no attribute 'get'` filed as `fallback:offline` (issue #264).
  err = payload.get("error")
  msg = str(err.get("message", payload) if isinstance(err, dict) else (err or payload))[:500].lower()
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
