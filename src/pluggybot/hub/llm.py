"""HuggingFace Inference Providers as the overseer's client (issue #15).

The overseer's injection seam has always been "anything with
`.messages.create(**kwargs)` returning an object with `.content` and
`.usage`" -- the Anthropic client's shape, and the seam every fake in
tests/test_overseer.py already speaks. This module adapts the HF router's
OpenAI-compatible chat API to THAT seam, so `Overseer._call`, its validation,
its metering and its fallback machinery do not know or care which vendor
answered. A model id containing "/" (every HF id is `org/name`) selects this
backend; a bare id stays Anthropic. `$PLUGGY_MODEL` picks the model without a
flag, like every other deploy knob.

Why the router: Ben's hardware runs ~8B models at a decent speed, so the plan
is to MEASURE model quality over the API first (`scripts/overseer_probe.py
--model Qwen/...`) and switch to local hosting only once an 8B has proven
out. The router speaks one protocol for all of them and publishes per-provider
pricing in `/v1/models`, which is where `pricing()` gets honest numbers for
the cost-per-sim-hour report instead of a hardcoded table.

Differences from the Anthropic path, and how they are handled:

  - No prompt caching. The router bills full input tokens every call; the
    stable/volatile prompt split is kept anyway (some providers reuse KV
    cache for latency), and the usage shim reports the cache fields as 0, so
    `cacheHitRate: 0` is the HONEST reading here rather than the 4096-token
    floor the Anthropic path documents.
  - Structured outputs are provider-dependent. The request carries the same
    JSON schema as OpenAI-style `response_format`; a provider that rejects it
    (a 4xx naming the field) gets ONE retry with the schema spelled out in
    the system text instead, and the overseer's `_extract_json` + `validate`
    remain the last word either way -- they always were, because the schema
    is enforced server-side and checked sim-side on both vendors.
  - A missing token fails at CONSTRUCTION, not on the first request (the
    opposite of `anthropic.Anthropic()`, whose late failure is why the
    cool-off exists). The overseer's client property already catches this and
    resolves to `fallback:no-client`, which is the right story: there is
    nothing to back off from when nothing could ever have been dialled.
  - Reasoning models may prefix their answer with a `<think>` block. It is
    stripped before parsing, because "the model reasoned first" and "the
    model did not answer JSON" are different events and only the second
    should become `fallback:ValueError`.

Stdlib `urllib` on purpose: the serving image installs six pinned packages
(deploy/requirements-serve.txt) and this backend should not grow that list to
talk to one HTTPS endpoint sixty times an hour.
"""

import json
import os
import urllib.error
import urllib.request
from types import SimpleNamespace

TOKEN_ENV = "HF_TOKEN"
ROUTER = "https://router.huggingface.co/v1"

#: Answer-format instructions for the one-retry path when a provider rejects
#: `response_format`. Kept terse: the schema itself rides along, and the
#: overseer's validate() is what actually enforces it.
FORMAT_NOTE = ("\n\nANSWER FORMAT\n\nAnswer with a single JSON object and "
               "nothing else -- no prose around it, no code fences. It must "
               "match this JSON schema exactly:\n")


def is_hf_model(model: str) -> bool:
  """Every HF id is `org/name`; no Anthropic id contains a slash."""
  return "/" in (model or "")


def _default_fetch(url: str, body: dict | None, headers: dict,
                   timeout: float) -> tuple[int, dict]:
  """One HTTPS exchange: (status, parsed JSON). The injection seam for tests.

  An HTTP error status is RETURNED rather than raised, because a 4xx body is
  data this module reads (the retry decision, the router's error message);
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


class HFClient:
  """The HF router, wearing the Anthropic client's shape.

  `fetch` is the test seam: (url, body, headers, timeout) -> (status, json).
  """

  def __init__(self, token: str | None = None, timeout: float = 8.0,
               fetch=None) -> None:
    self.token = token or os.environ.get(TOKEN_ENV, "").strip()
    if not self.token:
      # Failing HERE is deliberate -- see the module docstring. The overseer
      # catches this at client construction and runs scripted with
      # `fallback:no-client`, exactly as it does for a missing SDK.
      raise ValueError(f"${TOKEN_ENV} is not set")
    self.timeout = timeout
    self.fetch = fetch or _default_fetch
    # `.messages.create(...)` -- the seam, shaped like the SDK's.
    self.messages = SimpleNamespace(create=self._create)

  def _headers(self) -> dict:
    return {"Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"}

  def _create(self, model: str, max_tokens: int, system, output_config=None,
              messages=()) -> SimpleNamespace:
    """`messages.create`, translated: Anthropic call shape in, shim out."""
    # The system prompt arrives as Anthropic content blocks (the cached-prefix
    # shape); the router wants one system message. cache_control is dropped --
    # there is nothing here for it to mark.
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
    status, payload = self.fetch(f"{ROUTER}/chat/completions", body,
                                 self._headers(), self.timeout)
    if (400 <= status < 500 and schema is not None
        and _blames_format(payload)):
      # This provider does not do constrained decoding. Spell the schema out
      # in the system text instead and ask once more -- validate() remains
      # the enforcement either way.
      retry = dict(body)
      retry.pop("response_format")
      retry["messages"] = [{"role": "system",
                            "content": sys_text + FORMAT_NOTE
                            + json.dumps(schema, sort_keys=True)},
                           *[dict(m) for m in messages]]
      status, payload = self.fetch(f"{ROUTER}/chat/completions", retry,
                                   self._headers(), self.timeout)
    if status != 200 or "error" in payload:
      raise RuntimeError(_error_line(status, payload))
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

  def pricing(self, model: str) -> tuple[float, float] | None:
    """(USD per Mtok in, out) for a model, off the router's own catalogue.

    The router picks the provider per request, so this is the CHEAPEST live
    provider's rate -- an estimate, and an honest one: the catalogue is where
    HF publishes what it bills, and a hardcoded table here would be stale by
    the second model Ben tries. None when the catalogue does not answer, and
    the caller reports cost as unknown rather than as zero.
    """
    try:
      status, payload = self.fetch(f"{ROUTER}/models", None,
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


def _blames_format(payload: dict) -> bool:
  msg = str((payload.get("error") or {}).get("message", payload))[:500].lower()
  return ("response_format" in msg or "json_schema" in msg
          or "structured" in msg or "grammar" in msg)


def _error_line(status: int, payload: dict) -> str:
  err = payload.get("error")
  msg = err.get("message") if isinstance(err, dict) else err
  return f"HF router {status}: {str(msg or payload)[:160]}"


def _answer_text(payload: dict) -> str:
  choices = payload.get("choices") or []
  message = (choices[0].get("message") or {}) if choices else {}
  text = str(message.get("content") or "")
  # A reasoning model thinks out loud before answering. Keep the answer.
  if "</think>" in text:
    text = text.rsplit("</think>", 1)[-1]
  return text.strip()
