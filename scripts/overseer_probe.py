"""Measure the overseer against the REAL API (issue #15).

The unit tests fly the overseer against fakes, which is right -- a suite that
needs a key and a network is a suite that fails for reasons that are not about
the code. But two of the issue's acceptance criteria are *measurements* and
cannot be faked into existence:

  - the prompt-cache hit rate (`cache_read_input_tokens` non-zero across calls)
  - the cost per sim-hour

This script is where those numbers come from. It makes N real decisions against
a synthetic (but honest) robot state, prints the token accounting per call, and
says plainly whether caching engaged.

  ANTHROPIC_API_KEY=... uv run python scripts/overseer_probe.py
  ... --calls 5 --world home
  ... --tokens-only     # count the prefix and stop -- no DECISIONS, no tokens
                        # billed. ⚠ It still needs a key: `count_tokens` is a
                        # real (free) endpoint, not a local tokenizer, and
                        # there is no offline way to count Claude tokens that
                        # is worth trusting.
  HF_TOKEN=... uv run python scripts/overseer_probe.py \
      --model Qwen/Qwen3-4B-Instruct-2507   # any `org/name` id goes to the
                        # HuggingFace router instead (mind/llm.py) -- THIS is
                        # how candidate models are measured before one is
                        # picked for a served world ($PLUGGY_MODEL). Rates
                        # come off the router's own catalogue; the cache
                        # section does not apply (the router bills full input
                        # every call) and says so.
  uv run python scripts/overseer_probe.py --backend local
                        # ...and a model on THIS MACHINE (issue #19): ollama
                        # or llama.cpp on $PLUGGY_OVERSEER_URL, no key, no
                        # network, no bill. The numbers that matter here are
                        # the SECONDS per decision -- a 30-second answer is a
                        # robot standing still in the garden -- and how often
                        # the answer is valid, which is what the schema is
                        # for. Cost prints as "no API cost" rather than as
                        # $0.00000, because those are different claims.

⚠ EXPECT A CACHE HIT RATE OF ZERO unless the stable prefix is over 4096
tokens. That is Claude Haiku 4.5's minimum cacheable prefix, and below it a
`cache_control` marker is silently inert -- no error, no warning, just
`cache_creation_input_tokens: 0` forever. `--tokens-only` prints the prefix
size next to the threshold so the two are read together, because the honest
options at that point are "the prefix genuinely has more to say" and "this
model does not cache prompts this small", and padding it until the number
looks right is neither.
"""

import argparse
import json
import time
from dataclasses import replace

from pluggybot.lifecycle import board_book
from pluggybot.mind import llm
from pluggybot.mind.thoughts import ThoughtFiles
from pluggybot.telemetry.protocol import ROBOT_ROOT
from pluggybot.mind.overseer import MODEL, Menu, Overseer


def synthetic_state(menu: Menu, i: int) -> dict:
  """A plausible robot, drifting between calls.

  Deliberately NOT identical per call: an unchanging user turn would make the
  cache reading meaningless (the whole request would hit, prefix or no), and
  the real volatile block changes every time.
  """
  return {
    "simTimeS": round(120.0 + 97.3 * i, 1),
    "battery": {"fraction": round(0.92 - 0.07 * i, 3), "wh": 0.9,
                "reserveWh": 0.55, "charging": False},
    "mapDone": i > 0,
    "points": 12 * i,
    "recentTasks": [{"task": "draw", "ok": True, "points": 18,
                     "reason": "inked 6/6 strokes on whiteboard_a"}][:i],
    "tasksThisMission": ["draw"][:min(i, 1)],
    "boards": {b: {"fill": 0.11 * i, "strokes": 6 * i, "programs": []}
               for b in menu.boards},
    "journal": ["whiteboard_a was already full when I got there"][:i],
    # The two thought files that ride the VOLATILE half (issue #38). Here
    # rather than in the prefix on purpose, and carried by the probe because
    # they are real input tokens on every call -- a measurement that left
    # them out would under-report what a decision costs.
    "thoughts": {"History.md": [f"[t={120 * i}s] carry: fetched and stowed "
                                "module_lcd (+2 points)"][:i],
                 "Knowledge_and_Opinions.md":
                   "whiteboard_b is the one people look at" if i else ""},
    "visitorSuggestions": [],
    # A claimable offer, so the probe exercises `take_task` -- the action the
    # acceptance run measured small models getting WRONG (the kind "draw" in
    # `task` instead of the id, 23 times in 4 sim-hours before the prompt
    # spelled the id shape out). A decision naming `t_0007` is the fix
    # working; `fallback:ValueError ... not on offer` is it not.
    "offeredTasks": [{"id": "t_0007", "kind": "artwork",
                      "description": "Draw a sun on whiteboard_a for people "
                                     "to rate.",
                      "paysUpTo": 30, "claimable": True,
                      "needsAnswer": False}],
    "decisions": i,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--world", choices=("room_hub", "home"), default="home")
  parser.add_argument("--calls", type=int, default=4,
                      help="real decisions to make (each one costs money)")
  parser.add_argument("--model", default=None,
                      help=f"default {MODEL} on Anthropic, "
                           f"{llm.LOCAL_MODEL} on the local backend")
  parser.add_argument("--backend", default=None, choices=llm.BACKENDS,
                      help="which mind answers (issue #19); default auto -- "
                           "the model id's shape")
  parser.add_argument("--url", default=None, metavar="URL",
                      help="base URL for the local / openai-compatible "
                           f"backend (default {llm.LOCAL_URL})")
  parser.add_argument("--goals", default=None, metavar="PATH")
  parser.add_argument("--thoughts", default=None, metavar="DIR",
                      help="the robot's thought files (issue #38). Point it "
                           "at a real directory to measure the prefix a "
                           "deployment actually sends -- Main.md and Goals.md "
                           "ride in it, so an edited persona changes the "
                           "number below")
  parser.add_argument("--escalate-to", default=None, metavar="ID",
                      help="also measure the ESCALATION path (issue #37): "
                           "the bigger mind a decision can be bought from. "
                           "⚠ This spends real money -- one call per "
                           "decision below, at that model's rates")
  parser.add_argument("--force-escalate", action="store_true",
                      help="ask the big mind on EVERY decision, ignoring the "
                           "cadence gates. For measurement only: it is the "
                           "only way to get a per-escalation number without "
                           "waiting out the ten-minute interval between "
                           "them, and it is exactly what the gates exist to "
                           "stop the robot doing")
  parser.add_argument("--tokens-only", action="store_true",
                      help="count the prefix and stop -- no API calls")
  args = parser.parse_args()

  book = board_book(args.world)
  menu = Menu.for_world(args.world, book)
  # The REAL memory, so the prefix measured here is the prefix a deployment
  # sends: `Main.md` and `Goals.md` are in it (issue #38), and the two
  # writable files are deliberately not -- they ride the user turn below.
  memory = ThoughtFiles.open(args.thoughts, goals_path=args.goals)
  backend = llm.resolve_backend(args.backend, args.model or "")
  model = args.model or (llm.LOCAL_MODEL if backend == "local" else MODEL)
  boss = Overseer(menu, thoughts=memory, model=model, backend=backend,
                  base_url=args.url, escalate_to=args.escalate_to)
  prefix = boss.system[0]["text"]

  # The prefix now states the robot's NAME (issue #39), resolved from
  # $PLUGGY_ROBOT_NAME like a deployment's would be -- so the probe reports
  # who it measured, not just how big the measurement was.
  print(f"robot        : {boss.robot_name} (a {ROBOT_ROOT})")
  print(f"world        : {args.world}")
  print(f"model        : {model} on {backend}")
  if args.escalate_to:
    print(f"escalates to : {args.escalate_to}"
          + ("  (forced on every decision -- measurement only)"
             if args.force_escalate else ""))
  print(f"actions      : {', '.join(menu.available())}")
  print(f"prefix chars : {len(prefix)}")
  print(f"memory       : {', '.join(memory.stable())} cached; "
        f"{', '.join(memory.volatile())} per call")

  hf = backend == "huggingface"
  local = backend in ("local", "openai-compatible")
  client = boss.client
  if client is None:
    key = {"huggingface": "$HF_TOKEN",
           "anthropic": "$ANTHROPIC_API_KEY",
           "local": ("a runtime listening on "
                     f"{args.url or llm.default_url(backend)}"),
           "openai-compatible": "$PLUGGY_OVERSEER_KEY and --url"}[backend]
    print(f"\nno client -- needs {key} (the sim would run scripted)")
    for err in boss.usage.errors:
      print(f"  {err}")
    return
  prefix_tokens = None
  minimum = 4096
  if local:
    # Nothing to price and nothing to count: a local runtime publishes no
    # rates and has no count_tokens endpoint. What it DOES have is a KV
    # cache, which is why the prompt is still split stable/volatile -- the
    # saving is latency rather than money, and the per-call seconds below
    # are where it shows up.
    print(f"endpoint     : {args.url or llm.default_url(backend)}")
    if args.tokens_only:
      print("(--tokens-only needs count_tokens, which is Anthropic-only; "
            "prefix chars above are the size measure here)")
      return
  elif hf:
    # The router has no count_tokens endpoint and no billed prompt cache --
    # `cacheHitRate: 0` below is the honest reading, not the Haiku floor.
    # What it DOES publish is per-provider pricing, which is where the
    # cost report's rates come from.
    rates = ((boss.usage.usd_per_mtok_in, boss.usage.usd_per_mtok_out)
             if boss.usage.priced else None)
    print("router rates : "
          + (f"${rates[0]}/Mtok in, ${rates[1]}/Mtok out (cheapest live "
             "provider)" if rates else "UNKNOWN -- catalogue did not answer; "
             "usd below will read 0"))
    if args.tokens_only:
      print("(--tokens-only needs count_tokens, which is Anthropic-only; "
            "prefix chars above are the size measure here)")
      return
  else:
    try:
      counted = client.messages.count_tokens(
        model=model, system=boss.system,
        messages=[{"role": "user", "content": "?"}])
    except Exception as e:                  # noqa: BLE001
      # Almost always a missing or rejected key. Say so in one line rather
      # than in a twelve-frame traceback -- this script exists to report
      # numbers, and "I could not get one, here is why" is a report.
      print(f"\ncould not count tokens: {type(e).__name__}: "
            f"{str(e).splitlines()[0][:160]}")
      print("set ANTHROPIC_API_KEY -- count_tokens is a free endpoint, but it"
            " is still an authenticated one")
      return
    prefix_tokens = counted.input_tokens
    # 4096 on Haiku 4.5. Named here rather than imported because it is a
    # property of the MODEL, and the probe is the thing that gets pointed at
    # a different one.
    print(f"prefix tokens: {prefix_tokens} (cacheable minimum on Haiku 4.5:"
          f" {minimum})")
    if prefix_tokens < minimum:
      print(f"             ⚠ {minimum - prefix_tokens} short -- the"
            " cache_control marker will be INERT and the hit rate below will"
            " read 0. That is the model's floor, not a bug.")
    if args.tokens_only:
      return

  print(f"\nmaking {args.calls} real decision(s)...\n")
  wall0 = time.monotonic()
  for i in range(args.calls):
    before = dict(boss.usage.as_dict())
    t0 = time.monotonic()
    state = synthetic_state(menu, i)
    decision = boss.decide(state)
    if args.force_escalate and not decision.escalated:
      # Straight past the gates, on purpose and only here: the interval is
      # ten minutes and the point of this run is the per-escalation number.
      boss._last_escalation = None
      boss.escalations = 0
      esc0 = time.monotonic()
      decision = boss._maybe_escalate(replace(decision, escalate=True), state)
      print(f"   escalation took {time.monotonic() - esc0:5.2f}s")
    dt = time.monotonic() - t0
    now = boss.usage.as_dict()
    delta = {k: now[k] - before[k] for k in
             ("inputTokens", "outputTokens", "cacheReadTokens",
              "cacheWriteTokens")}
    print(f"{i + 1}. {decision.summary()}")
    print(f"   {dt:5.2f}s  in={delta['inputTokens']}"
          f" out={delta['outputTokens']}"
          f" cache_read={delta['cacheReadTokens']}"
          f" cache_write={delta['cacheWriteTokens']}")
  wall = time.monotonic() - wall0

  stats = boss.stats()
  print("\n" + json.dumps(stats, indent=1))
  if args.escalate_to and boss.escalation_usage.input_tokens:
    eu = boss.escalation_usage
    made = max(1, len(boss.decisions) if args.force_escalate
               else boss.escalations)
    print(f"\nescalation cost        : ${eu.usd / made:.5f} per call "
          f"({eu.input_tokens // made} in, {eu.output_tokens // made} out at "
          f"${eu.usd_per_mtok_in}/${eu.usd_per_mtok_out} per Mtok)"
          if eu.priced else
          "\nescalation cost        : unknown -- no published rates")
  # The acceptance criteria, in the units they were written in. A decision
  # every ~2 minutes of sim time is the design doc's cadence.
  per_call = stats["usd"] / max(1, stats["llmCalls"])
  valid = sum(1 for d in boss.decisions if not d.scripted)
  print(f"\nmean wall per decision : {wall / max(1, args.calls):.2f} s")
  print(f"valid decisions        : {valid}/{args.calls}"
        + ("" if stats["constrained"] else
           "   (UNCONSTRAINED -- this endpoint refused the schema)"))
  if backend == "local":
    print("cost                   : none -- the model is on this machine")
  elif not stats["priced"]:
    print("cost                   : unknown -- this backend publishes no "
          "rates; the token counts above are the honest measure")
  else:
    print(f"cost per decision      : ${per_call:.6f}")
    print(f"cost per sim-hour      : ${per_call * 30:.4f}"
          "   (at one decision per 2 sim-minutes)")
    print(f"cost per sim-day       : ${per_call * 30 * 24:.3f}")
  hit = stats["cacheHitRate"]
  if local:
    print(f"prompt-cache hit rate  : {hit:.0%}  -- a local runtime reuses its "
          "KV cache for LATENCY and bills nothing either way")
  elif hf:
    print(f"prompt-cache hit rate  : {hit:.0%}  -- the router bills full "
          "input every call; 0 is expected")
  else:
    print(f"prompt-cache hit rate  : {hit:.0%}"
          + ("" if hit else f"  -- prefix is {prefix_tokens} tokens, below"
                            f" the {minimum}-token floor"))


if __name__ == "__main__":
  main()
