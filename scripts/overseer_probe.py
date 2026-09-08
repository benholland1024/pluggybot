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

⚠ THE LATENCY DISTRIBUTION IS THE POINT OF THE `--calls` NUMBER (issue #117).
A fallback rate is a deterministic function of (latency distribution,
deadline), and the distribution needs no physics -- so `--calls 50` here is
how `overseer.CALL_TIMEOUT_S` gets chosen, in ten minutes, instead of by
five hours of flying the sim and reading the timeouts off the wreckage.
Two rules the report obeys:

  - **A mean hides the tail, and the tail is the whole measurement.** The
    deadline is a CAP on this distribution: the loaded baseline's median was
    6.5 s against 8 s and it still lost a third of its decisions. So min /
    median / p90 / p95 / max and the raw values, the way the record schema
    reports every other distribution (`Evaluation.md` section 4).
  - **The probe holds calls to `--timeout`, not to the deadline.** A
    distribution measured through the deadline it is meant to justify is
    CENSORED at that deadline -- every slow call reads as exactly 8 s and
    the tail the number is chosen from is the one part that was thrown away.

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

from pluggybot.evaluation.record import LATENCY_PERCENTILES, dist

from pluggybot.lifecycle import board_book
from pluggybot.mind import llm
from pluggybot.mind.thoughts import ThoughtFiles
from pluggybot.telemetry.protocol import ROBOT_ROOT
from pluggybot.mind.overseer import CALL_TIMEOUT_S, MODEL, Menu, Overseer

#: What the probe holds a call to, and deliberately not the deadline under
#: test: see the docstring. DERIVED from that deadline rather than fixed, so
#: raising `CALL_TIMEOUT_S` can never quietly bring the probe's own cap down
#: onto the distribution it is supposed to measure -- which is the same
#: censoring mistake one level up. Long enough that nothing on a working
#: endpoint is clipped (the measured max is 7.38 s), short enough that a
#: hung call does not hold the probe for the afternoon; a call that IS
#: clipped is reported as clipped rather than folded into the tail.
PROBE_TIMEOUT_S = max(120.0, 2 * CALL_TIMEOUT_S)

#: The ladder the timeout share is reported at. It brackets the original
#: 8 s, the local backend's 45 s and today's 90 s, so the number is read off
#: a curve rather than argued for one value at a time. The row for the
#: deadline actually in force is marked, whatever it is.
CANDIDATE_DEADLINES = (5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0)


def timeout_share(latencies: list[float], deadline: float) -> float:
  """The fraction of these calls a deadline would have cut off.

  The whole reason this issue does not need the sim: a fallback rate is
  this function of a measured distribution, and only the residual --
  answers that arrived and were malformed -- has to be flown for.
  """
  if not latencies:
    return 0.0
  return sum(1 for v in latencies if v > deadline) / len(latencies)


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
    # working; a `fallback:garbled` is it not.
    "offeredTasks": [{"id": "t_0007", "kind": "artwork",
                      "description": "Draw a sun on whiteboard_a for people "
                                     "to rate.",
                      "paysUpTo": 30, "claimable": True,
                      "needsAnswer": False}],
    "decisions": i,
  }


def report_orders(orders: list[tuple]) -> None:
  """THE CAPABILITY GATE (issue #115): does the agent use the affordance?

  A standing order is what the robot does when nobody can be reached, and
  it is the agent's to set -- `idle` until it says otherwise. Three things
  are worth knowing and none needs a sim: whether it sets one at all, WHAT
  it sets, and whether that changes as the battery falls.

  ⚠ AN ORDER THAT IS ALWAYS `idle` IS NOT THE SAME AS NO ORDER. The floor
  is `idle` too, so an agent that never engages with the field and one that
  deliberately chooses to stand still produce the same behaviour and
  different records -- which is why `set` and `unset` are counted apart in
  the run record, and why this prints the ORDER beside the action rather
  than only counting them.
  """
  if not orders:
    return
  set_any = [o for _, o, _ in orders if o]
  print(f"\nstanding orders        : {len(set_any)}/{len(orders)} decisions "
        "left one")
  if set_any:
    kinds: dict = {}
    for o in set_any:
      kinds[o] = kinds.get(o, 0) + 1
    print(f"  what               : {dict(sorted(kinds.items()))}")
  print("  battery  order       action")
  for frac, order, action in orders:
    print(f"  {frac:6.0%}   {order or '(none)':<11} {action}")


def report_latency(latencies: list[float], boss: Overseer,
                   held_to: float) -> None:
  """The distribution, then what each candidate deadline would cost.

  ⚠ Two shares, and they are not the same number. The deadline curve is
  what the LATENCY would have cost; the residual below it is what arrived
  in time and was still unusable -- a malformed answer, a refused schema,
  a spent budget. A deadline can only ever buy back the first, so the
  second is the floor any fallback-rate threshold has to clear.
  """
  d = dist(latencies, LATENCY_PERCENTILES)
  if not d["n"]:
    return
  print("\nwall seconds per decision (n=%d)" % d["n"])
  print(f"  min {d['min']:.2f}   median {d['median']:.2f}   "
        f"p90 {d['p90']:.2f}   p95 {d['p95']:.2f}   max {d['max']:.2f}")
  print(f"  raw: {' '.join(f'{v:.2f}' for v in latencies)}")
  clipped = sum(1 for v in latencies if v >= held_to)
  if clipped:
    print(f"  ⚠ {clipped} call(s) hit the probe's own {held_to:g} s cap: the "
          "tail is RIGHT-CENSORED and every number below is a lower bound")
  print("\ndeadline  would time out")
  for cand in CANDIDATE_DEADLINES:
    share = timeout_share(latencies, cand)
    mark = "   <- today" if cand == CALL_TIMEOUT_S else ""
    print(f"  {cand:5.1f} s   {share:6.1%}  "
          f"{'#' * round(share * 40)}{mark}")
  # What a deadline cannot buy back: answers that arrived and were no good.
  late = sum(1 for d_ in boss.decisions
             if d_.scripted and d_.source == "fallback:timeout")
  other = [d_.source for d_ in boss.decisions
           if d_.scripted and d_.source != "fallback:timeout"]
  print(f"\nresidual (arrived, unusable) : {len(other)}/{len(boss.decisions)}"
        + (f"  {dict(sorted({s: other.count(s) for s in set(other)}.items()))}"
           if other else "  -- every answer that arrived was valid"))
  if late:
    print(f"late even at {held_to:g} s        : {late}")


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
  parser.add_argument("--arm", choices=("guarded", "autonomous"),
                      default="guarded",
                      help="which ARM's prompt and context to measure "
                           "(issue #115). `autonomous` takes the three rails "
                           "off in the prompt, hides the verdicts code "
                           "computed, and offers the agent a STANDING ORDER "
                           "-- which is the capability gate this answers: "
                           "does it set one, what, and at what battery")
  parser.add_argument("--timeout", type=float, default=PROBE_TIMEOUT_S,
                      metavar="S",
                      help="wall seconds a call is held to here. NOT the "
                           f"deadline under test ({CALL_TIMEOUT_S} s): a "
                           "latency distribution measured through that "
                           "deadline is censored at it, and the tail is what "
                           "the deadline has to be chosen from")
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
  autonomous = args.arm == "autonomous"
  boss = Overseer(menu, thoughts=memory, model=model, backend=backend,
                  base_url=args.url, escalate_to=args.escalate_to,
                  # THE CAPABILITY GATE (issue #115). A standing order is
                  # the one affordance this arm gives the agent that no
                  # earlier arm had, and whether it USES it is a
                  # prompt-response question -- no physics, ten minutes.
                  autonomous=autonomous, standing_orders=autonomous,
                  # ⚠ NOT the deadline under test -- see `PROBE_TIMEOUT_S`.
                  # It also keeps the measurement honest in a second way:
                  # a call that outlives its deadline stays in flight, and
                  # `start` answers the NEXT one `fallback:busy` in
                  # microseconds, which would land in this distribution as
                  # a very fast decision.
                  timeout_s=args.timeout)
  prefix = boss.system[0]["text"]

  # The prefix now states the robot's NAME (issue #39), resolved from
  # $PLUGGY_ROBOT_NAME like a deployment's would be -- so the probe reports
  # who it measured, not just how big the measurement was.
  print(f"robot        : {boss.robot_name} (a {ROBOT_ROOT})")
  print(f"world        : {args.world}   (arm: {args.arm})")
  print(f"model        : {model} on {backend}")
  print(f"call held to : {args.timeout:g} s   (the deadline under test is "
        f"{CALL_TIMEOUT_S:g} s; a censored distribution cannot justify one)")
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
  latencies: list[float] = []
  orders: list[tuple] = []
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
    latencies.append(round(dt, 3))
    orders.append((state["battery"]["fraction"], decision.standing_order,
                   decision.action))
    print(f"{i + 1}. {decision.summary()}")
    print(f"   {dt:5.2f}s  in={delta['inputTokens']}"
          f" out={delta['outputTokens']}"
          f" cache_read={delta['cacheReadTokens']}"
          f" cache_write={delta['cacheWriteTokens']}")

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
  report_latency(latencies, boss, args.timeout)
  if boss.standing_orders:
    report_orders(orders)
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
