"""The state diagram in README.md is read against the code (issue #221;
since #714 it sits under the first screen, which is for people): a diagram
in the README is where a developer looks first,
and one that has drifted from `lifecycle.State`, the death causes, the
event types or the memory verbs is worse than none.

Nothing here flies anything. The README's two Mermaid blocks are parsed
with regexes that know just enough of the syntax to pull names out.
"""

import ast
import re
from pathlib import Path
from typing import get_args

from pluggybot import lifecycle
from pluggybot.mind import events, text
from pluggybot.telemetry.protocol import DEATH_CAUSES

README = Path(__file__).resolve().parents[1] / "README.md"

#: Where the README's first screen, the one written for people (#714), ends.
FIRST_SCREEN_END = "## For developers"

#: Mermaid caps a `stateDiagram-v2` edge label at 200 px and a longer line
#: wraps (Chromium) or is cut off (Firefox) -- issue #259's clipped labels.
#: At the renderer's 16 px sans that is ~7 px a character; 26 leaves margin.
LABEL_LINE_CHARS = 26


def _mermaid_blocks() -> list[str]:
  blocks = re.findall(r"```mermaid\n(.*?)```", README.read_text(), re.S)
  assert len(blocks) == 2, "the README carries two diagrams: the loop and the memory"
  return blocks


def _state_names(block: str) -> set[str]:
  """Every node of a `stateDiagram-v2` transition, `[*]` excluded."""
  names = set()
  for line in block.splitlines():
    m = re.match(r"\s*(\S+)\s*-->\s*(\S+)", line)
    if m:
      names.update(n for n in m.groups() if n != "[*]")
  return names


def _label_lines(block: str) -> list[str]:
  """Every `<br/>`-separated line of every transition label."""
  lines = []
  for line in block.splitlines():
    m = re.match(r"\s*\S+\s*-->\s*\S+\s*:\s*(.*)", line)
    if m:
      lines.extend(part.strip() for part in m.group(1).split("<br/>"))
  return lines


def _diagram_verbs(block: str) -> set[str]:
  """`add / remove` pairs written into a memory node's label -- or one
  verb alone, for a document with an add and no remove (the ticket desk:
  closing is the operator's)."""
  return {v for pair in re.findall(r'<br/>(\w+)(?: / (\w+))?"\]', block)
          for v in pair if v}


def test_the_loop_diagram_names_exactly_the_lifecycle_states():
  loop, _ = _mermaid_blocks()
  assert _state_names(loop) == set(get_args(lifecycle.State))


def test_the_loop_diagram_names_every_death_cause_and_event_type():
  loop, _ = _mermaid_blocks()
  section = README.read_text().split("## Scripts")[0]
  for cause in DEATH_CAUSES:
    assert re.search(rf"\b{cause}\b", loop), f"death cause {cause!r} is not on a transition"
  for event in events.EVENT_TYPES:
    assert f"`{event}`" in section, f"event type {event!r} is not in the README's list"
  for event in events.INTERRUPTING_EVENTS:
    assert event in loop, f"interrupting event {event!r} does not leave PROCEDURE"


def test_every_loop_label_line_fits_the_renderers_width_cap():
  loop, _ = _mermaid_blocks()
  lines = _label_lines(loop)
  assert lines, "the loop diagram has labelled transitions"
  long = [line for line in lines if len(line) > LABEL_LINE_CHARS]
  assert not long, f"label lines over {LABEL_LINE_CHARS} chars would wrap or clip: {long}"


def test_the_memory_diagram_names_exactly_the_robots_verbs_and_real_states():
  loop, memory = _mermaid_blocks()
  # EVERY document's verbs, not the `.md` files' alone: the procedural tier
  # (the library) is memory too, and comparing against `text.FILES` is how
  # it went missing from the first draft.
  verbs = {v for s in text.DOCUMENTS for v in s.verbs}
  assert _diagram_verbs(memory) == verbs
  # The states drawn beside the tiers are the loop's, not invented for the
  # picture; a node like `RECALL / LOOK` would name two.
  drawn = set(re.findall(r"\b([A-Z][A-Z_]{3,})\b", memory))
  drawn -= {"LR", "TB", "MAIN", "GOALS", "TOP", "NOTES", "FIND", "HIST", "USE",
            "VISITOR", "PROCS", "TICKETS"}
  assert drawn <= _state_names(loop), drawn - _state_names(loop)


def _slug(heading: str) -> str:
  """GitHub's anchor for a heading: lower case, punctuation dropped, spaces
  as hyphens."""
  return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_the_first_screen_links_only_to_what_is_there():
  """The first screen is a visitor's (#714): a renamed doc, a moved picture,
  a heading reworded or a scene the script no longer offers fails here, not
  on a person following it."""
  readme = README.read_text()
  assert FIRST_SCREEN_END in readme, "the first screen ends where the developers' part begins"
  first = readme.split(FIRST_SCREEN_END)[0]
  targets = [t for t in re.findall(r"\]\(([^)\s]+)\)", first) if "://" not in t]
  files = [t for t in targets if not t.startswith("#")]
  assert files, "the first screen links the docs and the picture"
  assert not [t for t in files if not (README.parent / t).exists()]
  anchors = {_slug(h) for h in re.findall(r"^#+ (.+)$", readme, re.M)}
  assert not [t for t in targets if t.startswith("#") and t[1:] not in anchors]
  script = README.parent / "scripts" / "humanoid_spike.py"
  scenes = next({ast.literal_eval(k) for k in node.value.keys}
                for node in ast.parse(script.read_text()).body
                if isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "SCENES" for t in node.targets))
  named = set(re.findall(r"humanoid_spike\.py --(?:view|strip) (\w+)", first))
  assert named, "the first screen names a scene to run"
  assert named <= scenes, named - scenes
