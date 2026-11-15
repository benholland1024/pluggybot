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


def _prose(markdown: str) -> str:
  """The text outside fenced code blocks (``` or ~~~, indented under a list
  item too), where a `#` is a heading and a `](...)` a link."""
  return re.sub(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[^\n]*$", "", markdown,
                flags=re.S | re.M)


def _anchors(markdown: str) -> set[str]:
  """GitHub's anchor for each heading, off its text as rendered (a link's
  words, no tags): lower case, punctuation dropped, spaces as hyphens, and
  `-1`, `-2` after a heading's repeats."""
  seen, anchors = {}, set()
  for heading in re.findall(r"^#{1,6} (.+)$", _prose(markdown), re.M):
    shown = re.sub(r"<[^>]+>", "", re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading))
    slug = re.sub(r"[^\w\- ]", "", shown.strip().lower()).replace(" ", "-")
    anchors.add(f"{slug}-{seen[slug]}" if slug in seen else slug)
    seen[slug] = seen.get(slug, 0) + 1
  return anchors


def _links(markdown: str) -> list[str]:
  """Every link's target in the prose, inline (with a title or none) or by
  reference; one with a scheme (`https:`, `mailto:`) is no file here."""
  text = _prose(markdown)
  inline = re.findall(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)", text)
  by_reference = re.findall(r"^ {0,3}\[[^\]]+\]:\s*<?([^\s>]+)", text, re.M)
  return [t for t in inline + by_reference if not re.match(r"[A-Za-z][\w+.-]*:", t)]


def _broken(source: Path, targets: list[str]) -> list[str]:
  """The targets that name no file, or a heading their file does not have."""
  broken, anchors = [], {}
  for target in targets:
    path, _, anchor = target.partition("#")
    doc = source.parent / path if path else source
    if not doc.exists():
      broken.append(target)
    elif anchor and doc.suffix == ".md":
      if doc not in anchors:
        anchors[doc] = _anchors(doc.read_text())
      if anchor not in anchors[doc]:
        broken.append(target)
  return broken


def _scenes(script: Path) -> set[str]:
  """The scenes `--view` and `--strip` offer: the keys of the script's
  `SCENES` dict."""
  for node in ast.parse(script.read_text()).body:
    target = node.target if isinstance(node, ast.AnnAssign) else (
      node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
    if getattr(target, "id", None) == "SCENES" and isinstance(node.value, ast.Dict):
      return {ast.literal_eval(k) for k in node.value.keys}
  raise AssertionError(f"{script.name} has no `SCENES = {{...}}` to read the scenes from")


def _strips(script: Path) -> set[str]:
  """The scenes whose filmstrip `--strip` draws: one with a branch of its
  own (`args.strip == "walk"`), a stage-2 scene `_scene_gen` lists, and the
  body's, the default. Any other falls through to another scene's."""
  drawn = {"body"}
  for node in ast.walk(ast.parse(script.read_text())):
    if (isinstance(node, ast.Compare) and getattr(node.left, "attr", None) == "strip"
        and isinstance(node.ops[0], (ast.Eq, ast.In))):
      drawn |= {c.value for c in ast.walk(node.comparators[0])
                if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    if isinstance(node, ast.FunctionDef) and node.name == "_scene_gen":
      drawn |= {k.value for d in ast.walk(node) if isinstance(d, ast.Dict)
                for k in d.keys if isinstance(k, ast.Constant)}
  return drawn


def test_the_first_screen_links_only_to_what_is_there():
  """The first screen is a visitor's (#714): a renamed doc, a moved picture,
  a heading reworded or a scene the script no longer offers fails here, not
  on a person following it."""
  readme = README.read_text()
  assert FIRST_SCREEN_END in readme, "the first screen ends where the developers' part begins"
  first = readme.split(FIRST_SCREEN_END)[0]
  targets = _links(first)
  assert [t for t in targets if not t.startswith("#")], \
      "the first screen links the docs and the picture"
  broken = _broken(README, targets)
  assert not broken, broken
  script = README.parent / "scripts" / "humanoid_spike.py"
  named = set(re.findall(r"humanoid_spike\.py --(?:view|strip) (\w+)", first))
  assert named, "the first screen names a scene to run"
  assert named <= _scenes(script), named - _scenes(script)
  stripped = set(re.findall(r"humanoid_spike\.py --strip (\w+)", first))
  assert stripped <= _strips(script), stripped - _strips(script)


def test_the_start_here_page_links_only_to_what_is_there():
  """CONTRIBUTING.md is where the first screen sends a person next (#714)."""
  page = README.parent / "CONTRIBUTING.md"
  targets = _links(page.read_text())
  assert targets, "the start-here page links the docs"
  broken = _broken(page, targets)
  assert not broken, broken
