"""Where the robot's documents are kept (issue #217): one interface, and
the files on the volume as its only implementation.

Every document in `mind/text.py`'s registry -- the thought files, the
procedure library, the tool records -- is persisted through a `Store`, and
NOTHING ELSE in `mind/`, `procedure/` or `workshop/` touches the disk
(`tests/test_text.py` walks their syntax trees for a write). That is the
whole point of the seam: the memory mechanism is to be rethought once
(#221, "better than `.md` for everything"), and a rethink is one new
`Store` rather than a change per surface.

Two implementations, and their behaviour is asserted identical:

  `FileStore(root)`   the volume. A key is a relative path under `root`
                      (`Main.md`, `procedures/sun.procedure`,
                      `tools/scoop.tool.json`); a write is ATOMIC (a temp
                      file and `os.replace`, so a crash mid-write keeps
                      the old document); `archive` renames a document to
                      `<stem>.<n><suffix>` beside a fresh one (a true
                      death, issue #136: the evidence stays on the volume
                      where the operator looks).
  `MemoryStore()`     a dict, for a unit test and a demo with no state
                      directory. Reads back what was written and nothing
                      else; `archive` keeps the old text under the same
                      `<stem>.<n><suffix>` key.

The store knows NOTHING about writers, caps or verbs -- those are the
registry's, enforced at `text.admit` before a store is ever asked to
write. A store that refused a write on its own account would be a second
rule nobody can see from the table.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable


@runtime_checkable
class Store(Protocol):
  """What the registry's documents need of their storage."""

  def read(self, key: str) -> str | None:
    """The document's text, or None where there is none."""

  def write(self, key: str, text: str) -> None:
    """Replace the document whole. Present afterwards, atomically."""

  def remove(self, key: str) -> None:
    """Take the document away. Removing what is not there is nothing."""

  def keys(self, prefix: str = "", suffix: str = "") -> list[str]:
    """Every key under `prefix` ending in `suffix`, sorted."""

  def archive(self, key: str) -> str | None:
    """Put the document aside under a numbered name and return that
    name, or None where there was nothing to put aside."""


def archived_name(key: str, taken) -> str:
  """`History.md` -> `History.1.md`, or `.2.md` where that is `taken`.
  Shared by both stores so an archive on disk and in memory is one name."""
  p = PurePosixPath(key)
  n = 1
  while taken(str(p.with_suffix(f".{n}{p.suffix}"))):
    n += 1
  return str(p.with_suffix(f".{n}{p.suffix}"))


class MemoryStore:
  def __init__(self, texts: dict[str, str] | None = None) -> None:
    self.texts: dict[str, str] = dict(texts or {})

  def read(self, key: str) -> str | None:
    return self.texts.get(key)

  def write(self, key: str, text: str) -> None:
    self.texts[key] = text

  def remove(self, key: str) -> None:
    self.texts.pop(key, None)

  def keys(self, prefix: str = "", suffix: str = "") -> list[str]:
    return sorted(k for k in self.texts
                  if k.startswith(prefix) and k.endswith(suffix))

  def archive(self, key: str) -> str | None:
    if key not in self.texts:
      return None
    name = archived_name(key, lambda k: k in self.texts)
    self.texts[name] = self.texts.pop(key)
    return name


class FileStore:
  """The volume (`/var/lib/pluggybot` in the image; `$PLUGGY_THOUGHTS`).

  `aliases` maps a key to a path OUTSIDE the root: the pre-#38 goals file
  (`$PLUGGY_GOALS`) still means "this file is Goals.md", so a deploy that
  has been editing it keeps its goals. An alias is honoured with or
  without a root; a root-less store with no alias for a key reads and
  writes nothing for it, because reading a file should not create one.
  """

  def __init__(self, root: str | os.PathLike | None,
               aliases: dict[str, str | os.PathLike] | None = None) -> None:
    self.root = Path(root) if root is not None else None
    self.aliases = {k: Path(v) for k, v in (aliases or {}).items()}
    if self.root is not None:
      self.root.mkdir(parents=True, exist_ok=True)

  def path(self, key: str) -> Path | None:
    if key in self.aliases:
      return self.aliases[key]
    if self.root is None:
      return None
    return self.root / key

  def read(self, key: str) -> str | None:
    path = self.path(key)
    if path is None or not path.exists():
      return None
    return path.read_text()

  def write(self, key: str, text: str) -> None:
    target = self.path(key)
    if target is None:
      return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, target)          # a crash mid-write keeps the old file

  def remove(self, key: str) -> None:
    path = self.path(key)
    if path is not None and path.exists():
      path.unlink()

  def keys(self, prefix: str = "", suffix: str = "") -> list[str]:
    if self.root is None:
      return sorted(k for k in self.aliases
                    if k.startswith(prefix) and k.endswith(suffix)
                    and self.aliases[k].exists())
    base = self.root / prefix if prefix else self.root
    if not base.exists():
      return []
    return sorted(str(p.relative_to(self.root)) for p in base.rglob(f"*{suffix}")
                  if p.is_file())

  def archive(self, key: str) -> str | None:
    path = self.path(key)
    if path is None or not path.exists():
      return None
    name = archived_name(key, lambda k: (path.parent / PurePosixPath(k).name).exists())
    os.replace(path, path.parent / PurePosixPath(name).name)
    return name
