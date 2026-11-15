"""The license travels with the code (#714; CLAUDE.md's Conventions): every
third-party license text that NOTICE names sits in `models/`, goes into the
wheel through `license-files`, and goes into the serving image. Nothing
here flies anything."""

import fnmatch
import hashlib
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: apache.org's LICENSE-2.0.txt, byte for byte: the same file MuJoCo's and
#: ONNX's wheels ship. The appendix's placeholders stay as they are; the
#: copyright line is NOTICE's.
APACHE_2_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def _project(path: Path) -> dict:
  return tomllib.loads(path.read_text())["project"]


def test_the_license_is_apache_2_in_both_projects():
  assert hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest() == APACHE_2_SHA256
  assert _project(ROOT / "pyproject.toml")["license"] == "Apache-2.0"
  assert _project(ROOT / "training" / "pyproject.toml")["license"] == "Apache-2.0"


def test_every_license_text_notice_names_travels_with_the_code():
  named = set(re.findall(r"\bmodels/[\w.-]+\.LICENSE\b", (ROOT / "NOTICE").read_text()))
  in_tree = {f"models/{p.name}" for p in (ROOT / "models").glob("*.LICENSE")}
  assert named == in_tree, f"named in NOTICE or in models/, not both: {named ^ in_tree}"
  wanted = {"LICENSE", "NOTICE"} | in_tree
  shipped = set(_project(ROOT / "pyproject.toml")["license-files"])
  assert shipped == wanted, f"license-files differs by {shipped ^ wanted}"
  copied = {src.rstrip("/") for src in _copied((ROOT / "Dockerfile").read_text())}
  uncopied = [path for path in sorted(wanted)
              if not any(src in ("", ".") or path == src or path.startswith(src + "/")
                         for src in copied)]
  assert not uncopied, f"no COPY brings {uncopied} into the image"
  kept_out = [path for path in sorted(wanted) if _ignored(path)]
  assert not kept_out, f".dockerignore keeps {kept_out} out of the image's build"


def _copied(dockerfile: str) -> set[str]:
  """Every source a `COPY` brings into the image: its words but the flags
  and the last, the destination."""
  sources = set()
  for line in re.findall(r"^COPY\s+(.+)$", dockerfile, re.M):
    words = [w for w in line.split() if not w.startswith("--")]
    sources |= set(words[:-1])
  return sources


def _ignored(path: str) -> bool:
  """Whether a `.dockerignore` pattern keeps the path out of the build: one
  that matches it or a folder it is in, name by name as Docker reads it
  (`*` within a name, `**` across names; a `!` exception is not read: none
  is written)."""
  patterns = [line.strip().strip("/").split("/")
              for line in (ROOT / ".dockerignore").read_text().splitlines()
              if line.strip() and not line.lstrip().startswith(("#", "!"))]
  parts = path.split("/")
  return any(_matches(pattern, parts[:i]) for pattern in patterns
             for i in range(1, len(parts) + 1))


def _matches(pattern: list[str], parts: list[str]) -> bool:
  if not pattern:
    return not parts
  if pattern[0] == "**":
    return any(_matches(pattern[1:], parts[i:]) for i in range(len(parts) + 1))
  return bool(parts) and fnmatch.fnmatchcase(parts[0], pattern[0]) \
      and _matches(pattern[1:], parts[1:])
