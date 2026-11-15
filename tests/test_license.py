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
  copied = _copied((ROOT / "Dockerfile").read_text())
  assert {"LICENSE", "NOTICE"} <= copied, "the image carries LICENSE and NOTICE"
  assert "models/" in copied or "models" in copied, "the image carries models/ and its texts"
  kept_out = [path for path in wanted if _ignored(path)]
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
  """Whether a pattern of `.dockerignore` matches the path or a folder it
  is in (a `!` exception is not read: none is written)."""
  patterns = [line.strip().strip("/") for line in (ROOT / ".dockerignore").read_text().splitlines()
              if line.strip() and not line.lstrip().startswith(("#", "!"))]
  parts = path.split("/")
  prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
  return any(fnmatch.fnmatch(prefix, pattern) for pattern in patterns for prefix in prefixes)
