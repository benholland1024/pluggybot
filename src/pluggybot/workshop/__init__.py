"""The workshop (issue #168): where an agent-described tool is specified,
validated against the coupling envelope, built into a module and priced.

`spec.py` is what the agent emits and what it means; `validate.py` is
docs/ToolPattern.md §2 as code. Nothing here writes MJCF for the agent --
the agent names parts from `rack/catalog.py` and where they go, and the
generator in `rack/coupling.py` does the rest, contact rules included.
"""
