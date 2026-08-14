"""PluggyWorld wire protocol (webserver v0): scene transpiler + recorder.

pluggybot is the protocol's PRODUCER: the scene JSON and telemetry JSONL
emitted here are the canonical artifacts (committed under protocol/), and
the website repo vendors recorded copies into its own tests. Design doc:
rooftop-media-2026/docs/pluggyworld.md, § "The scene protocol".

Import from the submodules directly (scene, recorder, protocol) -- this
package init stays empty so `python -m pluggybot.telemetry.scene` does not
re-import the module it is about to run.
"""
