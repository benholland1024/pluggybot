"""The robot's imagination (issue #466, Track A of #465): a world of its own,
built from what it sensed, never from the world's source.

  scene.py    the scene language: a document of parts, joints and catches
  compile.py  a document compiled with the robot's own body into a world
  record.py   the record of a probe: what the robot sent and sensed
  rollout.py  a record's commands replayed in a world: the drivers' readings
  worker.py   the rollout in a process of its own, given only plain data
  fit.py      a template's unknowns found by rollouts against a record
  author.py   a language model writes a template's structure (stage 3)
  model.py    the rounds: author, place, fit, judge, revise
  ghost.py    the model on the wire, for the site to draw (`imagined`)

THE FENCE: nothing the robot's model is built from may read the world's
model of a mechanism. The worker is given a document and a record, as JSON
and float arrays, and nothing that could carry an `MjModel`; and nothing in
this package imports a mechanism's activity or the spike that chose it
(`tests/test_imagination.py` walks the imports). Nothing here is specific
to a lid.
"""
