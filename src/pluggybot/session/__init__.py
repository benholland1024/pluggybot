"""User control (issue #530): a person's procedure, in the robot's own
language, drives the humanoid in a world of its own -- one attempt of the
attempt kit's task (`evaluation.attempts`), streamed live to the site and
graded as the robot's attempts are. Never Luca or Rowan, and never the
served world: every action of a mind comes from its LLM (CLAUDE.md).

  wire      what a session's stream carries beyond the world's frames, and
            what the host and the site say to each other
  attempt   one attempt flown and streamed: the child process a run is
  host      the long-lived process on the server that the site hands runs
            to, one at a time
"""
