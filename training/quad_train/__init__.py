"""The training entry point (`python -m quad_train.train <task>`) and the
exporter (`python -m quad_train.export`, ONNX to the `.npz` the served sim
runs). It registers no task of its own: the humanoid's are `humanoid_train`'s.
Its name is the quadruped's, whose tasks lived here until it was retired
(#559; `quadruped-final` has them), kept because `pod.sh`, the README and
the docs' commands name it."""
