# Training the bodies' policies (#377, #507)

A uv project of its own. Nothing here is imported by `pluggybot`, and none
of it enters the serving image: the served sim runs the exported policies in
plain numpy (`pluggybot.legs.policy`, `pluggybot.humanoid.policy`).
`quad_train` trains the quadruped and `humanoid_train` the humanoid; one
entry point registers both (`python -m quad_train.train <task>`).

- **mjlab 1.5.x** (MuJoCo Warp + PyTorch + rsl_rl), because it is the line
  that pins `mujoco ~=3.10.0`, the served sim's own version, and because the
  actuator honesty #377 needs is configuration there: a DC-motor torque-speed
  curve, command delay, randomised armature, friction and effort limits.
  Playground was ~2.5× faster on this card but carries none of that stock
  (the trainer comparison is in the PR for #377).
- It reads the body from `models/quadruped.xml` and `models/quadruped.json`,
  which `uv run python -m pluggybot.legs.model` writes: mjlab 1.5 caps numpy
  below pluggybot's, so the two cannot share an environment.

```bash
cd training && uv sync
# train (checkpoints every 50 iterations; --agent.resume True --agent.load-run
# <dir> carries on after an interruption)
WARP_CACHE_PATH=... .venv/bin/python -m quad_train.train Pluggy-Quad-Flat \
  --env.scene.num-envs 2048 --agent.max-iterations 2000 \
  --agent.logger tensorboard --agent.run-name flat --log-root <runs>
# export: numpy is checked against the ONNX file before anything is written
.venv/bin/python -m quad_train.export <run>/<run>.onnx ../models/quadruped_policy.npz \
  --gpu "<card>" --wall "<h:mm>" --envs 2048 --iterations 2000
# fly it in the served sim's physics, and hash it twice
cd .. && uv run python scripts/quad_spike.py --policy
uv run python scripts/quad_spike.py --determinism
```

**Tasks:** `Pluggy-Quad-Flat` (walking on flat ground) and
`Pluggy-Quad-Rough` (stairs, blocks to 0.15 m, rough ground, slopes; blind —
the critic sees the terrain, the actor does not), `-Perceptive` with the
actor shown the height scan. The stairs (#388) are flights of eight on a
0.28 m tread, risers from 0.10 to 0.26 m across the ten levels (the house's
0.18 in the middle); on a stair tile most commands walk straight out along a
flight (`quad_train/stairs.py`), because a robot is promoted for ending an
episode 4 m from its tile's centre and a turning, sidestepping command seldom
gets it there. `Pluggy-Quad-Getup` rises rather than springs (#389: the
stand pays from 2 s after the release, and charges on torque, joint speed
and the rise ramp up over training) and `Pluggy-Quad-Posture` tracks a
commanded height and tilt; SimNotes has both.

**The humanoid's tasks** (#507; `humanoid_train/`, its body read off
`models/humanoid.{xml,json}`, each joint its own RobStride part):
`Pluggy-Humanoid-Stand` balances on planted feet at a commanded posture
(the pelvis's height and the trunk's pitch, a crouch and a bow) while its
arms are SCRIPTED -- sent somewhere new every few seconds with their weight
fed forward, as the served reach drives them -- and its hands carry and
pull; shoves and pulls ramp up over the first 1200 iterations
(`DISTURBANCE_STAGES`, `--env.curriculum.disturbances.params.scale`), and
each sole is charged for its distance from where the episode's stance put
it (`feet_anchor`; SimNotes, "The humanoid stands, reaches and grasps").
`Pluggy-Humanoid-Walk` is mjlab's G1 velocity recipe on our body, on flat
ground: the actor blind to its base velocity as the quadruped's is, the arms
scripted more quietly than a reach moves them, the legs charged for meeting
each other, and the hips' yaw given twice the G1's tolerance -- with no
waist a turn is the hips' yaw alone, and at the G1's a policy asked 0.5
rad/s turned 0.08 (SimNotes, "The humanoid walks").
`Pluggy-Humanoid-Walk-Rough` carries the flat walker on over rough patches,
slopes and low blocks, every difficulty at once (`--agent.resume True
--agent.load-run '.*_walk2$' --agent.load-checkpoint model_2999.pt`): its
observations are the flat task's, so the checkpoint carries over, and
mjlab's terrain curriculum is off because it promotes a robot for ending an
episode 4 m from its tile's centre, which a turning command seldom does
(rough1's levels sat at 0.03 of 9). Export any of them with
`--body humanoid`, which records the leg joints it drives and each one's
gains. `humanoid_train/g1_check.py` flies mjlab's stock G1
velocity policy outside mjlab -- numpy, MuJoCo's C engine, at 5 ms and the
served 2 ms -- the pipeline's check on a body its maker walks (#507, rule 6).

Every reward's weight and parameter and every curriculum stage is a flag
(`--env.rewards.belly-down.params.budget-s 0`,
`--env.curriculum.gentle-rise-speed.params.stages.3.step 0`), so variants of
one task share a checkout: #389 flew three at once on one rented 4090.

⚠ **The GPU is shared** with the test suite, whose EGL renderers hold 3–4 GB
of a 6 GB card: stay at ≤ 2048 environments while it runs, or take
`flock -x ~/pluggybot-worktrees/.box.lock` for a full run.

## Renting a pod (`pod.sh`, #510)

`pod.sh` rents on Vast.ai, or on Runpod with `PROVIDER=runpod`. Its header
has the commands, and its `TERMS` the offers Vast is asked for. A card is
picked by what a million steps cost, never by its hourly rate, and the first
run on a card records its steps a second:

| card | where | $/h | steps/s | $ a million steps | flown |
|---|---|---|---|---|---|
| RTX 4090 | Runpod, secure cloud | 0.89 | 120,242 | 0.0021 | rough3, its iterations 3009–3308 |
| RTX 4090 | Vast, a Ryzen 9 3900X host | 0.414 | 103,472 | 0.0011 | rough3 again: its code, flags and checkpoint |
| RTX 4090 | Vast, the same host | 0.414 | 77,749 | 0.0015 | today's task, fresh, iterations 10–309 |
| RTX 5090 | Vast, a Ryzen 7 7800X3D host | 0.516 | 218,000 | 0.00066 | mjlab's stock G1 flat task, 3000 iterations in 22.5 min (#507) |
| RTX 5090 | Vast, the same host | 0.516 | 209,000 | 0.00069 | `Pluggy-Humanoid-Stand`, 4000 iterations in 31 min (#507) |

Two runs on one card share it rather than add to it: two humanoid runs on
one 5090 each took 1.0 s an iteration against 0.46 s alone.

A host moves the rate as much as its card does: the Vast 4090 ran at its full
450 W on 16 lanes, and was 14 % slower behind a 3.8 GHz Zen 2. A first run far
below the table is a slow host, so try another before trusting its price.

- **Both bill a stopped pod's disk until it is deleted,** and Vast bills
  traffic by the byte (a `setup` downloaded 3.4 GB, a cent).
- **At a zero balance** Runpod stops every pod and deletes one with no network
  volume (ours) with its checkpoints. Vast stops every instance, bills its
  disk on, and destroys it after a grace period when no card is saved. Keep
  the balance above what is running, and pull a long run's checkpoints as it
  goes.
- **The spend is read off the provider:** on Vast `vastai show user` (the
  credit) and `vastai show invoices-v1 --charges -s DAY -e DAY` (both dates:
  1.8.3 crashes on one); on Runpod its billing page.
