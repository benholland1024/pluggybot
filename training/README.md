# Training the humanoid's policies (#377, #507)

A uv project of its own. Nothing here is imported by `pluggybot`, and none
of it enters the serving image: the served sim runs the exported policies in
plain numpy (`pluggybot.humanoid.policy`). `humanoid_train` holds the
humanoid's tasks; the entry point that registers them and the exporter are
`quad_train`'s (`python -m quad_train.train <task>`, `python -m
quad_train.export`), the package named for the quadruped whose tasks lived
there until it was retired (#559; the tag `quadruped-final` has them).

- **mjlab 1.5.x** (MuJoCo Warp + PyTorch + rsl_rl), because it is the line
  that pins `mujoco ~=3.10.0`, the served sim's own version, and because the
  actuator honesty #377 needs is configuration there: a DC-motor torque-speed
  curve, command delay, randomised armature, friction and effort limits.
  Playground was ~2.5× faster on this card but carries none of that stock
  (the trainer comparison is in the PR for #377).
- It reads the body from `models/humanoid.xml` and `models/humanoid.json`,
  which `uv run python -m pluggybot.humanoid.model` writes: mjlab 1.5 caps
  numpy below pluggybot's, so the two cannot share an environment.

```bash
cd training && uv sync
# train (checkpoints every 50 iterations; --agent.resume True --agent.load-run
# <dir> carries on after an interruption)
WARP_CACHE_PATH=... .venv/bin/python -m quad_train.train Pluggy-Humanoid-Walk \
  --env.scene.num-envs 4096 --agent.max-iterations 3000 \
  --agent.logger tensorboard --agent.run-name walk --log-root <runs>
# export: numpy is checked against the ONNX file before anything is written,
# and the task it names is recorded with it
.venv/bin/python -m quad_train.export <run>/<run>.onnx ../models/humanoid_walk.npz \
  --task Pluggy-Humanoid-Walk --gpu "<card>" --wall "<h:mm>" --envs 4096 --iterations 3000
# fly it in the served sim's physics, and the pair's day twice, hashed
cd .. && uv run python scripts/humanoid_spike.py --track
uv run python scripts/determinism_spike.py --world home_humanoid --pair --sim-s 300
```

The quadruped's tasks (#377-#389: walking flat and rough, blind and
perceptive, getting up, holding a posture) are at `quadruped-final`; the
stairs recipe they carried (#388) is `humanoid_train/stairs.py`: on a stair
tile most commands walk straight out along a flight, because a robot is
promoted for ending an episode 4 m from its tile's centre and a turning,
sidestepping command seldom gets it there.

**The humanoid's tasks** (#507; `humanoid_train/`, its body read off
`models/humanoid.{xml,json}`, each joint its own RobStride part):
`Pluggy-Humanoid-Stand` balances on planted feet at a commanded posture
(the pelvis's height and the trunk's pitch, a crouch and a bow) while its
arms and its waist's yaw (#560's stage (c3)) are SCRIPTED -- sent somewhere
new every few seconds with their weight fed forward, as the served reach
drives them -- and its hands carry and pull; shoves and pulls ramp up over
the first 1200 iterations to 0.6 of their full size (`DISTURBANCE_STAGES`,
`DISTURBANCE_SCALE`: at full size it fell rather than stand, stand1's and
the waist's first retrains), and
each sole is charged for its distance from where the episode's stance put
it (`feet_anchor`; SimNotes, "The humanoid stands, reaches and grasps").
#544's retrains against the `pull` verb's 60 N (a one-hand draw, a capped
moment, long holds, a ramp, a fall charge, and `-Stand-Felt` shown each
hand's force) are in git, bee276e to 34f0ed3: none held more on the
robot's own senses, so `pull` asks at most 20 N instead (SimNotes,
"Holding the pull verb's 60 N at a hand").
`Pluggy-Humanoid-Walk` is mjlab's G1 velocity recipe on our body, on flat
ground: the actor blind to its base velocity as the quadruped's was, the arms
scripted more quietly than a reach moves them, the legs charged for meeting
each other, and the hips' yaw given twice the G1's tolerance -- with no
waist a turn is the hips' yaw alone, and at the G1's a policy asked 0.5
rad/s turned 0.08 (SimNotes, "The humanoid walks").
`Pluggy-Humanoid-Walk-Rough` carries the flat walker on over blocks, low
steps up and down and tilted plates (`--agent.resume True --agent.load-run
'.*_walk2$' --agent.load-checkpoint model_2999.pt`; its observations are
the flat task's, so the checkpoint carries over) under mjlab's terrain
curriculum, its tracking rewards sharper than the flat task's. ⚠ Its
terrain is BOXES ALONE: on a height field the box soles blew up (a world
in seven on the rough patches within 20 s), and two runs ended in it. ⚠
Blind, it stood still for any command that risked a fall, on flat ground
too -- backward, sideways -- while the stock widths paid 37 % for it
(SimNotes, "The humanoid walks"). `Pluggy-Humanoid-Walk-Seeing` (stage 5) is a walker that SEES THE GROUND:
its actor shown a height scan -- the terrain under a grid 1.6 x 1.0 m at
0.1 m, from 0.5 m behind the pelvis to 1.1 m ahead (`walk.SCAN`) -- over
the rough task's boxes and flights up and down at the house's 0.30 m tread
(risers 0.02-0.23 m across the ten rows: at 0.05 m on the easiest a
fresh walker never climbed a row), with #388's stairs recipe (on a
stair tile most commands walk straight out along a flight; a swinging foot
must clear the nose ahead). On the robot the scan is the head's D435 laid
into a map (`pluggybot.humanoid.scan`). ⚠ Carried on from walk2 (its
checkpoint widened to the new inputs, their weights zero), it kept walk2's
2.5 cm shuffle and stopped at the first riser for 800 iterations whatever
the swing's charges; it is trained from scratch, every robot starting on
the easiest row (seeing7, 8000 iterations); on mjlab's wide tracking terms
it stood still asked to sidestep or back off slowly. `-Walk-Seeing-Fine`
carries it on (`--agent.resume True --agent.load-run '.*_seeing7$'`) with
the tracking paid NARROWLY too, beside the wide terms: the planar velocity
within 0.15 m/s and the turn within 0.10 rad/s (`FINE_*`, the last round's;
widths and weights are flags, `--env.rewards.track-planar-fine.weight`,
`--env.rewards.track-yaw-fine.params.std` and the like), its command's
curriculum gone and the terrain from row five. Each round 1500-2000
iterations on the last, two ways at once (planar weight, turn weight, turn
width): fine1 (1.0, 0.5, 0.25) and fine2 (2.0, 1.0, 0.25) from seeing7;
fine3 (2.0, 3.0, 0.25) from fine2; fine4a (3.0, 3.0, 0.25) and fine4b (2.0,
4.5, 0.25) from fine3; fine5a (2.0, 3.0, 0.10) and fine5b (2.0, 3.0, 0.07)
from fine3; fine6a (3.0, 3.0, 0.10) from fine5a and fine6b (the same) from
fine4a. The walker that sees the ground is fine5a's
iteration 12500 (`models/humanoid_walk_seeing.npz`): of six checkpoints
100 iterations apart it alone passed Ben's flat-floor rule on the rule's
walks (SimNotes, "One walker that sees the ground"); it is not served.
mjlab writes the ONNX file of the LAST save alone: an earlier checkpoint's
is the trainer resumed from it with `learn` swapped for the save's export
(bit-identical on the last checkpoint). `Pluggy-Humanoid-Stairs` is the
same walker on flights alone (risers 0.02-0.22 m) and a little flat floor,
nine commands in ten walking straight out along a flight (stairs1, the
stairs' walker until the fine-tuned seeing walker climbed better; git has
it). `Pluggy-Humanoid-Getup`
gets up off the floor and falls on purpose: the legs' and the arms'
twenty-four joints the policy's (the waist held home), episodes starting on its back, its front, dropped any way up or
TIPPING from its stand at a shove's speed; a lift at the pelvis holds part
of its weight early in training (`ASSIST_STAGES`, to nothing by iteration
2400), the rise paid as it comes and the stand from 3 s, and a hit on
anything but a sole past two body weights charged. `-Getup-Head` carries a
get-up that stands on (`--agent.resume True --agent.load-run '.*_getup1$'`)
with the head's own hits charged 20 a body weight past half a body weight
(the other parts' 0.2 past two; `HEAD_WEIGHT`: the rewards are scaled by
the step, so a 5 kN hit costs a tenth of an episode's pay),
four episodes in ten a fall, no lift, every charge at full weight, and a
side among the starts: at the shared charge alone the get-up let the head
strike at 5.5-6.9 m/s falling forward or sideways, and with no side start
it lost getting up off one side (getup3, the served get-up until #679, is
getup1's 4450 iterations, 2000 without the side, then 1000 with it).
`-Getup-Wall` (#679) carries getup3 on (`--agent.resume True --agent.load-run
'.*_getup3$' --agent.load-checkpoint model_7448.pt`) with a box beside the
fallen body in a share of its episodes (`OBSTACLE`): a counter's top
(0.60-0.95 m) or a wall's (2.4 m), a mocap body sunk into the floor to it;
beside the trunk (at a body's front or back, its left or right, by a
share), beyond the head or the feet of a body lying, and into or along the
way a body tips, its face a gap past the body's own reach (so no start
begins inside it); its hits charged as the floor's. Trained on open floor
alone, the get-up never stood a body lying on its side with its front to a
face, which is how the counter's falls lay. Its stand is paid as the served
get-up hands back, within 10 deg of upright and touching nothing of the box
(paid from 25 deg, it stood leaning on the counter, and the served get-up
never handed back), and staying down is charged (`down`).
`-Getup-Wall-Mirror` trains on each sample and its mirror image as well
(rsl_rl's symmetry extension, `getup.mirror`): fine-tuned without it, the
get-up came to rise off its back by rolling to one side, and with a box on
that side lay still beside it. Both carry the recipe as their defaults, each
part measured: 20 s episodes (`WALL_EPISODE_S`: a slower stand beside a
face still earns), twice the side starts (`WALL_STARTS`), the entropy bonus
kept and small steps from getup3 (`WALL_PPO`: at no entropy the action's
noise fell to 0.06 and the get-up shook standing in the served physics; at
`desired_kl` 0.01 the get-up off its back beside a box fell from 0.99 to
0.51 in 450 iterations). The first round's pick, never served, is
getupwall10's iteration 8450: `-Getup-Wall-Mirror` from getup3, 1002
iterations in 38 min on a 5090 shared with a second run, chosen by the
training physics' scan below before the bar was read on it (SimNotes,
"Getting up beside a wall or the counter", has its readings). Later checkpoints stood more often from a side
facing the box and lost the back beside a box on its left: read a run's
checkpoints there with `python -m humanoid_train.scan RUN [EVERY] [N]
[BASE.pt] [TASK]` (fifteen starts against a box, on open floor and posed,
every checkpoint flown at once in its own block of worlds beside a base, in
TASK's env), and make any checkpoint the served npz with
`python -m quad_train.from_checkpoint` (mjlab writes the ONNX of the last
save alone; checked identical to `quad_train.export`'s of the same
checkpoint; `TASK WALL` name the run where the template's do not). ⚠ A
dict's tuple is no flag (`--env.events.reset-base.params.
obstacle.gap-m.along 0 0.15` is refused): change those in code.
`-Getup-Wall2-Mirror` (#679's second round) carries getupwall10 at 8450 on
with POSED starts beside the wall task's (0.2 to their 1.15,
`WALL2_STARTS`): the four served falls getupwall10 lost, as it ended them --
sitting still on the floor in three, propped on a wall in the fourth: fixed
points of a blind policy -- and as the get-up was handed each, moving
(`humanoid_train/getup_poses.json`, `humanoid_spike.py --fall-ends`), each
mirrored half the time, at any heading, one resting on its face or falling
against it started there; and a wall FAR in the way a body tips
(`OBSTACLE2`), which alone teaches nothing (the training physics' clean tip
stood 32 of 32 against it on getupwall10). `-Getup-Wall3-Mirror` (the third)
HANDS BACK as served (`HandBackAction`): upright and within 5 cm of the
stand's height for 0.5 s, the served standing policy
(`models/humanoid_stand.npz`) has the legs, the arms held at its pose, until
a fall the served watch would take; handed back is paid and waiting
charged, the stand paid only handed back, the get-up's own joint and torque
charges charged only while it holds the legs, and no gyro test -- ⚠ each,
and what went wrong without it, in SimNotes ("The wall get-up's second and
third rounds"): never handed back, a get-up stands where the standing policy
cannot keep it.
`models/humanoid_getup.npz`, SERVED since #679, is getupwall14's iteration
9850: `-Getup-Wall3-Mirror` from getupwall12's 9050 (`-Getup-Wall2-Mirror`
from getupwall10's 8450), 800 iterations in ~25 min on a 5090 shared with a
second run, chosen by the scan in `-Getup-Wall3`'s physics before the bar was
read on it (SimNotes, "The wall get-up's second and third rounds"). Its
second run, the same from getupwall10, never learned to rise from a sit (as
the second round's other run did not): the sit was learned once, by
getupwall12, and carried on. ⚠ It trained with a bug since fixed (getup.py's
review): a reset that drew a posed start gave every start drawn with it that
pose's joint speeds (up to 7.7 rad/s), so some of its lying and tipping
starts began with their joints swinging.
⚠ A get-up SEES THE JOINTS IT
DRIVES: over every joint the waist made 80 inputs, where getup3's 78 carry
on; and mjlab's metadata lists every joint of the body, so the export writes
the joints its task's actor names (`quad_train.export.seen_joints`, as the
carrying walker's, #682), or the served runtime builds 80 inputs for a
78-input net.
⚠ The head is its fourteen parts (#581's stage 3), each charged as the
head. Export any of
them naming its task (`--task`), which records the joints it drives and
each one's gains (`--joints all` for the get-up, its arms at their served
scale). `humanoid_train/g1_check.py` flies mjlab's stock G1
velocity policy outside mjlab -- numpy, MuJoCo's C engine, at 5 ms and the
served 2 ms -- the pipeline's check on a body its maker walks (#507, rule 6).

`Pluggy-Humanoid-Walk-Carry` (#682's stage B) is walk2 fine-tuned to carry
(`--agent.resume True --agent.load-run '.*_walk2$' --agent.load-checkpoint
model_2999.pt`): each arm in segments of 4-15 s hanging, moving as walk2's
did, or holding one pose of the carry box the whole segment
(`walk.CarryArms`; `--env.actions.arms.close-share 0.5` draws half of the
carries near the served carry pose, carry3's), a 0-1 kg payload at each grip
site drawn at every reset (`robot.get_carry_spec`'s welded bodies), the actor
seeing walk2's 24 joints in walk2's order so its checkpoint carries over (the
exporter writes the joints a task sees). ⚠ ITS MODEL IS RIGID AT THE WAIST, as
walk2's body was: on the waist's joint the pelvis's yaw rang against the trunk
at the policy's rate, and carry1 phased that ringing to the instant the reward
samples the root -- by the reward it tracked its turn, by its heading it spun
(SimNotes, "A walker trained to carry"). carry2 is 2500 iterations from walk2,
carry3 1500 more from carry2 at a close share of 0.5;
`models/humanoid_walk_carry.npz` is not served.

Every reward's weight and parameter and every curriculum stage is a flag
(`--env.rewards.belly-down.params.budget-s 0`,
`--env.curriculum.gentle-rise-speed.params.stages.3.step 0`), so variants of
one task share a checkout: #389 flew three of the quadruped's at once on
one rented 4090.

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
| RTX 4090 | Runpod, secure cloud | 0.89 | 120,242 | 0.0021 | the quadruped's rough3, its iterations 3009–3308 |
| RTX 4090 | Vast, a Ryzen 9 3900X host | 0.414 | 103,472 | 0.0011 | the quadruped's rough3 again: its code, flags and checkpoint |
| RTX 4090 | Vast, the same host | 0.414 | 77,749 | 0.0015 | the quadruped's walking task then, fresh, iterations 10–309 |
| RTX 5090 | Vast, a Ryzen 7 7800X3D host | 0.516 | 218,000 | 0.00066 | mjlab's stock G1 flat task, 3000 iterations in 22.5 min (#507) |
| RTX 5090 | Vast, the same host | 0.516 | 209,000 | 0.00069 | `Pluggy-Humanoid-Stand`, 4000 iterations in 31 min (#507) |
| RTX 5090 | Vast, three hosts in South Korea | 0.490-0.516 | 198,000-205,000 | 0.0007 | `Pluggy-Humanoid-Stand` and `-Stand-Felt` (stand7-stand13), 4000 iterations in 32-33 min each (#544) |
| RTX 5090 | Vast, another 7800X3D host | 0.490 | 215,000 | 0.00063 | `Pluggy-Humanoid-Walk`, 3000 iterations in 23 min (#507) |
| RTX 5090 | Vast, two 7800X3D hosts in South Korea | 0.516 | 193,000 | 0.00074 | `Pluggy-Humanoid-Stand` on the waist (waist1-waist17), 4000 iterations in 34 min; two runs to a card took 68 min each, 16 % slower than one after the other (#560 (c3)) |
| RTX 5090 | Vast, the same host | 0.490 | 122,000 | 0.0011 | `Pluggy-Humanoid-Walk-Rough` (rough4: 50 tiles of boxes), 2000 iterations in 27 min (#507) |
| RTX 5090 | Vast, a Ryzen 9 9950X host | 0.641 | 114,000 | 0.0016 | `Pluggy-Humanoid-Walk-Seeing` from walk2 (50 tiles of boxes and flights, the 187-ray scan) alone (#507) |
| RTX 5090 | Vast, an EPYC 7B12 host (DLPerf 86) | 0.540 | 11,000 | 0.014 | the same, and rough4's task: a slow card (above) |
| RTX 5090 | Vast, a Ryzen 5 9600X host | 0.689 | 181,000 | 0.0011 | `Pluggy-Humanoid-Stairs` (stairs1: 30 tiles of flights), 8000 iterations in 72 min alone (#507) |
| RTX 5090 | Vast, the same host | 0.689 | 249,000 | 0.00077 | `Pluggy-Humanoid-Getup-Head` (getup3, a plane), 1000 iterations in 6.6 min alone (#507) |
| RTX 5090 | Vast, a Ryzen 9 9950X host | 0.647 | 96,000 | 0.0019 | `Pluggy-Humanoid-Getup` (getup1) beside a seeing walker's run on the same card (#507) |
| RTX 5090 | Vast, an EPYC 9B14 host | 0.633 | 95,000-98,000 | 0.0018 | `Pluggy-Humanoid-Walk-Seeing-Fine` (fine2, fine3, fine4a, fine5a, fine6a), 1500 iterations in 25 min alone (#507) |
| RTX 5090 | Vast, a Ryzen 7 7800X3D host | 0.516 | 108,000 | 0.0013 | the same (fine4b, fine5b, fine6b), 1500 iterations in 23 min alone (#507) |
| RTX 5090 | Vast, a Ryzen 7 7800X3D host in South Korea | 0.516 | 155,000 | 0.00092 | `Pluggy-Humanoid-Getup-Wall` (getupwall1), 1500 iterations in 16 min alone; two runs to the card ~65,000 each (getupwall10: 1002 iterations in 38 min) (#679) |
| RTX 5090 | Vast, another 7800X3D host in South Korea | 0.516 | 70,000 each, two to the card | 0.0010 | `Pluggy-Humanoid-Getup-Wall2-Mirror` and `-Wall3-Mirror` (getupwall12-15), 1000 iterations in 22-25 min with two runs to the card (#679's second and third rounds) |
| RTX 5090 | Vast, a Ryzen 7 7800X3D host | 0.516 | 118,000 | 0.0012 | `Pluggy-Humanoid-Walk-Carry` on the waist's joint (carry1, stopped: #682) at 0.83 s an iteration |
| RTX 5090 | Vast, an EPYC 7K62 host (DLPerf 200) | 0.514 | 61,000 | 0.0023 | `Pluggy-Humanoid-Walk-Carry` (carry2), 2500 iterations in 68 min alone: a slow host by its CPU |
| RTX 5090 | Vast, a Xeon w7-3455 host | 0.514 | 101,000 | 0.0014 | `Pluggy-Humanoid-Walk-Carry` (carry3), 1500 iterations in 25 min alone |

Two runs on one card share it rather than add to it: two humanoid runs on
one 5090 each took 1.0 s an iteration against 0.46 s alone.

A host moves the rate as much as its card does: the Vast 4090 ran at its full
450 W on 16 lanes, and was 14 % slower behind a 3.8 GHz Zen 2. A first run far
below the table is a slow host, so try another before trusting its price.
⚠ One machine's 5090s (EPYC 7B12) scored a DLPerf of 86 against the other
5090 offers' ~195, and ran `Pluggy-Humanoid-Walk-Seeing` and rough4's own
task at a TENTH of another 5090's speed (114,000 steps/s on a Ryzen 9 9950X
host): `pod.sh create` passes over an offer under 70 % of its card's median
DLPerf (`DLPERF_SHARE`). A host's download speed varies as much: one took 40
minutes to install what another installed in 5.

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
