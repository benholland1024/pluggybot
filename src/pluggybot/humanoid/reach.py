"""Reaching (issue #507, stage 2): a hand sent to a pose in the stance frame
while the standing policy balances (`policy.StandDriver`), as a
manipulator's reach is made -- the posture and the arm's last joints
PLANNED on the robot's model of itself (`plan`), the arm MOVED there in
joint space to a pre-grasp pose short of the target, then the hand
APPROACHING it along a straight line, its joints solved again every
command as the trunk sways (`arms.SelfModel`), and HOLDING there. A
retreat goes back out along that line, then home in joint space by a way
judged on the same model (`way_home`). The grippers are driven and read
beside it (`arms.Gripper`). Everything it reads is the robot's own senses.

⚠ From a hanging arm a big move solved in one jump falls into the wrong
basin (the shoulder thrown back, the elbow over): a hanging arm is nearly
singular for a hand moving up. So the plan's solve starts from several
arms (`SEEDS`) and the motion's from wherever the arm last was.

A hand that is OUT moves on CARRIED (#493): along a straight line to a
point, or through a path's points in turn (`carry`), from where the hand
IS, no faster than `HOLD_SPEED`, in the least-bent posture that reaches
(`posture_for`) or follows the path (`path_for`), stopping where the
arm's own solve misses rather than command the miss. Its turn is the one
it is asked to keep -- a path may turn it point by point -- and never
yields to what twists it: a drop handle in a hand that pivoted, or whose
turn followed a twist past 4 or 10 deg, swung round its pin under the
first push and buckled (SimNotes, "The humanoid opens the chest").

`Reach.step()` runs once a physics step, after the standing policy's and
before `mj_step`; the arms take new commands every `EVERY_S`.
"""

from dataclasses import dataclass
import itertools
import math

import mujoco
import numpy as np
from scipy.optimize import least_squares

from pluggybot.humanoid import friction
from pluggybot.humanoid import model as hm
from pluggybot.humanoid.arms import (RANGE_MARGIN, REACH_POS_TOL, REACH_ROT_TOL, Arm, Gripper,
                                     Motion, HandGoal, SelfModel, Waist, _rot_error,
                                     kept_value, value_back)
from pluggybot.humanoid.policy import HEIGHT_RANGE, PITCH_RANGE, StandDriver


#: The arms' commands are refreshed this often, s: 100 Hz, inside the RS
#: MIT mode's 1 kHz, and a trunk's sway between two is under a millimetre
#: at the hand.
EVERY_S = 0.01
#: A hand's fastest along its line, m/s, and its grip's fastest turn, rad/s
#: (a person's careful reach is 0.2-0.5 m/s).
HAND_SPEED, HAND_TURN = 0.2, 1.0
#: An arm moving in joint space does so no faster than this at its fastest
#: joint, rad/s, every joint arriving together.
JOINT_RATE = 1.0
#: The pre-grasp: this far short of the target, m, back along the way the
#: fingers point.
APPROACH_M = 0.08
#: A plan's slack: the arm must also reach this much further out, m, away
#: from its shoulder -- a goal planned at full stretch was 2.8 cm out of
#: reach of the posture the policy held, 2 cm under the one asked. It is
#: slack over the robot as it stands (`Lean`), never the lean itself: kept
#: over a statue balanced at the soles' middle, the live trunk stood 1.6-6.6
#: cm behind it and the hand arrived short (#627).
REACH_MARGIN_M = 0.015
#: A hand whose reach's posture, read as the robot stands with its arm out
#: at its pre-grasp, does not put the goal in reach takes another this many
#: times at most before it goes in (`Reach._look`, #627): the lean an arm
#: brings is the standing policy's own and no table predicts it -- over
#: random arms the CoM moved -2 to +5 cm, a fit to the arm's angles still
#: off 0.7-1.7 cm rms -- but the robot reads it, out, to 0.1-0.2 cm.
LOOKS = 2
#: A hand's force (`force`) is the WHOLE BODY's reading while the hand is at
#: work in one of these phases and the standing policy has the legs
#: (`standing`; `arms.SelfModel.felt`, #547), else its arm's alone
#: (`SelfModel.hand_force`). ⚠ ALONG A STRAIGHT ARM THE ARM IS BLIND: 20 N
#: at the grip read 5 pushing away and 2 away and down, where the whole
#: body read 19.7-20.2 every way of 14. ⚠ WALKING, A SOLE IS IN THE AIR,
#: which the balance has planted: a swinging leg's own torques read up to
#: 400 N (`humanoid.odometry`).
FELT_PHASES = ("approach", "hold", "pull", "blocked", "carry", "retreat")
#: WHAT THE ROBOT READS of a hand's force (`sensed_force`: `Body.hand_force`,
#: a procedure's `read`, its stop conditions, a verb's report) takes the
#: body's own motion in (`arms.Motion`, #631) while the hand goes its own way
#: -- swung out or home, on its line in or out -- never in these phases,
#: where it holds or is held and is read as still: pressing slowly on what it
#: holds, its motion is smaller than its estimate's noise. ⚠ THE BODY'S OWN
#: LOOPS NEVER READ IT (`force`: the reach's stop on a push, the ease, the
#: force move's loop): taken into `force` everywhere, the hands checks lost
#: 18 set-outs of 480 for 5 won on one pod (the cube 46 of 96 where 53, a
#: pull's give following the estimate's noise); kept out of these phases,
#: still 7 for 1, its approaches stopped on a push at other moments, and 3
#: drawers fell.
HELD_PHASES = ("hold", "pull", "blocked", "carry")
#: A hand on its way in STOPS where it is (`blocked`) once its own solve
#: misses its command by this, m -- the goal out of reach from the posture
#: the policy truly holds -- or once the force it reads at the hand
#: (`force`) passes this, N, pushing on something it was not sent to. ⚠
#: Unstopped, a crouch that sat further back than the plan's statue left a
#: chest's knob 10.7 cm out of reach, and the arm drove the gripper into
#: the counter's edge until it pushed the robot over backwards.
BLOCKED_M, BLOCKED_N = 0.03, 30.0
#: ...for this long, s: the force read off the drivers assumes the body is
#: still, and a deep bow's folding read 81.5 N at the hand for an instant
#: as its approach began. A push lasts; a transient does not.
BLOCKED_S = 0.15
#: A hand on its WAY OUT to its pre-grasp (`move`, in joint space) stops
#: once what its arm reads would TIP THE BODY TOWARD IT past this, N m --
#: its pull toward the body at the hand's height, less its press down at
#: the hand's reach (`_tipping`) -- held where its drivers read its joints,
#: never where they were commanded (`_struck`; SimNotes, "A hand on its way
#: out that meets something"). ⚠ Unstopped, a jaw caught on a chest's
#: bracket was driven on until the robot went over its toes (#708): such
#: catches pass this 0.09-0.20 s after the touch, 10 N m at most before it.
#: ⚠ NEVER ON A PUSH OR A PRESS: our door's way out drags a jaw over the
#: door's top pressing down on it, and stopped on the whole reading or on
#: its pull alone the door passed fewer.
OUT_TIP_NM = 20.0
#: ...past it this long running, s, while the body still takes the posture
#: it was asked (`POSTURE_TOL`); in it, the tick it passes (`_tipped`). ⚠ A
#: TRANSIENT IS NO CATCH: the reading assumes the body still, and as the
#: standing policy stopped a deep bow within 0.01 s (the cube check's
#: set-out 38, its waist 0.05 rad short of its posture), the arm's drivers
#: caught the arm up and read 44 N at a hand that touched nothing, past the
#: line for two of the arm's ticks (0.02 s). ⚠ NO HOLD IN THE POSTURE:
#: there #708's catches pass the line, and held 0.02 s past it at the
#: chest's draws 321-336, 3 of 16 fell where 1 did, the hand pressing the
#: bracket at 41-104 N by the stop.
OUT_TIP_S = 0.03
#: A hand held whose force pushes on along its fingers past this, N, is
#: eased before its jaws shut (`ease`, #560): pressed on a cube's board at
#: 40-120 N, the jaws dragged across it and the feet swivelled 4-6 deg
#: (SimNotes, "Setting a table").
EASE_N = 10.0
#: The posture's own loop: the pelvis's height and the trunk's pitch as the
#: robot's model of itself has them (its legs' encoders, its IMU) held to
#: the plan's by trimming what the policy is asked -- this much of the
#: miss a second, at most this far, m and rad. Unheld, the policy stood 2
#: cm under the height it was asked.
TRIM_RATE, TRIM_MAX = 1.0, (0.05, 0.15)
#: A hand's approach starts once the posture the robot measures is this
#: near the plan's -- the pelvis's height, m, the trunk's pitch and the
#: waist's yaw, rad -- and its trunk is still -- its gyro under
#: `TRUNK_STILL_RATE`, rad/s -- or this long after its command arrived, s: a
#: full-stretch reach begun before the body had folded into a deep bow
#: stopped short. ⚠ The force read off the drivers assumes the body still
#: (`BLOCKED_N`): the floor's approach begun as its bow stopped turning
#: (0.59 rad/s) read 492 N at the hand for 0.2 s and stopped 6 cm short of
#: it; begun under 0.05, 62 N for an instant, and it held there (#520).
POSTURE_TOL, TRUNK_STILL_RATE, POSTURE_WAIT_S = (0.025, 0.06, 0.02), 0.05, 4.0
#: The waist turns to its posture's angle no faster than this, rad/s: the
#: trunk's pitch's ramp (`policy.PITCH_RATE`) -- and is set going and stopped
#: at this, rad/s^2 (`turned`), as training's scripted waist turns it
#: (`training/humanoid_train/stand.py`): the stand is served the turns it
#: trained on. (Set going at once, the stand that knew no waist turned its
#: soles no more, 60 deg with both arms out: 1.85 deg to 1.81.)
WAIST_RATE, WAIST_ACCEL = 0.6, 1.5
#: A hand that is out moves no faster than this when carried,
#: m/s: at the reach's own 0.2 a lid's pull outran the posture, which
#: comes a second or two behind its command, and the arm's solve missed
#: its command by 2-5 cm all the way up (#493).
HOLD_SPEED = 0.06
#: A path's slack (`path_for`): each of its points further from the
#: shoulder than its first is also reached this much further AHEAD, m: the
#: lean its load still brings. Lifting, the live trunk stood 1.5-2 cm behind
#: a statue balanced at the soles' middle all the way up a lid's arc and 2 cm
#: more at its top, where a path planned with no slack stopped at 57 deg
#: (#493); the statue stands as the robot reads itself standing now
#: (`Lean`, #627), and only this last 2 cm is to come.
LIVE_MARGIN_M = 0.02
#: ...and each is followed within this, m: a solve trades position for the
#: turn it holds (`arms.IK_ROT_WEIGHT`), and down a lid's arc, the wrist near
#: its stop, left points 1.1-1.2 cm off that the live hand passed -- which
#: stops only past `BLOCKED_M`.
PATH_POS_TOL = 0.02
#: A carried hand whose solve misses its command by this, m, or
#: its turn by this, rad, takes a posture that reaches it, planned at most
#: once this often, s (`posture_for` is a fraction of a second of the
#: processor's). ⚠ THE TURN TOO: a pull up a lid's arc standing held its
#: place to a millimetre and missed its turn by 0.19-0.2 rad, its wrist at
#: its stop pushing the hand 2 cm sideways, until it stopped (#493).
REPLAN_M, REPLAN_ROT, REPLAN_S = 0.015, 0.12, 1.0
#: A carried hand's grip may turn this far off its plan, rad: a held drop
#: handle swings on its own pin, so 10 deg of it moves the knob under a
#: centimetre, which the arm's give takes up -- held to a reach's 0.1, a
#: lid's arc reached within a millimetre failed on the wrist's last 7-14
#: deg (#493).
HOLD_ROT_TOL = 0.2
#: A pull's arm joint its driver reads within this of an end of its range,
#: or past it, rad, is AT ITS END (`_on_stops`, #744's `pull.joint_end`). X5's
#: chest flights flown again with the truth beside the reading: all 640 joints
#: pressing their limit as 1,096 pulls ended (MuJoCo's limit is soft: pressed,
#: up to 47 mrad past it) read at most 2 mrad inside their end; at
#: `arms.STOP_RAD`'s 20 mrad, 476 joints pressing nothing read as at their end
#: too, a shoulder's yaw up to 21 mrad short among them, and at this, 118, each
#: within 7 mrad of it (SimNotes, "What a pull and a reach read of the body").
JOINT_END_RAD = 0.005
#: A carried hand has arrived once within this of its goal, m; one whose
#: command arrived and which is still short of it this long after, s, is
#: held back by something (`carry`'s report says how far short).
ARRIVE_M, HELD_BACK_S = 0.01, 1.0
#: A carried hand's solve that misses its command by this, m, has
#: diverged and is never commanded; and a move is given this long, s, to
#: begin before a miss can end it.
WILD_M, SETTLE_S = 0.06, 0.3
#: A PULL (`pull`, #525): a force move. The hardest it pushed and whether it
#: is held fast are read off the hand's reading along its way over this
#: long, s; a hand that has gone no further than this, m, in this long, s,
#: while it pushes with this share of its force -- or that is not going
#: (below) with its command as far ahead as it may go -- is held fast...
MEAN_S, FAST_M, FAST_S, FAST_SHARE = 0.1, 0.004, 0.6, 0.8
#: ...the force it is CLOSED on is that reading over this long, s, carried
#: this far ahead, s, at the rate it rises, read over this long, s: the move
#: slows before the force reaches its ask, not once it has. ⚠ Closed on
#: `MEAN_S`'s reading as it stood, it built over 100 N/s on a locked bar and
#: put 1.7 times its ask on it before it read the ask -- closed even on the
#: bar's true force (#547)...
ADMIT_SMOOTH_S, AHEAD_S, RISE_SMOOTH_S = 0.03, 0.1, 0.08
#: ...and it backs off past its force up to this many times its speed: no
#: faster than it walked in, the bar took 1.3 times a 10 N ask while a body
#: leaning on the held hand kept the force up, and 1.2 at twice (#547).
BACK_SPEED = 2.0
#: ...a hand GOES while it gets this far along its way, m, in this long, s
#: (and at that pace over `FAST_S`, or it is not held fast pinned). One that
#: does not -- HELD -- whose command is out of the arm's reach takes no other
#: posture where the arm reaches the hand itself: its command stays where
#: the arm last reached -- but backs off toward the hand past its force, its
#: arm's joints toward those that reach it (held still, a pull on a locked
#: bar stood at 21-24 N for 20 while its body leaned on it), and pinned
#: there pushing under `FAST_SHARE` of its force, it is out of reach, not
#: held fast (in free air, a pull past the arm's reach was "held fast" after
#: 1.1 cm, #547). ⚠ AGAINST A HAND HELD, THE COMMAND'S LEAD IS THE
#: SOLVE'S MISS: postures taken for it bowed, crouched and stood the body up
#: with the hand on a locked bar, 30-100 N on the bar for a pull asked
#: 10-20, and 7 ways in 14 never ended (#547). ⚠ BUT A HAND ITS ARM CANNOT
#: HOLD WHERE IT IS TAKES ONE: a lid's knob gripped 15-18 mm past the arm's
#: reach opened only from another posture, before the lid had moved; and
#: no lead tells the two apart -- moving lids took theirs with the command
#: up to 17 mm ahead, and on the bar a push took six with it under 15.
#: ⚠ ALONG ITS WAY, AND FASTER THAN A GRIP SLIDES: held on the bar, the
#: hand's sensed place moved 4 mm in a tenth of a second, and its grip slid
#: 3-7 mm a second, 10-12 as a push leaned the body in; but held fast at its
#: force off its way alone, a door swung by its hinge across the hand's way
#: was let go of...
GO_ON_M, GO_S = 0.015, 1.0
#: ...its command leads it by at most this along its way, m, whatever the
#: force asks, and moves no faster than this times its speed along its way
#: and across it: what runs away with the hand -- a lid falling toward the
#: robot across a way down -- the arm's spring holds back...
LEAD_MAX_M, ACROSS_SPEED = 0.08, 2.0
#: ...A HAND THAT HOLDS NOTHING TAKES A POSTURE ONLY WHILE IT READS LIGHT
#: (#699; SimNotes, "A pull's posture and the hand it rests on"): at its
#: start or on its way, only while it reads no more than this many times the
#: pull's ask, whichever way -- as it reads as the pull begins, over `MEAN_S`
#: on its way -- an ask of at least this, N: the most a procedure's `pull`
#: asks (`steps.PULL_MOST_N`), what the standing body holds at a hand
#: (#544), so 40 N for every pull a robot writes -- and one that is coming
#: is GIVEN UP where the hand reads past it: the body BACKS OFF toward the
#: posture the pull began in until the hand reads no more than the ask
#: again (of this at least: 20 N), is held there, and that pull takes no
#: other. ⚠ A posture bows, crouches or turns the body over a hand that
#: rests on something, and where the arm's solve can no longer follow, its
#: joints stand and the body bears on the hand: 45-150 N down on a cupboard
#: door's top for a 15-20 N pull. ⚠ NOT AT THE ASK ALONE: a hand reading
#: little as its posture is asked is loaded as the body moves. ⚠ A FORCE,
#: NOT A SHARE OF A SMALL ASK: at twice the ask (20 N for a 10 N press) more
#: of what passes was refused. ⚠ BACK ONLY AS FAR AS UNLOADS IT: held where
#: it had got to, a body 1 cm into its posture stayed leant on what it held
#: at 80-92 N for 10 s; sent all the way back, the waist swung a low arm
#: into the counter (935 N). ⚠ A HAND THAT GRIPS WHAT IT PULLS TAKES ITS
#: POSTURES AS IT DID (the grip it began with, `gripped`): two of our door's
#: 72 held-out passes opened only by the body leaning on its lever (45-155 N
#: read), though a gripped fork or knob falls the same way (SimNotes).
PRESSED_SHARE, PRESSED_ASK_N = 2.0, 20.0
#: ...AND A PROCEDURE'S CARRY (`move_hand`) OF A HAND THAT HOLDS NOTHING,
#: WHICH ASKS NO FORCE, READS A LOAD PAST THIS SHARE OF `PRESSED_ASK_N` --
#: 20 N -- at its start and on its way alike, and gives a posture up so too,
#: backing off until it reads no more (#699 on a carry; SimNotes, "A pull's
#: posture and the hand it rests on"): an open hand at a cube's board read
#: 25-35 N while a carry took four postures, and at Rowan's live draw the
#: body went down from a crouch of 0.16-0.22 m (2 of 24 flown); at the
#: pull's 40 N none was refused. Never the loop's own (a charge's). ⚠ ITS
#: LOAD IS READ OFF THE WHOLE BODY'S BALANCE ALONE, WHERE IT CLOSES
#: (`Reach.closed`), AND A POSTURE IS GIVEN UP ONLY FOR ONE READ PAST IT FOR
#: `MEAN_S` RUNNING: carried on purpose in free air from the stand into a
#: crouch, a bow and a turn at once, nothing touching the robot but its soles,
#: the arm's own estimate -- which stands where the balance does not close --
#: read 430 N (its mean past the line for 0.21 s, and the posture was given
#: up); where it closed, at most 60 N, its mean 15 N, bowed to 0.95 rad.
CARRY_PRESSED_SHARE = 1.0
#: ...giving, its turn gives to what it holds this share a command (100 a
#: second: a tenth of a second's lag) -- held, a wrist twisted by a drop
#: handle's swing put 20-30 N across the pull at the grip and turned the
#: robot 22 deg on its feet...
GIVE_RATE = 0.1
#: ...the way it reports at its end is the hand's over this much of its last
#: travel, m, its sensed place read each time the hand has gone this far, m
#: (at 6 cm/s a hand goes 0.6 mm a command, about what its sensed place
#: scatters)...
TRAIL_M, STEP_M = 0.02, 0.006
#: ...and its arm holds at this share of its gains: the force is closed on
#: the hand's reading, whatever the stiffness.
PULL_STIFFNESS = 0.6
#: THE WAY HOME (`way_home`, #539; SimNotes, "The way home"): straight to
#: the rest in joint space -- unless that takes some part of the arm ahead
#: of the robot's front (`front`), where whatever the hand worked over is,
#: more than this much lower, m, than the lowest of it there now -- else
#: the SIDE WAY, where that dips less: rolled out to its side to at least
#: this much, rad, its other joints home with it held there, then lowered
#: at its side. ⚠ Straight, an arm came down onto a 0.72 m counter's edge
#: 0.28 m ahead from 25 of 30 places a hand works over it, and onto a 0.90
#: m one from 23 of 26; this way onto neither: held out this far its elbow
#: stays near its shoulder's height, and lowered at its side the arm
#: reaches 9 cm ahead.
HOME_DIP_M, HOME_ROLL = 0.01, 1.3
#: ...its roll in -- lowered at its side -- with its shoulder pitched this
#: much back, rad, then forward to its rest (#560). ⚠ Lowered in the hanging
#: arm's line, it dragged a jaw over a counter the robot stood turned to for
#: 10 s; on its model this far back clears one at the toes turned 10 deg.
HOME_BACK = 0.2
#: ...sampled every this much of its furthest joint's walk, rad.
HOME_SAMPLE_RAD = 0.05
#: ...and home is the rest commanded and its drivers reading every joint
#: within this of it, rad (they read to 0.0026).
HOME_RAD = 0.05
#: A WAY HOME SOMETHING HOLDS (#693's live family; SimNotes, "A way home
#: something holds"): its command at its rest and its drivers reading the
#: arm more than `HOME_RAD` off it this long, s, it is held -- on a
#: counter's top, or on its own thigh in a crouch -- and its command is set
#: where its drivers read it (phase "held"), pressing on nothing. ⚠ Pressed
#: on toward its rest, a jaw read 39-69 N on a counter's top for as long as
#: nothing else moved the arm: 12 s into a withdraw's wait the robot went
#: over backward, and a walk's hand-over walked it with the jaw on the top.
HELD_S = 0.5
#: ...first lifted this far straight up, m, and sent home again from there,
#: at most this many times a way home -- never with another robot by
#: (`keep_out`): a way judged from the top would take the way that comes
#: least near it rather than be refused, as a verb's must (#678) -- and a
#: raise its arm did not follow, `BLOCKED_M` short of it for `BLOCKED_S`
#: after its command arrived, is left held (`_raise_short`). ⚠ TWICE, every
#: way home held: held mid-way by a stall (`STALL_S`), a door's withdraw
#: raised once stalled on the door's top again and was left by it, the
#: door's grade lost 5 times in 32; raised twice, 4 of those passed -- but a
#: hand-over's shove hold raised again pushed its door 55 -> 49 deg (SimNotes,
#: "A way home whose arm stalls").
LIFT_M, LIFTS = 0.05, 2
#: ...and a walk's hand-over's is held so at once, on its way, where what its
#: arm reads tips the body away from the hand (minus `_tipping`) past this, N
#: m, this long running, s (`_shoved`, `watch_home`; SimNotes, "A way home
#: that shoves the body back"): the 11-18 family's shoves held 17-34 N m for
#: 0.2 s 1.0-1.7 s before their falls. ⚠ NEVER A WITHDRAW'S: its word is the
#: arm's. ⚠ HELD, NEVER AT ONCE: a push on a prop that gives passes 20 N m for
#: under 0.2 s; held 0.2 s, a way home that touches nothing read 9 at most.
HOME_TIP_NM, HOME_TIP_S = 15.0, 0.2
#: ...and any way home is held so where its drivers read the arm more than
#: this off its command, rad, this long running, s (`_stalled`; SimNotes,
#: "A way home whose arm stalls"): something holds it while its command walks
#: on. ⚠ Walked on, a jaw on a top's edge wound the arm up 0.7 rad over 1.3 s,
#: pressing 16-57 N and shoving the robot 13 cm back, and slipped off at the
#: rest: the arm snapped home and the robot went over backward. ⚠ HELD, NEVER
#: AT ONCE: the door check's first ways home catch a jaw on the door's top up
#: to 0.97 rad behind, and slip free inside 0.45 s.
STALL_RAD, STALL_S = 0.5, 0.3
#: The phases of an arm that keeps no hand at work: at its rest, carried at
#: the side a walk carries it at (#682), or left where its way home was held
#: (`HELD_S`) -- what the stand's return waits on (`_go_home`), and what a
#: walk's hand-over and a restart's save let be.
AT_EASE = ("rest", "carried", "held")
#: ...and of an arm that is no hand out, for the walk's hand-over's let-go
#: and for rest (`HumanoidWay._hand_out`): an arm left held is one -- an
#: aside takes no arm home first.
NOT_OUT = ("rest", "carried")
#: Each hand's other, by side (`keep`).
OTHER_SIDE = {"left": "right", "right": "left"}
#: The way OUT is judged so from its other end (`way_out`, #560): straight, a
#: hanging hand at a table 13 cm out was driven into its top at 353-359 N.

#: THE CARRY POSE (#682; Ben, 2026-11-12: `close`, one piece in one hand):
#: what a walk's hand-over walks an arm to whose gripper says it holds
#: something, in joint space, its way judged as the way home is
#: (`way_home`), and holds it at through the walk -- the LEFT arm's shoulder
#: pitch, roll and yaw and its elbow, rad (the right mirrors roll and yaw):
#: the forearm folded up, the hand at the hips 0.19-0.23 m ahead. Its wrist
#: keeps the hand turned as it was against the floor (`carry_pose`). ⚠ NEVER
#: WHERE THE REACH LEFT IT: over the table's top, the shoulder rolled
#: 2.5-2.9 rad, walk2 fell in 16 of 37 walks; at this pose, holding the
#: plate or the cup, in none of 56, and its own sensors kept off the plate
#: (SimNotes, "Carrying while walking, measured").
CARRY_POSE = (0.15, 0.25, 0.0, 2.00)
#: ...its wrist found on a grid this many ways in pitch and in roll, then
#: once more this fine round the best.
CARRY_GRID, CARRY_FINE = (33, 41), 0.1

#: The postures a reach may take, in the order they are tried -- the least
#: bent first: the pelvis's height off the stand's, m, and the trunk's
#: pitch, rad, inside what the standing policy trained on
#: (`policy.HEIGHT_RANGE`, `PITCH_RANGE`).
POSTURES = ((0.0, 0.0), (-0.08, 0.0), (-0.16, 0.0), (-0.16, 0.3), (-0.2, 0.55),
            (-0.22, 0.8), (-0.25, 0.95), (-0.25, 1.1))
#: The arms a plan's solve starts from, the LEFT side's (shoulder pitch,
#: roll, yaw, elbow, wrist pitch, wrist roll; the right mirrors roll and
#: yaw): hanging, held out level, out and down, up, down to the floor, and
#: hanging plumb from a trunk bowed forward.
SEEDS = ((0.0, 0.1, 0.0, 0.3, 0.0, 0.0), (-1.4, 0.1, 0.0, 0.3, 0.0, 0.0),
         (-0.9, 0.1, 0.0, 0.6, 0.6, 0.0), (-2.4, 0.1, 0.0, 0.3, 0.0, 0.0),
         (-0.5, 0.1, 0.0, 0.2, -0.3, 0.0), (0.8, 0.1, 0.0, 0.3, 0.3, 0.0),
         (1.1, 0.1, 0.0, 0.1, 0.0, 0.0))
_MIRRORED = (1, 2, 5)
#: ...and the postures a hand that is out may take while it is carried
#: (`carry`, `path_for`): a reach's, with a bow at full
#: height beside the stand (#493). Bowed 0.15-0.3 rad its shoulder comes
#: ~6 cm forward, and a lid's arc out of reach from the stand -- the knob
#: 0.5 m ahead and over the shoulder's height by 60 deg -- was in it.
HOLD_POSTURES = ((0.0, 0.0), (0.0, 0.15), (0.0, 0.3), (-0.08, 0.0), (-0.08, 0.15),
                 (-0.16, 0.0), (-0.16, 0.3), (-0.2, 0.55), (-0.22, 0.8), (-0.25, 0.95),
                 (-0.25, 1.1))
#: The waist's yaw a posture may take, rad, in the order tried at each of
#: the legs' (#560's stage (c3), as #620 measured them, `waist_spike.py`):
#: none, then either way in steps of 15 deg to its range's 60 -- and how bent
#: a turned one is: this much of a turn as bent as the legs' next posture
#: (#620's `bent`).
WAIST_YAWS = tuple(math.radians(a) for a in (0, 15, -15, 30, -30, 45, -45, 60, -60))
WAIST_BENT_RAD = math.radians(20)


def with_waist(legs, yaws=WAIST_YAWS) -> tuple[tuple[float, float, float], ...]:
  """Every posture of `legs` (height, pitch) at each of the waist's `yaws`,
  (height, pitch, yaw): every one with the waist square first, in `legs`'
  order, then the turned ones the least bent first -- a legs' posture's
  place in `legs` and its yaw over `WAIST_BENT_RAD`, ties in `legs`' order,
  then `yaws'`. ⚠ #620'S COST ALONE TURNED THE WAIST BEFORE IT BOWED
  DEEPER, and the served door and chest passed 23 and 14 of 32 draws from
  the tour where square first they passed 28 and 25 (today's body 31 and
  20; the stand that knew no waist, one pod): the waist reaches where no
  square posture does."""
  rows = [(w != 0, i + abs(w) / WAIST_BENT_RAD, i, k, (float(h), float(p), float(w)))
          for i, (h, p) in enumerate(legs) for k, w in enumerate(yaws)]
  return tuple(r[4] for r in sorted(rows, key=lambda r: r[:4]))


#: ...a reach's postures, and a carried hand's (`posture_for`, `path_for`).
REACH_POSTURES = with_waist(POSTURES)
CARRY_POSTURES = with_waist(HOLD_POSTURES)
#: However the robot reads itself standing, its shoulder stands within this
#: of the statue balanced over the soles' middle with its arms hanging, m: a
#: goal further out than that of its reach is skipped on that statue, kept,
#: before one is balanced as it stands (`Lean`; it stood them up to 6.6 cm
#: back and, an arm over the top, 4.4 cm of weight forward, #627).
LEAN_MOST_M = 0.10
#: A posture whose trunk above the waist comes this far into a thigh or a
#: shank stands nowhere, m: the robot's own drawing meets its own parts only
#: in the pairs it names, and none is the trunk's against a leg -- at the
#: hips' deepest bow a yaw of 15-30 deg put the torso's corner 1-2.5 cm into
#: a thigh (#620).
TRUNK_INTO_M = 0.01


def turned(at: np.ndarray, speed: np.ndarray, goal: np.ndarray, dt: float
           ) -> tuple[np.ndarray, np.ndarray]:
  """A command walked toward `goal` one step of `dt`: no faster than
  `WAIST_RATE`, its speed changed by no more than `WAIST_ACCEL` a second,
  slowing in time to stop there. (command, speed)."""
  to_go = goal - at
  # the fastest that can still stop there, a step's travel counted in
  brake = WAIST_ACCEL * dt * (np.sqrt(0.25 + 2.0 * np.abs(to_go) / (WAIST_ACCEL * dt * dt)) - 0.5)
  want = np.sign(to_go) * np.minimum(brake, WAIST_RATE)
  speed = speed + np.clip(want - speed, -WAIST_ACCEL * dt, WAIST_ACCEL * dt)
  step = speed * dt
  # ...there once a step toward it reaches it, or it is there and all but stopped
  # (never on a step away: a goal moved just behind a turning command would
  # stop it dead)
  there = (((np.sign(step) == np.sign(to_go)) & (np.abs(step) >= np.abs(to_go)))
           | ((np.abs(to_go) < 1e-9) & (np.abs(speed) <= WAIST_ACCEL * dt)))
  return np.where(there, goal, at + step), np.where(there, 0.0, speed)


def seed(side: str, s) -> np.ndarray:
  q = np.array(s, dtype=float)
  if side == "right":
    q[list(_MIRRORED)] *= -1
  return q


def _balance(sm: SelfModel, height: float, pitch: float, arms: dict | None, waist: float,
             com: float):
  """What a statue's legs zero (`static_legs`): of (the pelvis's place
  ahead, the hip's, the knee's and the ankle's pitch), the sole's bottom on
  the floor, the sole flat, and the centre of mass `com` ahead of it -- the
  pelvis `height` off the stand's, pitched `pitch`, its arms and waist so."""
  m, d = sm.m, sm.d
  root, sole = sm.root, sm.soles[0]
  adr = {n: sm.qadr[k] for k, n in enumerate(hm.JOINT_NAMES)}
  h = hm.stand_height() + height
  quat = np.zeros(4)
  mujoco.mju_euler2Quat(quat, np.array([0.0, pitch, 0.0]), "XYZ")
  stand = {**hm.stand_qpos(), **{j: float(waist) for j in hm.WAIST}, **(arms or {})}

  def resid(v):
    x, hip, knee, ankle = v
    d.qpos[:] = 0.0
    d.qpos[0], d.qpos[2] = x, h
    d.qpos[3:7] = quat
    for n, val in stand.items():
      d.qpos[adr[n]] = val
    for s in hm.SIDES:
      d.qpos[adr[f"{s}_hip_pitch"]] = hip
      d.qpos[adr[f"{s}_knee"]] = knee
      d.qpos[adr[f"{s}_ankle_pitch"]] = ankle
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    r = d.geom_xmat[sole].reshape(3, 3)
    bottom = d.geom_xpos[sole][2] - m.geom_size[sole][2]
    return [10 * bottom, 3 * r[0, 2], 10 * (d.subtree_com[root][0] - d.geom_xpos[sole][0] - com)]
  return resid


def static_legs(height: float, pitch: float, model: SelfModel | None = None,
                arms: dict | None = None, waist: float = 0.0, com: float = 0.0) -> dict:
  """The legs' sagittal joints holding the pelvis `height` off the stand's
  and the trunk pitched `pitch`, both soles flat and the centre of mass
  `com` ahead of their middle, m (over it, by default): a statue's answer
  -- with the arms where `arms` puts them (joint name -> angle), else
  hanging, and the waist turned `waist`. ⚠ An arm held out moves the centre
  of mass, and the policy moves its pelvis back for it: balanced with its
  arms hanging, the statue stood 3 cm in front of the robot reaching a lid's
  knob and 7 cm in front of it at the lid's top, where the plan's reach was
  out of the robot's (#493). ⚠ AND THE POLICY HOLDS ITS WEIGHT WHERE IT
  HOLDS IT: 1.2 cm behind the soles' middle at the stand, 2.9 with an arm
  out after a step (#627; `Lean`). Joint name -> angle; with its arms
  hanging, the model's own answer kept (`SelfModel.statues`)."""
  sm = model or SelfModel()
  # ...the one over the soles' middle kept: one at a lean read is that read's
  kept = arms is None and not com
  key = (float(height), float(pitch), float(waist))
  if kept and key in sm.statues:
    return dict(sm.statues[key][0])
  resid = _balance(sm, height, pitch, arms, waist, com)
  best = None
  guesses = ([0.0, -0.2 - pitch, 0.4, -0.2], [0.0, -0.8 - pitch, 1.6, -0.8])
  if not kept and key in sm.statues:
    # ...one weighed otherwise starts from the statue's own legs: a fit, not two
    legs = sm.statues[key][0]
    guesses = ([0.0, legs["left_hip_pitch"], legs["left_knee"], legs["left_ankle_pitch"]],)
  for guess in guesses:
    fit = least_squares(resid, guess, bounds=([-1.0, -3.0, 0.0, -0.9], [1.0, 2.2, 2.6, 0.6]))
    if best is None or fit.cost < best.cost:
      best = fit
  _, hip, knee, ankle = best.x
  legs = {f"{s}_{k}": v for s in hm.SIDES
          for k, v in (("hip_pitch", hip), ("knee", knee), ("ankle_pitch", ankle))}
  if kept:
    sm.statues[key] = (dict(legs), None)
  return legs


def _posed(sm: SelfModel, height: float, pitch: float, q0: np.ndarray,
           arms: dict | None = None, waist: float = 0.0, com: float = 0.0,
           roll: float = 0.0) -> None:
  q = q0.copy()
  q[sm.waist] = waist
  for n, v in (arms or {}).items():
    q[hm.JOINT_NAMES.index(n)] = v
  for n, v in static_legs(height, pitch, sm, arms, waist, com).items():
    q[hm.JOINT_NAMES.index(n)] = v
  # gravity in the trunk's frame for a roll, then a pitch (`SelfModel.pose`)
  sm.pose(q, np.array([math.sin(pitch), -math.cos(pitch) * math.sin(roll),
                       -math.cos(pitch) * math.cos(roll)]))


@dataclass(frozen=True)
class Lean:
  """Where the robot reads itself standing (#627), off its model of itself
  posed off its senses (`Lean.read`): its centre of mass `com` ahead of its
  soles' middle, m, and its pelvis shifted `dx`, `dy`, m, and turned `turn`,
  rad, off the statue balanced there -- the stance frame's -- and its trunk
  rolled `roll`, rad, as its IMU reads it. A statue posed so (`statue`)
  stands where the live trunk stands. ⚠ THE STANDING POLICY
  HOLDS ITS WEIGHT WHERE IT HOLDS IT, AND NO TABLE PREDICTS IT: 1.2 cm behind
  the soles' middle at the stand and none in a deep bow, and an arm out
  moves it -2 to +5 cm by how the arm is bent, not by its weight alone (a
  natural arm and an arm over the top, the same weight, -0.8 and +4.4) --
  so a plan reads it, and reads it again with the arm out (`Reach._look`).
  The robot's own model read it to 0.05-0.2 cm of the truth at every
  posture, step and arm measured. The posture it was read in (`at`, as
  asked) is posed as the robot holds it (`held`): asked the deepest bow,
  the standing policy held 0.05-0.07 rad less of it, and the statue as
  asked stood a shoulder 1.1 cm ahead of the truth."""
  com: float = 0.0
  dx: float = 0.0
  dy: float = 0.0
  turn: float = 0.0
  roll: float = 0.0
  at: tuple | None = None
  held: tuple | None = None

  @staticmethod
  def read(live: SelfModel, posture, planner: SelfModel) -> "Lean":
    """The lean of `live`, posed off the robot's senses, asked `posture`
    (height, pitch, yaw): its centre of mass, the posture it holds as it
    measures it (its pelvis's height and pitch, its waist), and its pelvis
    off the statue of that posture balanced there with its arms as `live`
    has them (posed on `planner`)."""
    com = float(live.d.subtree_com[live.root][0])
    arms = {n: float(live.q[hm.JOINT_NAMES.index(n)])
            for s in hm.SIDES for n in (f"{s}_{j}" for j in hm.ARM)}
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    r = live.d.xmat[live.root].reshape(3, 3)
    held = (float(live.d.qpos[2]) - hm.stand_height(),
            math.asin(max(-1.0, min(1.0, -float(r[2, 0])))), float(live.q[live.waist][0]))
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    _posed(planner, *held[:2], q0, arms, held[2], com, roll)
    (x0, y0), t0 = _pelvis(planner)
    (x1, y1), t1 = _pelvis(live)
    turn = math.remainder(t1 - t0, 2 * math.pi)
    c, s = math.cos(turn), math.sin(turn)
    return Lean(com, x1 - (c * x0 - s * y0), y1 - (s * x0 + c * y0), turn, roll,
                tuple(float(v) for v in posture), held)


def _pelvis(sm: SelfModel) -> tuple[tuple[float, float], float]:
  """The pelvis's place on the floor, stance frame, and its heading."""
  r = sm.d.xmat[sm.root].reshape(3, 3)
  return ((float(sm.d.xpos[sm.root][0]), float(sm.d.xpos[sm.root][1])),
          math.atan2(r[1, 0], r[0, 0]))


#: A statue balanced at a lean sooner (`statue(fast=True)`): its legs the
#: kept statue's moved by their own change as its weight's place moves,
#: worked out across this much of it, m, either way, and kept -- then this
#: many Newton steps on its balance from there.
SLOPE_M, FAST_STEPS = 0.02, 2


def _slope(sm: SelfModel, height: float, pitch: float, waist: float) -> np.ndarray:
  """How the statue's legs (hip, knee, ankle pitch) move as its centre of
  mass is balanced further ahead of the soles' middle, rad a metre; kept."""
  key = ("slope", height, pitch, waist)
  if key not in sm.statues:
    ends = [static_legs(height, pitch, sm, None, waist, c) for c in (SLOPE_M, -SLOPE_M)]
    sm.statues[key] = np.array([(ends[0][f"left_{j}"] - ends[1][f"left_{j}"]) / (2 * SLOPE_M)
                                for j in ("hip_pitch", "knee", "ankle_pitch")])
  return sm.statues[key]


def _posed_fast(sm: SelfModel, height: float, pitch: float, q0: np.ndarray,
                arms: dict | None, waist: float, com: float, roll: float) -> None:
  """`_posed`, its balance solved by Newton's method (`FAST_STEPS`) from the
  kept statue's legs moved by the lean, less what the arms' weight moves it
  ahead (`_slope`): the same balance, without a least squares' cost a
  posture -- a plan weighs dozens."""
  legs = static_legs(height, pitch, sm, None, waist)
  resid = _balance(sm, height, pitch, arms, waist, com)
  v = np.array([0.0, *(legs[f"left_{j}"] for j in ("hip_pitch", "knee", "ankle_pitch"))])
  # ...as far as the weight stands off `com` there, the arms set (the
  # residual's third row is ten times it)
  v[1:] -= resid(v)[2] / 10 * _slope(sm, height, pitch, waist)
  for _ in range(FAST_STEPS):
    r0 = np.array(resid(v))
    jac = np.empty((3, 3))
    for k in range(3):
      e = np.zeros(4)
      e[k + 1] = 1e-6
      jac[:, k] = (np.array(resid(v + e)) - r0) / 1e-6
    v[1:] -= np.linalg.solve(jac, r0)
  q = q0.copy()
  q[sm.waist] = waist
  for n, val in (arms or {}).items():
    q[hm.JOINT_NAMES.index(n)] = val
  for s in hm.SIDES:
    for k, j in enumerate(("hip_pitch", "knee", "ankle_pitch")):
      q[hm.JOINT_NAMES.index(f"{s}_{j}")] = v[k + 1]
  sm.pose(q, np.array([math.sin(pitch), -math.cos(pitch) * math.sin(roll),
                       -math.cos(pitch) * math.cos(roll)]))


def statue(sm: SelfModel, posture, q0: np.ndarray, arms: dict | None = None,
           lean: Lean | None = None, fast: bool = False) -> None:
  """`sm` posed as the robot would stand in `posture` (height, pitch, yaw):
  a statue's legs (`static_legs`), its arms where `arms` puts them (joint
  name -> angle; else hanging), balanced with its centre of mass where
  `lean` reads it and its pelvis shifted and turned as `lean` says (#627),
  the posture it was read in as it held it; with no lean, over the soles'
  middle. `fast`: its balance solved from the kept statue's (`_posed_fast`):
  within 0.01 mm of the least squares over random arms and leans of -3.5 to
  +5 cm at every carry posture, at 0.3 ms where it takes 12."""
  lean = lean or Lean()
  if lean.at is not None and np.allclose(posture, lean.at, atol=1e-9):
    # ...as it holds it, balanced afresh: a held posture's statue is a read's own
    posture, fast = lean.held, False
  height, pitch, waist = (float(v) for v in posture)
  if fast:
    _posed_fast(sm, height, pitch, q0, arms, waist, lean.com, lean.roll)
  else:
    _posed(sm, height, pitch, q0, arms or None, waist, lean.com, lean.roll)
  if lean.dx or lean.dy or lean.turn:
    d = sm.d
    c, s = math.cos(lean.turn), math.sin(lean.turn)
    x, y = float(d.qpos[0]), float(d.qpos[1])
    d.qpos[0], d.qpos[1] = c * x - s * y + lean.dx, s * x + c * y + lean.dy
    turn, out = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), lean.turn)
    mujoco.mju_mulQuat(out, turn, d.qpos[3:7].copy())
    d.qpos[3:7] = out
    mujoco.mj_kinematics(sm.m, d)
    mujoco.mj_comPos(sm.m, d)


def stands(sm: SelfModel, posture, q0: np.ndarray) -> bool:
  """Whether a posture (height, pitch, yaw) stands, its statue posed, its
  arms hanging (`_posed`): its trunk clear of its legs (`TRUNK_INTO_M`),
  kept with its statue (`SelfModel.statues`)."""
  height, pitch, waist = posture
  _posed(sm, height, pitch, q0, waist=waist)
  key = (float(height), float(pitch), float(waist))
  legs, ok = sm.statues[key]
  if ok is None:
    ok = trunk_clear(sm) > -TRUNK_INTO_M
    sm.statues[key] = (legs, ok)
  return bool(ok)


#: The trunk's parts above the waist, and the legs' they may fold into.
UPPER_GEOMS = ("torso",) + hm.HEAD_GEOMS
LEG_PARTS = tuple(f"{s}_{g}" for s in hm.SIDES for g in ("thigh", "shank"))


def trunk_clear(sm: SelfModel) -> float:
  """How far the trunk above the waist is from the thighs and the shanks as
  posed, m (negative: through them; `mj_geomDistance`, as #620 measured)."""
  m, d = sm.m, sm.d
  return min(float(mujoco.mj_geomDistance(m, d, m.geom(a).id, m.geom(b).id, 0.1, None))
             for a in UPPER_GEOMS for b in LEG_PARTS)


def arm_reach(sm: SelfModel, side: str) -> float:
  """The furthest the grip gets from its shoulder's roll axis, m: every
  link of the arm hangs on the one line below it, so it is the arm hanging
  straight -- a bound no solve can beat, kept (`SelfModel.statues`), and
  measured on a pose of its own: ⚠ measured on the model as posed, the first
  in a process left it with straight legs, and the plan solved its first
  posture on that."""
  key = ("reach", side)
  if key not in sm.statues:
    d = mujoco.MjData(sm.m)
    mujoco.mj_kinematics(sm.m, d)
    shoulder = d.xpos[sm.m.body(f"{side}_shoulder_roll_link").id]
    sm.statues[key] = float(np.linalg.norm(d.site_xpos[sm.grip[side]] - shoulder))
  return sm.statues[key]


def out_of_reach(sm: SelfModel, side: str, pos, further: float = 0.0) -> bool:
  """Whether `pos` is further from `side`'s shoulder than any solve could
  bring the grip within the reach's tolerance (`REACH_POS_TOL`), `further`
  on top, the model as posed: a plan skips the posture there, nothing lost."""
  shoulder = sm.d.xpos[sm.m.body(f"{side}_shoulder_roll_link").id]
  return float(np.linalg.norm(np.asarray(pos) - shoulder)) + further >= (
    arm_reach(sm, side) + REACH_POS_TOL)


def _arm(side: str, q_arm) -> dict:
  """One arm's joints by name, for a statue to balance."""
  return {f"{side}_{j}": float(v) for j, v in zip(hm.ARM, q_arm)}


def _ranked(sm: SelfModel, side: str, pos, rot, start: np.ndarray | None = None
            ) -> list[tuple[bool, np.ndarray, float, float]]:
  """`SelfModel.solve` from `start` and each of `SEEDS`, every one, ranked:
  within tolerance and nearest `start` -- or, with none, nearest the arm
  hanging -- first, then the rest by how far off, ties in their seeds' order;
  (within tolerance, q, ep, er) each."""
  tries = [seed(side, s) for s in SEEDS]
  if start is not None:
    tries.insert(0, np.asarray(start, dtype=float))
  near = tries[0]
  got = []
  for k, q0 in enumerate(tries):
    sm.q[sm.arm[side]] = q0
    q, ep, er = sm.solve(side, pos, rot, iters=80)
    ok = ep < REACH_POS_TOL and er < REACH_ROT_TOL
    got.append(((not ok, float(np.abs(q - near).max()) if ok else ep + 0.1 * er), k, (ok, q, ep, er)))
  return [g[2] for g in sorted(got, key=lambda g: g[:2])]


def solve_seeded(sm: SelfModel, side: str, pos, rot, start: np.ndarray | None = None
                 ) -> tuple[np.ndarray, float, float]:
  """`SelfModel.solve` from each of `SEEDS` (and `start`), the best kept:
  within tolerance and nearest `start` -- or, with none, nearest the arm
  hanging -- else the least off (`_ranked`'s first). ⚠ NEAREST, NEVER THE
  FIRST TO REACH: kept in its seeds' order, the hanging arm's own solve took
  the shoulder thrown back and the arm over the top where five other seeds
  reached the same point with the arm in front -- and an arm so bent leant
  the standing policy 4.4 cm forward where the other leant it 0.8 back
  (#627)."""
  return _ranked(sm, side, pos, rot, start)[0][1:]


def _solutions(sm: SelfModel, side: str, pos, rot, start: np.ndarray | None = None
               ) -> list[np.ndarray]:
  """Every arm `solve_seeded` solves that puts the grip at `pos` (turned to
  `rot`) within the reach's tolerance as `sm` is posed, in the order it
  ranks them: its own pick first (#726)."""
  return [q for ok, q, _, _ in _ranked(sm, side, pos, rot, start) if ok]


def _reaching(sm: SelfModel, side: str, pos, rot, start=None, seeded: bool = True):
  """The arm that puts the grip at `pos` (turned to `rot`) as `sm` is
  posed, within the reach's tolerance, nearest `start` (`solve_seeded`) --
  from `start` alone first where `seeded` is False -- or None."""
  if not seeded and start is not None:
    sm.q[sm.arm[side]] = np.asarray(start, dtype=float)
    q, ep, er = sm.solve(side, pos, rot, iters=80)
    if ep < REACH_POS_TOL and er < REACH_ROT_TOL:
      return q
  q, ep, er = solve_seeded(sm, side, pos, rot, start)
  return q if ep < REACH_POS_TOL and er < REACH_ROT_TOL else None


def _goal_arm(sm: SelfModel, side: str, goal: HandGoal, slack: bool, start=None,
              seeded: bool = True):
  """The arm that puts the grip at `goal` as `sm` is posed -- and, with
  `slack`, at `REACH_MARGIN_M` further from its shoulder too -- nearest
  `start`; or None. ⚠ EACH POINT IS ALSO SOLVED FROM THE OTHER'S ARM: from
  the seeds alone, the two missed by 0.4-0.5 rad of turn at points 5 mm
  from points that reached, none of seven seeds converging, and the plan
  took a deeper bow there -- with no tag in view (#627, the table's cup)."""
  q_goal = _reaching(sm, side, goal.pos, goal.rot, start, seeded)
  if not slack:
    return q_goal
  shoulder = sm.d.xpos[sm.m.body(f"{side}_shoulder_roll_link").id]
  out = np.asarray(goal.pos, dtype=float) - shoulder
  far = np.asarray(goal.pos, dtype=float) + REACH_MARGIN_M * out / np.linalg.norm(out)
  q_far = _reaching(sm, side, far, goal.rot, start if q_goal is None else q_goal,
                    seeded and q_goal is None)
  if q_far is not None and q_goal is None:
    q_goal = _reaching(sm, side, goal.pos, goal.rot, q_far, seeded=False)
  return q_goal if q_far is not None else None


def _reached_from(sm: SelfModel, side: str, posture, q0: np.ndarray, goal: HandGoal,
                  pre: HandGoal, slack: bool, lean: "Lean | None", arms: dict | None, start,
                  seeded: bool = True):
  """(the arm at the pre-grasp, the arm at the goal) where `posture` puts
  `goal` in `side`'s reach -- with its slack -- on the robot as it would
  stand there (`statue`): with this arm where `arms` puts it (else hanging),
  then balanced with it where that solve put it and solved again from
  there, as an arm held out moves the trunk; or None. The goal's first
  solve nearest `start` (`solve_seeded`), or -- not `seeded` -- from `start`
  alone first."""
  others = dict(arms or {})
  statue(sm, posture, q0, others, lean, fast=True)
  if out_of_reach(sm, side, goal.pos, REACH_MARGIN_M if slack else 0.0):
    return None
  q = _goal_arm(sm, side, goal, slack, start, seeded)
  if q is None:
    return None
  statue(sm, posture, q0, {**others, **_arm(side, q)}, lean, fast=True)
  q = _goal_arm(sm, side, goal, slack, q, seeded=False)
  if q is None:
    return None
  q_pre = _reaching(sm, side, pre.pos, pre.rot, q, seeded=False)
  return None if q_pre is None else (q_pre, q)


#: Two of the goal's solves are one arm within this of each other, rad,
#: every joint: from seeds that end in one branch the solve lands on one
#: arm to well under this.
SAME_ARM_RAD = 0.01


def _reached(sm: SelfModel, side: str, posture, q0: np.ndarray, goal: HandGoal,
             pre: HandGoal, slack: bool, lean: "Lean | None", arms: dict | None, start,
             clear: bool = False):
  """`_reached_from`'s (the arm at the pre-grasp, the arm at the goal), or
  None. ⚠ `clear` (#726): A REACH'S PLAN KEEPS ITS ARM OFF THE ROBOT'S OWN
  TRUNK, at the pre-grasp and at the goal (`in_trunk`) -- today's, else the
  goal's other solves in turn as `solve_seeded` ranks them (`_solutions`),
  each planned on from there, the first that keeps off it: (q_pre, q,
  None); with none, today's and how near it comes ((q_pre, q, {part, m}))."""
  got = _reached_from(sm, side, posture, q0, goal, pre, slack, lean, arms, start)
  if not clear or got is None:
    return got
  near = in_trunk(side, *got)
  if near is None:
    return (*got, None)
  # ...the goal's other solves, as the first was solved: on the statue with
  # this arm hanging
  statue(sm, posture, q0, dict(arms or {}), lean, fast=True)
  solves = _solutions(sm, side, goal.pos, goal.rot, start)
  # ...the first is `solve_seeded`'s pick, today's plan's own start: planned
  # on from there it is today's again
  tried = solves[:1]
  for q in solves[1:]:
    if any(float(np.abs(q - t).max()) <= SAME_ARM_RAD for t in tried):
      continue
    tried.append(q)
    other = _reached_from(sm, side, posture, q0, goal, pre, slack, lean, arms, q, seeded=False)
    if other is not None and in_trunk(side, *other) is None:
      return (*other, None)
  return (*got, near)


def plan(side: str, goal: HandGoal, model: SelfModel | None = None,
         postures=REACH_POSTURES, approach: float = APPROACH_M, lean: "Lean | None" = None,
         arms: dict | None = None, start=None, own: dict | None = None):
  """The least-bent posture from which `side`'s hand reaches `goal` -- and
  its pre-grasp, `approach` short of it -- WITH `REACH_MARGIN_M` to spare,
  else the least bent that reaches it at full stretch; and the arm's
  joints at both, nearest `start` (the arm as it is; else hanging). Tried
  on the robot's model of itself as it would stand in each posture
  (`statue`): a statue's legs and waist (`stands`), balanced where `lean`
  reads it (#627; with none, over the soles' middle), the other arm where
  `arms` puts it and this one where its solve does (`_reached`). Returns
  ((height, pitch, yaw), arm at the pre-grasp, arm at the goal, whether it
  has slack), or None if no posture reaches it. With `own`, a dict (#726):
  the arm at both kept off the robot's own trunk (`_reached`'s `clear`) --
  where every posture that reaches puts it in the trunk, None, and `own`
  says how near the first of them came (`part`, `m`) and holds its plan,
  the one planned without the trunk (`plan`)."""
  sm = model or SelfModel()
  q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  pre = pregrasp(goal, approach)
  first = None
  for slack in (True, False):
    for posture in postures:
      if not stands(sm, posture, q0) or out_of_reach(sm, side, goal.pos, -LEAN_MOST_M):
        continue
      got = _reached(sm, side, posture, q0, goal, pre, slack, lean, arms, start,
                     clear=own is not None)
      if got is None:
        continue
      found = (tuple(posture), got[0], got[1], slack)
      if own is None or got[2] is None:
        return found
      if first is None:
        first = (found, got[2])
  if first is not None:
    own.update(first[1], plan=first[0])
  return None


def pregrasp(goal: HandGoal, approach: float = APPROACH_M) -> HandGoal:
  """`approach` short of `goal`, back along the way its fingers point (the
  grip's -z); without an orientation, straight back toward the body."""
  if goal.rot is None:
    back = np.array([-1.0, 0.0, 0.0])
  else:
    back = np.asarray(goal.rot)[:, 2]
  return HandGoal(np.asarray(goal.pos) + approach * back, goal.rot, goal.stiffness)


#: An arm's solid parts, as a way home keeps them clear (`way_home`).
ARM_PARTS = hm.ARM_GEOMS + tuple(f"{j}_pad" for j in hm.JAWS)


def toes(sm: SelfModel) -> float:
  """How far ahead of the soles' middle its toes are, m: the soles' front as
  the robot's model of itself stands."""
  d, m = sm.d, sm.m
  return max(float(d.geom_xpos[g][0] + np.abs(d.geom_xmat[g].reshape(3, 3)[0]) @ m.geom_size[g])
             for g in sm.soles)


def outline(sm: SelfModel, side: str) -> np.ndarray:
  """The arm's solid parts as (how far ahead, how high), m, stance frame: a
  box's corners, a capsule's two ends with its radius ahead of them and
  under them."""
  m, d = sm.m, sm.d
  out = []
  for part in ARM_PARTS:
    g = m.geom(f"{side}_{part}").id
    pos, rot, size = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
      out += [(e[0] + size[0], e[2] - size[0])
              for e in (pos + size[1] * rot[:, 2], pos - size[1] * rot[:, 2])]
    else:
      corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * size
      out += [(p[0], p[2]) for p in pos + corners @ rot.T]
  return np.array(out)


def front(sm: SelfModel, side: str, rest) -> float:
  """How far ahead the robot's own front is, m: its toes, or its arm
  hanging at `rest` where that reaches further -- where every way home
  ends, so never a dip. ⚠ Posed as a reach's statue, the hanging grip is
  0.8 cm past the toes: judged from the toes alone, straight home tied the
  side way there and took it down onto the counter. Leaves the arm at
  rest."""
  sm._set_arm(side, rest)
  return max(toes(sm), float(outline(sm, side)[:, 0].max()))


def dip(sm: SelfModel, side: str, q0, ways, low: float, ahead: float) -> float:
  """How far under `low` any part of the arm goes further ahead than
  `ahead`, m, as it walks from `q0` through `ways` in joint space, every
  joint arriving together (`Reach._joints`)."""
  worst, prev = 0.0, np.asarray(q0, dtype=float)
  for w in ways:
    n = max(1, math.ceil(float(np.abs(w - prev).max()) / HOME_SAMPLE_RAD))
    for k in range(1, n + 1):
      sm._set_arm(side, prev + (w - prev) * (k / n))
      p = outline(sm, side)
      p = p[p[:, 0] > ahead]
      if len(p):
        worst = max(worst, low - float(p[:, 1].min()))
    prev = w
  return worst


def _side_way(side: str, q_out, q_home) -> list[np.ndarray]:
  """The side way from an arm at `q_out` to `q_home` in joint space: rolled
  out to its side (`HOME_ROLL`, or further where it is already), its other
  joints to `q_home`'s with its shoulder pitched `HOME_BACK` back, its roll
  then in, and its pitch forward -- the targets in turn."""
  k, p = hm.ARM.index("shoulder_roll"), hm.ARM.index("shoulder_pitch")
  out = seed(side, (0.0, HOME_ROLL, 0.0, 0.0, 0.0, 0.0))[k]
  raised, lowered, back = (np.asarray(q, dtype=float).copy() for q in (q_out, q_home, q_home))
  raised[k] = lowered[k] = raised[k] if abs(raised[k]) > HOME_ROLL else out
  lowered[p] = back[p] = back[p] + HOME_BACK
  return [raised, lowered, back, np.asarray(q_home, dtype=float).copy()]


#: ANOTHER ROBOT'S BODY, AS A HAND'S WAY KEEPS OFF IT (#678; SimNotes, "The
#: arms keep off the other robot"): a cylinder standing round the middle of
#: its soles where this robot's LIDAR sees it (`HumanoidMission.keep_out`),
#: this wide, m -- the standing body, its arms hanging, its grippers 0.28 m
#: out, on the drawing -- and this tall, its head's top. ⚠ ITS SIGHTED
#: PLACE, NEVER ITS HEADING: a robot standing square to the counter beside
#: this one is narrower that way than round, so a refusal can come where
#: its arm would have passed it (#678's stage 1: 15 of 40 at 0.71-0.95 m)...
OTHER_BODY_M, OTHER_TOP_M = 0.28, 1.55
#: ...with this much room between it and every part of the arm, m.
OTHER_CLEAR_M = 0.05
#: The way drawn in: the arm's elbow bent this far, rad, before its shoulder
#: moves (its range ends at 2.10).
TUCK_ELBOW = 1.9


def arm_points(sm: SelfModel, side: str) -> tuple[np.ndarray, np.ndarray]:
  """The arm's solid parts as points (x, y, z), stance frame, each with the
  radius round it: a capsule's axis at five points, a box's corners and its
  middle."""
  m, d = sm.m, sm.d
  pts, rad = [], []
  for part in ARM_PARTS:
    g = m.geom(f"{side}_{part}").id
    pos, rot, size = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
      pts += [pos + f * size[1] * rot[:, 2] for f in (-1.0, -0.5, 0.0, 0.5, 1.0)]
      rad += [size[0]] * 5
    else:
      corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * size
      pts += list(pos + corners @ rot.T) + [pos.copy()]
      rad += [0.0] * 9
  return np.array(pts), np.array(rad)


def room(sm: SelfModel, side: str, q0, ways, keep_out) -> float:
  """The least room, m, between any part of the arm and any robot of
  `keep_out` ((x, y) of its soles' middle, stance frame: `OTHER_BODY_M`
  round, `OTHER_TOP_M` tall) as the arm walks from `q0` through `ways` in
  joint space (as `dip` walks it), its start included; inf with none."""
  if not len(keep_out):
    return math.inf
  centres = np.asarray(keep_out, dtype=float).reshape(-1, 2)
  worst, prev = math.inf, np.asarray(q0, dtype=float)
  for w in [prev] + list(ways):
    w = np.asarray(w, dtype=float)
    n = max(1, math.ceil(float(np.abs(w - prev).max()) / HOME_SAMPLE_RAD))
    for k in range(1, n + 1):
      sm._set_arm(side, prev + (w - prev) * (k / n))
      p, r = arm_points(sm, side)
      under = p[:, 2] - r < OTHER_TOP_M
      if under.any():
        gap = np.hypot(p[under, 0, None] - centres[:, 0], p[under, 1, None] - centres[:, 1])
        worst = min(worst, float((gap - r[under, None]).min()) - OTHER_BODY_M)
    prev = w
  return worst


#: A WAY OUT FROM A HAND ALREADY OUT KEEPS OFF THE ROBOT'S OWN TRUNK (#722;
#: SimNotes, "A way out from a hand already out"): the arm's parts that meet
#: the trunk (`model.ARM_GEOMS`' pairs) keep this much room, m, from the trunk
#: above the waist (`UPPER_GEOMS`) all the way, on the robot's model of itself
#: -- or, begun nearer, go no nearer. ⚠ Unjudged, a gripper was driven into
#: the torso at 165 N and the head at 75 N; over the 11-15 wake's 53 runs
#: flown on ac2538e8, the 7 such ways that met the trunk (36-56 N) were judged
#: 0.9-3.3 cm through it, or 2 mm deeper than a gripper begun on it, and the 9
#: judged 0.28-2 cm clear met nothing.
OWN_CLEAR_M = 0.0
#: ...sampled every this much of its furthest joint's walk, rad: about 3 mm
#: at the grip, so a way may cut no more than half that between two samples.
OWN_SAMPLE_RAD = 0.005
#: ...and read no further off than this, m (`mj_geomDistance`'s `distmax`).
OWN_READ_M = 0.2
#: ...two readings of one depth may part by this, m: MuJoCo's box against a
#: box and `_box_box`'s agree to 0.07 mm over 596 poses through each other.
OWN_SLACK_M = 1e-4


def _box_box(m, d, a: int, b: int) -> float:
  """The signed distance between box geoms `a` and `b` as posed, m: apart,
  the least of each one's corners against the other box and of their edges
  against each other's; overlapping, minus their least overlap over the
  fifteen separating axes."""
  ca, cb = d.geom_xpos[a], d.geom_xpos[b]
  ra, rb = d.geom_xmat[a].reshape(3, 3), d.geom_xmat[b].reshape(3, 3)
  sa, sb = m.geom_size[a], m.geom_size[b]
  axes = [ra[:, i] for i in range(3)] + [rb[:, i] for i in range(3)]
  axes += [np.cross(ra[:, i], rb[:, j]) for i in range(3) for j in range(3)]
  overlap = math.inf
  for ax in axes:
    n = float(np.linalg.norm(ax))
    if n < 1e-9:
      continue
    ax = ax / n
    o = (float(np.abs(ra.T @ ax) @ sa) + float(np.abs(rb.T @ ax) @ sb)
         - abs(float((cb - ca) @ ax)))
    overlap = min(overlap, o)
  if overlap > 0.0:
    return -overlap
  signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))

  def corners(c, r, s):
    return c + (signs * s) @ r.T

  def outside(p, c, r, s):
    return np.linalg.norm(np.maximum(np.abs((p - c) @ r) - s, 0.0), axis=1)

  pa, pb = corners(ca, ra, sa), corners(cb, rb, sb)
  best = min(float(outside(pa, cb, rb, sb).min()), float(outside(pb, ca, ra, sa).min()))
  # ...and every edge against every edge: corners one axis apart
  edges = [(i, j) for i in range(8) for j in range(8) if i < j and (signs[i] != signs[j]).sum() == 1]
  p0, p1 = pa[[i for i, _ in edges]], pa[[j for _, j in edges]]
  q0, q1 = pb[[i for i, _ in edges]], pb[[j for _, j in edges]]
  u, v = (p1 - p0)[:, None, :], (q1 - q0)[None, :, :]
  w = p0[:, None, :] - q0[None, :, :]
  aa, bb, cc = (u * u).sum(-1), (u * v).sum(-1), (v * v).sum(-1)
  dd, ee = (u * w).sum(-1), (v * w).sum(-1)
  den = aa * cc - bb * bb
  s = np.clip(np.where(den > 1e-12, (bb * ee - cc * dd) / np.where(den > 1e-12, den, 1.0), 0.0),
              0.0, 1.0)
  t = np.clip((bb * s + ee) / cc, 0.0, 1.0)
  s = np.clip((bb * t - dd) / aa, 0.0, 1.0)
  gap = np.linalg.norm(w + s[..., None] * u - t[..., None] * v, axis=-1)
  return min(best, float(gap.min()))


def apart(m, d, a: int, b: int) -> float:
  """How far apart geoms `a` and `b` are as posed, m, negative through each
  other (`mj_geomDistance`, read no further than `OWN_READ_M`). ⚠ MUJOCO
  3.10 READS TWO BOXES APART AS 0.0 AT SOME POSES (the gripper against the
  torso at 1.4 % of an arm's random poses, its from-to then a segment
  centimetres long): a box against a box that reads 0.0 is read again
  (`_box_box`). Over 4,000 random arm poses only a box against a box read
  so: the capsules', cylinders' and the sphere's readings each match their
  own from-to."""
  got = mujoco.mj_geomDistance(m, d, a, b, OWN_READ_M, None)
  box = mujoco.mjtGeom.mjGEOM_BOX
  if got == 0.0 and m.geom_type[a] == box and m.geom_type[b] == box:
    return min(OWN_READ_M, _box_box(m, d, a, b))
  return got


def own_rooms(sm: SelfModel, side: str, q0, ways) -> np.ndarray:
  """The least room, m, between each of the arm's parts that meet the trunk
  (`model.ARM_GEOMS`) and each of the trunk's above the waist (`UPPER_GEOMS`:
  the torso, the head and the LIDAR in it, fixed to the shoulders, so a
  function of the arm's joints alone) -- a pair each, the arm's part by the
  trunk's -- as the arm walks from `q0` through `ways` in joint space, every
  joint arriving together (`Reach._joints`), its start included; negative
  through it."""
  m, d = sm.m, sm.d
  pairs = [(m.geom(f"{side}_{p}").id, m.geom(g).id) for p in hm.ARM_GEOMS for g in UPPER_GEOMS]
  worst, prev = np.full(len(pairs), math.inf), np.asarray(q0, dtype=float)
  for w in [prev] + list(ways):
    w = np.asarray(w, dtype=float)
    n = max(1, math.ceil(float(np.abs(w - prev).max()) / OWN_SAMPLE_RAD))
    for k in range(1, n + 1):
      sm._set_arm(side, prev + (w - prev) * (k / n))
      worst = np.minimum(worst, [apart(m, d, a, b) for a, b in pairs])
    prev = w
  return worst


def own_room(sm: SelfModel, side: str, q0, ways) -> tuple[float, str]:
  """The least of `own_rooms`, m, and the trunk's part it comes nearest."""
  rooms = own_rooms(sm, side, q0, ways)
  i = int(np.argmin(rooms))
  return float(rooms[i]), UPPER_GEOMS[i % len(UPPER_GEOMS)]


#: The one model of itself a way is judged against its own trunk on
#: (`own_model`): posed by an arm's joints alone, so shared.
_OWN: SelfModel | None = None


def own_model() -> SelfModel:
  """The model of itself an arm is judged against its own trunk on
  (`own_rooms`): posed by that arm's joints alone, apart from the
  planner's, whose pose a judgment's callers go on reading -- one for every
  robot (`_OWN`)."""
  global _OWN
  if _OWN is None:
    _OWN = SelfModel()
  return _OWN


def in_trunk(side: str, *arms) -> dict | None:
  """Where some arm of `arms` (each its six joints) lies within the
  robot's own trunk on its model of itself (#726; `own_rooms`, each pair of
  parts closer than `OWN_CLEAR_M`, less the `OWN_SLACK_M` two readings may
  part by): the deepest's part of the trunk ("torso", or "head" for the
  head and the LIDAR in it) and its room, m, negative through it, as #722's
  `own` says them; else None."""
  sm = own_model()
  worst = None
  for q in arms:
    rooms = own_rooms(sm, side, np.asarray(q, dtype=float), [])
    i = int(np.argmin(rooms))
    if rooms[i] < OWN_CLEAR_M - OWN_SLACK_M and (worst is None or rooms[i] < worst[1]):
      worst = (UPPER_GEOMS[i % len(UPPER_GEOMS)], float(rooms[i]))
  return None if worst is None else {"part": "torso" if worst[0] == "torso" else "head",
                                     "m": round(worst[1], 3)}


def _same_way(a, b) -> bool:
  return len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b))


def _tucked_way(side: str, q0, q1) -> list[np.ndarray]:
  """The way drawn in from `q0` to `q1` in joint space: its elbow bent to
  `TUCK_ELBOW` where it stands, its shoulder and wrist then to `q1`'s, the
  elbow held, then its elbow to `q1`'s."""
  e = hm.ARM.index("elbow")
  a, b = (np.asarray(q, dtype=float).copy() for q in (q0, q1))
  a[e] = b[e] = max(float(a[e]), TUCK_ELBOW)
  return [a, b, np.asarray(q1, dtype=float).copy()]


def _aside(side: str, q0, q1, out: bool) -> list[np.ndarray]:
  """The side way from `q0` to `q1`: home's (`_side_way`), or going `out`
  that walked back."""
  return _side_way(side, q1, q0)[::-1][1:] + [q1] if out else _side_way(side, q0, q1)


def _dips(sm: SelfModel, side: str, q0, at, ahead: float):
  """How far a way from `q0` dips under the lowest of the arm ahead of the
  robot's front (`ahead`) as it stands at `at`, over the work (`dip`): a
  function of the way, m. ⚠ None of it ahead works over nothing there, and
  every way dips 0: judged as if all under it were solid, a drawer pulled
  close took the side way, which swept a jaw into its face and pushed it 4
  cm back in."""
  sm._set_arm(side, at)
  p = outline(sm, side)
  there = p[p[:, 0] > ahead]
  if not len(there):
    return lambda w: 0.0
  low = float(there[:, 1].min())
  return lambda w: dip(sm, side, q0, w, low, ahead)


class Reach:
  """The arms and grippers of one body beside its standing policy (module
  docstring). A hand's PHASE: `rest` (hanging), `move` (joint space to the
  pre-grasp, the posture walking to its plan), `approach` (a straight line
  in), `hold`, `pull` (the goal moved while holding), `blocked` (stopped on
  its way, `BLOCKED_M`), `carry` (a hand that is out carried on, `carry`),
  `joints` (an arm walked joint by joint, `set_joints`), `retreat` (a
  straight line out), `home` (joint space home, its own way: `way_home`) --
  or, a hand a walk carries (`carry_q`), to its carry pose -- and `carried`
  (held there, #682)."""

  def __init__(self, model, data, stand: StandDriver, prefix: str = "",
               key: str = "humanoid"):
    self.m, self.d, self.stand = model, data, stand
    self.self_model = SelfModel()
    self.arms = {s: Arm(model, data, s, prefix) for s in hm.SIDES}
    self.grippers = {s: Gripper(model, data, s, prefix, key) for s in hm.SIDES}
    #: The waist, turned to the posture's angle (`posture`'s last), its
    #: command ramped there (`WAIST_RATE`).
    self.waist = Waist(model, data, prefix)
    self.waist_cmd = np.zeros(len(hm.WAIST))
    self.waist_speed = np.zeros(len(hm.WAIST))
    stand_q = hm.stand_qpos()
    self.rest = {s: np.array([stand_q.get(f"{s}_{j}", 0.0) for j in hm.ARM])
                 for s in hm.SIDES}
    self.phase = {s: "rest" for s in hm.SIDES}
    #: Each hand's goal, the pose its command is at on its line, the joint
    #: target its arm walks to in joint space, and what it last commanded.
    self.goal: dict[str, HandGoal | None] = {s: None for s in hm.SIDES}
    self.cmd: dict[str, HandGoal | None] = {s: None for s in hm.SIDES}
    self.q_to = {s: self.rest[s].copy() for s in hm.SIDES}
    self.q_cmd = {s: self.rest[s].copy() for s in hm.SIDES}
    #: Each arm's way home still to walk after `q_to` (`way_home`), and its
    #: way out (`way_out`); and every joint as its driver last read it
    #: (`home`), unread at first.
    self.homes: dict[str, list] = {s: [] for s in hm.SIDES}
    self.outs: dict[str, list] = {s: [] for s in hm.SIDES}
    self.sensed = np.full(len(hm.JOINT_NAMES), np.nan)
    #: How far the last solve left the model's grip off its command, m, and
    #: whether the reach's plan had slack (`plan`).
    self.miss = {s: 0.0 for s in hm.SIDES}
    self.slack = {s: True for s in hm.SIDES}
    #: How long each hand has been out of reach or pushing (`BLOCKED_S`).
    self._off = {s: 0.0 for s in hm.SIDES}
    #: ...and what stopped each hand's last reach on its way (`blocked`), by
    #: the rule that did (#743): `by` "pushed" (on its way in, its force past
    #: `BLOCKED_N`; `n` the hardest it read, N), "reach" (on its way in, its own
    #: solve past `BLOCKED_M` off its command) or "caught" (on its way out,
    #: `_struck`); None where none stopped it so. Words alone read it.
    self.stopped: dict[str, dict | None] = {s: None for s in hm.SIDES}
    #: ...and of the run of ticks behind a stop on its way in (`_off`), how
    #: many pushed, how many in all, and the hardest push read, N: the push
    #: is said where it held at least half the run (#743's review: judged on
    #: the stop's tick alone, a push's last dip or a lone spike named the
    #: other cause).
    self._pushes = {s: (0, 0, 0.0) for s in hm.SIDES}
    #: ...and how long its way out's reading has tipped the body past the
    #: line, running (`OUT_TIP_S`).
    self._tip = {s: 0.0 for s in hm.SIDES}
    #: ...and how long its way home's reading has tipped the body away from
    #: the hand past its line, running (`HOME_TIP_S`) -- a way home a walk's
    #: hand-over sent, watched so (`watch_home`, `retreat(watch=)`).
    self._shove = {s: 0.0 for s in hm.SIDES}
    self.watch_home = {s: False for s in hm.SIDES}
    #: ...and how long its way home's drivers have read the arm past
    #: `STALL_RAD` behind its command, running (`STALL_S`).
    self._stall = {s: 0.0 for s in hm.SIDES}
    self.every = max(1, round(EVERY_S / model.opt.timestep))
    self.steps = 0
    #: The posture the plan wants -- the pelvis's height off the stand's,
    #: the trunk's pitch, the waist's yaw -- and the trim on what the
    #: standing policy is asked (`TRIM_RATE`), its height and pitch.
    self.posture = np.zeros(2 + len(hm.WAIST))
    self.trim = np.zeros(2)
    self.tau = np.zeros(len(hm.JOINT_NAMES))
    self.qd = np.zeros(len(hm.JOINT_NAMES))
    #: The body's own motion as its senses have it, taken into its hands'
    #: force (`arms.Motion`, #631).
    self.motion = Motion(self.every, model.opt.timestep)
    #: The arm joints' and the waist's friction as calibrated and stored (`friction.FILE`),
    #: taken off the drivers' torques where a joint turns; and what every
    #: joint's may hold still (`friction.still`), which the whole body's
    #: reading weighs it by.
    self.friction = friction.load()
    sm = self.self_model
    self.still = friction.still(self.friction, sm.m.dof_frictionloss[sm.vadr])
    #: Whether the standing policy has the legs, both soles down: the body's
    #: owner says, every step (`HumanoidMission`); the whole body's reading
    #: assumes it, so until it is said the arm's alone is read.
    self.standing = False
    #: The force each hand pushes with, N, stance frame: the whole body's
    #: reading for a hand at work, standing, else its arm's (`FELT_PHASES`)
    #: -- and its arm's alone, whichever is read.
    self.force = {s: np.zeros(3) for s in hm.SIDES}
    self.arm_force = {s: np.zeros(3) for s in hm.SIDES}
    #: ...whether each hand's `force` this command is the whole body's balance,
    #: closed: where it is not, the arm's alone, which takes the body as still,
    #: stands -- and a carry's rule reads no load there (#699; `carry`, `_held`).
    #: Not kept: read afresh each command before anything reads it, and no hand
    #: that holds nothing is out across a save (`Body.working`).
    self.closed = {s: False for s in hm.SIDES}
    #: ...and what the robot reads of it (`HELD_PHASES`, #631).
    self.sensed_force = {s: np.zeros(3) for s in hm.SIDES}
    #: Where each grip was by its senses at the last command, stance frame
    #: (`_update`): what a stop condition reads (#554), so that reading it
    #: draws nothing more on the drivers' noise; None before the first.
    self.sensed_grip: dict[str, np.ndarray | None] = {s: None for s in hm.SIDES}
    #: How fast each hand's command goes along its line, m/s; a carry's
    #: state; and how its last carry ended (`report`).
    self.speed = {s: HAND_SPEED for s in hm.SIDES}
    self.pulls: dict[str, dict | None] = {s: None for s in hm.SIDES}
    self.report: dict[str, dict | None] = {s: None for s in hm.SIDES}
    #: Each reach's looks (`_look`, #627): the postures it may take, those it
    #: has left, how often it has looked, and whether its hand waits at its
    #: pre-grasp while its body takes the posture a look chose.
    self.allowed: dict[str, list] = {s: [] for s in hm.SIDES}
    self.left: dict[str, list] = {s: [] for s in hm.SIDES}
    self.looks = {s: 0 for s in hm.SIDES}
    self.looking = {s: False for s in hm.SIDES}
    self._planner: SelfModel | None = None
    #: The other robots' bodies a hand's way keeps off (#678): each one's
    #: soles' middle (x, y), stance frame, as the body's look placed them
    #: as it began a reach, a withdraw or a walk's hand-over, and none once
    #: it is over (`HumanoidMission.keep_out`); and, by side, how the last
    #: way judged against them was refused (`_kept_off`), else None.
    self.keep_out: list = []
    self.kept: dict[str, dict | None] = {s: None for s in hm.SIDES}
    #: ...and, by side, how near the robot's own trunk the last way out
    #: judged against it came where none kept clear (`_own_kept`, #722): of
    #: the ways tried, the least near's part of the trunk (`part`, torso or
    #: head) and its least room, m (`m`); else None.
    self.own: dict[str, dict | None] = {s: None for s in hm.SIDES}
    #: ...and, by side, the way home the last judgment found clear and the
    #: arm it began from (`home_clear`): the way `_go_home` then takes.
    self.judged_home: dict[str, tuple | None] = {s: None for s in hm.SIDES}
    #: By side, the carry pose a walk's hand-over takes that arm to and
    #: holds it at (#682, `carry_pose`) -- where its way home ends -- else
    #: None: home is its rest. A new reach, a joint moved by hand and a
    #: let-go end it (`drop_carry`).
    self.carry_q: dict[str, np.ndarray | None] = {s: None for s in hm.SIDES}
    #: By side, the other hand whose grip kept the body in its posture as
    #: this hand's last reach, carry or pull was asked (#719's option 2), else
    #: None (`Body.posture_kept_by`).
    self.kept_by: dict[str, str | None] = {s: None for s in hm.SIDES}
    #: Whether the stand waits for an arm on its way home (`_go_home`, #693):
    #: asked once every arm is home or carried (`_update`).
    self.stand_home = False
    #: By side, whether its way home is still to be read home by its drivers
    #: (`_go_home`), how long they have read it held at its rest, s, and how
    #: often it has been lifted since its way home began (`HELD_S`, `LIFTS`).
    self.homing = {s: False for s in hm.SIDES}
    self.held_for = {s: 0.0 for s in hm.SIDES}
    self.lifted = {s: 0 for s in hm.SIDES}

  # ---- what a caller asks ----

  def reach(self, side: str, goal: HandGoal, approach: float = APPROACH_M,
            postures=REACH_POSTURES, flip: bool = True, refuse: bool = True,
            verb: bool = False) -> bool:
    """A hand to `goal`: planned, then moved, approached and held. False
    (nothing moves) if no posture the policy holds puts it in reach. ⚠ A
    TWO-JAW GRIP IS THE SAME GRIP HALF A TURN ROUND ITS FINGERS (#493): where
    the one asked plans no reach, the other is planned -- the left hand
    reached a chest's knob with its jaws closing to its right in twice the
    places it did with them closing to its left, its wrist's roll at the
    end of its range. `postures` are the ones it may take, `flip` whether
    it may take the other grip. Where another robot stands by (`keep_out`,
    #678), its way out and its way home from there keep off it: none that
    does, and nothing moves (`kept` says how) -- or, not `refuse`, the way
    out that comes least near it is taken (`_kept_off`). From a hand already
    out, its way out keeps off its own trunk too (#722, `_own_kept`): none
    that does, and a `verb`'s moves nothing (`own` says how); the loop's own
    (a charge's: no wait frees it) takes the way that comes least near it.
    And its PLAN keeps the arm off its own trunk at the pre-grasp and at the
    goal (#726, `plan`'s `own`): where every plan that reaches puts it in
    the trunk, a `verb`'s moves nothing (`own` says how near the first came),
    and the loop's own takes that one. ⚠ NO OTHER POSTURE
    WHILE THE OTHER HAND HOLDS SOMETHING AT ITS WORK (#719's option 2,
    `keep`): only the one it is in -- a `verb`'s reach, never the loop's own
    (a charge's)."""
    self.stopped[side] = None
    kept = self.keep(side, verb)
    if kept is not None:
      postures = (kept,)
    plan_, goal, own = self._planned(side, goal, approach, postures, flip)
    self.kept[side] = self.judged_home[side] = self.own[side] = None
    if own is not None:
      # every plan that reaches puts the arm in its own trunk (#726): a
      # verb's moves nothing; the loop's own -- no wait frees it -- takes it
      self.own[side] = {"part": own["part"], "m": own["m"]}
      if refuse and verb:
        return False
      plan_, goal = own["plan"], own["goal"]
    if plan_ is None:
      return False
    posture, q_pre, _, slack = plan_
    outs = self.way_out(side, q_pre, posture, refuse=refuse, own=refuse and verb)
    if outs is None:
      # every way out meets another robot's body (#678), or its own (#722):
      # nothing moves, a hand carried still carried
      return False
    if refuse and not self._returns(side, q_pre, posture):
      # ...nor goes where every way home from would: stranded there, its
      # withdraw and every walk the verbs ask are refused (#678)
      self.kept[side]["back"] = True
      return False
    # ...a hand reached out again carries no longer (#682): its way home is
    # its rest once more
    self.carry_q[side] = None
    self.slack[side], self.outs[side] = slack, outs
    self.hold_posture(*posture)
    self._off[side] = self._tip[side] = 0.0
    self.speed[side] = HAND_SPEED
    self.goal[side] = goal
    self.q_to[side] = self.outs[side].pop(0)
    self.allowed[side] = [tuple(float(v) for v in p_) for p_ in postures]
    self.left[side], self.looks[side], self.looking[side] = [], 0, False
    self.phase[side] = "move"
    # ...a way home it cut short is over: no later rest reads it held
    self.homing[side] = False
    return True

  def clear_way(self, side: str, goal: HandGoal, approach: float = APPROACH_M,
                postures=REACH_POSTURES, flip: bool = True) -> bool | str | None:
    """Whether `reach` would plan `goal` for `side` and take a way out that
    keeps off every other robot's body (#678) -- None where it plans none,
    "own" where every plan, or every way out, would put the arm in its own
    trunk (#726, #722); nothing moved, `kept` and `own` as they were."""
    plan_, _, in_own = self._planned(side, goal, approach, postures, flip)
    if plan_ is None:
      return "own" if in_own is not None else None
    kept, own = self.kept[side], self.own[side]
    self.own[side] = None
    out = self.way_out(side, plan_[1], plan_[0])
    clear = ("own" if out is None and self.own[side] is not None
             else out is not None and self._returns(side, plan_[1], plan_[0]))
    self.kept[side], self.own[side] = kept, own
    return clear

  def _returns(self, side: str, q_pre, posture) -> bool:
    """Whether some way home from the pre-grasp `q_pre` keeps off every
    other robot's body (#678), in `posture` and stood up; True with none
    by. False sets `kept`."""
    return not len(self.keep_out) or self._judged(
      side, np.asarray(q_pre, dtype=float).copy(), self.rest[side].copy(), out=False,
      poses=(posture, (0.0, 0.0, 0.0)), refuse=True) is not None

  def plans(self, side: str, goal: HandGoal, approach: float = APPROACH_M,
            postures=REACH_POSTURES, flip: bool = True) -> bool:
    """Whether a procedure's `reach` would plan `goal` for `side`: its arm
    kept off its own trunk (#726) -- nothing moved."""
    return self._planned(side, goal, approach, postures, flip)[0] is not None

  def _planned(self, side, goal, approach, postures, flip):
    """(the plan, its goal -- half a turn round where that grip was tried --
    and None), its arm kept off its own trunk (#726, `plan`'s `own`); or,
    where every plan that reaches puts it in the trunk, (None, the goal,
    `own`: how near the first came, its `plan` and its `goal`)."""
    # ...on the robot as it reads itself standing, the other arm where it is
    # and this one's answer nearest it as it is (#627)
    sm = self._planner_model()
    kw = {"lean": self._lean(), "arms": self._arms(but=side), "start": self.q_cmd[side]}
    own: dict = {}
    plan_ = plan(side, goal, sm, approach=approach, postures=postures, own=own, **kw)
    if own:
      own["goal"] = goal
    if plan_ is None and goal.rot is not None and flip:
      flipped = np.asarray(goal.rot, dtype=float) @ np.diag([-1.0, -1.0, 1.0])
      goal = HandGoal(np.asarray(goal.pos, dtype=float), flipped, goal.stiffness)
      # ...where the grip asked reaches only through the trunk too: the same
      # grip half a turn round may keep off it (#493)
      also: dict = {}
      plan_ = plan(side, goal, sm, approach=approach, postures=postures, own=also, **kw)
      if also and not own:
        own = dict(also, goal=goal)
    return plan_, goal, (own or None) if plan_ is None else None

  def _lean(self) -> Lean | None:
    """How the robot reads itself standing now (`Lean.read`): its model of
    itself as its arms' loop last posed it -- never its joints read again,
    which would draw on their noise -- in the posture it was asked; None
    before its first reading."""
    if np.isnan(self.sensed).any():
      return None
    return Lean.read(self.self_model, self.posture, self._planner_model())

  def _arms(self, but: str | None = None) -> dict:
    """The arms where they are commanded, by joint name -- all but `but`'s."""
    return {k: v for s in hm.SIDES if s != but for k, v in _arm(s, self.q_cmd[s]).items()}

  def stop(self, side: str, why: str) -> None:
    """A carry ended from outside (its caller's patience), held where the
    hand IS, as its own stops are."""
    if self.phase[side] == "carry":
      self._end(side, why, *self.grip_now(side))

  def halt(self, side: str, why: str) -> bool:
    """A hand's move stopped now, held where it is (#554: a procedure's stop
    condition): a carry or a pull where the hand IS, as its own stops hold
    it (`stop`, its report saying `why`); a hand on its way in, or holding
    with its goal moved (`move_goal`), where its command is, as one stopped
    on its way is (`blocked`); an arm on its way out to its pre-grasp where
    its joints are commanded, and the body in the posture it has got to --
    out there, its goal and its command the grip they put it at, so it may
    be carried on or withdrawn as any hand that is out. ⚠ THE POSTURE TOO:
    held at its joints alone, an arm stopped 0.15 s into a reach rode the
    crouch its plan was still taking 17 cm down, and the next verb, starting
    from a command the body had left, stepped a joint 1.4 rad in one
    command. True where a move was under way; False, nothing changed, where
    none was."""
    phase = self.phase[side]
    if phase == "carry":
      self.stop(side, why)
      return True
    if phase in ("approach", "pull"):
      self.phase[side], self.looking[side] = "blocked", False
      return True
    if phase != "move":
      return False
    self.hold_posture(*self.posture_now())
    # the grip its commanded joints put it at, the body as last posed
    sm = self._planner_model()
    sm.q = self.self_model.q.copy()
    sm.d.qpos[:] = self.self_model.d.qpos
    sm._set_arm(side, self.q_cmd[side])
    pos, rot = sm.grip_pose(side)
    k = self.goal[side].stiffness
    self.goal[side], self.cmd[side] = HandGoal(pos.copy(), rot.copy(), k), HandGoal(pos, rot, k)
    self.q_to[side] = self.q_cmd[side].copy()
    self.outs[side] = []
    self.phase[side] = "blocked"
    return True

  def _tipping(self, side: str) -> float:
    """How hard what a hand's arm reads would tip the body toward the hand,
    N m (`OUT_TIP_NM`, #708): its pull toward the body at the hand's height,
    less its press down at the hand's reach -- off the force its arm reads it
    push with (`force`) and where its senses put it, stance frame."""
    at = self.sensed_grip[side]
    if at is None:
      return 0.0
    reach_m = math.hypot(float(at[0]), float(at[1]))
    if reach_m < 1e-6:
      return 0.0
    f = self.force[side]
    pull = -(float(f[0]) * float(at[0]) + float(f[1]) * float(at[1])) / reach_m
    return float(at[2]) * pull + reach_m * float(f[2])

  def _tipped(self, side: str, dt: float) -> bool:
    """Whether what a hand's arm reads has tipped the body toward the hand
    past `OUT_TIP_NM` (#708): the tick it does with the body in the posture
    it was asked (`POSTURE_TOL`), else once it has for `OUT_TIP_S` running;
    `dt` its tick."""
    self._tip[side] = self._tip[side] + dt if self._tipping(side) > OUT_TIP_NM else 0.0
    if not self._tip[side]:
      return False
    there = (np.abs(self.posture_now() - self.posture) <= POSTURE_TOL).all()
    return bool(there) or self._tip[side] >= OUT_TIP_S - 1e-9

  def _struck(self, side: str) -> None:
    """A hand on its way out caught on something (`OUT_TIP_NM`, #708):
    stopped as `halt` stops an arm on its way out -- the body in the posture
    it has got to, the hand's goal and command the grip its arm then puts it
    at, `blocked` -- but its arm held where its drivers read its joints, so
    that it no longer drives on into what it met."""
    self.q_cmd[side] = self.sensed[self.self_model.arm[side]].copy()
    self.halt(side, "blocked")
    self.stopped[side] = {"by": "caught"}

  def ease(self, side: str) -> bool:
    """A hand held where it is -- holding (`hold`) or stopped on its way
    (`blocked`, pressed hardest) -- whose force pushes on along its fingers
    past `EASE_N`: held where its senses put it, its turn kept, its command
    walked there along its line at `HAND_SPEED` (a setpoint is ramped), its
    arm's spring no longer pressing it on what it met. True where it was."""
    c, at = self.cmd[side], self.sensed_grip[side]
    if self.phase[side] not in ("hold", "blocked") or c is None or at is None:
      return False
    # ...its fingers' way as its loop last posed it: a fresh read of its
    # joints would draw on their noise
    rot = self.self_model.grip_pose(side)[1] if c.rot is None else np.asarray(c.rot, dtype=float)
    if float(self.force[side] @ -rot[:, 2]) <= EASE_N:
      return False
    self.goal[side] = HandGoal(at.copy(), None if c.rot is None else rot.copy(), c.stiffness)
    self.speed[side], self.phase[side] = HAND_SPEED, "hold"
    return True

  def out(self, side: str) -> bool:
    """Whether a hand is out at a pose of its own: holding it, moving on
    from it or stopped on its way (what `carry` moves on)."""
    return self.goal[side] is not None and self.phase[side] in (
      "hold", "pull", "blocked", "carry")

  def _planner_model(self) -> SelfModel:
    if self._planner is None:
      self._planner = SelfModel()
    return self._planner

  def posture_for(self, side: str, pos, rot=None, onward: bool = False,
                  past=(), postures=None) -> tuple | None:
    """The least-bent posture (`CARRY_POSTURES`: the legs' and the waist's)
    from which `side`'s grip reaches `pos` -- turned to `rot`, or at any
    turn -- on the robot's model of itself as it would stand in each, its
    arms where they are (`statue`, as it reads itself standing: `Lean`), the
    arm starting where it is and from `SEEDS`; or None if none reaches it.
    `onward`: a held hand going on, whose arm missed `pos` in the posture it
    holds -- another posture, from the arm as it is alone, and none of
    `past`; `postures`, those alone (`keep`). ⚠ AS IT STANDS, NEVER A STATUE
    BALANCED AT ITS SOLES' MIDDLE: a carry re-planned on one took the
    posture it held again, and ended out of reach (#626, #627)."""
    sm = self._planner_model()
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    lean, arms = self._lean(), self._arms()
    starts = [self.q_cmd[side]] + ([] if onward else [seed(side, s_) for s_ in SEEDS])
    # the posture it holds first, where it still reaches: a change of
    # posture mid-move is a change of the arm's whole branch -- ⚠ and going
    # on, never the posture it holds, from a fresh seed: up a lid's arc the
    # arm stood at its elbow's stop 2 cm short of its way for 9 s, while
    # another branch of the same posture "reached" (#525)
    if postures is not None:
      order = list(postures)
    else:
      here = tuple(float(v) for v in self.posture)
      gone = {here} | {tuple(float(v) for v in p_) for p_ in past}
      order = ([p_ for p_ in CARRY_POSTURES if tuple(p_) not in gone] if onward
               else [here] + [p_ for p_ in CARRY_POSTURES if tuple(p_) != here])
    for posture in order:
      if not stands(sm, posture, q0) or out_of_reach(sm, side, pos, -LEAN_MOST_M):
        continue
      statue(sm, posture, q0, arms, lean, fast=True)
      if out_of_reach(sm, side, pos):
        continue
      for start in starts:
        sm._set_arm(side, start)
        _, ep, er = sm.solve(side, pos, rot, iters=80)
        if ep < REACH_POS_TOL and er < REACH_ROT_TOL:
          return tuple(posture)
    return None

  def path_for(self, side: str, poses, rot=None, postures=None) -> tuple | None:
    """A posture (`CARRY_POSTURES`, the least bent first) from which `side`'s grip
    FOLLOWS `poses` (stance frame) in turn at the turn `rot` -- or at each
    pose its own, where `rot` is a list -- the first
    solved from the seeds, each next ON from the last, as a moving arm must
    -- or None: on the robot as it stands in the posture it holds (below),
    else as it would stand in another (`statue`, as it reads itself
    standing: `Lean`), its arms where they are and this one, point by point,
    where the last point put it. Each point further from its shoulder than
    the first is reached `LIVE_MARGIN_M` further ahead too: the lean its
    load still brings. ⚠ A point
    reached from a fresh seed is not a path: the arm went up a lid's arc in
    one branch, and where only another reached on, it swung between them
    and jerked the lid shut at 3.3 rad/s (#493). `postures`: those alone
    (`keep`)."""
    sm = self._planner_model()
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    poses = [np.asarray(p, dtype=float) for p in poses]
    rots = list(rot) if isinstance(rot, (list, tuple)) else [rot] * len(poses)
    ahead = np.array([LIVE_MARGIN_M, 0.0, 0.0])
    shoulder_id = sm.m.body(f"{side}_shoulder_roll_link").id
    # a hand that is out goes on from the arm as it IS, its posture first:
    # the branch its grip left it in is the branch its path must follow --
    # where the path starts at the hand (a reach planned while a hand is
    # out, stopped short, starts elsewhere)
    out = (self.out(side)
           and float(np.linalg.norm(poses[0] - self.grip_now(side)[0])) < PATH_POS_TOL)
    here = tuple(float(v) for v in self.posture)
    order = ([here] + [p_ for p_ in CARRY_POSTURES if tuple(p_) != here]) if out \
      else CARRY_POSTURES
    if postures is not None:
      order = [tuple(float(v) for v in p_) for p_ in postures]
    lean, arms = self._lean(), self._arms()
    for posture in order:
      # ⚠ ...AND THE ROBOT AS IT STANDS, its model posed off its senses, in
      # the posture it holds: up a lid's arc it stood 4 cm behind the
      # statue of that posture, which could not reach where its hand WAS
      live = out and tuple(posture) == here
      if live:
        # ...its arm as its encoders read it, the branch its grip is in,
        # never as commanded: a lid's weight held the hand 3.5 cm off its
        # command at the arc's top, from which the path down solved short
        sm.q = self.self_model.q.copy()
        sm.d.qpos[:] = self.self_model.d.qpos
        mujoco.mj_kinematics(sm.m, sm.d)
        q, ep, er = sm.q[sm.arm[side]].copy(), 0.0, 0.0
      else:
        # another posture than the one it holds: where it is held there,
        # from the arm as it is or any other way (the live solve carries the
        # arm there as the body moves)
        if not stands(sm, posture, q0) or out_of_reach(sm, side, poses[0], -LEAN_MOST_M):
          continue
        statue(sm, posture, q0, arms, lean, fast=True)
        if out_of_reach(sm, side, poses[0]):
          continue
        q, ep, er = solve_seeded(sm, side, poses[0], rots[0],
                                 self.q_cmd[side] if out else None)
      if ep >= PATH_POS_TOL or er >= HOLD_ROT_TOL:
        continue
      reach0 = float(np.linalg.norm(poses[0] - sm.d.xpos[shoulder_id]))
      for p, r_ in zip(poses[1:], rots[1:]):
        if not live:
          # ...balanced with its arm out where it is going (`static_legs`)
          statue(sm, posture, q0, {**arms, **_arm(side, q)}, lean, fast=True)
        further = float(np.linalg.norm(p - sm.d.xpos[shoulder_id])) > reach0
        target = p + ahead if further else p
        sm._set_arm(side, q)
        q, ep, er = sm.solve(side, target, r_, iters=80)
        if ep >= PATH_POS_TOL or er >= HOLD_ROT_TOL:
          break
      else:
        return tuple(posture)
    return None

  def carry(self, side: str, pos, stiffness: float | None = None,
            speed: float = HOLD_SPEED, path=None, turns=None, verb: bool = False) -> bool:
    """A hand that is out (`out`) carried from where it IS along a straight
    line to `pos`, stance frame, at `speed`, its turn kept, at `stiffness`
    (or its goal's), the posture the least bent that reaches the end --
    through `path`'s points first, from one posture that follows them all
    (`path_for`), turning to `turns`' at each (one a point of `path` and
    one at `pos`, None keeping the one before), where given. False (nothing moves) for a hand that is
    not out, or an end no posture reaches. A procedure's carry (`verb`) of a
    hand that holds nothing takes no other posture while the hand reads a
    load (`CARRY_PRESSED_SHARE`, #699): as it is asked, the one the body is
    in (or where it has got to); on its way, none until it reads light; and
    one coming is given up where it reads a load, the body backing off until
    it reads none. How it ends is its `report` -- a `verb`'s that began on a
    held grip ends "let go" where that grip is lost (`_held`)."""
    kept = self.keep(side, verb)
    if not self.out(side):
      return False
    h, r = self.grip_now(side)
    # ⚠ THE TURN IT WAS ASKED TO KEEP, never the one it sagged to: planned
    # from a hand stretched up a lid's arc, the wrist's 0.1-0.15 rad of give
    # put the path back down out of every posture's reach (#493)
    if self.cmd[side] is not None and self.cmd[side].rot is not None:
      r = np.asarray(self.cmd[side].rot, dtype=float)
    end = np.asarray(pos, dtype=float)
    ways = [np.asarray(p, dtype=float) for p in (path or ())]
    rots, last = [], r
    for t in (turns if turns is not None else [None] * (len(ways) + 1)):
      # a point with no turn keeps the one before: the hand's own, first
      last = last if t is None else np.asarray(t, dtype=float)
      rots.append(last)
    if len(rots) != len(ways) + 1:
      raise ValueError(f"a turn for each of the path's {len(ways)} points and its end, "
                       f"not {len(rots)}")
    # ...none but the one it is in while the other hand holds something, a
    # `verb`'s carry (#719)
    only = {} if kept is None else {"postures": (kept,)}
    if ways or turns is not None:
      # a PATH: one posture that follows it all, its points walked in turn
      posture = self.path_for(side, [h] + ways + [end], rot=[r] + rots, **only)
    else:
      posture = self.posture_for(side, end, rot=r, **only)
    if posture is None:
      self.report[side] = {"why": "out of reach", "shortM": float(np.linalg.norm(end - h)),
                           "moved": 0.0, "seconds": 0.0}
      return False
    # ...nor -- a `verb`'s hand that holds nothing -- another while it reads
    # a load (`CARRY_PRESSED_SHARE`, #699): it goes as far as its arm reaches
    # from where the body is, held there where a posture still comes
    open_ = verb and not self.grippers[side].held()
    loaded = False
    if open_ and self.closed[side] and self._pressed(self.force[side], 0.0, CARRY_PRESSED_SHARE):
      here = self._in_or_got_to()
      loaded, posture = tuple(posture) != here, here
    k = self.goal[side].stiffness if stiffness is None else float(stiffness)
    moved_posture = tuple(posture) != tuple(float(v) for v in self.posture)
    began = self._standing_in() if moved_posture and open_ else None
    self.hold_posture(*posture)
    self.cmd[side] = HandGoal(h.copy(), r.copy(), k)
    self.goal[side] = HandGoal(ways[0] if ways else end, rots[0].copy(), k)
    self.speed[side], self._off[side] = float(speed), 0.0
    fixed = bool(ways) or turns is not None
    self.pulls[side] = {"start": h.copy(), "t": 0.0,
                        "ways": list(zip(ways[1:] + [end], rots[1:])) if ways else [],
                        "fixed": fixed,
                        "open": open_, **({"loaded": True} if loaded else {}),
                        # ...withheld: a path's posture planned again once it reads light (`_held`)
                        **({"recheck": True} if loaded and fixed else {}),
                        # ...a `verb`'s, a grip it began with: lost, it ends there (`_held`)
                        **({"gripped": True} if verb and not open_ else {})}
    if moved_posture:
      # the hand holds still while the body takes the posture (`_held`) --
      # withheld, the one it is held in where it has got to: none to give up
      self.pulls[side]["postured"] = 0.0
      if loaded:
        self.pulls[side]["held_here"] = True
    if began is not None:
      self.pulls[side]["began"] = began
    if loaded:
      # ...its reading the mean it holds: read light on a command whose
      # balance did not close, a replan took the posture withheld (`_held`)
      self.pulls[side]["fm"] = np.asarray(self.force[side], dtype=float).copy()
    self.report[side] = None
    self.phase[side] = "carry"
    return True

  def pull(self, side: str, way, force: float, distance_m: float,
           speed: float = HOLD_SPEED, give: bool = True, verb: bool = False) -> bool:
    """A hand that is out PULLED along `way` (stance frame) -- or pushed: a
    FORCE MOVE (#525) -- with at most `force`, N, as the hand reads it,
    and at most `speed`, m/s. Across its way, GIVING, the hand yields to
    what pushes it, no faster than `ACROSS_SPEED` times its speed, and its
    turn gives to what it holds (`GIVE_RATE`) but about the jaws' own line;
    not giving, it holds its line and its turn -- hybrid force and
    position, the robot arm's old division. Until it has gone `distance_m` along its way ("distance"), it
    is held fast, a grip it began with is lost ("let go"), or it can reach
    no further. The posture is the one from which the hand reaches its
    way's end, taken first; a hand held takes no other (`GO_ON_M`); a hand
    that holds nothing takes none while it reads a load, and one that comes
    is given up where it reads a load meanwhile, the body backing off until
    it reads none (`PRESSED_SHARE`, #699).
    False (nothing moves) for a hand that is not out. How it ends is its
    `report`."""
    if force <= 0.0:
      raise ValueError(f"a pull pushes with some force, not {force} N")
    kept = self.keep(side, verb)
    if not self.out(side):
      return False
    h, r = self.grip_now(side)
    if self.cmd[side] is not None and self.cmd[side].rot is not None:
      r = np.asarray(self.cmd[side].rot, dtype=float)
    w = np.asarray(way, dtype=float)
    w = w / max(float(np.linalg.norm(w)), 1e-9)
    # the posture its way's end is reached from, the hand held still while
    # it comes: a posture changed partway jerks what the hand holds -- and
    # none while a hand that holds nothing reads a load (`PRESSED_SHARE`)
    gripped = self.grippers[side].held()
    # ...kept (#719), the one the body stands in: one still coming held where it got to
    posture = (None if not gripped and self._pressed(self.force[side], force)
               else kept if kept is not None
               else self.path_for(side, [h, h + float(distance_m) * w], rot=r))
    moved = posture is not None and tuple(posture) != tuple(float(v) for v in self.posture)
    began = self._standing_in() if moved else None
    if moved:
      self.hold_posture(*posture)
    self.cmd[side] = HandGoal(h.copy(), r.copy(), PULL_STIFFNESS)
    self.goal[side] = HandGoal(h.copy(), r.copy(), PULL_STIFFNESS)
    self.speed[side], self._off[side] = float(speed), 0.0
    self.pulls[side] = {"start": h.copy(), "t": 0.0, "ways": [], "fixed": True, "peak": 0.0,
                        **({"postured": 0.0, "began": began} if moved else {}),
                        "pull": {"way": w, "force": float(force), "distance": float(distance_m),
                                 "give": bool(give), "gripped": gripped,
                                 "last": h.copy(), "went": 0.0, "trail": [h.copy()],
                                 "still": 0.0, "anchor": h.copy(), "pinned": 0.0,
                                 "past": [(0.0, 0.0)]}}
    self.report[side] = None
    self.phase[side] = "carry"
    return True

  @staticmethod
  def _pressed(read, ask: float, share: float = PRESSED_SHARE) -> bool:
    """Whether a pull's hand reads a load (`PRESSED_SHARE`): `read`, N, any
    way, past `share` times its `ask`, of `PRESSED_ASK_N` at least."""
    return float(np.linalg.norm(read)) > share * max(float(ask), PRESSED_ASK_N)

  def keep(self, side: str, verb: bool = False) -> tuple | None:
    """As a hand's reach, carry or pull is asked (#719's option 2): the
    posture it is kept in (`keeps`), else None; which hand held, by side,
    in `kept_by`, which the move's posture changes on its way read, and
    every way into them is a move that sets it here."""
    kept = self.keeps(side, verb)
    self.kept_by[side] = None if kept is None else OTHER_SIDE[side]
    return kept

  def keeps(self, side: str, verb: bool = False) -> tuple | None:
    """The posture a hand's reach, carry or pull asked now would be kept in
    (`keep`), nothing recorded: the posture the body is in, where the other
    hand is out at its work (`out`) and its gripper reads it holds something
    -- a verb's move, never the loop's own (`verb`: a charge's, a walk's
    lift) -- else None. A piece carried at the side (#682) holds the body in
    nothing. The posture kept is the one it holds where the body is within
    `POSTURE_TOL` of it, else where it has got to: never a sensed reading
    in place of the plan for a trunk that only sways."""
    other = OTHER_SIDE[side]
    held = verb and self.out(other) and self.grippers[other].held()
    if not held:
      return None
    return self._in_or_got_to()

  def _in_or_got_to(self) -> tuple:
    """The posture the body is in, as a move is asked that may take no other
    (`keeps`, a loaded carry's): the one it holds where the body is within
    `POSTURE_TOL` of it, else where it has got to -- never a sensed reading
    in place of the plan for a trunk that only sways."""
    now = self.posture_now()
    there = (np.abs(now - self.posture) <= POSTURE_TOL).all()
    return tuple(float(v) for v in (self.posture if there else now))

  def _standing_in(self) -> tuple:
    """The posture the body stands in as another is asked of it: the one it
    holds, or -- one still coming -- where it has got to, as a move stopped
    is held (`halt`)."""
    return tuple(float(v) for v in (self.posture if self._posture_held() else self.posture_now()))

  def _look(self, side: str) -> bool:
    """Whether a hand at its pre-grasp goes in from the posture it holds
    (#627): as the robot reads itself standing now, its arm out (`Lean`),
    the least bent of the postures its reach may take that puts its goal in
    reach -- with its slack, else at full stretch, as `plan` chooses -- is
    the one it holds, or none is, or it has looked `LOOKS` times. Else that
    posture is taken, never one it has left this reach, and it looks again
    once there; its hand waits at its pre-grasp meanwhile (`looking`). ⚠
    EITHER WAY: the table's cup is reached by an arm over the top, which leant
    the policy 1.5 cm forward where a natural arm leans it 1.1 back, so its
    plan bows deeper than it need -- out of its tags' view. ⚠ A POSTURE
    WHOSE PLAN FROM THE ARM AS IT IS PUTS IT IN ITS OWN TRUNK IS PASSED OVER
    (#726, `in_trunk`): at the goal it goes in to, and -- another posture --
    at the pre-grasp its arm holds the hand at while the body takes it; in
    the one it holds the hand is at its pre-grasp already (judged there too,
    a look crouched over a plan 0.3 mm into the torso). Never for another
    seed's arm: the hand goes in from where it is. With none that keeps off
    it, the least bent that reaches is taken all the same."""
    if self.looks[side] >= LOOKS or self.kept_by[side]:
      return True
    self.looks[side] += 1
    sm = self._planner_model()
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    goal = self.goal[side]
    pre = pregrasp(goal)
    lean, arms = self._lean(), self._arms()
    here = tuple(float(v) for v in self.posture)
    gone = {tuple(float(v) for v in p_) for p_ in self.left[side]}
    chosen = first = None
    for slack in (True, False):
      for posture in (tuple(float(v) for v in p_) for p_ in self.allowed[side]):
        if (posture in gone or not stands(sm, posture, q0)
            or out_of_reach(sm, side, goal.pos, -LEAN_MOST_M)):
          continue
        got = _reached(sm, side, posture, q0, goal, pre, slack, lean, arms, self.q_cmd[side])
        if got is None:
          continue
        first = first or posture
        q_pre, q = got[:2]
        if (in_trunk(side, q) if posture == here else in_trunk(side, q_pre, q)) is None:
          chosen = posture
          break
      if chosen is not None:
        break
    chosen = chosen or first
    if chosen is None or chosen == here:
      return True
    self.left[side].append(here)
    self.hold_posture(*chosen)
    return False

  def _reaching(self, side: str, pos: np.ndarray, rot) -> np.ndarray | None:
    """The arm's joints that put the grip at `pos` and `rot` in the posture
    held -- on the robot's model of itself -- or None where none does."""
    q, ep, er = self.self_model.solve(side, pos, rot)
    return q if ep <= REPLAN_M and er <= REPLAN_ROT else None

  def hold_posture(self, height: float, pitch: float, waist: float = 0.0) -> None:
    """The posture to stand in, its legs' held by their own loop
    (`TRIM_RATE`), the waist turned to `waist`, rad, to the left positive --
    in place of a stand an arm's way home was keeping (`stand_home`)."""
    self.stand_home = False
    self.posture = np.array([height, pitch, waist], dtype=float)
    self.stand.ask(*(self.posture[:2] + self.trim))

  def turn_waist(self, waist: float) -> None:
    """The posture's waist alone turned to `waist`, rad (its command walks
    there, `turned`): the friction's calibration (`friction`)."""
    self.posture[2] = float(waist)

  def waist_home(self) -> bool:
    """Whether the waist is home: its posture's angle zero, commanded there,
    and its driver reading it within `HOME_RAD` (a walk and a get-up hold it
    there: neither policy knows it) -- or, a fresh reach's before its first
    reading, commanded there alone: a body boots and stands up with it home."""
    sensed = self.sensed[self.self_model.waist]
    read = ~np.isnan(sensed)
    return bool(not self.posture[2:].any() and not self.waist_cmd.any()
                and (np.abs(sensed[read]) <= HOME_RAD).all())

  def hold_waist(self) -> None:
    """Once a physics step while the get-up has every other joint: the
    waist walked home to zero (`turned`) and held there stiff, nothing
    read -- no model of itself is posed while it tumbles."""
    if self.steps % self.every == 0:
      self.posture[2:] = 0.0
      self._turn_waist(self.every * self.m.opt.timestep, np.zeros(len(hm.WAIST)))
    self.steps += 1

  def _turn_waist(self, dt: float, feed: np.ndarray) -> None:
    """The waist's command one step toward the posture's angle (`turned`),
    sent."""
    self.waist_cmd, self.waist_speed = turned(self.waist_cmd, self.waist_speed,
                                              self.posture[2:], dt)
    self.waist.command(self.waist_cmd, feed)

  def held_pitch(self) -> float:
    """The trunk's pitch the posture is held at, rad, by the robot's own
    reckoning: what the standing policy is asked once it is shown it (what
    it is shown on its way there), less the trim that ask carries, which
    only takes up what the policy misses by -- none where the ask is held
    at the deepest bow the policy trained on (`PITCH_RANGE`): taken off
    there too, an 8 s stand's trim (1.3 deg) read the floor's held bow
    2.3 deg tipped, not 1.5. Exactly 0 at the stand. The fall watch reads
    a tip off it (`getup.FallWatch`, #520)."""
    shown = self.stand.goal if self.stand.settled else self.stand.posture
    plan = float(self.posture[1])
    carried = min(max(plan + float(self.trim[1]), PITCH_RANGE[0]), PITCH_RANGE[1]) - plan
    return float(shown[1]) - carried

  def set_joints(self, side: str, q) -> None:
    """The arm to joint angles `q` (its six, `hm.ARM` order) in joint space
    and held there: the motor level's (`Body.ramp_routine`), its hand's
    goal let go, and a carry it held (#682)."""
    self.goal[side] = self.cmd[side] = None
    self.outs[side] = []
    self.carry_q[side] = None
    self.q_to[side] = np.asarray(q, dtype=float).copy()
    self.looking[side] = False
    self.phase[side] = "joints"
    self.homing[side] = False

  def stiffen(self, side: str, stiffness: float) -> None:
    """The stiffness the hand's commands carry from now (a share of
    `model.ARM_GAIN_SCALE`'s gains)."""
    if self.goal[side] is not None:
      self.goal[side].stiffness = stiffness

  def move_goal(self, side: str, pos) -> None:
    """While holding: the hand's goal moved (a pull is the goal drawn back
    with a low stiffness), its command walking there on its line -- a
    `pull`, which pushes on what it holds and so never stops for it."""
    self.goal[side].pos = np.asarray(pos, dtype=float)
    self.speed[side] = HAND_SPEED
    self.phase[side] = "pull"

  def retreat(self, side: str, approach: float = APPROACH_M, watch: bool = False) -> None:
    """Out along the line it came in on, then home its own way
    (`way_home`); the posture back to the stand. An arm on its way home
    already goes on. `watch`, a walk's hand-over's: the way home is held
    where it would tip the body away from the hand (`_shoved`); a
    withdraw's never, its word the arm's -- whichever began a way home
    under way, the caller now waiting on it says which."""
    self.watch_home[side] = watch
    if self.phase[side] in ("rest", "home", "retreat"):
      return
    self.lifted[side], self._shove[side] = 0, 0.0
    if self.phase[side] in ("joints", "move", "held") or self.cmd[side] is None:
      # an arm moved joint by joint (`set_joints`), still walking out to its
      # pre-grasp, or held where its way home stopped (`HELD_S`), has no
      # line to come out along: home from where its command is (unhandled,
      # a withdraw waited forever)
      self._go_home(side)
      return
    self.goal[side] = pregrasp(self.goal[side], approach)
    self.speed[side] = HAND_SPEED
    self.looking[side] = False
    self.phase[side] = "retreat"

  def way_home(self, side: str, refuse: bool = False) -> list[np.ndarray] | None:
    """The joint targets an arm walks home through from its command, judged
    on the robot's model of itself as it stands (`HOME_DIP_M`): its rest
    alone -- or, where that walk takes some part of the arm further ahead
    than the robot's front (`front`) and lower than the lowest of it there
    now, the side way: rolled out to its side (`HOME_ROLL`), its other
    joints home but its shoulder pitched back (`HOME_BACK`), its roll, then
    its pitch. The side way only where it dips less. Where another robot
    stands by (`keep_out`, #678), the way judged against its body as it
    stands and stood up (`_kept_off`): None where every way crosses it and
    `refuse`, else the one that comes least near it."""
    return self._judged(side, self.q_cmd[side].copy(), self.home_to(side), out=False,
                        poses=(None, (0.0, 0.0, 0.0)), refuse=refuse)

  def home_to(self, side: str, carry=None) -> np.ndarray:
    """Where an arm's way home ends: the carry pose a walk carries it at
    (`carry_q`, or `carry` given), else its rest."""
    q = self.carry_q[side] if carry is None else carry
    return (self.rest[side] if q is None else np.asarray(q, dtype=float)).copy()

  def home_clear(self, side: str, carry=None) -> bool:
    """Whether some way home -- to its carry pose, carried (`home_to`) --
    keeps off every other robot's body (#678), judged from where it will
    begin (`_home_from`): False sets `kept`, nothing moved; True with none
    by."""
    self.judged_home[side] = None
    if not len(self.keep_out):
      return True
    q0 = self._home_from(side)
    way = self._judged(side, q0, self.home_to(side, carry), out=False,
                       poses=(None, (0.0, 0.0, 0.0)), refuse=True)
    if way is not None:
      self.judged_home[side] = (q0, [np.asarray(w, dtype=float).copy() for w in way])
    return way is not None

  def carry_pose(self, side: str) -> np.ndarray:
    """The arm's six joints a walk carries a hand at (#682): `CARRY_POSE`'s
    shoulder and elbow (mirrored for the right), its wrist's pitch and roll
    the ones that keep the hand turned as it is now against the floor --
    the floor's up in the hand's own frame as it is held now -- the turn its
    command keeps (as `carry` keeps it), else its model as last posed, never
    its joints read again (each read draws on their noise, and a check
    before a walk would move the flight) -- so what it holds is kept as it
    is: level, lifted level. ⚠ SOLVED ON THE STATUE OF THE STAND (`statue`,
    posture 0), where a walk carries it, never the posture it holds now:
    solved bowed, a plate lifted from a reach's 0.3 rad bow was carried 15-17
    deg tipped once the trunk stood up. Each joint `arms.RANGE_MARGIN` inside
    its range; never what the hand holds, which it cannot know."""
    if self.cmd[side] is not None and self.cmd[side].rot is not None:
      r_now = np.asarray(self.cmd[side].rot, dtype=float)
    else:
      r_now = self.self_model.grip_pose(side)[1]
    up = r_now.T @ np.array([0.0, 0.0, 1.0])
    sm = self._planner_model()
    idx = sm.arm[side]
    lo = sm.ranges[idx, 0] + RANGE_MARGIN
    hi = sm.ranges[idx, 1] - RANGE_MARGIN
    q = seed(side, (*CARRY_POSE, 0.0, 0.0))
    q[:4] = np.clip(q[:4], lo[:4], hi[:4])
    arms = self._arms()
    arms.update({f"{side}_{j}": float(v) for j, v in zip(hm.ARM, q)})
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    statue(sm, (0.0, 0.0, 0.0), q0, arms=arms, fast=True)
    wp, wr = hm.ARM.index("wrist_pitch"), hm.ARM.index("wrist_roll")
    adr = sm.qadr[idx]
    grip = sm.grip[side]

    def off(p_: float, r_: float) -> float:
      q[wp], q[wr] = p_, r_
      sm.d.qpos[adr] = q
      mujoco.mj_kinematics(sm.m, sm.d)
      have = sm.d.site_xmat[grip].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
      return float(np.arccos(np.clip(have @ up, -1.0, 1.0)))

    best = (math.inf, 0.0, 0.0)
    for p_ in np.linspace(lo[wp], hi[wp], CARRY_GRID[0]):
      for r_ in np.linspace(lo[wr], hi[wr], CARRY_GRID[1]):
        e = off(p_, r_)
        if e < best[0]:
          best = (e, p_, r_)
    _, p0, r0 = best
    for p_ in np.linspace(max(lo[wp], p0 - CARRY_FINE), min(hi[wp], p0 + CARRY_FINE), 21):
      for r_ in np.linspace(max(lo[wr], r0 - CARRY_FINE), min(hi[wr], r0 + CARRY_FINE), 21):
        e = off(p_, r_)
        if e < best[0]:
          best = (e, p_, r_)
    q[wp], q[wr] = best[1], best[2]
    sm._set_arm(side, q)
    return q.copy()

  def drop_carry(self, side: str) -> None:
    """An arm carried no longer (#682): its way home is its rest again --
    a let-go's, a withdraw's."""
    self.carry_q[side] = None

  def carried(self, side: str) -> bool:
    """Whether an arm is held at the carry pose a walk took it to (#682)."""
    return self.phase[side] == "carried" and self.carry_q[side] is not None

  def _home_from(self, side: str) -> np.ndarray:
    """Where an arm's way home will begin (`retreat`): a hand on its line in
    comes out along it first, to its pre-grasp `APPROACH_M` back along its
    fingers -- solved on its model of itself as it stands, from where the
    arm is commanded -- else its command."""
    q = self.q_cmd[side].copy()
    if (self.phase[side] in ("rest", "home", "joints", "move") or self.cmd[side] is None
            or self.goal[side] is None):
      return q
    pre = (self.goal[side] if self.phase[side] == "retreat"
           else pregrasp(self.goal[side], APPROACH_M))
    sm = self._planner_model()
    sm.q = self.self_model.q.copy()
    sm.d.qpos[:] = self.self_model.d.qpos
    sm._set_arm(side, q)
    got, ep, _ = sm.solve(side, pre.pos, pre.rot)
    return got if ep <= REPLAN_M else q

  def way_out(self, side: str, q_to, posture=None, refuse: bool = True,
              own: bool | None = None) -> list[np.ndarray] | None:
    """The joint targets an arm walks out through from its command to `q_to`,
    a reach's pre-grasp: the way home's judgment from its other end --
    `q_to` alone, or, from an arm with none of it ahead of the robot's front
    whose walk there takes some part of it further ahead and lower than the
    lowest of it there at `q_to`, the side way home walked back. The side
    way only where it dips less. Judged as the robot stands, never in the
    posture the reach then takes: of 151 reaches planned in a crouch or a
    bow, 10 judged otherwise there, each straight where standing it was the
    side way (#560). Where another robot stands by (`keep_out`, #678),
    judged against its body as it stands and in `posture` (`_kept_off`):
    None where every way crosses it and `refuse`, else the one that comes
    least near it. From a hand already out, judged against its own trunk
    too (`_own_kept`, #722): None where every way passes through it and
    `own` (None: as `refuse`), else the one that comes least near it."""
    return self._judged(side, self.q_cmd[side].copy(), np.asarray(q_to, dtype=float).copy(),
                        out=True, poses=(None, posture), refuse=refuse,
                        own=refuse if own is None else own)

  def _judged(self, side: str, q0, q1, out: bool, poses=(None,),
              refuse: bool = False, own: bool = False) -> list[np.ndarray] | None:
    """The joint targets from `q0` to `q1` on the robot's model of itself
    as it stands (`HOME_DIP_M`): `q1` alone, or the side way (`_side_way`,
    walked back going `out`) where straight takes some part of the arm
    further ahead than the robot's front (`front`) and lower than the
    lowest of it there at the end over the work -- `q0` going home, `q1`
    going out -- and the side way dips less; a hand already out going out,
    straight, kept off its own trunk (`_own_kept`). With another robot by,
    the first way of today's, straight, the side way and the way drawn in
    that keeps off it (`_kept_off`)."""
    sm = self._planner_model()
    sm.q = self.self_model.q.copy()
    sm.d.qpos[:] = self.self_model.d.qpos
    pick, dips = self._picked(sm, side, q0, q1, out)
    clear = None
    if out and dips is None:
      # ...a hand already out (`_picked`): its way judged against its trunk
      pick, clear = self._own_kept(sm, side, q0, q1, pick, own)
      if pick is None:
        return None
    if not len(self.keep_out):
      return pick
    return self._kept_off(sm, side, q0, q1, out, pick, dips, poses, refuse, clear)

  def _own_kept(self, sm, side, q0, q1, pick, refuse):
    """A WAY OUT FROM A HAND ALREADY OUT KEEPS OFF THE ROBOT'S OWN TRUNK
    (#722; `OWN_CLEAR_M`; SimNotes, "A way out from a hand already out"):
    `pick` (today's, straight) where its arm keeps that much room from the
    trunk above the waist all the way, its end included (`own_rooms`) -- or,
    each of its parts begun nearer one of the trunk's, goes no nearer that
    one -- else the first of the side way and the
    way drawn in that keeps so and dips no deeper over the work than going
    straight on (as `_kept_off` judges a hand already out); else `own` saying
    how, and None where `refuse`, or of those ways the one that comes least
    near it, `own` saying how near -- straight where the pre-grasp itself
    lies in the trunk, every way ending there. Returns that way, and the
    judge of any other (`_kept_off`'s) -- none where no way keeps clear: then
    another robot's ways are weighed as #678 weighs them. ⚠ NEVER A WAY FROM A HAND
    AT REST: there the plan's pre-grasp itself could lie in the trunk, and
    refused, our cube's set-out 7 lost its pass; since #726 a procedure's
    plan keeps it off (`plan`'s `own`), so it could be judged -- not yet."""
    own = own_model()
    rooms: dict = {}

    def rooms_of(w) -> np.ndarray:
      key = b"".join(np.asarray(x, dtype=float).tobytes() for x in w)
      if key not in rooms:
        rooms[key] = own_rooms(own, side, q0, w)
      return rooms[key]
    # ...each pair of parts no nearer than it began, where it began nearer
    need = np.minimum(OWN_CLEAR_M, rooms_of([])) - OWN_SLACK_M

    def short(w) -> float:
      return float((rooms_of(w) - need).min())

    def clear(w) -> bool:
      return short(w) >= 0.0
    if clear(pick):
      return pick, clear
    best_way = pick
    # ...every way ends at the pre-grasp: one in the trunk leaves no way clear
    if (own_rooms(own, side, q1, []) >= need).all():
      self._pose_planner(sm, None)
      dips = _dips(sm, side, q0, q1, front(sm, side, self.rest[side]))
      deepest = max(HOME_DIP_M, dips(pick))
      # ...today's is straight (`_picked`): the side way, then the way drawn in
      for w in (_aside(side, q0, q1, True), _tucked_way(side, q0, q1)):
        if dips(w) > deepest + 1e-9:
          continue
        if clear(w):
          return w, clear
        if short(w) > short(best_way):
          best_way = w
    r = rooms_of(best_way)
    i = int(np.argmin(r - need))
    part = UPPER_GEOMS[i % len(UPPER_GEOMS)]
    self.own[side] = {"part": "torso" if part == "torso" else "head", "m": round(float(r[i]), 3)}
    return (None, clear) if refuse else (best_way, None)

  def _picked(self, sm, side: str, q0, q1, out: bool):
    """`_judged`'s way with nobody by, and how far a way dips (`_dips`; None
    for a hand already out, whose way is never judged so: `_own_kept` and
    `_kept_off` judge another against it)."""
    ahead = front(sm, side, self.rest[side])
    if out:
      sm._set_arm(side, q0)
      if (outline(sm, side)[:, 0] > ahead).any():
        # a hand already out sweeps nothing aside on its way to the next
        return [q1], None
    dips = _dips(sm, side, q0, q1 if out else q0, ahead)
    straight = dips([q1])
    if straight <= HOME_DIP_M:
      return [q1], dips
    aside = _aside(side, q0, q1, out)
    return (aside if dips(aside) < straight else [q1]), dips

  def _kept_off(self, sm, side, q0, q1, out, pick, dips, poses, refuse, clear=None):
    """A HAND'S WAY KEEPS OFF ANOTHER ROBOT'S BODY (#678; SimNotes, "The
    arms keep off the other robot"): the first of today's way (`pick`),
    straight, the side way and the way drawn in (`_tucked_way`) whose arm
    keeps `OTHER_CLEAR_M` off every body of `keep_out` (`room`) -- an arm
    begun nearer only going no nearer -- in each of `poses` (None as the
    robot stands, else a posture's statue), and dips no deeper over the
    work than `pick` (a hand already out: than going straight on, judged as
    the arm stands at its end): #539's and #560's rule kept whole, so at a
    counter, where today's way out is the side way, none other is taken.
    Else, `kept` saying how (the nearest robot's middle, stance frame),
    None where `refuse`, or of those ways the one that comes least near it:
    what the loop's own moves take once they have waited
    (`HumanoidMission.ARM_WAIT_S`) -- at a counter, today's. A way out
    another than `pick` keeps off the robot's own trunk too (`clear`,
    `_own_kept`'s)."""
    ways = [pick] + [w for w in ([np.asarray(q1, dtype=float).copy()], _aside(side, q0, q1, out),
                                 _tucked_way(side, q0, q1)) if not _same_way(w, pick)]
    self._pose_planner(sm, None)
    if dips is None:
      # ...a hand already out: another way dips no deeper under the lowest
      # of the arm ahead at its end than going straight on does
      dips = _dips(sm, side, q0, q1, front(sm, side, self.rest[side]))
    deepest = max(HOME_DIP_M, dips(pick))
    posed: dict = {}
    best, best_way = -math.inf, pick
    for i, w in enumerate(ways):
      self._pose_planner(sm, None)
      if i and (dips(w) > deepest + 1e-9 or (clear is not None and not clear(w))):
        continue
      worst = math.inf
      for k, pose in enumerate(poses):
        # ...each posture's statue balanced once a judgment, and the room
        # its arm begins with there
        if k in posed:
          sm.q[:], sm.d.qpos[:], start = posed[k]
          mujoco.mj_kinematics(sm.m, sm.d)
        else:
          self._pose_planner(sm, pose)
          start = room(sm, side, q0, [], self.keep_out)
          posed[k] = (sm.q.copy(), sm.d.qpos.copy(), start)
        worst = min(worst, room(sm, side, q0, w, self.keep_out) - min(OTHER_CLEAR_M, start))
      if worst > best:
        best, best_way = worst, w
      if worst >= 0.0:
        self.kept[side] = None
        return w
    c = np.asarray(self.keep_out, dtype=float).reshape(-1, 2)
    k = int(np.argmin(np.hypot(c[:, 0], c[:, 1])))
    self.kept[side] = {"home": not out, "m": round(float(np.hypot(*c[k])), 2),
                       "x": round(float(c[k, 0]), 2), "y": round(float(c[k, 1]), 2),
                       "short": round(-best, 3)}
    return None if refuse else best_way

  def _pose_planner(self, sm, pose) -> None:
    """`sm` as the robot stands (None), or in posture `pose`'s statue."""
    if pose is None:
      sm.q = self.self_model.q.copy()
      sm.d.qpos[:] = self.self_model.d.qpos
      mujoco.mj_kinematics(sm.m, sm.d)
      return
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    statue(sm, tuple(float(v) for v in pose), q0, arms=self._arms(), lean=self._lean(),
           fast=True)

  def _ahead(self, side: str) -> bool:
    """Whether some part of the arm as commanded stands further ahead than
    the robot's front (`front`), on its model of itself as it stands: where
    none does, its way home was judged straight, unchecked (`_picked`)."""
    sm = self._planner_model()
    sm.q = self.self_model.q.copy()
    sm.d.qpos[:] = self.self_model.d.qpos
    ahead = front(sm, side, self.rest[side])
    sm._set_arm(side, self.q_cmd[side])
    return bool((outline(sm, side)[:, 0] > ahead).any())

  def _go_home(self, side: str) -> None:
    """The arm on its way home (`way_home`), its goal let go; the posture
    back to the stand once no hand is out -- or, off the stand with none of
    the arm ahead of its front, once its command is home."""
    self.goal[side] = self.cmd[side] = None
    self.outs[side] = []
    self.looking[side] = False
    # ...the way home a judgment found clear, where the arm begins it as it
    # was judged (#678: judged afresh from a start a few mm off, a way clear
    # by a hair could cross the other robot unrefused)
    judged, self.judged_home[side] = self.judged_home[side], None
    if judged is not None and np.abs(self.q_cmd[side] - judged[0]).max() <= HOME_RAD:
      self.homes[side] = [w.copy() for w in judged[1]]
    else:
      self.homes[side] = self.way_home(side)
    self.q_to[side] = self.homes[side].pop(0)
    self.phase[side] = "home"
    self.homing[side], self.held_for[side], self._stall[side] = True, 0.0, 0.0
    # ...and the stand once no hand is out at its work -- one carried at the
    # side is none (#682); kept the reach's bow, a withdraw left it bowed --
    # but off the stand, an arm with none of it ahead of the robot's front
    # has its command home first (`_update`). ⚠ ITS WAY HOME WAS JUDGED
    # STRAIGHT, UNCHECKED, ON ITS MODEL AS IT STANDS (#693): asked as the arm
    # set off, the stand carried a hand its bow kept behind the toes 3.4 cm
    # past them and down onto a counter's top, and tipped the robot over
    # backward. ⚠
    # NEVER AN ARM AHEAD: held as it came home, a crouch's side way swept the
    # jaws into a cupboard's opened door, and a turn held to the end swung
    # the hanging arm into a table's leg (the check lost 3 doors and 23
    # tables of 32). ⚠ NOR WITH THE OTHER ARM GOING HOME TOO, judged apart,
    # NOR A POSTURE STILL WALKING to a plan a retreat cut short (`settled`):
    # neither is the posture the way was judged in
    if all(p in (*AT_EASE, "home") for p in self.phase.values()):
      alone = [s for s in hm.SIDES if self.phase[s] == "home"] == [side]
      if self.posture.any() and alone and self.stand.settled and not self._ahead(side):
        self.stand_home = True
      else:
        self.hold_posture(0.0, 0.0)

  def home(self, side: str) -> bool:
    """Whether an arm is home: its rest commanded, and every joint of it
    read there by its driver (`HOME_RAD`)."""
    return self.phase[side] == "rest" and bool(
      np.abs(self.sensed[self.self_model.arm[side]] - self.rest[side]).max() <= HOME_RAD)

  def held(self, side: str) -> bool:
    """Whether an arm's way home stopped where something holds it, its
    command eased there (`HELD_S`): out, until a way home begins again."""
    return self.phase[side] == "held"

  def _ease(self, side: str) -> np.ndarray:
    """An arm left held: its command where its drivers read it, pressing on
    nothing (phase "held"); returns that reading."""
    q = self.sensed[self.self_model.arm[side]].copy()
    self.q_cmd[side], self.q_to[side] = q, q.copy()
    self.goal[side] = self.cmd[side] = None
    self.homing[side], self.held_for[side] = False, 0.0
    self.phase[side] = "held"
    return q

  def _stand_at_ease(self) -> None:
    """The stand a way home kept from the robot (`stand_home`, #693), asked
    once every arm is at ease (`AT_EASE`) -- however the last came to it:
    home, carried, or left held (#693b's fourth review: left held, a raise held
    too, the stand never came and the robot stayed bowed)."""
    if self.stand_home and all(p in AT_EASE for p in self.phase.values()):
      self.stand_home = False
      self.hold_posture(0.0, 0.0)

  def _raise_short(self, side: str) -> bool:
    """Whether a raise (`_held_home`'s: `_homed`, `_shoved`, `_stalled`)
    whose command has arrived has its arm short of it: its grip, by its
    senses, more than `BLOCKED_M` from it."""
    c, at = self.cmd[side], self.sensed_grip[side]
    return (self.lifted[side] > 0 and not self.homing[side] and c is not None
            and at is not None and float(np.linalg.norm(at - c.pos)) > BLOCKED_M)

  def home_over(self, side: str) -> bool:
    """Whether an arm's last way home is over by its drivers (`_homed`):
    read home or held -- False only for one whose command reached its rest
    and whose drivers have not yet read it either way, `HELD_S` at most."""
    return not (self.homing[side] and self.phase[side] == "rest")

  def _homed(self, side: str, dt: float) -> None:
    """A way home's end as its drivers read it: home, or -- not home
    `HELD_S` running with its rest commanded -- held, its command set where
    they read it, pressing on nothing: lifted `LIFT_M` straight up and sent
    home again from there (`_go_home`'s judgment) `LIFTS` times a way home,
    then left there ("held")."""
    if self.home(side):
      self.homing[side], self.held_for[side] = False, 0.0
      self.watch_home[side] = False
      return
    self.held_for[side] += dt
    if self.held_for[side] < HELD_S:
      return
    self._held_home(side)

  def _shoved(self, side: str, dt: float) -> bool:
    """Whether an arm on a walk's hand-over's way to its rest (`watch_home`)
    -- in joint space, or its rest commanded and not yet read home
    (`homing`, `home`) -- reads a tip of the body away from its hand past
    `HOME_TIP_NM` for `HOME_TIP_S` running (minus `_tipping`, off the force
    its arm reads it push with); `dt` its tick."""
    homing = self.watch_home[side] and self.carry_q[side] is None and (
      self.phase[side] == "home"
      or (self.phase[side] == "rest" and self.homing[side] and not self.home(side)))
    over = homing and -self._tipping(side) > HOME_TIP_NM
    self._shove[side] = self._shove[side] + dt if over else 0.0
    return self._shove[side] >= HOME_TIP_S - 1e-9

  def _stalled(self, side: str, dt: float) -> bool:
    """Whether an arm on its way home in joint space (`way_home`) reads its
    drivers more than `STALL_RAD` off its command, any joint either way, for
    `STALL_S` running -- behind it, as an arm something holds while its
    command walks on; `dt` its tick. A carry's way to its pose never is
    (#682)."""
    arm = self.self_model.arm[side]
    over = (self.phase[side] == "home" and self.carry_q[side] is None
            and float(np.abs(self.sensed[arm] - self.q_cmd[side]).max()) > STALL_RAD)
    self._stall[side] = self._stall[side] + dt if over else 0.0
    return self._stall[side] >= STALL_S - 1e-9

  def _held_home(self, side: str) -> None:
    """A way home held (`_homed`, `_shoved`, `_stalled`): eased where its
    drivers read it, pressing on nothing, then raised `LIFT_M` and sent home
    again `LIFTS` times a way home, else left there ("held"). Its counts begun
    again: a raise's line never reads them, and the next way home has its
    whole hold; its way still to walk forgotten, and left there, watched no
    longer."""
    self._shove[side] = self._stall[side] = 0.0
    self.homes[side] = []
    q = self._ease(side)
    if self.lifted[side] >= LIFTS or len(self.keep_out):
      self.watch_home[side] = False
      self._stand_at_ease()
      return
    self.lifted[side] += 1
    # ...its grip where its drivers put it, on its model of itself as it
    # stands, its line straight up from there; then `_go_home` from the top
    sm = self._planner_model()
    self._pose_planner(sm, None)
    sm._set_arm(side, q)
    pos, rot = sm.grip_pose(side)
    self.cmd[side] = HandGoal(pos.copy(), rot.copy(), 1.0)
    self.goal[side] = HandGoal(pos + np.array([0.0, 0.0, LIFT_M]), rot.copy(), 1.0)
    self.speed[side], self.looking[side] = HAND_SPEED, False
    self.phase[side] = "retreat"

  def done(self, side: str) -> bool:
    """Whether a hand has arrived where it was sent (or hangs), or stopped
    on its way (`blocked`), or an arm sent in joint space is there."""
    if self.phase[side] == "joints":
      return bool(np.allclose(self.q_cmd[side], self.q_to[side]))
    return self.phase[side] in ("hold", "rest", "blocked")

  # ---- the loop ----

  def _pose(self, arms: dict | None = None) -> None:
    q, qd, tau = self.stand.joints.read(self.d)
    self.tau, self.qd, self.sensed = tau, qd, q.copy()
    if arms is not None:
      for s in hm.SIDES:
        q[self.self_model.arm[s]] = arms[s]
    self.self_model.pose(q, self.stand.trunk.gravity())

  def grip_now(self, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Where the robot believes its grip is, stance frame: its model posed
    from its senses."""
    self._pose()
    return self.self_model.grip_pose(side)

  def _posture_there(self, dt: float) -> bool:
    """Whether the body has reached its plan's posture (`POSTURE_TOL`) and
    stopped there (`TRUNK_STILL_RATE`), or has been given `POSTURE_WAIT_S` to."""
    self._posture_wait += dt
    near = (np.abs(self.posture_now() - self.posture) <= POSTURE_TOL).all()
    still = float(np.linalg.norm(self.stand.trunk.rate)) < TRUNK_STILL_RATE
    if (near and still) or self._posture_wait >= POSTURE_WAIT_S:
      self._posture_wait = 0.0
      return True
    return False

  _posture_wait = 0.0

  def _posture_held(self) -> bool:
    """Whether the body stands still in the posture it holds (`POSTURE_TOL`,
    `TRUNK_STILL_RATE`)."""
    near = (np.abs(self.posture_now() - self.posture) <= POSTURE_TOL).all()
    return bool(near and float(np.linalg.norm(self.stand.trunk.rate)) < TRUNK_STILL_RATE)

  def _line(self, side: str, dt: float) -> bool:
    """The command one step along its line to the goal; True once there."""
    g, c = self.goal[side], self.cmd[side]
    step = g.pos - c.pos
    n = float(np.linalg.norm(step))
    v = self.speed[side]
    c.pos = g.pos.copy() if n <= v * dt else c.pos + step * (v * dt / n)
    if g.rot is not None:
      e = _rot_error(g.rot, c.rot)
      a = float(np.linalg.norm(e))
      if a <= HAND_TURN * dt:
        c.rot = np.array(g.rot, dtype=float)
      else:
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, e / a, HAND_TURN * dt)
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, q)
        c.rot = r.reshape(3, 3) @ c.rot
    c.stiffness = g.stiffness
    return n <= v * dt

  def _joints(self, side: str, dt: float) -> bool:
    """The arm one step toward `q_to` in joint space; True once there."""
    gap = self.q_to[side] - self.q_cmd[side]
    worst = float(np.abs(gap).max())
    if worst <= JOINT_RATE * dt:
      self.q_cmd[side] = self.q_to[side].copy()
      return True
    self.q_cmd[side] = self.q_cmd[side] + gap * (JOINT_RATE * dt / worst)
    return False

  def step(self) -> None:
    for g in self.grippers.values():
      g.sample()
    self.motion.sample(self.steps, self.stand, self.d)
    if self.steps % self.every == 0:
      self._update(self.every * self.m.opt.timestep)
    self.steps += 1

  #: What a restart carries of a reach (#559, `HumanoidMission.kept_state`):
  #: every hand's phase and commands, its filters and what it last read.
  KEPT = ("phase", "goal", "cmd", "q_to", "q_cmd", "homes", "outs", "sensed", "miss", "slack",
          "_off", "_tip", "steps", "posture", "trim", "tau", "qd", "standing", "force",
          "arm_force", "sensed_grip", "speed", "pulls", "report", "waist_cmd", "waist_speed",
          "allowed", "left", "looks", "looking", "carry_q", "stand_home", "homing",
          "held_for", "lifted", "kept_by", "_shove", "watch_home", "_stall")

  def kept_state(self) -> dict:
    return {"reach": {k: kept_value(getattr(self, k)) for k in self.KEPT},
            "grippers": {s: {"shut": g.shut_asked, "force": g.force, "gap": g.gap,
                             "gapMean": g.gap_mean, "squeeze": g.squeeze,
                             "rng": g.sense.rng.bit_generator.state}
                         for s, g in self.grippers.items()},
            "motion": self.motion.kept_state(),
            # ...and its model of itself as its loop last posed it: what a
            # spot is placed through between two of the loop's ticks
            "selfModel": self.self_model.kept_state()}

  def restore_kept(self, state: dict) -> None:
    for k, v in state["reach"].items():
      setattr(self, k, value_back(v))
    self.motion.restore_kept(state.get("motion", {}))
    if "selfModel" in state:                  # a save from before it: posed at the next tick
      self.self_model.restore_kept(state["selfModel"])
    self.sensed_force = {s: np.array(f, dtype=float) for s, f in self.force.items()}
    for s, kept in state["grippers"].items():
      g = self.grippers[s]
      g.shut_asked, g.force = bool(kept["shut"]), float(kept["force"])
      g.gap, g.squeeze = float(kept["gap"]), float(kept["squeeze"])
      g.gap_mean = float(kept.get("gapMean", kept["gap"]))     # a save from before it
      g.sense.rng.bit_generator.state = kept["rng"]

  def posture_now(self) -> np.ndarray:
    """The posture as the robot's model of itself has it: the pelvis's
    height off the stand's (its legs' encoders), the trunk's pitch (its
    IMU) and the waist's yaw (its driver), as last posed."""
    sm = self.self_model
    return np.array([sm.d.qpos[2] - hm.stand_height(),
                     math.asin(max(-1.0, min(1.0, float(self.stand.trunk.gravity()[0])))),
                     *sm.q[sm.waist]])

  def _trim(self, dt: float) -> None:
    """The posture as the robot's model has it against the plan's; the
    policy asked that much more, once the posture it was shown has arrived
    (a ramp's lag is not a miss), never past what it trained on. ⚠ IT TRIMS
    ON WHILE THE WALKER HOLDS THE LEGS, off the walker's posture: a walk
    leaves the standing goal pitched forward, 0.008-0.017 rad after a short
    step and 0.020 after a walk of a second or more (the edge of the band
    `settled` reads), taken up again a second or two after the hand-back.
    Trimmed only while the standing policy steps, our table's reference
    laid 20 of 32 from the tour where it lays 27 (#668): every hand job is
    measured on it as it is."""
    have = self.posture_now()[:2]
    if self.stand.settled:
      trim = np.clip(self.trim + TRIM_RATE * (self.posture[:2] - have) * dt,
                     -np.array(TRIM_MAX), TRIM_MAX)
      asked = self.posture[:2] + trim
      inside = (asked >= [HEIGHT_RANGE[0], PITCH_RANGE[0]]) & (asked <= [HEIGHT_RANGE[1], PITCH_RANGE[1]])
      self.trim = np.where(inside, trim, self.trim)
      self.stand.ask(*(self.posture[:2] + self.trim))

  # ---- a hand that is out, carried (#493) ----

  def _end(self, side: str, why: str, h: np.ndarray, r: np.ndarray,
           at_goal: bool = False) -> None:
    """A carry over: held at its goal (one that arrived) or
    where the hand IS -- never at a command the arm could not follow -- and
    its `report`: why, how long, how far it went and where it ended up
    from where it began (stance frame, m), and the hardest it read along
    its way where it watched its force (N) -- a pull's, its arm's joints
    read on their stop as it ended (`stops`, #744)."""
    st = self.pulls[side]
    c = self.cmd[side]
    if st.get("backing") and st.get("pull") is None:
      # ...a carry's body backing off a posture given up held where it has got
      # to, as its back-off holds it (`_held`, #699): left going, it walked all
      # the way back to where the carry began, as a pull's swung a low arm
      # into the counter (935 N)
      self.hold_posture(*self.posture_now())
    short = float(np.linalg.norm(self.goal[side].pos - h))
    if not at_goal:
      # where the hand IS, at the turn it was ASKED (`carry`): held at the
      # turn it sagged to, a pull at a time crept the drop handle round
      turn = (np.asarray(c.rot, dtype=float) if c.rot is not None else r).copy()
      self.goal[side] = HandGoal(h.copy(), turn.copy(), c.stiffness)
      self.cmd[side] = HandGoal(h.copy(), turn, c.stiffness)
    disp = h - st["start"]
    rep = {"why": why, "seconds": round(st["t"], 3), "moved": round(float(np.linalg.norm(disp)), 4),
           "shortM": round(short, 4), "dx": round(float(disp[0]), 4),
           "dy": round(float(disp[1]), 4), "dz": round(float(disp[2]), 4)}
    if st.get("loaded"):
      # ...a procedure's carry whose hand, holding nothing, read a load (#699)
      rep["loaded"] = True
    if st.get("pull") is not None:
      pv = st["pull"]
      rep["force"] = round(st["peak"], 2)
      rep["went"] = round(pv["went"] + float((h - pv["last"]) @ pv["way"]), 4)
      # the way the hand was going at its end: its last `TRAIL_M` of travel
      went = h - pv["trail"][0]
      n = float(np.linalg.norm(went))
      rep["way"] = [round(float(v), 4) for v in (went / n if n > STEP_M else pv["way"])]
      rep["stops"] = self._on_stops(side)
    self.report[side] = rep
    self.pulls[side] = None
    self.phase[side] = "hold"

  def _on_stops(self, side: str) -> list[str]:
    """The joints of `side`'s arm its drivers last read at an END of the
    range its model of itself gives (#744: held fast at its own wrist's end,
    a pull read as at a lid's open stop): within `JOINT_END_RAD` of it or
    past it -- its arm's solve keeps every joint `RANGE_MARGIN` inside, so
    one read there was pushed onto its stop. Off `sensed`, never read
    afresh: a read draws noise."""
    sm = self.self_model
    arm = sm.arm[side]
    q, lo, hi = self.sensed[arm], sm.ranges[arm, 0], sm.ranges[arm, 1]
    return [j for j, on in zip(hm.ARM, (q - lo < JOINT_END_RAD) | (hi - q < JOINT_END_RAD))
            if on]

  def _held(self, side: str, dt: float, h: np.ndarray, r: np.ndarray) -> float:
    """One command of a carried hand (`carry`), `h` and `r` where its grip
    IS by its senses; its stiffness. Its solve is
    commanded as the reach's always is, near as the arm comes -- ⚠ BUT NEVER
    ONE THAT DIVERGED (`WILD_M`: a miss of 10-15 cm commanded swung an arm
    round a chest's knob and out of it) -- and a miss past `BLOCKED_M` held
    `BLOCKED_S` ends the move where the hand is (`out of reach`), once the
    move has had `SETTLE_S` to begin: a carry stopped at the arm's stretch
    starts the next where its solve cannot quite put it. ⚠ A PROCEDURE'S
    CARRY (`carry`'s `verb`) ENDS WHERE A GRIP IT BEGAN WITH IS LOST, AS A PULL
    DOES (`_pulled`, "let go"; #742): carried on, a jaw slid off a board's edge
    and the empty hand "arrived", the cube where it was set out, 16 carries of
    16. Never the loop's own (a charge's plug), bit for bit as before."""
    st = self.pulls[side]
    st["t"] += dt
    if st.get("gripped") and not self.grippers[side].held():
      self._end(side, "let go", h, r)
      return self.cmd[side].stiffness
    # ⚠ A HAND HOLDS STILL WHILE ITS BODY TAKES A NEW POSTURE: carried on
    # meanwhile, a lid was jerked down at 1.7-2.3 rad/s and up to 81 deg
    settling = (st.get("postured") is not None and not self._posture_held()
                and st["t"] - st["postured"] < POSTURE_WAIT_S)
    # ...a procedure's hand that holds nothing, read over `MEAN_S` as a pull's
    # (#699): a posture coming is GIVEN UP where it reads a load, the body
    # backing off toward the posture the carry began in -- the hand waiting
    # on it -- until it reads no more, held there; that carry takes no other
    loaded = False
    if st.get("open"):
      # ...off the whole body's balance alone, where it closes (`closed`): one
      # that does not reads nothing, the mean and its run held as they were
      if self.closed[side]:
        km = min(1.0, dt / MEAN_S)
        fm = st["fm"] = (1.0 - km) * st.get("fm", self.force[side]) + km * self.force[side]
        # ...a posture given up only for a load read past the line for
        # `MEAN_S` running (`CARRY_PRESSED_SHARE`)
        st["over"] = st.get("over", 0.0) + dt if self._pressed(fm, 0.0, CARRY_PRESSED_SHARE) else 0.0
      loaded = st.get("fm") is not None and self._pressed(st["fm"], 0.0, CARRY_PRESSED_SHARE)
    # ...never the body held where it stood at a loaded ask (`held_here`): no
    # posture came, and given up there, the carry took none once light
    if (settling and st.get("over", 0.0) >= MEAN_S - 1e-9 and not st.get("given")
        and not st.get("held_here")):
      self.hold_posture(*st["began"])
      st["postured"], st["given"], st["backing"], st["loaded"] = st["t"], True, True, True
    elif st.get("backing") and not (settling and loaded):
      if settling:
        self.hold_posture(*self.posture_now())
      st["backing"] = False
    there = False if settling else self._line(side, dt)
    c, sm = self.cmd[side], self.self_model
    q, ep, er = sm.solve(side, c.pos, c.rot)
    if ((ep > REPLAN_M or er > REPLAN_ROT) and not st.get("fixed") and not self.kept_by[side]
        and st["t"] - st.get("planned", -REPLAN_S) >= REPLAN_S):
      # The arm runs out of reach from this posture: the least bent that
      # reaches the command now, while there is one (`posture_for`) -- none
      # while that hand reads a load, nor once one was given up: asked as
      # soon as it reads light, as a pull's is
      if loaded:
        st["loaded"] = True
      elif not st.get("given"):
        st["planned"] = st["t"]
        posture = self.posture_for(side, c.pos, rot=c.rot)
        # ...and one decided reading light, whatever it found, withheld
        # nothing for a load
        st.pop("loaded", None)
        if posture is not None and tuple(posture) != tuple(self.posture):
          if st.get("open") and "began" not in st:
            st["began"] = self._standing_in()
          st.pop("held_here", None)
          self.hold_posture(*posture)
          st["postured"] = st["t"]
    if st.get("recheck") and not loaded:
      # ...a path's or a turned carry's, which plans none on its way: the one
      # withheld at its ask planned again over what is left of it, once, at
      # the first command that reads light -- one reading at the ask kept the
      # whole move from the posture its path needed
      del st["recheck"]
      st.pop("loaded", None)
      g = self.goal[side]
      posture = self.path_for(side, [c.pos, g.pos] + [p for p, _ in st["ways"]],
                              rot=[c.rot, g.rot] + [t for _, t in st["ways"]])
      if posture is not None and tuple(posture) != tuple(float(v) for v in self.posture):
        st.setdefault("began", self._standing_in())
        st.pop("held_here", None)
        self.hold_posture(*posture)
        st["postured"] = st["t"]
    if ep < WILD_M:
      self.q_cmd[side], self.miss[side] = q, ep
    if ep > BLOCKED_M or er > HOLD_ROT_TOL:
      # ...given a posture it took the time to come before it is out of reach
      # -- none once it was given up and the body is held (#699)
      coming = (st["t"] < SETTLE_S or ((not st.get("given") or st.get("backing"))
                and st["t"] - st.get("postured", -POSTURE_WAIT_S) < POSTURE_WAIT_S))
      self._off[side] = 0.0 if coming else self._off[side] + dt
      if self._off[side] >= BLOCKED_S:
        self._end(side, "out of reach", h, r)
      return c.stiffness
    self._off[side] = 0.0
    if there and self.phase[side] == "carry" and st.get("ways"):
      self.goal[side].pos, self.goal[side].rot = st["ways"].pop(0)
      return c.stiffness
    if there and self.phase[side] == "carry":
      # arrived once the HAND is there, never its command alone: on a soft
      # arm a lid held the hand 5 cm back while every command arrived (#493)
      short = float(np.linalg.norm(self.goal[side].pos - h))
      st["there"] = st.get("there", 0.0) + dt
      if short <= ARRIVE_M:
        self._end(side, "arrived", h, r, at_goal=True)
      elif st["there"] >= HELD_BACK_S:
        self._end(side, "held back", h, r, at_goal=True)
    return c.stiffness

  def _pulled(self, side: str, dt: float, h: np.ndarray, r: np.ndarray) -> float:
    """One command of a PULL (`pull`), `h` and `r` where its grip IS by its
    senses: the command walked along the way as the force allows, following
    the hand across it, no faster than its speed, its turn giving; then
    solved and commanded as a carry's is. Its stiffness."""
    st = self.pulls[side]
    pv = st["pull"]
    c, sm = self.cmd[side], self.self_model
    st["t"] += dt
    w, limit = pv["way"], pv["force"]
    if pv["gripped"] and not self.grippers[side].held():
      # A GRIP LOST ENDS THE PULL where the hand is: else it pulls on through
      # the air to its distance, and its report is that free motion's
      self._end(side, "let go", h, r)
      return c.stiffness
    dh = h - pv["last"]
    if float(np.linalg.norm(dh)) >= STEP_M:
      pv["went"] += float(dh @ w)
      pv["last"] = h.copy()
      pv["trail"] = (pv["trail"] + [h.copy()])[-1 - int(round(TRAIL_M / STEP_M)):]
    # held, or GOING along its way (`GO_ON_M` in `GO_S`, or at that pace over
    # `FAST_S`); off its place (`FAST_M`), not held fast at its force
    gone = float((h - st["start"]) @ w)
    pv["past"] = [p for p in pv["past"] if st["t"] - p[0] <= GO_S] + [(st["t"], gone)]
    going = gone - min(p[1] for p in pv["past"]) >= GO_ON_M
    pace = gone - min(p[1] for p in pv["past"] if st["t"] - p[0] <= FAST_S) >= (
      GO_ON_M * FAST_S / GO_S)
    moved = float(np.linalg.norm(h - pv["anchor"])) > FAST_M
    if moved:
      pv["anchor"], pv["still"] = h.copy(), 0.0
    settling = (st.get("postured") is not None and not self._posture_held()
                and st["t"] - st["postured"] < POSTURE_WAIT_S)
    # THE FORCE, CLOSED ON WHAT THE HAND FEELS (admittance): along its way the
    # command walks at its speed while the force the hand pushes with along
    # it, a moment ahead (`AHEAD_S`), is under half its limit, slows to a
    # stand as it reaches it, and backs off past it (`BACK_SPEED`) -- through
    # the arm's own springs, its model put 15 N on a drawer's stop for a 5 N
    # limit and 27 for 10, where the hand's reading was within 0.5 N of the
    # truth
    k, km, kr = (min(1.0, dt / s_) for s_ in (ADMIT_SMOOTH_S, MEAN_S, RISE_SMOOTH_S))
    before = st.get("f", self.force[side])
    f = st["f"] = (1.0 - k) * before + k * self.force[side]
    fm = st["fm"] = (1.0 - km) * st.get("fm", self.force[side]) + km * self.force[side]
    rise = st["rise"] = (1.0 - kr) * st.get("rise", 0.0) + kr * float((f - before) @ w) / dt
    along = float(fm @ w)
    st["peak"] = max(st["peak"], along)
    x = 2.0 * (limit - float(f @ w) - AHEAD_S * rise) / limit
    if going and x < 0.0:
      # ...but a hand GOING along its way is held by nothing: where its force
      # is going only slows it, and it backs off past its force as read, at
      # its speed -- the whole body's reading rides 4-5 N on a hand moving in
      # free air, and backed off on where that went, a 4 N pull up 0.3 m
      # stopped after 1.1 cm (#547). ⚠ One that meets a stop goes on reading
      # as going for up to `GO_S`.
      rate = self.speed[side] * min(0.0, max(-1.0, 2.0 * (limit - float(f @ w)) / limit))
    else:
      rate = self.speed[side] * (min(1.0, x) if x >= 0.0 else BACK_SPEED * max(-1.0, x))
    if settling:
      # ...standing while its body takes a posture, but backing off: stood
      # still, it held on reading 15 times its force as a posture stood the
      # body up with the hand on a locked bar (#547)
      rate = min(rate, 0.0)
    pressed = not pv["gripped"] and self._pressed(fm, limit)
    if settling and pressed and not st.get("given"):
      # ...AND THE POSTURE GIVEN UP WHERE THE HAND READS A LOAD (#699): the
      # body backs off toward the posture the pull began in -- the hand
      # still waiting on it -- and this pull takes no other
      self.hold_posture(*st["began"])
      st["postured"], st["given"], st["backing"] = st["t"], True, True
    elif st.get("backing") and not (settling and self._pressed(fm, limit, 1.0)):
      # ...only until the hand reads no more than the ask again (20 N at
      # least), and is held there, as a move stopped is (`halt`) -- or is
      # back; still loaded when its time to come is out, it goes on back
      if settling:
        self.hold_posture(*self.posture_now())
      st["backing"] = False
    prev, prev_rot = c.pos.copy(), c.rot
    to = c.pos + rate * dt * w
    ahead = float((to - h) @ w)
    pinned = ahead > LEAD_MAX_M
    if pinned:
      to = to - (ahead - LEAD_MAX_M) * w
    # ACROSS its way, GIVING, the command follows the hand -- what it holds
    # takes it -- no faster than `ACROSS_SPEED` times its speed (yielding
    # only to a force read past a dead band, it lagged a drop handle's flip
    # and ran out of reach on every lid); HOLDING, it keeps its line (giving,
    # a cube set down slid 13 cm along what it touched)
    if pv["give"]:
      off = to - h
      to = to - (off - float(off @ w) * w)
    step = to - prev
    n = float(np.linalg.norm(step))
    most = ACROSS_SPEED * self.speed[side] * dt
    c.pos = prev + (step if n <= most else step * (most / n))
    # ...and its TURN gives to what it holds, giving -- ⚠ but never about
    # the jaws' own line, which what it holds turns about between the pads:
    # a lid's weight bent the wrist about that line, the give followed it
    # 70 deg down, and the arm ran out of reach at 46-54 deg of lid on 4
    # chests of 8; held there, 7 of 8 opened past 57 (#525)
    if pv["give"] and c.rot is not None:
      e = _rot_error(r, c.rot)
      jaws = c.rot[:, 0]
      e = e - float(e @ jaws) * jaws
      a_ = float(np.linalg.norm(e))
      if a_ > 1e-9:
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, e / a_, GIVE_RATE * a_)
        m9 = np.zeros(9)
        mujoco.mju_quat2Mat(m9, quat)
        c.rot = m9.reshape(3, 3) @ c.rot
    q, ep, er = sm.solve(side, c.pos, c.rot)
    pv["beyond"] = False
    if ep > REPLAN_M or er > REPLAN_ROT:
      there = None if going else self._reaching(side, h, c.rot)
      if there is not None:
        # HELD (`GO_ON_M`): the command stays where the arm last reached --
        # backing off, a share of the way to the hand, its arm's joints as
        # far toward those that reach it, none faster than `JOINT_RATE` (a
        # setpoint is ramped, never written across a gap: the solve that
        # reaches the hand begins where one that missed left the arm)
        lead = float((prev - h) @ w)
        back = min(1.0, -rate * dt / lead) if rate < 0.0 and lead > 0.0 else 0.0
        gap = float(np.abs(there - self.q_cmd[side]).max())
        back = min(back, JOINT_RATE * dt / gap) if gap > 0.0 else back
        c.pos, c.rot, pinned = prev + back * (h - prev), prev_rot, True
        q, ep, er = self.q_cmd[side] + back * (there - self.q_cmd[side]), self.miss[side], 0.0
        pv["beyond"] = True
      elif pressed or st.get("given"):
        # ...none while the hand reads a load, nor once one was given up
        # (`PRESSED_SHARE`, #699): it goes as far as its arm reaches
        pass
      elif not self.kept_by[side] and st["t"] - st.get("planned", -REPLAN_S) >= REPLAN_S:
        st["planned"] = st["t"]
        # ...never one it has left this move: free to, two postures could take
        # turns, each switch a fresh wait, and the move never end out of reach
        left_ = st.setdefault("left", [])
        posture = self.posture_for(side, c.pos, rot=c.rot, onward=True, past=left_)
        if posture is not None and tuple(posture) != tuple(self.posture):
          left_.append(tuple(float(v) for v in self.posture))
          st.setdefault("began", self._standing_in())
          self.hold_posture(*posture)
          st["postured"] = st["t"]
    self.goal[side].pos, self.goal[side].rot = c.pos.copy(), c.rot
    # held fast: the hand stays put while it pushes with its whole force, or
    # goes nowhere with its command as far ahead as it may go -- as far as its
    # arm reaches, only where it pushes its whole force: else it can reach no
    # further
    if not moved and along >= FAST_SHARE * limit and not settling:
      pv["still"] += dt
    pv["pinned"] = pv["pinned"] + dt if pinned and not pace and not settling else 0.0
    if ep < WILD_M:
      self.q_cmd[side], self.miss[side] = q, ep
    if ep > BLOCKED_M or er > HOLD_ROT_TOL:
      # ...given a posture, the time it takes to come -- none once it was
      # given up and the body is held (#699): an arm that cannot follow
      # from there is out of reach now
      coming = (st["t"] < SETTLE_S or ((not st.get("given") or st.get("backing"))
                and st["t"] - st.get("postured", -POSTURE_WAIT_S) < POSTURE_WAIT_S))
      self._off[side] = 0.0 if coming else self._off[side] + dt
      if self._off[side] >= BLOCKED_S:
        self._end(side, "out of reach", h, r)
      return c.stiffness
    self._off[side] = 0.0
    if pv["went"] >= pv["distance"]:
      self._end(side, "distance", h, r)
    elif pv["still"] >= FAST_S:
      self._end(side, "held fast", h, r)
    elif pv["pinned"] >= FAST_S:
      short = pv["beyond"] and along < FAST_SHARE * limit
      self._end(side, "out of reach" if short else "held fast", h, r)
    return c.stiffness

  def _feel(self) -> None:
    """Each hand's force off the drivers' torques as just read, the model
    posed off the senses (`_pose`), a turning joint's friction taken off
    (the arms' and the waist's): every hand's arm alone, then the whole
    body's for those at work, standing (`FELT_PHASES`) -- and what the robot
    reads of it (`sensed_force`), the same with the body's own motion in it
    as its senses have it for a hand going its own way (`arms.Motion`,
    `HELD_PHASES`, #631)."""
    sm = self.self_model
    motion = self.motion.read(sm, self.qd, self.stand)
    tau = self.tau.copy()
    tau[sm.waist] -= friction.taken_by(self.friction, hm.WAIST, self.qd[sm.waist])
    for s in hm.SIDES:
      arm = sm.arm[s]
      tau[arm] -= friction.taken(self.friction, s, self.qd[arm])
      self.arm_force[s] = sm.hand_force(s, tau[arm])
    self.force = dict(self.arm_force)
    working = [s for s in hm.SIDES if self.phase[s] in FELT_PHASES]
    whole = self.standing and bool(working)
    felt = sm.felt(working, tau, self.still) if whole else {}
    self.force.update(felt)
    self.closed = {s: s in felt for s in hm.SIDES}
    self.sensed_force = dict(self.force)
    going = [s for s in hm.SIDES if self.phase[s] not in HELD_PHASES + ("rest",)]
    if motion is None or not going:
      return
    for s in going:
      if not (whole and s in working):
        self.sensed_force[s] = sm.hand_force(s, tau[sm.arm[s]], motion=motion)
    if whole and all(s in going for s in working):
      self.sensed_force.update(sm.felt(working, tau, self.still, motion))

  def _update(self, dt: float) -> None:
    sm = self.self_model
    self._pose()
    self._feel()
    # Where each hand IS, by its senses (#493): a carried hand's to steer by,
    # and every hand's for a stop condition to read (#554)
    seen = {s: sm.grip_pose(s) for s in hm.SIDES}
    self.sensed_grip = {s: seen[s][0].copy() for s in hm.SIDES}
    have = {s: seen[s] for s in hm.SIDES if self.phase[s] == "carry"}
    self._trim(dt)
    self._pose(self.q_cmd)
    for s in hm.SIDES:
      phase, stiffness = self.phase[s], 1.0
      if phase == "carry":
        pulled = self.pulls[s] is not None and self.pulls[s].get("pull") is not None
        stiffness = (self._pulled if pulled else self._held)(s, dt, *have[s])
        sm._set_arm(s, self.q_cmd[s])
      elif phase == "move" and self._tipped(s, dt):
        self._struck(s)
        sm._set_arm(s, self.q_cmd[s])
        self.miss[s], stiffness = 0.0, self.cmd[s].stiffness
      elif phase in ("rest", "move", "joints", "home", "carried", "held"):
        if self._shoved(s, dt) or self._stalled(s, dt):
          # ...a way home that would tip the body away from its hand, or whose
          # arm stalls as its command walks on, is held where its drivers read
          # it, as one its rest is held from (#693's live family, its withdraw
          # case): eased, or raised, never on toward its rest
          self._held_home(s)
          phase = self.phase[s]
        elif phase == "rest" and self.homing[s]:
          # ...a way home is over once its drivers read it home, or held
          self._homed(s, dt)
        there = self._joints(s, dt)
        sm._set_arm(s, self.q_cmd[s])
        self.miss[s] = 0.0
        if phase == "move" and there and self.outs[s]:
          self.q_to[s] = self.outs[s].pop(0)
        elif phase == "move" and there and self.stand.settled and self._posture_there(dt):
          pos, rot = sm.grip_pose(s)
          self.cmd[s] = HandGoal(pos, rot, self.goal[s].stiffness)
          self.phase[s] = "approach"
          # ...once it has looked where it stands (#627): else its hand waits
          # at its pre-grasp while its body takes the posture the look chose
          self.looking[s] = not self._look(s)
        elif phase == "home" and there:
          if self.homes[s]:
            self.q_to[s] = self.homes[s].pop(0)
          else:
            self.phase[s] = "rest" if self.carry_q[s] is None else "carried"
            # ...a carry's way ends at its pose, never read home (#682)
            self.homing[s] = self.homing[s] and self.carry_q[s] is None
            # ...its command home: the stand its way home kept from it
            # (`_go_home`), its drivers not yet read
            self._stand_at_ease()
      elif phase == "blocked":
        sm._set_arm(s, self.q_cmd[s])
        stiffness = self.cmd[s].stiffness
      else:
        # approach, pull, hold or retreat: a line (`_line`), solved again.
        held = HandGoal(self.cmd[s].pos.copy(), self.cmd[s].rot, self.cmd[s].stiffness)
        waiting = phase == "approach" and self.looking[s]
        if waiting:
          # ...but a hand waiting at its pre-grasp holds there while its body
          # takes the posture its look chose (`_look`), and looks again there
          there = False
          if self.stand.settled and self._posture_there(dt):
            self.looking[s] = not self._look(s)
        else:
          there = self._line(s, dt)
        c = self.cmd[s]
        q, ep, _ = sm.solve(s, c.pos, c.rot)
        # Out of reach on its way or while it holds; pushing only on its
        # way in (a hold may carry a weight, and a pull pushes on purpose) --
        # never while its body takes a posture with the hand waiting: the
        # force off the drivers assumes the body still (`BLOCKED_S`).
        pushing = float(np.linalg.norm(self.force[s]))
        pushed = phase == "approach" and pushing > BLOCKED_N
        off = phase != "retreat" and not waiting and (ep > BLOCKED_M or pushed)
        if off:
          # ...the run behind a stop, counted afresh wherever `_off` begins again
          n_push, n_all, hardest = (0, 0, 0.0) if self._off[s] == 0.0 else self._pushes[s]
          self._pushes[s] = (n_push + pushed, n_all + 1, max(hardest, pushing) if pushed else hardest)
        self._off[s] = self._off[s] + dt if off else 0.0
        if self._off[s] >= BLOCKED_S:
          # Stopped where it was last commanded: the goal is out of its reach,
          # or the hand pushes on something -- the push said where it held at
          # least half the run, both or not (`_pushes`)
          n_push, n_all, hardest = self._pushes[s]
          self.stopped[s] = ({"by": "pushed", "n": hardest} if 2 * n_push >= n_all
                             else {"by": "reach"})
          self.cmd[s] = held
          q, ep, _ = sm.solve(s, held.pos, held.rot)
          self.phase[s], there = "blocked", False
        self.q_cmd[s], self.miss[s], stiffness = q, ep, c.stiffness
        if there and phase in ("approach", "pull"):
          self.phase[s] = "hold"
        elif there and phase == "retreat":
          if self._raise_short(s):
            # ...a raise its arm did not follow -- held from above, or under
            # an edge -- is no start for a way home: given `BLOCKED_S` to
            # catch up, then left held where it is
            self.held_for[s] += dt
            if self.held_for[s] >= BLOCKED_S:
              self._ease(s)
              self._stand_at_ease()
              sm._set_arm(s, self.q_cmd[s])
          else:
            self._go_home(s)
      self.arms[s].command(self.q_cmd[s], sm.gravity(s), stiffness)
    # ...and the waist to the posture's angle, the trunk above it held as
    # the arms are commanded
    self._turn_waist(dt, sm.waist_gravity())
