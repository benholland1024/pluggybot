"""A failed grade's reading against its line (issue #737), each pinned by a
direct call on its grader with a made-up reading (docs/Testing.md):

  1. a reading near its line is printed so that it never reads as the line
     it failed -- "came out 14.7 cm, short of the 15", never "15 cm, short
     of the 15" (Luca's drawer, 11-22), nor "called 960 s after the start,
     past the 960 s" (Rowan's, 11-22) -- in every sentence of the five
     graders that says a reading against its line: the drawer's, the
     door's, the lid's, the three speed limits, the fork's turn, the time
     limit and the hold;
  2. a reading far from its line keeps its whole words (`test_lid.py`'s
     "rose to 50 deg, short of the 57", `test_hand_jobs.py`'s "came out 0
     cm, short of the 15"), and the cube's and the table's distances say
     what they said;
  3. the lines said once over several of the table's pieces are exact at
     the places they are said at, so no piece's reading needs them longer.
"""

import math
import re

import pytest

from pluggybot.challenge import door, drawer, lid, place, table

LID = {"topDeg": 63.8, "downRadS": 0.51, "tiltDeg": 2.8, "falls": 0, "touchedDuringHold": [],
       "angleAtCallDeg": -0.2, "angleDeg": -0.2, "caught": True, "holdS": 5.0,
       "elapsedS": 84.0, "catchN": 3.89, "limitS": 960.0}
DRAWER = {"mostM": 0.18, "fastestMS": 0.1, "tiltDeg": 2.0, "falls": 0, "outAtCallM": 0.17,
          "outM": 0.17, "holdS": 5.0, "elapsedS": 20.0, "limitS": 960.0, "touchedDuringHold": []}
DOOR = {"mostDeg": 70.0, "pressedDeg": 45.0, "fastestRadS": 1.0, "tiltDeg": 2.0, "falls": 0,
        "angleAtCallDeg": 65.0, "angleDeg": 64.0, "holdS": 5.0, "elapsedS": 30.0,
        "limitS": 960.0, "touchedDuringHold": []}
CUBE = {"liftedM": 0.05, "fastestDownMS": 0.2, "fastestFreeMS": 0.2, "robotTiltDeg": 2.0,
        "falls": 0, "offAtCallM": 0.01, "offM": 0.01, "aboveM": 0.0005, "tiltDeg": 1.0,
        "holdS": 5.0, "elapsedS": 20.0, "limitS": 960.0, "touchedDuringHold": []}
PIECE = {"liftedM": 0.05, "fastestFreeMS": 0.1, "offAtCallM": 0.01, "offM": 0.01,
         "aboveM": 0.0, "tiltDeg": 1.0}


def _table(**kw) -> dict:
  m = {"robotTiltDeg": 5.0, "falls": 0, "touchedDuringHold": [], "holdS": 5.0,
       "elapsedS": 90.0, "limitS": 960.0, "plate": dict(PIECE), "fork": dict(PIECE),
       "cup": dict(PIECE)}
  m["fork"]["turnDeg"] = 3.0
  for k, v in kw.items():
    m[k] = {**m[k], **v} if isinstance(v, dict) else v
  return m


GRADERS = {"lid": (lid.eval_lid, LID), "drawer": (drawer.eval_drawer, DRAWER),
           "door": (door.eval_door, DOOR), "cube": (place.eval_place, CUBE),
           "table": (table.eval_table, _table())}


def test_the_passing_set_outs_pass():
  for name, (grade, passing) in GRADERS.items():
    assert grade(passing)[0], name


# ---- 1. near its line, never printed as the line ------------------------------------

@pytest.mark.parametrize("grade, m, said", [
  # the issue's: Luca's drawer, the door, the lid's 57.1 deg, a descent of 1.04 rad/s
  (drawer.eval_drawer, DRAWER | {"mostM": 0.147, "outAtCallM": 0.147, "outM": 0.147},
   "the drawer came out 14.7 cm, short of the 15 it must"),
  (drawer.eval_drawer, DRAWER | {"outAtCallM": 0.1497, "outM": 0.1497},
   "the drawer came out 18 cm but was 14.97 cm out when you finished, short of the 15 it must"),
  (drawer.eval_drawer, DRAWER | {"outM": 0.1496}, "the drawer 17 cm out at the call, but 14.96 cm 5 s later"),
  (door.eval_door, DOOR | {"mostDeg": 49.6, "angleAtCallDeg": 49.6, "angleDeg": 49.6},
   "the door opened 49.6 deg, short of the 50 it must"),
  (door.eval_door, DOOR | {"angleAtCallDeg": 49.96, "angleDeg": 49.96},
   "the door opened 70 deg but was 49.96 deg open when you finished, short of the 50 it must"),
  (door.eval_door, DOOR | {"angleDeg": 49.7}, "the door 65 deg open at the call, but 49.7 deg 5 s later"),
  (lid.eval_lid, LID | {"topDeg": 57.1}, "the lid rose to 57.1 deg, short of the 57.3 it must open past"),
  (lid.eval_lid, LID | {"topDeg": 57.0}, "the lid rose to 57.0 deg, short of the 57.3 it must open past"),
  (lid.eval_lid, LID | {"downRadS": 1.04},
   "came down at 1.04 rad/s while open -- let fall, not lowered (no faster than 1.0)"),
  (place.eval_place, CUBE | {"fastestFreeMS": 0.504},
   "the cube came down at 0.504 m/s with nothing holding it -- dropped, not set down (no "
   "faster than 0.5)"),
  (table.eval_table, _table(cup={"fastestFreeMS": 0.53}),
   "the cup came down at 0.53 m/s with nothing holding it -- dropped or knocked over, not set "
   "down (no faster than 0.5)"),
  (table.eval_table, _table(fork={"turnDeg": 15.3}),
   "the fork lies 15.3 deg off the place's line: it must lie along it, its head toward the "
   "card, within 15 deg"),
] + [
  # Rowan's t_0758 (W1's comment on #737): every grader's time limit, and its hold
  (grade, passing | {"elapsedS": 960.3}, "called 960.3 s after the start, past the 960 s the job allows")
  for grade, passing in GRADERS.values()
] + [
  (grade, passing | {"holdS": 4.97}, "measured 4.97 s after the call, not the 5 s the job asks for")
  for grade, passing in GRADERS.values()
])
def test_a_reading_near_its_line_is_never_printed_as_the_line(grade, m, said):
  ok, _, why = grade(m)
  assert not ok and said in why, why


#: Each sentence that says a reading against its line: the grader, the
#: measurements set to the reading, the line, the scale it is printed at
#: (the drawer's metres as cm), a pattern over the reason catching the
#: reading and the line it prints, and which way the reading failed it. The
#: end of a hold prints no line: its reading is read against the job's.
SENTENCES = [
  ("drawer", ("mostM", "outAtCallM", "outM"), drawer.OPEN_M, 100,
   r"came out ([\d.]+) cm, short of the ([\d.]+) it must", -1),
  ("drawer", ("outAtCallM", "outM"), drawer.OPEN_M, 100,
   r"was ([\d.]+) cm out when you finished, short of the ([\d.]+)", -1),
  ("drawer", ("outM",), drawer.OPEN_M, 100, r"but ([\d.]+) cm 5 s later()", -1),
  ("door", ("mostDeg", "angleAtCallDeg", "angleDeg"), math.degrees(door.OPEN_RAD), 1,
   r"door opened ([\d.]+) deg, short of the ([\d.]+) it must", -1),
  ("door", ("angleAtCallDeg", "angleDeg"), math.degrees(door.OPEN_RAD), 1,
   r"was ([\d.]+) deg open when you finished, short of the ([\d.]+)", -1),
  ("door", ("angleDeg",), math.degrees(door.OPEN_RAD), 1, r"but ([\d.]+) deg 5 s later()", -1),
  ("lid", ("topDeg",), math.degrees(lid.OPEN_RAD), 1,
   r"rose to ([\d.]+) deg, short of the ([\d.]+)", -1),
  ("lid", ("downRadS",), lid.LOWER_RAD_S, 1,
   r"came down at ([\d.]+) rad/s .*\(no faster than ([\d.]+)\)", 1),
  ("cube", ("fastestFreeMS",), place.DROP_M_S, 1,
   r"came down at ([\d.]+) m/s .*\(no faster than ([\d.]+)\)", 1),
  ("cube", ("offM",), place.PLACE_M, 100, r"ended ([\d.]+) cm .* within ([\d.]+) cm", 1),
  ("table", ("cup.fastestFreeMS",), table.DROP_M_S, 1,
   r"came down at ([\d.]+) m/s .*\(no faster than ([\d.]+)\)", 1),
  ("table", ("cup.offM",), table.PLACE_M, 100, r"ended ([\d.]+) cm .* within ([\d.]+) cm", 1),
  ("table", ("fork.turnDeg",), math.degrees(table.FORK_RAD), 1,
   r"lies ([\d.]+) deg .* within ([\d.]+) deg", 1),
] + [
  (name, ("elapsedS",), 960.0, 1, r"called ([\d.]+) s after the start, past the ([\d.]+) s", 1)
  for name in GRADERS
] + [
  (name, ("elapsedS", "limitS"), 67.5, 1, r"called ([\d.]+) s after the start, past the ([\d.]+) s", 1)
  for name in GRADERS
] + [
  (name, ("holdS",), lid.HOLD_S, 1, r"measured ([\d.]+) s after the call, not the ([\d.]+) s", -1)
  for name in GRADERS
]
#: How far past its line each sentence is tried, in the units it is printed in.
OFF = (1e-5, 1e-4, 0.001, 0.004, 0.0049, 0.0051, 0.01, 0.04, 0.049, 0.051, 0.3, 0.49, 0.5,
       0.51, 3.0)


@pytest.mark.parametrize("name, keys, line, scale, pattern, way", SENTENCES)
def test_every_reading_against_its_line_prints_as_it_was_judged(name, keys, line, scale, pattern, way):
  """The rule itself, over readings from a hair to well past the line: the
  printed reading and its printed line compare as the measured ones do."""
  grade, passing = GRADERS[name]
  for off in OFF:
    reading = line + way * off / scale
    m = {k: (dict(v) if isinstance(v, dict) else v) for k, v in passing.items()}
    for key in keys:
      piece, _, field = key.rpartition(".")
      # a limit set is the line itself; every other key is the reading
      value = line if field == "limitS" else reading
      (m[piece] if piece else m)[field] = value
    ok, _, why = grade(m)
    hit = re.search(pattern, why)
    assert not ok and hit, (reading, why)
    shown = float(hit.group(1))
    said = float(hit.group(2)) if hit.group(2) else scale * line
    assert (shown > said) - (shown < said) == way, (reading, why)


# ---- 2. far from its line, its words as they were ------------------------------------

def test_a_reading_far_from_its_line_keeps_its_whole_words():
  assert lid.against(0.0, 15.0) == ("0", "15")
  assert lid.against(50.0, math.degrees(lid.OPEN_RAD)) == ("50", "57")
  assert lid.against(2.4, lid.LOWER_RAD_S, 1) == ("2.4", "1.0")
  assert lid.against(1200.4, 960.0) == ("1200", "960")
  assert lid.against(4.0, lid.HOLD_S, 1, 0) == ("4.0", "5")
  # the cube's and the table's distances round up past their line, as before
  assert lid.against(2.6, 100 * place.PLACE_M, 0, 1) == ("3", "2.5")
  assert lid.against(2.501, 100 * place.PLACE_M, 0, 1) == ("3", "2.5")
  # a reading on its line is the line
  assert lid.against(15.0, 15.0) == ("15", "15")


# ---- 3. a line said once over several pieces -----------------------------------------

def test_the_lines_said_once_over_the_table_pieces_are_exact_at_their_places():
  """The drop's and the place's line are said once beside every piece at
  fault, at their own places; exact there, every piece's reading is printed
  against the same line `lid.against` prints."""
  for line, places in ((table.DROP_M_S, 1), (100 * table.PLACE_M, 1)):
    assert float(f"{line:.{places}f}") == line
  why = table.eval_table(_table(plate={"fastestFreeMS": 0.501}, cup={"fastestFreeMS": 4.6}))[2]
  assert why.startswith("the plate came down at 0.501 m/s and the cup came down at 4.6 m/s, "
                        "each with nothing holding it"), why
