"""Where the humanoids' jobs are set out (#559, #560): the research center
storeroom's counter (`home.storeroom`) and the restaurant's table
(`home.restaurant`), each a module of the same members, so the lifecycle
offers, sets out, restores and puts away a job's things one way, whichever
area its target is in. A target names its area's thing: a prop on the
counter, the table a place is laid at.

  TARGETS                      {target: its `TaskKind.target_kind`}
  present(model)               whether the compiled world has the area
  set_out(model, data, t, k, kind)
                               an offer's: the thing as draw k of the job's
                               kind sets it out (the table's place laid
                               from beside it, from the pass or from the
                               kitchen's shelf)
  redraw(model, data, t, k)    draw k's hidden dynamics alone (a restart's)
  drawn(t, k)                  draw k's truth, its activity's
  touched_by(model, data, t, geoms)
  put_away(model, data, t), away(model, data, t)
                               what no job holds put away (the table's
                               dishware; the counter's props stay)

...and, where an area keeps some of a job's things put away until it is
taken (the table's pieces, #775: an offer stands untaken most of its time,
and a piece out costs every step and every ray), the members that say how;
an area without `take` sets everything out at the offer:

  offered(model, data, t, k, kind)
                               an offer's there: what it shows while nobody
                               holds the job (the card), the rest put away;
                               `set_out` is then a taken job's whole set-out
  take(model, data, t, k, kind)
                               a claim's: the rest where draw k lays it
  pieces_away(model, data, t), as_placed(model, data, t)
                               what a restore reads and sets: the rest put
                               away, and the geoms as a saved world had them

Each area's activity is named by its target (`HubLifecycle._target_activity`).
"""

from pluggybot.home import restaurant, storeroom

AREAS = (storeroom, restaurant)


def area_of(target: str):
  """The area a target is in, or None."""
  for area in AREAS:
    if target in area.TARGETS:
      return area
  return None


def targets() -> dict[str, str]:
  """Every area's targets, by their target kind."""
  return {t: kind for area in AREAS for t, kind in area.TARGETS.items()}
