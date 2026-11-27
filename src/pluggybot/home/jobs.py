"""Where the humanoids' jobs are set out (#559, #560): the research center
storeroom's counter (`home.storeroom`), the restaurant's table
(`home.restaurant`) and the shop's shelf (`home.store`, #736: present in no
served house yet), each a module of the same members, so the lifecycle
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

Each area's activity is named by its target (`HubLifecycle._target_activity`).
"""

from pluggybot.home import restaurant, storeroom, store

AREAS = (storeroom, restaurant, store)


def area_of(target: str):
  """The area a target is in, or None."""
  for area in AREAS:
    if target in area.TARGETS:
      return area
  return None


def targets() -> dict[str, str]:
  """Every served area's targets, by their target kind: a world's `jobs`
  (`world_config`), what its producer may offer -- never an area no served
  house has (`SERVED` False: the shop's shelf, #736), whose job only a
  flight that puts it into its own house offers (`solve.py --feature
  shelf`)."""
  return {t: kind for area in AREAS if getattr(area, "SERVED", True)
          for t, kind in area.TARGETS.items()}


def words(target: str, k: int) -> dict:
  """What an offer at draw `k` of `target` names of what it sets out, the
  words its sentence is filled with (`TaskProducer._build`): the shelf's
  labels (`store.words`); none for any other area."""
  area = area_of(target)
  return dict(area.words(target, k)) if area is not None and hasattr(area, "words") else {}
