import os as _os

from pluggybot.batch import STACKS_ENV as _STACKS_ENV
from pluggybot.batch import stacks_on_sigusr1 as _stacks_on_sigusr1

# ...a flight a watched batch started (`pluggybot.batch`, #684): its stacks
# on SIGUSR1, however it was started -- a pool's worker runs no script's main
if _os.environ.get(_STACKS_ENV):
  _stacks_on_sigusr1()


def hello() -> str:
    return "Hello from pluggybot!"
