"""Learn's verb on the umbrella CLI — `astro-mine train`.

Training is a **headline** action, and the one the commons turns on: a policy is the platform's
unit of exchange, so the command that produces one should be reachable by naming the action rather
than the package (RFC-0011 §2; astro-mine/docs#57). Hence a per-verb adapter at the top level
rather than an `astro-mine learn …` passthrough.

**Nothing here imports the umbrella.** The contract is structural — ``name``, ``help``,
``add_arguments(parser)``, ``run(args) -> int`` — so a component is reachable from `astro-mine`
without depending on it (``conventions.md §1.1``). ``astro-mine-cli`` is not a dependency of this
package and must not become one.

**Nothing here re-declares a flag.** Learn's CLI is flat — no subcommands — so the whole parser
*is* this verb, and the adapter attaches the same ``add_train_arguments`` that
:func:`astro_mine.learn.train.run._parser` uses and calls the same ``main``-equivalent body.
`astro-mine train --export …` therefore takes exactly the flags `astro-mine-train --export …` does.

The umbrella imports this module only when `train` actually runs, which matters here: Learn's
import graph reaches Ray and Torch, and a user typing `astro-mine score` should never pay for them.
"""

from __future__ import annotations

import argparse

from astro_mine.learn.train.run import add_train_arguments, run_from_args

__all__ = ["train"]


class _Train:
    name = "train"
    help = "train a policy and export it"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        add_train_arguments(parser)

    def run(self, args: argparse.Namespace) -> int:
        return run_from_args(args)


train = _Train()
