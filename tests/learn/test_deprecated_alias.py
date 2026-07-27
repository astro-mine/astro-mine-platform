"""The deprecated `astro-mine-train` alias (RFC-0011 §5).

``astro-mine-train`` already carried the prefix; what it got wrong was the noun — it was named
after the verb rather than the package that ships it (``conventions.md §13``). The old name
keeps working for one deprecation cycle and is removed at the first public-benchmark milestone.

Two properties are worth holding onto until then. The alias must actually *work* — a deprecation
that breaks the command is a removal wearing a warning label. And its notice must go to
**stderr**, because the scripts most likely to still use the old name are exactly the ones piping
stdout somewhere that a stray line would corrupt.
"""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest

from astro_mine.learn.train.run import deprecated_alias, main

OLD = "astro-mine-train"
NEW = "astro-mine-learn"


def test_both_console_scripts_are_declared() -> None:
    """The new name is the command; the old one still resolves, to the shim rather than to
    ``main`` — so it cannot quietly become a silent second entry point."""
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts.get(NEW) == "astro_mine.learn.train.run:main"
    assert scripts.get(OLD) == "astro_mine.learn.train.run:deprecated_alias"


def test_the_notice_goes_to_stderr_and_names_the_replacement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        deprecated_alias(["--help"])
    captured = capsys.readouterr()
    assert OLD in captured.err and NEW in captured.err
    assert "deprecated" in captured.err
    # The point of the whole exercise: nothing on stdout.
    assert OLD not in captured.out or "deprecated" not in captured.out


def test_stdout_is_byte_identical_to_the_new_name(capsys: pytest.CaptureFixture[str]) -> None:
    """A pipeline that switches names must see no difference at all on stdout."""
    with pytest.raises(SystemExit):
        main(["--help"])
    canonical = capsys.readouterr().out
    with pytest.raises(SystemExit):
        deprecated_alias(["--help"])
    assert capsys.readouterr().out == canonical
