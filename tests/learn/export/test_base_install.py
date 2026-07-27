"""`import astro_mine.learn` must work with every optional extra absent (#30).

The base wheel exists to be light: `pyproject.toml` keeps `onnx`/`onnxruntime` in `[export]` and
Torch in `[rllib]` because *"a policy consumer needs neither"*, and `conventions.md §7` says the
tier that "MUST always work" is the one you get from a plain install. Until #30 that promise was
empty — `astro_mine/learn/__init__.py` reached `astro_mine.learn.export.tensors`, which ran
`export/__init__.py`, which imported `onnxruntime` and Torch. Nothing in the package imported.

**These tests run in a subprocess, deliberately.** An import guarantee cannot be tested in-process:
by the time this module runs, the rest of the suite has already imported `astro_mine.learn` with
every extra present, and `sys.modules` would hide exactly the regression being watched for. A fresh
interpreter with the optional distributions blocked is the only honest form of the question — and
it is the same shape as the environment the guarantee is about.

The blocker refuses the optional trees rather than the package, so a regression surfaces as the
real failure a user would see (`ModuleNotFoundError: torch`) rather than as a test-harness artifact.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

#: Every optional distribution Learn declares behind an extra. Blocking all of them at once is the
#: point: the guarantee is not "onnxruntime is deferred", it is "a base install works".
BLOCKED = ("torch", "onnx", "onnxruntime", "jax", "jaxlib", "mlflow", "ray", "pyarrow")

_BLOCKER = f"""
import sys
from importlib.abc import MetaPathFinder

BLOCKED = {BLOCKED!r}


class _Blocker(MetaPathFinder):
    \"\"\"Refuse the optional trees, exactly as a base install would.\"\"\"

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{fullname!r}}", name=fullname)
        return None


sys.meta_path.insert(0, _Blocker())
# Anything already resident would defeat the blocker; nothing should be, in a fresh interpreter.
for name in list(sys.modules):
    if name.split(".")[0] in BLOCKED:
        del sys.modules[name]
"""


def _run(body: str) -> subprocess.CompletedProcess[str]:
    """Run `body` in a fresh interpreter with the optional distributions blocked."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_package_imports_with_every_extra_absent() -> None:
    """The claim the base wheel makes about itself. If this fails, `[export]` and `[rllib]` are
    not extras — they are dependencies with an inaccurate declaration."""
    result = _run(
        """
        import astro_mine.learn
        print(astro_mine.learn.SwarmEnv.__name__)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "SwarmEnv"


def test_the_eval_surface_imports_with_every_extra_absent() -> None:
    """`eval/onnx.py` documents that it — and the whole `eval` surface — imports without the
    extra. It was already written that way and was defeated by a package `__init__` it did not
    own, which is the whole of #30."""
    result = _run(
        """
        from astro_mine.learn.eval import onnx_policy_id, onnx_policy_under_test
        print(onnx_policy_id.__name__, onnx_policy_under_test.__name__)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["onnx_policy_id", "onnx_policy_under_test"]


def test_the_tensor_vocabulary_costs_nothing() -> None:
    """The constants are the contract a host binds, not export machinery — `tensors` imports
    nothing at all, and reaching it must not execute the export stack. This is the specific import
    that used to drag in Torch and ONNX Runtime."""
    result = _run(
        """
        import sys
        from astro_mine.learn.export import OBS_INPUT, HIDDEN_INPUT, MESSAGE_INPUT
        from astro_mine.learn.export.tensors import STATE_TENSORS
        assert "astro_mine.learn.export.equivalence" not in sys.modules, "export machinery loaded"
        print(OBS_INPUT, HIDDEN_INPUT, MESSAGE_INPUT, len(STATE_TENSORS))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_the_umbrella_verb_and_scaffolds_are_reachable() -> None:
    """What #30 actually cost a user: the umbrella's `train` verb and both plugin scaffolds are
    entry points into this package, so an unimportable package made all three fail at the command
    line with a traceback about ONNX Runtime."""
    result = _run(
        """
        from astro_mine.learn.umbrella import train
        from astro_mine.learn.scaffolds import algorithm_scaffold, curriculum_scaffold
        print(train.name, algorithm_scaffold.name, curriculum_scaffold.name)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["train", "algorithm", "curriculum"]


@pytest.mark.parametrize(
    ("symbol", "extra"),
    [
        # The extra named is the first one the symbol's own module actually trips on, which is a
        # property of that module rather than of the symbol: `package`/`onnx` reach for Torch
        # (`[rllib]`), `host`/`equivalence` reach for ONNX Runtime (`[export]`). Both are correct
        # answers to "what do I install to use this", and neither is guessable from the name — so
        # they are pinned per symbol rather than asserted generically.
        ("export_policy_package", "rllib"),
        ("to_onnx_bytes", "rllib"),
        ("onnx_policy", "export"),
        ("assert_onnx_equivalence", "export"),
    ],
)
def test_touching_the_machinery_names_the_extra_it_needs(symbol: str, extra: str) -> None:
    """The other half of laziness: deferring an import must not make the failure *worse*.

    A name that genuinely needs an extra still fails — but when it is used, and with an install
    line, rather than as a bare `ModuleNotFoundError` raised while some unrelated module was being
    imported. Getting this wrong would trade one confusing error for a quieter one.
    """
    result = _run(
        f"""
        import astro_mine.learn.export as export
        try:
            export.{symbol}
        except ModuleNotFoundError as exc:
            print(exc)
        else:
            raise AssertionError("expected the missing extra to be reported")
        """
    )
    assert result.returncode == 0, result.stderr
    message = result.stdout
    assert f"[{extra}]" in message
    assert f"astro-mine-learn[{extra}]" in message


def test_an_unknown_attribute_is_still_an_attribute_error() -> None:
    """A lazy module must not turn every typo into an import error — `hasattr` and duck-typing
    across the package depend on the normal failure mode surviving."""
    result = _run(
        """
        import astro_mine.learn.export as export
        assert not hasattr(export, "definitely_not_a_symbol")
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_dir_still_lists_the_whole_surface() -> None:
    """Tab-completion and introspection are how people find this API; laziness must not empty it."""
    result = _run(
        """
        import astro_mine.learn.export as export
        names = dir(export)
        missing = [n for n in export.__all__ if n not in names]
        print("missing:", missing)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "missing: []"
