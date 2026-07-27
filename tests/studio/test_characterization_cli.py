"""Characterization tests for the reachable (non-serve-composition) parts of `studio.cli`.

astro-mine-platform deliberately does not ship the Studio REST surface
(``astro_mine.studio.api``), so :func:`build_serve_app` always raises ImportError at its guard and
everything below it — the app composition, the 503 degradation matrix, the uvicorn run — is dead
code in this distribution (lines after ``from .api import create_app`` in ``build_serve_app``, and
the post-build half of ``_cmd_serve``). These tests pin the *current* behavior of everything else:
argument parsing, the ``_resolve_*`` precedence chains, the seam-wiring and banner helpers, the UI
mount, the seed attach, and the honest ImportError degradation of ``serve`` itself.

No behavior is added or fixed here — oddities are pinned with comments.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astro_mine.core.registry import CapabilityTag
from astro_mine.hub.registry import Registry
from astro_mine.studio.cli import (
    CACHE_DIR_ENV,
    DEFAULT_SIGNING_KEY_NAMES,
    DEFAULT_TRUSTED_KEY_NAMES,
    REGISTRY_ENV,
    SIGNING_KEY_ENV,
    TRUSTED_KEY_ENV,
    UI_DIR_ENV,
    SeamState,
    ServeReport,
    _attach_seed,
    _build_parser,
    _cmd_serve,
    _mount_ui,
    _read_key,
    _resolve_cache_dir,
    _resolve_key,
    _resolve_registry,
    _resolve_ui_dir,
    _wire_hub_seams,
    build_serve_app,
    main,
    render_banner,
)

from .test_cli import _registry_with_anchor_keys
from .test_hub_catalog import _publish_asset


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The workspace convention exports ``ASTRO_MINE_HUB_REGISTRY``; clear every CLI env default so
    each test pins the precedence chain explicitly rather than inheriting the workspace's."""
    for env in (REGISTRY_ENV, TRUSTED_KEY_ENV, SIGNING_KEY_ENV, CACHE_DIR_ENV, UI_DIR_ENV):
        monkeypatch.delenv(env, raising=False)


# --- argument parsing / main() dispatch -----------------------------------------------------


def test_top_level_help_exits_zero_and_names_the_prog(capsys: pytest.CaptureFixture) -> None:
    """``--help`` is a clean exit 0 with the prog name and the one subcommand on stdout."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "astro-mine-studio" in out
    assert "serve" in out
    assert "design front door" in out


def test_serve_help_lists_every_flag(capsys: pytest.CaptureFixture) -> None:
    """``serve --help`` documents each flag, including the env-var defaults in the help text."""
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--host",
        "--port",
        "--registry",
        "--trusted-key",
        "--signing-key",
        "--cache-dir",
        "--ui-dir",
        "--no-ui",
        "--no-seed",
    ):
        assert flag in out
    assert REGISTRY_ENV in out  # the registry default names the env convention


def test_no_subcommand_is_a_usage_error(capsys: pytest.CaptureFixture) -> None:
    """A subcommand is required; bare invocation is an argparse usage error (exit 2, stderr)."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_unknown_subcommand_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code == 2


def test_non_integer_port_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--port", "not-a-number"])
    assert exc.value.code == 2


def test_serve_is_the_only_subcommand() -> None:
    """The CLI offers exactly one verb — no version/banner/etc. subcommands exist today."""
    parser = _build_parser()
    sub = next(action for action in parser._actions if action.dest == "command")
    assert list(sub.choices) == ["serve"]


def test_serve_defaults_and_dispatch_target() -> None:
    """``serve`` parses to the documented defaults and dispatches to ``_cmd_serve``."""
    args = _build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.registry is None
    assert args.trusted_key is None
    assert args.signing_key is None
    assert args.cache_dir is None
    assert args.ui_dir is None
    assert args.no_ui is False
    assert args.seed is True
    assert args.func is _cmd_serve


# --- _resolve_registry ----------------------------------------------------------------------


def test_registry_flag_wins_over_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(REGISTRY_ENV, str(tmp_path / "from-env"))
    assert _resolve_registry(str(tmp_path / "from-flag")) == tmp_path / "from-flag"


def test_registry_env_used_when_no_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(REGISTRY_ENV, str(tmp_path / "from-env"))
    assert _resolve_registry(None) == tmp_path / "from-env"


def test_registry_absent_everywhere_is_none() -> None:
    assert _resolve_registry(None) is None


def test_registry_empty_string_flag_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinned quirk: an explicit empty ``--registry ''`` is falsy and resolves to None — it does
    NOT fall back to the env var (the arg short-circuits env lookup) and does not error."""
    monkeypatch.setenv(REGISTRY_ENV, "/somewhere")
    assert _resolve_registry("") is None


def test_registry_tilde_is_expanded() -> None:
    assert _resolve_registry("~/reg") == Path.home() / "reg"


# --- _resolve_key ---------------------------------------------------------------------------


def test_key_flag_wins_even_when_the_file_does_not_exist(tmp_path: Path) -> None:
    """Pinned: an explicit ``--trusted-key`` path is returned verbatim without an existence check
    (missing files degrade later, in ``_read_key``)."""
    ghost = tmp_path / "no-such-key.pem"
    got = _resolve_key(str(ghost), TRUSTED_KEY_ENV, None, DEFAULT_TRUSTED_KEY_NAMES)
    assert got == ghost


def test_key_env_wins_over_registry_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    keys = tmp_path / "reg" / "keys"
    keys.mkdir(parents=True)
    (keys / DEFAULT_TRUSTED_KEY_NAMES[0]).write_bytes(b"registry-key")
    monkeypatch.setenv(TRUSTED_KEY_ENV, str(tmp_path / "env-key.pem"))
    got = _resolve_key(None, TRUSTED_KEY_ENV, tmp_path / "reg", DEFAULT_TRUSTED_KEY_NAMES)
    assert got == tmp_path / "env-key.pem"  # env wins, existence again unchecked


def test_key_empty_env_falls_through_to_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pinned quirk: an empty env var is falsy, so the registry default is still consulted."""
    keys = tmp_path / "reg" / "keys"
    keys.mkdir(parents=True)
    expected = keys / DEFAULT_TRUSTED_KEY_NAMES[0]
    expected.write_bytes(b"registry-key")
    monkeypatch.setenv(TRUSTED_KEY_ENV, "")
    got = _resolve_key(None, TRUSTED_KEY_ENV, tmp_path / "reg", DEFAULT_TRUSTED_KEY_NAMES)
    assert got == expected


def test_key_cosign_name_beats_legacy_anchor_dev(tmp_path: Path) -> None:
    """When both default names exist under ``<registry>/keys/``, the first name (cosign) wins."""
    keys = tmp_path / "keys"
    keys.mkdir()
    (keys / DEFAULT_TRUSTED_KEY_NAMES[0]).write_bytes(b"cosign")
    (keys / DEFAULT_TRUSTED_KEY_NAMES[1]).write_bytes(b"legacy")
    got = _resolve_key(None, TRUSTED_KEY_ENV, tmp_path, DEFAULT_TRUSTED_KEY_NAMES)
    assert got == keys / DEFAULT_TRUSTED_KEY_NAMES[0]


def test_key_none_when_registry_has_no_keys_dir(tmp_path: Path) -> None:
    assert _resolve_key(None, SIGNING_KEY_ENV, tmp_path, DEFAULT_SIGNING_KEY_NAMES) is None


def test_key_none_when_nothing_is_available() -> None:
    assert _resolve_key(None, SIGNING_KEY_ENV, None, DEFAULT_SIGNING_KEY_NAMES) is None


# --- _resolve_ui_dir ------------------------------------------------------------------------


def test_ui_dir_default_is_cwd_relative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no flag and no env, the UI dir is ``<cwd>/ui/dist-harness`` — cwd-relative, so the
    answer changes with the working directory (pinned as-is)."""
    monkeypatch.chdir(tmp_path)
    assert _resolve_ui_dir(None) == tmp_path / "ui" / "dist-harness"


def test_ui_dir_env_overrides_the_cwd_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(UI_DIR_ENV, str(tmp_path / "custom-ui"))
    assert _resolve_ui_dir(None) == tmp_path / "custom-ui"


def test_ui_dir_flag_wins_over_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(UI_DIR_ENV, str(tmp_path / "from-env"))
    assert _resolve_ui_dir(str(tmp_path / "from-flag")) == tmp_path / "from-flag"


def test_no_ui_flag_suppresses_ui_resolution_entirely(tmp_path: Path) -> None:
    """``--no-ui`` short-circuits in ``_cmd_serve``: ``_resolve_ui_dir`` is never consulted and
    the ui_dir passed onward is None. Characterized at the parser level here (the ternary lives in
    ``_cmd_serve``, exercised in the degradation test below)."""
    args = _build_parser().parse_args(["serve", "--no-ui"])
    assert args.no_ui is True
    assert args.ui_dir is None


# --- _resolve_cache_dir ---------------------------------------------------------------------


def test_cache_dir_flag_wins_and_is_created(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "cache"
    assert not target.exists()
    got = _resolve_cache_dir(str(target))
    assert got == target
    assert target.is_dir()  # created eagerly, parents included


def test_cache_dir_env_used_when_no_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "env-cache"))
    got = _resolve_cache_dir(None)
    assert got == tmp_path / "env-cache"
    assert got.is_dir()


def test_cache_dir_defaults_under_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The last-resort default is ``~/.cache/astro-mine-studio`` (HOME redirected here so the
    test never touches the real home directory)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    got = _resolve_cache_dir(None)
    assert got == tmp_path / ".cache" / "astro-mine-studio"
    assert got.is_dir()


def test_cache_dir_creation_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "cache"
    assert _resolve_cache_dir(str(target)) == _resolve_cache_dir(str(target))


# --- _read_key ------------------------------------------------------------------------------


def test_read_key_degrades_to_none_not_a_crash(tmp_path: Path) -> None:
    """Missing keys are a degrade signal: None path → None, absent file → None, real file →
    its bytes."""
    assert _read_key(None) is None
    assert _read_key(tmp_path / "missing.pem") is None
    real = tmp_path / "key.pem"
    real.write_bytes(b"PEM-BYTES")
    assert _read_key(real) == b"PEM-BYTES"


# --- ServeReport / render_banner ------------------------------------------------------------


def test_serve_report_url_and_wired_properties() -> None:
    report = ServeReport(host="0.0.0.0", port=9001)
    assert report.url == "http://0.0.0.0:9001"
    report.seams = [SeamState("a", True, ""), SeamState("b", True, "")]
    assert report.all_seams_wired
    report.seams.append(SeamState("c", False, "down"))
    assert not report.all_seams_wired


def test_serve_report_with_no_seams_counts_as_all_wired() -> None:
    """Pinned quirk: ``all_seams_wired`` is vacuously True on an empty seam list (``all([])``)."""
    assert ServeReport(host="h", port=1).all_seams_wired


def test_banner_renders_every_seam_with_honest_marks() -> None:
    """The banner is a pure render of the report: URL, UI state, ✓/○ per seam, seed detail."""
    report = ServeReport(host="127.0.0.1", port=8123)
    report.seams = [
        SeamState("publishing", False, "no signing key — publish disabled"),
        SeamState("terrain", True, "verifying against cosign.pub"),
    ]
    report.ui_detail = "UI mount disabled"
    report.seed_detail = "seeding disabled (--no-seed)"
    banner = render_banner(report)
    assert "Astro-Mine Studio" in banner
    assert "http://127.0.0.1:8123" in banner
    assert "○ publishing: no signing key — publish disabled" in banner
    assert "✓ terrain: verifying against cosign.pub" in banner
    assert "UI:   UI mount disabled" in banner
    assert "Seed: seeding disabled (--no-seed)" in banner


# --- _wire_hub_seams (reachable: it needs only the Hub client, not studio.api) --------------


def test_wire_hub_seams_with_full_keys_wires_everything(tmp_path: Path) -> None:
    reg_path, trusted, signing = _registry_with_anchor_keys(tmp_path)
    kwargs, seams = _wire_hub_seams(reg_path, trusted, signing, tmp_path / "cache")

    assert all(seam.wired for seam in seams)
    assert [seam.name for seam in seams] == ["publishing", "terrain", "catalog", "asset preview"]
    assert "publisher" in kwargs
    publishing = seams[0]
    assert publishing.detail == f"signing with {signing.name}"
    terrain = next(seam for seam in seams if seam.name == "terrain")
    assert terrain.detail == f"verifying against {trusted.name}"
    # The caches are created eagerly under the given root.
    assert (tmp_path / "cache" / "worlds").is_dir()
    assert (tmp_path / "cache" / "assets").is_dir()


def test_wire_hub_seams_without_signing_key_disables_only_publishing(tmp_path: Path) -> None:
    reg_path, trusted, _signing = _registry_with_anchor_keys(tmp_path)
    kwargs, seams = _wire_hub_seams(reg_path, trusted, None, tmp_path / "cache")

    assert "publisher" not in kwargs  # publish route would 503; read seams still wire
    publishing = next(seam for seam in seams if seam.name == "publishing")
    assert not publishing.wired
    assert "no signing key" in publishing.detail
    assert all(seam.wired for seam in seams if seam.name != "publishing")


def test_wire_hub_seams_without_trusted_key_wires_and_warns(tmp_path: Path) -> None:
    """Pinned: without a trusted key the read seams are still *wired* (True) — the detail carries
    the warning that verification will fail, rather than the seam going down."""
    reg_path, _trusted, _signing = _registry_with_anchor_keys(tmp_path)
    _kwargs, seams = _wire_hub_seams(reg_path, None, None, tmp_path / "cache")

    terrain = next(seam for seam in seams if seam.name == "terrain")
    assert terrain.wired
    assert "NO trusted key" in terrain.detail


# --- _mount_ui (reachable: needs fastapi, which the base install ships) ---------------------


def test_mount_ui_serves_a_built_dir_without_shadowing_routes(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>studio</title>hello-studio")
    app = FastAPI()

    @app.get("/ping")
    def _ping() -> dict:
        return {"ok": True}

    report = ServeReport(host="127.0.0.1", port=8000)
    _mount_ui(app, dist, report)

    assert report.ui_mounted
    assert report.ui_detail == f"mounted from {dist}"
    client = TestClient(app)
    assert "hello-studio" in client.get("/").text
    assert client.get("/ping").status_code == 200  # earlier API routes win over the mount


def test_mount_ui_absent_dir_serves_the_not_built_page(tmp_path: Path) -> None:
    app = FastAPI()
    report = ServeReport(host="127.0.0.1", port=8000)
    _mount_ui(app, tmp_path / "no-such-dist", report)

    assert not report.ui_mounted
    assert report.ui_detail == "UI not built — run `pnpm build:harness` in ui/"
    response = TestClient(app).get("/")
    assert response.status_code == 200  # never a 404 root
    assert "pnpm build:harness" in response.text


def test_mount_ui_disabled_still_answers_the_root(tmp_path: Path) -> None:
    """Pinned: ``--no-ui`` (ui_dir=None) reports 'UI mount disabled' but the root still serves
    the same 'not built' page rather than 404-ing — both non-mount branches share it."""
    app = FastAPI()
    report = ServeReport(host="127.0.0.1", port=8000)
    _mount_ui(app, None, report)

    assert not report.ui_mounted
    assert report.ui_detail == "UI mount disabled"
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "has not been built" in response.text


# --- _attach_seed ---------------------------------------------------------------------------


def test_attach_seed_disabled_reports_no_seed() -> None:
    report = ServeReport(host="h", port=1)
    _attach_seed(report, False, object())
    assert report.seed_detail == "seeding disabled (--no-seed)"


def test_attach_seed_without_publisher_degrades_honestly() -> None:
    report = ServeReport(host="h", port=1)
    _attach_seed(report, True, None)
    assert "example not pinned (no publisher)" in report.seed_detail


def test_attach_seed_pins_the_example_campaign_idempotently(tmp_path: Path) -> None:
    """With a real publisher the example campaign is pinned; a second attach resolves the same
    pin (idempotent by content). Mirrors the original serve-level test, minus the REST surface."""
    reg_path, trusted, signing = _registry_with_anchor_keys(tmp_path)
    _publish_asset(
        Registry(reg_path),
        signing.read_bytes(),
        asset_id="astro-mine.fleet.prospecting-rover",
        kind="rover",
        name="Prospecting Rover",
        tags=[CapabilityTag("mobility.wheeled")],
    )
    kwargs, _seams = _wire_hub_seams(reg_path, trusted, signing, tmp_path / "cache")

    report = ServeReport(host="h", port=1)
    _attach_seed(report, True, kwargs["publisher"])
    assert report.seed_detail.startswith("example campaign pinned"), report.seed_detail
    assert "example-lunar-ice:0.1.0" in report.seed_detail

    report2 = ServeReport(host="h", port=1)
    _attach_seed(report2, True, kwargs["publisher"])
    assert report2.seed_detail == report.seed_detail


def test_attach_seed_swallows_publisher_failures_into_the_detail() -> None:
    """A broken publisher never crashes serve — the failure lands in the seed detail."""
    report = ServeReport(host="h", port=1)
    _attach_seed(report, True, object())  # not a publisher; publish_campaign will blow up
    assert report.seed_detail.startswith("could not pin the example campaign:")


# --- build_serve_app / serve degradation in this distribution -------------------------------


def test_build_serve_app_raises_the_platform_import_error(tmp_path: Path) -> None:
    """In astro-mine-platform the guard always fires: ``astro_mine.studio.api`` is absent, so
    ``build_serve_app`` raises an ImportError naming this distribution, chained from the missing
    module. Everything below the guard (composition, 503 matrix) is dead code here."""
    with pytest.raises(ImportError, match=r"not included\s+in astro-mine-platform") as exc:
        build_serve_app(
            registry=None,
            trusted_key=None,
            signing_key=None,
            cache_dir=tmp_path / "cache",
            ui_dir=None,
            seed=False,
            host="127.0.0.1",
            port=8000,
        )
    assert "astro-mine-studio distribution" in str(exc.value)
    assert isinstance(exc.value.__cause__, ModuleNotFoundError)


def test_cmd_serve_degrades_with_the_platform_message(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The full ``serve`` path, end to end through ``main``: uvicorn *is* importable here, but the
    ImportError from ``build_serve_app`` is caught first, printed to stderr with an install hint,
    and the process exits 1 — no banner, no server, no traceback."""
    code = main(
        [
            "serve",
            "--registry",
            str(tmp_path / "registry"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-ui",
            "--no-seed",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # everything serve says goes to stderr
    assert "astro-mine-studio serve needs the [serve] extra:" in captured.err
    assert "not included" in captured.err
    assert "in astro-mine-platform" in captured.err
    assert "pip install astro-mine-studio[serve]" in captured.err
    # The banner is never printed — the failure happens before composition completes.
    assert "Astro-Mine Studio\n" not in captured.err
    # Resolution side effects still happen before the failure: the cache dir was created.
    assert (tmp_path / "cache").is_dir()
