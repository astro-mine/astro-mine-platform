"""Multi-fidelity profiles under one stable asset identity (RM-P0-FLEET-05)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_mine.core.sadf import enums, model
from astro_mine.fleet import cli, fidelity
from astro_mine.fleet.fidelity import FidelityError

T = enums.FidelityTier


def prof(
    tier: enums.FidelityTier,
    *,
    detail: str | None = None,
    determinism: enums.DeterminismClass | None = None,
    surrogate: model.SurrogateProfile | None = None,
) -> model.FidelityProfile:
    kwargs: dict[str, object] = {"tier": tier}
    if detail is not None:
        kwargs["detail"] = detail
    if determinism is not None:
        kwargs["determinism_class"] = determinism
    if surrogate is not None:
        kwargs["surrogate"] = surrogate
    return model.FidelityProfile(**kwargs)


def make_asset(
    profiles: list[model.FidelityProfile], *, asset_id: str = "test.rover"
) -> model.Asset:
    return model.Asset(
        identity=model.Identity(id=asset_id, name="Test", version="0.1.0", kind="rover"),
        core_interface_versions={"sadf": "0.1.0"},
        root_frame="base",
        fidelity_profiles=profiles,
    )


# --- ordering & selection --------------------------------------------------------


def test_profiles_are_ordered_coarse_to_fine() -> None:
    asset = make_asset([prof(T.ARTICULATED), prof(T.MASSMODEL), prof(T.KINEMATIC)])
    assert fidelity.tiers(asset) == [T.MASSMODEL, T.KINEMATIC, T.ARTICULATED]


def test_select_returns_the_tier_under_one_identity() -> None:
    # the acceptance: >=2 profiles under a single identity; switch tier without
    # re-instantiating (the same asset object / identity serves every tier).
    asset = make_asset(
        [prof(T.MASSMODEL, detail="mass/power only"), prof(T.ARTICULATED, detail="full linkage")]
    )
    assert fidelity.select(asset, T.MASSMODEL).detail == "mass/power only"
    assert fidelity.select(asset, T.ARTICULATED).detail == "full linkage"
    assert asset.identity.id == "test.rover"  # selection never changes identity


def test_has_tier() -> None:
    asset = make_asset([prof(T.MASSMODEL)])
    assert fidelity.has_tier(asset, T.MASSMODEL)
    assert not fidelity.has_tier(asset, T.ARTICULATED)


def test_coarsest_and_finest() -> None:
    asset = make_asset([prof(T.ARTICULATED), prof(T.MASSMODEL)])
    assert fidelity.coarsest(asset).tier is T.MASSMODEL
    assert fidelity.finest(asset).tier is T.ARTICULATED


def test_select_absent_tier_raises() -> None:
    asset = make_asset([prof(T.MASSMODEL)])
    with pytest.raises(FidelityError, match="no 'kinematic'"):
        fidelity.select(asset, T.KINEMATIC)


# --- validation ------------------------------------------------------------------


def test_single_fidelity_asset_is_valid() -> None:
    asset = make_asset([])
    fidelity.validate_profiles(asset)  # no raise
    assert fidelity.profiles(asset) == []


def test_coarsest_on_empty_raises() -> None:
    with pytest.raises(FidelityError, match="no fidelity profiles"):
        fidelity.coarsest(make_asset([]))


def test_finest_on_empty_raises() -> None:
    with pytest.raises(FidelityError, match="no fidelity profiles"):
        fidelity.finest(make_asset([]))


def test_duplicate_tier_rejected() -> None:
    asset = make_asset([prof(T.MASSMODEL), prof(T.MASSMODEL)])
    with pytest.raises(FidelityError, match="duplicate fidelity tier"):
        fidelity.profiles(asset)


def test_surrogate_tier_is_deferred() -> None:
    sp = model.SurrogateProfile(physics_domain=enums.SurrogatePhysicsDomain.GRANULAR_EXCAVATION)
    asset = make_asset([prof(T.SURROGATE, surrogate=sp)])
    with pytest.raises(FidelityError, match="deferred to Phase 1"):
        fidelity.validate_profiles(asset)


def test_surrogate_descriptor_on_structural_tier_is_deferred() -> None:
    sp = model.SurrogateProfile(physics_domain=enums.SurrogatePhysicsDomain.TERRAMECHANICS)
    asset = make_asset([prof(T.MASSMODEL, surrogate=sp)])
    with pytest.raises(FidelityError, match="deferred to Phase 1"):
        fidelity.validate_profiles(asset)


# --- CLI -------------------------------------------------------------------------


def run(*argv: str) -> int:
    try:
        cli.main(list(argv))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


def write_doc(path: Path, asset: model.Asset) -> Path:
    doc = model.SadfDocument(sadf_version="0.1", asset=asset)
    data = doc.model_dump(by_alias=True, mode="json", exclude_none=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_cli_fidelity_lists_profiles(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    asset = make_asset(
        [
            prof(T.MASSMODEL, detail="mass/power only"),
            prof(
                T.ARTICULATED, detail="full linkage", determinism=enums.DeterminismClass.BIT_EXACT
            ),
        ]
    )
    path = write_doc(tmp_path / "rover.sadf.json", asset)
    assert run("fidelity", str(path)) == 0
    out = capsys.readouterr().out
    assert "test.rover: 2 fidelity profile(s)" in out
    # coarse -> fine ordering and details rendered
    assert out.index("massmodel") < out.index("articulated")
    assert "mass/power only" in out and "[bit_exact]" in out


def test_cli_fidelity_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_doc(tmp_path / "a.json", make_asset([prof(T.MASSMODEL), prof(T.KINEMATIC)]))
    assert run("fidelity", str(path), "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["identity"] == "test.rover"
    assert [p["tier"] for p in payload["profiles"]] == ["massmodel", "kinematic"]


def test_cli_fidelity_single_fidelity_asset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_doc(tmp_path / "s.json", make_asset([]))
    assert run("fidelity", str(path)) == 0
    assert "single-fidelity asset" in capsys.readouterr().out


def test_cli_fidelity_rejects_duplicate_tiers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_doc(tmp_path / "d.json", make_asset([prof(T.MASSMODEL), prof(T.MASSMODEL)]))
    assert run("fidelity", str(path)) == 1
    assert "fleet fidelity" in capsys.readouterr().err


def test_cli_fidelity_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("fidelity", str(tmp_path / "nope.json")) == 1
    assert "cannot read file" in capsys.readouterr().err
