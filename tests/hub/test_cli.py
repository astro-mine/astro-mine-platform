"""CLI tests (RM-P1-HUB-06): publish/search/resolve/verify/pull end-to-end via main().

``--registry`` takes **either** transport (hub.md §7): a local OCI-layout directory *or* a remote
OCI registry over the Distribution Spec. Both are exercised here end to end — the remote against a
real HTTP registry (``tests/_fake_registry``) — because "the client/CLI resolves/verifies/pulls
against *any* OCI registry" is the deliverable, not just the library underneath it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.hub.client import main
from astro_mine.hub.registry import Registry
from astro_mine.hub.registry._oci import blob_path
from astro_mine.hub.supply_chain import generate_keypair

from ._fake_registry import FakeRegistry
from .conftest import make_manifest


def _manifest_file(tmp_path: Path, name: str = "pol", version: str = "1.0.0") -> str:
    path = tmp_path / f"{name}.json"
    path.write_text(make_manifest(name, version).model_dump_json())
    return str(path)


def _key_files(tmp_path: Path) -> tuple[str, str]:
    private_pem, public_pem = generate_keypair()
    (tmp_path / "key.pem").write_bytes(private_pem)
    (tmp_path / "pub.pem").write_bytes(public_pem)
    return str(tmp_path / "key.pem"), str(tmp_path / "pub.pem")


def _publish(tmp_path: Path, registry: str) -> None:
    manifest = _manifest_file(tmp_path)
    key, _ = _key_files(tmp_path)
    assert (
        main(
            [
                "publish",
                "--registry",
                registry,
                "--name",
                "pol",
                "--version",
                "1.0.0",
                "--kind",
                "policy",
                "--manifest",
                manifest,
                "--key",
                key,
            ]
        )
        == 0
    )


def test_cli_keygen_produces_a_usable_signing_keypair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`hub keygen` is the one signing-key command, so its output must be a real cosign keypair:
    the generated key signs a publish, and its public half verifies it."""
    keys = tmp_path / "keys"
    assert main(["keygen", "--out", str(keys)]) == 0
    key, pub = keys / "cosign.key", keys / "cosign.pub"
    assert key.exists() and pub.exists()
    capsys.readouterr()  # drop the "wrote …" line

    registry = str(tmp_path / "reg")
    manifest = _manifest_file(tmp_path)
    publish_ok = main(
        [
            "publish",
            "--registry",
            registry,
            "--name",
            "pol",
            "--version",
            "1.0.0",
            "--kind",
            "policy",
            "--manifest",
            manifest,
            "--key",
            str(key),
        ]
    )
    assert publish_ok == 0
    capsys.readouterr()
    assert main(["verify", "--registry", registry, "pol:1.0.0", "--trusted-key", str(pub)]) == 0


def test_cli_publish_search_resolve_verify_pullfile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = str(tmp_path / "reg")
    _publish(tmp_path, registry)
    digest = capsys.readouterr().out.strip()
    assert digest.startswith("sha256:")

    assert main(["search", "--registry", registry, "--text", "pol"]) == 0
    assert "pol:1.0.0" in capsys.readouterr().out

    assert main(["resolve", "--registry", registry, "--name", "pol"]) == 0
    assert digest in capsys.readouterr().out

    pub = str(tmp_path / "pub.pem")
    assert main(["verify", "--registry", registry, "pol:1.0.0", "--trusted-key", pub]) == 0
    assert capsys.readouterr().out.startswith("ok ")

    out = str(tmp_path / "pulled.json")
    assert (
        main(["pull", "--registry", registry, "pol:1.0.0", "--out", out, "--trusted-key", pub]) == 0
    )
    assert "pol" in Path(out).read_text()
    assert f"wrote {out}" in capsys.readouterr().out


def _publish_argv(registry: str, manifest: str, key: str) -> list[str]:
    return [
        "publish",
        "--registry",
        registry,
        "--name",
        "pol",
        "--version",
        "1.0.0",
        "--kind",
        "policy",
        "--manifest",
        manifest,
        "--key",
        key,
    ]


_MANIFEST_DOCUMENT = """\
# The shape `astro-mine-core validate` accepts, and the shape both shipped Core examples use.
manifest_version: "0.1"
manifest:
  name: pol
  version: 1.0.0
  kind: policy
  license: Apache-2.0
  description: A manifest document, not a bare PluginManifest.
  core_interfaces:
    policy: 0.1.0
  capability_tags:
    - mobility.wheeled
"""


def test_publish_accepts_the_manifest_document_form_core_validate_accepts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two readers of a manifest must not disagree (#46).

    `astro-mine-core validate` blesses a manifest *document* — `manifest_version` + a `manifest:`
    mapping, in YAML — which is what both shipped examples in `astro-mine-core/examples/plugins/`
    are. `publish --manifest` required a bare `PluginManifest` in strict JSON, so a reader who
    copied the example the guide points at, validated it, and got `OK` was rewarded with a pydantic
    traceback.
    """
    from astro_mine.core.registry import validate_manifest

    document = tmp_path / "my-plugin.manifest.yaml"
    document.write_text(_MANIFEST_DOCUMENT)
    validate_manifest(document.read_text())  # what `astro-mine-core validate` does

    key, _ = _key_files(tmp_path)
    registry = str(tmp_path / "reg")
    assert main(_publish_argv(registry, str(document), key)) == 0
    assert capsys.readouterr().out.strip().startswith("sha256:")
    # The published manifest is the *inner* model, not the wrapper.
    assert main(["search", "--registry", registry, "--text", "pol"]) == 0
    assert "pol:1.0.0" in capsys.readouterr().out


def test_publish_still_accepts_a_bare_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The historical form keeps working — this widens what is accepted, it does not move it."""
    key, _ = _key_files(tmp_path)
    assert main(_publish_argv(str(tmp_path / "reg"), _manifest_file(tmp_path), key)) == 0
    assert capsys.readouterr().out.strip().startswith("sha256:")


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("not: [a, valid, manifest", "not readable as YAML or JSON"),
        ("- just\n- a\n- list\n", "must be a YAML/JSON mapping"),
        ('{"name": "pol"}', "not a valid plugin manifest"),
        ("manifest_version: '0.1'\nmanifest: {}\n", "not a valid plugin manifest"),
    ],
    ids=["unparseable", "not-a-mapping", "bare-but-incomplete", "document-but-incomplete"],
)
def test_a_bad_manifest_is_one_error_line_not_a_traceback(
    contents: str,
    expected: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "bad.yaml"
    manifest.write_text(contents)
    key, _ = _key_files(tmp_path)
    assert main(_publish_argv(str(tmp_path / "reg"), str(manifest), key)) == 1
    err = capsys.readouterr().err
    assert err.startswith("astro-mine-hub publish: ")
    assert expected in err
    assert "Traceback" not in err


def test_a_missing_key_or_layer_is_an_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`resolve`/`pull`/`verify` have always degraded cleanly; `publish` now does too."""
    registry = str(tmp_path / "reg")
    manifest = _manifest_file(tmp_path)
    key, _ = _key_files(tmp_path)

    assert main(_publish_argv(registry, manifest, str(tmp_path / "absent.pem"))) == 1
    err = capsys.readouterr().err
    assert err.startswith("astro-mine-hub publish: cannot read signing key ")
    assert "Traceback" not in err

    argv = [*_publish_argv(registry, manifest, key), "--layer", str(tmp_path / "absent.onnx")]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("astro-mine-hub publish: cannot read payload layer ")
    assert "Traceback" not in err


def test_a_missing_trusted_key_is_an_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same unguarded read existed on `pull`/`verify`; `main`'s backstop covers them too."""
    registry = str(tmp_path / "reg")
    _publish(tmp_path, registry)
    capsys.readouterr()
    absent = str(tmp_path / "absent.pub")
    for argv in (
        ["verify", "--registry", registry, "pol:1.0.0", "--trusted-key", absent],
        ["pull", "--registry", registry, "pol:1.0.0", "--trusted-key", absent],
    ):
        assert main(argv) == 1
        err = capsys.readouterr().err
        assert "cannot read trusted key" in err
        assert "Traceback" not in err


def test_cli_pull_to_stdout(tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]) -> None:
    registry = str(tmp_path / "reg")
    _publish(tmp_path, registry)
    capfdbinary.readouterr()  # drop the publish digest
    assert main(["pull", "--registry", registry, "pol:1.0.0", "--no-verify"]) == 0
    assert b'"pol"' in capfdbinary.readouterr().out


def test_cli_resolve_error_exits_nonzero(tmp_path: Path) -> None:
    registry = str(tmp_path / "reg")
    Registry(registry)  # empty registry
    assert main(["resolve", "--registry", registry, "--name", "nope"]) == 1


def test_cli_pull_tampered_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = str(tmp_path / "reg")
    _publish(tmp_path, registry)
    capsys.readouterr()
    reg = Registry(registry)
    digest = reg.resolve("pol:1.0.0").digest
    config_digest = reg.read_manifest(digest)["config"]["digest"]
    blob_path(reg.path, config_digest).write_bytes(b'{"x":1}')
    assert main(["pull", "--registry", registry, "pol:1.0.0"]) == 1
    assert main(["verify", "--registry", registry, "pol:1.0.0"]) == 1  # verify fails closed too


def test_cli_pull_missing_artifact_exits_nonzero(tmp_path: Path) -> None:
    registry = str(tmp_path / "reg")
    Registry(registry)
    assert main(["pull", "--registry", registry, "absent:1.0.0"]) == 1
    assert main(["verify", "--registry", registry, "absent:1.0.0"]) == 1


# -- payload layers ------------------------------------------------------------------------------


def _publish_with_payload(tmp_path: Path, registry: str, *layers: bytes) -> None:
    paths: list[str] = []
    for index, data in enumerate(layers):
        path = tmp_path / f"layer{index}.bin"
        path.write_bytes(data)
        paths.append(str(path))
    key, _ = _key_files(tmp_path)
    argv = [
        "publish",
        "--registry",
        registry,
        "--name",
        "pol",
        "--version",
        "1.0.0",
        "--kind",
        "policy",
        "--manifest",
        _manifest_file(tmp_path),
        "--key",
        key,
    ]
    for path in paths:
        argv += ["--layer", path]
    assert main(argv) == 0


def test_cli_pull_payload_to_stdout(
    tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    registry = str(tmp_path / "reg")
    _publish_with_payload(tmp_path, registry, b"onnx-bytes")
    capfdbinary.readouterr()

    assert main(["pull", "--registry", registry, "pol:1.0.0", "--payload"]) == 0
    assert capfdbinary.readouterr().out == b"onnx-bytes"  # verified layer bytes, not the manifest


def test_cli_pull_payload_materializes_multiple_layers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = str(tmp_path / "reg")
    _publish_with_payload(tmp_path, registry, b"onnx-bytes", b"sadf-bytes")
    capsys.readouterr()

    out = tmp_path / "payload"
    assert main(["pull", "--registry", registry, "pol:1.0.0", "--payload", "--out", str(out)]) == 0
    written = sorted(path.read_bytes() for path in out.iterdir())
    assert written == [b"onnx-bytes", b"sadf-bytes"]
    assert len(capsys.readouterr().out.strip().splitlines()) == 2  # one content-addressed path each


def test_cli_pull_payload_refuses_ambiguous_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two layers cannot be written to one stdout stream — say so, don't concatenate silently."""
    registry = str(tmp_path / "reg")
    _publish_with_payload(tmp_path, registry, b"a", b"b")
    capsys.readouterr()

    assert main(["pull", "--registry", registry, "pol:1.0.0", "--payload"]) == 1
    assert "use --out DIR" in capsys.readouterr().err


def test_cli_pull_payload_fails_closed_on_tamper(tmp_path: Path) -> None:
    registry = str(tmp_path / "reg")
    _publish_with_payload(tmp_path, registry, b"onnx-bytes")

    reg = Registry(registry)
    manifest = reg.read_manifest(reg.resolve("pol:1.0.0").digest)
    blob_path(reg.path, manifest["layers"][0]["digest"]).write_bytes(b"malicious")

    assert main(["pull", "--registry", registry, "pol:1.0.0", "--payload"]) == 1


# -- the remote transport, driven through the CLI --------------------------------------------------


def test_cli_end_to_end_against_a_remote_oci_registry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--registry ghcr.io/...` — publish, resolve, verify, and pull over the Distribution Spec."""
    with FakeRegistry() as fake:
        registry = f"{fake.location}/astro-mine"
        _publish_with_payload(tmp_path, registry, b"onnx-bytes")
        digest = capsys.readouterr().out.strip()
        assert digest.startswith("sha256:")

        assert main(["resolve", "--registry", registry, "--name", "pol"]) == 0
        assert digest in capsys.readouterr().out

        pub = str(tmp_path / "pub.pem")
        assert main(["verify", "--registry", registry, "pol:1.0.0", "--trusted-key", pub]) == 0
        assert capsys.readouterr().out.startswith("ok ")

        out = str(tmp_path / "pulled.json")
        assert main(["pull", "--registry", registry, "pol:1.0.0", "--out", out]) == 0
        assert "pol" in Path(out).read_text()
        capsys.readouterr()

        payload = tmp_path / "payload"
        argv = ["pull", "--registry", registry, "pol:1.0.0", "--payload", "--out", str(payload)]
        assert main(argv) == 0
        assert [path.read_bytes() for path in payload.iterdir()] == [b"onnx-bytes"]


def test_cli_against_an_unreachable_remote_registry_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["pull", "--registry", "http://127.0.0.1:1/x", "pol:1.0.0"]) == 1
    assert "error:" in capsys.readouterr().err
