"""Supply-chain admission, enforced by a live Kyverno: signed admits, unsigned is *rejected*.

``tenancy/admission.admit()`` is a pure function that returns a decision object. Nothing ever
enforced it. The chart shipped a Kyverno ``ClusterPolicy`` -- and never installed Kyverno, while
the kind profile turned admission off entirely. So "Cloud never runs unsigned images in shared
tenancy" (``cloud.md`` §9) was an assertion about a dict.

Here the policy runs on a real admission controller. Two images, byte-identical but for a label;
one is cosign-signed with SLSA-provenance + SBOM attestations, the other is not. The signed one's
pod is created. The unsigned one's pod is **refused by the API server** -- the request fails, and
the failure names the policy.

Skipped unless ``cosign`` and ``docker`` are on PATH and the cluster was brought up with
admission on (``ADMISSION=1 ./platform/kind/up.sh``, which installs Kyverno and the policy from
``platform/profiles/kind-admission.yaml``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from astro_mine.cloud.tenancy.admission import ImageAttestation, admit
from tests.cloud.cluster.conftest import CLUSTER_MARKS, Kubectl, requires, run

pytestmark = [
    *CLUSTER_MARKS,
    requires("cosign", "docker"),
    pytest.mark.skipif(
        not os.environ.get("ASTRO_MINE_ADMISSION"),
        reason="bring the cluster up with ADMISSION=1 ./platform/kind/up.sh to install Kyverno",
    ),
]

#: The namespace the kind-admission profile scopes the policy to (see profiles/kind-admission.yaml).
#: Scoping matters: an `imageReferences: ["*"]` policy over every namespace would refuse the
#: platform's own operator pods and take the cluster down with it.
NAMESPACE = "astro-mine-admission"
POLICY = "require-signed-images"
#: The registry as *the cluster* resolves it. Not `localhost:5001`: Kyverno fetches signatures
#: from the registry itself, from inside its own pod, where `localhost` is the Kyverno pod.
REGISTRY = os.environ.get("ASTRO_MINE_CLUSTER_REGISTRY", "kind-registry:5000")
#: The self-signed CA the harness registry serves TLS with (up.sh exports it). crane must trust it
#: to push over HTTPS, because the registry is HTTPS-only now and crane's `--insecure` forces plain
#: HTTP, which the registry refuses. cosign uses --allow-insecure-registry (skip-verify) instead and
#: needs no CA. See astro-mine-cloud#30.
CA_CERT = os.environ.get("ASTRO_MINE_REGISTRY_CA")


def _keys(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """**The** cosign key pair -- the one whose public half is in the live Kyverno policy.

    Not a fresh one. ``up.sh`` generates the pair and installs the chart with the public half
    (``--set-file admission.cosignPublicKey``); a key minted here would be a key the cluster has
    never seen, so Kyverno would refuse the "signed" image -- correctly, because it was signed by a
    stranger -- and ``test_a_signed_image_is_admitted`` would quietly become a second copy of
    ``test_an_unsigned_image_is_rejected``. It did exactly that.

    Copied into a tmp dir rather than used in place: the container helpers chmod it and write a
    predicate alongside it, and the harness's own key directory is not ours to rearrange.
    """
    published = os.environ.get("ASTRO_MINE_COSIGN_KEYDIR")
    assert published, (
        "ASTRO_MINE_COSIGN_KEYDIR is unset. up.sh must export the key directory whose public half "
        "it handed to the Kyverno policy; signing with any other key tests only that Kyverno "
        "rejects strangers, which is what the *other* test is for."
    )

    directory = tmp_path_factory.mktemp("cosign")
    for name in ("cosign.key", "cosign.pub"):
        shutil.copy(Path(published) / name, directory / name)
    return directory


@pytest.fixture(scope="module")
def keys(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _keys(tmp_path_factory)


def _readable_by_container(path: Path) -> None:
    """Open *path* (a throwaway tmp dir and its contents) to the nonroot user inside a container.

    ``crane`` and ``cosign`` both run as uid 65532, while ``tmp_path_factory`` hands out 0700
    directories owned by the test user -- so a bind-mounted file is not merely unreadable, it
    cannot even be stat'd. These are throwaway keys and tars in a per-run tmp dir; opening them
    is cheaper and safer than running the containers as root.
    """
    path.chmod(0o755)
    for child in path.iterdir():
        if child.is_file():
            child.chmod(0o644)


def _crane(*args: str, mount: Path) -> subprocess.CompletedProcess[str]:
    """Run crane on the kind docker network, so it resolves the registry the way the cluster does.

    The host cannot push to ``kind-registry:5000`` directly (docker would demand TLS for a
    non-loopback name), and the *cluster* cannot pull from ``localhost:5001`` (that is the pod's
    own loopback). A container on the kind network sees the same registry the nodes and Kyverno
    do -- so there is exactly one image reference in play, and it is one everybody can resolve.

    The registry serves TLS, so crane connects over HTTPS and verifies against the harness CA
    (``SSL_CERT_FILE``, which Go's cert pool honours). ``--insecure`` is *not* used: it would force
    crane to plain HTTP, which the TLS registry refuses (astro-mine-cloud#30). The CA path comes
    from ``ASTRO_MINE_REGISTRY_CA`` (up.sh exports it).

    *mount* is bind-mounted at ``/work`` and is the container's working directory, so paths in
    *args* are relative to it. It must be the directory the artefact actually lives in: the image
    tars are written under ``tmp_path_factory``, not under the repo, so mounting the repo (as this
    once did) left crane unable to see them at all.
    """
    _readable_by_container(mount)
    argv = ["docker", "run", "--rm", "--network", "kind", "-v", f"{mount}:/work", "-w", "/work"]
    if CA_CERT:
        argv += ["-v", f"{CA_CERT}:/certs/ca.crt:ro", "-e", "SSL_CERT_FILE=/certs/ca.crt"]
    argv += ["gcr.io/go-containerregistry/crane:v0.21.7", *args]
    return run(argv)


def _push_tiny_image(tag: str, label: str, tmp_path: Path) -> str:
    """Build a minimal image, push it to the in-cluster registry, return its digest-pinned ref."""
    context = tmp_path / label
    context.mkdir(parents=True, exist_ok=True)
    # One file, so the image has one layer. `FROM scratch` plus a bare LABEL builds an image with
    # *no* layers at all, and `docker save` refuses it outright: "empty export - not implemented".
    # Admission never sees the image content -- Kyverno checks its signature -- but the image still
    # has to survive save/push to reach the registry at all.
    (context / "variant").write_text(f"{label}\n")
    (context / "Dockerfile").write_text(
        f"FROM scratch\nCOPY variant /variant\nLABEL org.astro-mine.variant={label}\n"
    )
    built = run(["docker", "build", "-t", tag, str(context)])
    assert built.returncode == 0, built.stderr

    saved = tmp_path / f"{label}.tar"
    exported = run(["docker", "save", tag, "-o", str(saved)])
    assert exported.returncode == 0, exported.stderr

    # Mount the directory the tar is actually in. It lives under tmp_path_factory, not under the
    # repo, so naming it relative to the cwd (as this once did) raised ValueError in the fixture
    # and errored every test that needed an image.
    pushed = _crane("push", saved.name, tag, mount=tmp_path)
    assert pushed.returncode == 0, pushed.stderr

    digest = _crane("digest", tag, mount=tmp_path)
    assert digest.returncode == 0, digest.stderr

    # rsplit, not split: the registry carries a *port*, so the first colon in
    # `kind-registry:5000/admission-signed-x:1` belongs to the host, not the tag. Splitting on it
    # (as this once did) yielded the bare name `kind-registry` -- and cosign dutifully went looking
    # for `index.docker.io/library/kind-registry`, where it got UNAUTHORIZED. Only the last colon
    # separates the tag.
    repository = tag.rsplit(":", 1)[0]
    return f"{repository}@{digest.stdout.strip()}"


def _cosign(keys: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Sign/attest from a container on the kind network, for the same reason crane runs there.

    The registry serves TLS with a self-signed cert, so ``--allow-insecure-registry`` (skip
    certificate *verification*, still over HTTPS) is enough and needs no CA plumbing. ``--allow-
    http-registry`` is *not* passed: it forces plaintext HTTP, which the TLS registry refuses
    (astro-mine-cloud#30). The flag goes straight after the subcommand, where cosign expects it.
    (Never do this against a real registry; the whole point of a signature is that you did not
    trust the transport in the first place.)
    """
    # Same nonroot bind-mount trap as crane: cosign runs as uid 65532 and the key dir is 0700.
    _readable_by_container(keys)
    subcommand, rest = args[0], args[1:]
    return run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "kind",
            "-e",
            "COSIGN_PASSWORD=",
            "-v",
            f"{keys}:/keys",
            "gcr.io/projectsigstore/cosign:v2.4.1",
            subcommand,
            "--allow-insecure-registry",
            *rest,
        ]
    )


def _pod(name: str, image: str) -> str:
    return json.dumps(
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": name, "namespace": NAMESPACE},
            "spec": {
                "restartPolicy": "Never",
                "containers": [{"name": "c", "image": image, "command": ["/nonexistent"]}],
            },
        }
    )


@pytest.fixture(scope="module")
def images(keys: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """One signed + attested image, one identical-but-unsigned image, both in the registry."""
    tmp = tmp_path_factory.mktemp("images")
    run_id = uuid.uuid4().hex[:8]
    signed = _push_tiny_image(f"{REGISTRY}/admission-signed-{run_id}:1", "signed", tmp)
    unsigned = _push_tiny_image(f"{REGISTRY}/admission-unsigned-{run_id}:1", "unsigned", tmp)

    # Only `signed` gets a signature -- and the SLSA-provenance + SBOM attestations the policy
    # demands, which is exactly what admit() requires of an ImageAttestation.
    for predicate, kind in (
        ({"buildType": "astro-mine/test"}, "https://slsa.dev/provenance/v1"),
        ({"bomFormat": "CycloneDX", "specVersion": "1.5"}, "https://cyclonedx.org/bom"),
    ):
        # The predicate must live where the cosign container can see it: the mounted key dir.
        (keys / "predicate.json").write_text(json.dumps(predicate))
        attested = _cosign(
            keys,
            "attest",
            "--key",
            "/keys/cosign.key",
            "--type",
            kind,
            "--predicate",
            "/keys/predicate.json",
            "--tlog-upload=false",
            "--allow-insecure-registry",
            "--yes",
            signed,
        )
        assert attested.returncode == 0, attested.stderr

    result = _cosign(
        keys,
        "sign",
        "--key",
        "/keys/cosign.key",
        "--tlog-upload=false",
        "--allow-insecure-registry",
        "--yes",
        signed,
    )
    assert result.returncode == 0, result.stderr
    return {"signed": signed, "unsigned": unsigned}


def test_the_policy_is_actually_enforcing(kubectl: Kubectl) -> None:
    """A policy in Audit mode would let everything through and still look installed."""
    policy = kubectl.json("get", "clusterpolicy", POLICY)
    assert policy["spec"]["validationFailureAction"] == "Enforce"
    kubectl("wait", "--for=condition=Ready", f"clusterpolicy/{POLICY}", "--timeout=120s")


def test_a_signed_image_is_admitted(kubectl: Kubectl, images: dict[str, str]) -> None:
    name = "signed-pod"
    kubectl.apply(_pod(name, images["signed"]))
    try:
        assert kubectl.json("get", "pod", name, "-n", NAMESPACE)["metadata"]["name"] == name
    finally:
        kubectl("delete", "pod", name, "-n", NAMESPACE, "--force", check=False)


def test_an_unsigned_image_is_rejected_by_the_api_server(
    kubectl: Kubectl, images: dict[str, str]
) -> None:
    """The pod is never created -- refusal happens at the cluster boundary, not after the fact.

    This asserts *why* it was refused, not merely that it was. Without that, the test passes on a
    Kyverno that cannot reach the registry at all -- the pod is denied, the policy is named, and the
    assertion is satisfied by an outcome that says nothing whatever about the missing signature.
    It was: for several runs this was green while Kyverno was refusing every image, signed ones
    included, because it could not fetch a signature over HTTP. A supply-chain test that cannot tell
    "no valid signature" from "could not look" is not a supply-chain test.
    """
    refused = kubectl(
        "apply", "-f", "-", stdin=_pod("unsigned-pod", images["unsigned"]), check=False
    )

    assert refused.returncode != 0, "an unsigned image was admitted"
    message = refused.stderr + refused.stdout
    assert POLICY in message, message

    # The refusal must be about the *signature*, not about the transport.
    assert "http: server gave HTTP response to HTTPS client" not in message, (
        "Kyverno refused the image because it could not reach the registry, not because the image "
        f"is unsigned -- this test proves nothing about signatures:\n{message}"
    )
    assert ".attestors" in message or "signature" in message.lower(), message

    listed = kubectl("get", "pod", "unsigned-pod", "-n", NAMESPACE, check=False)
    assert listed.returncode != 0, "the unsigned pod exists despite the policy"


def test_the_live_decision_agrees_with_the_in_process_one(images: dict[str, str]) -> None:
    """The two must not drift: `admit()` is the mirror of the policy, and mirrors can crack."""
    signed = ImageAttestation(
        image=images["signed"],
        cosign_verified=True,
        slsa_provenance=True,
        sbom=True,
        core_interface_version="0.1.0",
    )
    unsigned = signed.model_copy(
        update={
            "image": images["unsigned"],
            "cosign_verified": False,
            "slsa_provenance": False,
            "sbom": False,
        }
    )
    assert admit(signed).admitted is True
    assert admit(unsigned).admitted is False
