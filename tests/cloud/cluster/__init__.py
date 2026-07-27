"""Opt-in live-cluster integration tests (the ``cluster`` marker).

These need a real Kubernetes cluster and are skipped everywhere else -- see ``conftest.py`` for
the environment they expect and ``platform/kind/up.sh`` for the harness that provides it.
"""
