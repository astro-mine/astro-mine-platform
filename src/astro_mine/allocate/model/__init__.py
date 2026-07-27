"""The canonical constraint model: the solver-neutral Allocation IR.

``model/`` holds the problem representation the solver backends lower (allocate.md §3).
:mod:`.ir` is the intermediate representation — the versioned decision-variable / constraint
/ objective-term schema (JSON Schema + Protobuf), its compiler from an ``AllocationRequest``,
and the independent feasibility verifier. The IR→CP-SAT/MILP encoders land under
``model/compile/`` with RM-P1-ALLOC-02.
"""

from __future__ import annotations
