from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GateResult:
    name: str
    passed: bool
    detail: str


def require(condition: bool, name: str, detail: str) -> GateResult:
    return GateResult(name=name, passed=bool(condition), detail=detail)


def assert_gate_results(results: list[GateResult]) -> None:
    failed=[r for r in results if not r.passed]
    if failed:
        raise RuntimeError("QC gates failed: " + "; ".join(f"{r.name}: {r.detail}" for r in failed))
