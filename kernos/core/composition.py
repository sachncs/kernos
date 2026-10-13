"""Default dependency graph for the training loop.

The estimator keeps a convenient configuration-driven constructor, while the
orchestrator accepts this graph explicitly for fault injection, experiments,
and third-party strategy implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kernos.basis.whitening import Whitening
from kernos.core.factory import build_drift_metric, build_solver
from kernos.core.interfaces import (
    AnchorSampler,
    BudgetAccountant,
    DriftMetric,
    FeatureFusion,
    LocalFeatureMap,
    Orthogonalizer,
    OuterStep,
    RefreshPipeline,
    RefreshPolicy,
    ResidualSolver,
    WhiteningMap,
)
from kernos.core.plan import Plan
from kernos.correct.orth import Ridge
from kernos.correct.rbf import Rbf
from kernos.correct.sampler import Sampler
from kernos.fuse.fuse import Fuse
from kernos.fuse.scaler import Scaler
from kernos.loop.outerstep import Outerstep
from kernos.policy.budget import Budget
from kernos.policy.policy import Policy
from kernos.policy.refresh import Refresh


@dataclass
class LoopComponents:
    """Interchangeable services used by :class:`kernos.loop.loop.Loop`."""

    whitening: WhiteningMap
    scaler: Scaler
    sampler: AnchorSampler
    rbf: LocalFeatureMap
    solver: ResidualSolver
    fuse: FeatureFusion
    refresh_pipe: RefreshPipeline
    outerstep: OuterStep
    orth: Orthogonalizer
    drift_metric: DriftMetric
    budget: BudgetAccountant
    policy_factory: Callable[[Plan, float], RefreshPolicy]


def build_loop_components(plan: Plan) -> LoopComponents:
    """Build the default component graph from a validated plan."""
    whitening = Whitening(plan.stab_tau, plan.stab_alpha, plan.stab_eps)
    scaler = Scaler(plan.stab_eps)
    sampler = Sampler(plan.anchor_mix)
    rbf = Rbf(plan.local_bandwidth, plan.local_neighbors)
    solver = build_solver(plan)
    fuse = Fuse()
    return LoopComponents(
        whitening=whitening,
        scaler=scaler,
        sampler=sampler,
        rbf=rbf,
        solver=solver,
        fuse=fuse,
        refresh_pipe=Refresh(plan, whitening, scaler, sampler, rbf, fuse, solver),
        outerstep=Outerstep(plan),
        orth=Ridge(plan.stab_eta),
        drift_metric=build_drift_metric(plan.drift),
        budget=Budget(plan.budget),
        policy_factory=Policy,
    )
