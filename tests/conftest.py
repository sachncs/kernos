"""Shared fixtures for the kernos test suite."""

from __future__ import annotations

import numpy as np
import pytest

from kernos.basis.nystrom import Nystrom
from kernos.basis.whitening import Whitening
from kernos.bench.dataset import linear
from kernos.cache import Adaptive, Full, Stream
from kernos.core.plan import Buffer, Plan
from kernos.core.rng import Rng
from kernos.core.state import Bundle, Continuous, Discrete
from kernos.correct.orth import Ridge, Tikhonov
from kernos.correct.rbf import Rbf
from kernos.correct.sampler import Sampler
from kernos.embed.identity import Identity
from kernos.embed.kernel import Kernel
from kernos.embed.linear import Linear
from kernos.embed.projector import Projector
from kernos.fuse.fuse import Fuse
from kernos.fuse.gate import Gate
from kernos.fuse.scaler import Scaler
from kernos.loop.callback import Log, Profile, Snapshot
from kernos.loop.loop import Loop
from kernos.loop.outerstep import Outerstep
from kernos.policy.budget import Budget
from kernos.policy.drift import Frobenius, Spectral
from kernos.policy.policy import Policy
from kernos.policy.refresh import Refresh
from kernos.predict.predict import Predict
from kernos.solver.direct import Direct
from kernos.solver.iterative import Iterative
from kernos.solver.jacobi import Jacobi
from kernos.solver.woodbury import Woodbury


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def seeds() -> list[np.random.Generator]:
    return Rng.spawn(np.random.default_rng(0), 3)


@pytest.fixture
def plan() -> Plan:
    return Plan()


@pytest.fixture
def small_plan() -> Plan:
    return Plan(dim=4, mbasis=16, abasis=4, lk=4, steps=2, batch=8, drift_hi=0.5, cool=0, warm=0)


@pytest.fixture
def synthetic(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return linear(rng, 100, 4, noise=0.1)


@pytest.fixture
def continuous(rng: np.random.Generator) -> Continuous:
    embed = Linear(4, 4, rng)
    return Continuous(theta=embed, R=np.eye(4))


@pytest.fixture
def discrete() -> Discrete:
    return Discrete()


@pytest.fixture
def bundle(continuous: Continuous, discrete: Discrete) -> Bundle:
    return Bundle(continuous=continuous, discrete=discrete)


@pytest.fixture
def whitening(plan: Plan) -> Whitening:
    return Whitening(plan.stab_tau, plan.stab_alpha, plan.stab_eps)


@pytest.fixture
def scaler(plan: Plan) -> Scaler:
    return Scaler(plan.stab_eps)


@pytest.fixture
def sampler(plan: Plan) -> Sampler:
    return Sampler(plan.amix)


@pytest.fixture
def rbf(plan: Plan) -> Rbf:
    return Rbf(plan.ltau, plan.lk)


@pytest.fixture
def solver(plan: Plan) -> Direct:
    return Direct(plan.ridge, plan.stab_jitter, plan.stab_jitter_retry, 10.0, plan.stab_jitter_max, plan.stab_kappa)


@pytest.fixture
def fuse() -> Fuse:
    return Fuse()


@pytest.fixture
def gate() -> Gate:
    return Gate()


@pytest.fixture
def orth(plan: Plan) -> Ridge:
    return Ridge(plan.stab_eta)


@pytest.fixture
def tikhonov(plan: Plan) -> Tikhonov:
    return Tikhonov(plan.stab_eta)


@pytest.fixture
def budget() -> Budget:
    return Budget(total=10.0)


@pytest.fixture
def drift_frobenius() -> Frobenius:
    return Frobenius()


@pytest.fixture
def drift_spectral() -> Spectral:
    return Spectral()


@pytest.fixture
def refresh(plan: Plan, whitening, scaler, sampler, rbf, fuse, solver) -> Refresh:
    return Refresh(plan, whitening, scaler, sampler, rbf, fuse, solver)


@pytest.fixture
def outerstep(plan: Plan) -> Outerstep:
    return Outerstep(plan)


@pytest.fixture
def loop(plan: Plan) -> Loop:
    return Loop(plan)


@pytest.fixture
def small_loop(small_plan: Plan) -> Loop:
    return Loop(small_plan)


@pytest.fixture
def nystrom_basis(rng: np.random.Generator, whitening: Whitening) -> Nystrom:
    U = rng.standard_normal((40, 4))
    return Nystrom.fromdata(U, 16, whitening, rng)


@pytest.fixture
def linear_embed(rng: np.random.Generator) -> Linear:
    return Linear(4, 4, rng)


@pytest.fixture
def identity_embed() -> Identity:
    return Identity()


@pytest.fixture
def kernel_embed(rng: np.random.Generator) -> Kernel:
    centers = rng.standard_normal((8, 4))
    return Kernel(centers, gamma=1.0)


@pytest.fixture
def projector() -> Projector:
    return Projector(np.eye(4))


@pytest.fixture
def direct_solver(plan: Plan) -> Direct:
    return Direct(plan.ridge, plan.stab_jitter, plan.stab_jitter_retry, 10.0, plan.stab_jitter_max, plan.stab_kappa)


@pytest.fixture
def iterative_solver(plan: Plan) -> Iterative:
    return Iterative(plan.ridge)


@pytest.fixture
def woodbury_solver(plan: Plan) -> Woodbury:
    return Woodbury(plan.ridge, plan.stab_jitter, plan.stab_jitter_retry, 10.0, plan.stab_jitter_max)


@pytest.fixture
def jacobi(plan: Plan) -> Jacobi:
    return Jacobi()


@pytest.fixture
def full_cache() -> Full:
    return Full()


@pytest.fixture
def stream_cache() -> Stream:
    return Stream(mfeat=8)


@pytest.fixture
def adaptive_cache() -> Adaptive:
    return Adaptive(mfeat=8, threshold=20)


@pytest.fixture
def predict() -> Predict:
    weights = np.ones(4)
    return Predict(weights=weights)


@pytest.fixture
def callback_log() -> Log:
    return Log(log_every=1)


@pytest.fixture
def callback_snapshot() -> Snapshot:
    return Snapshot()


@pytest.fixture
def callback_profile() -> Profile:
    return Profile()
