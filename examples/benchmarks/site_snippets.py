# ruff: noqa: E402, F821, I001

"""Python snippets rendered on the public Kernos site.

The site reads the marked blocks directly from this file. Keep each block
syntactically complete so the repository test suite can compile it.
"""

# <snippet name="fit">
import numpy as np
from kernos import Kernos

rng = np.random.default_rng(42)
X_train = rng.standard_normal((80, 4))
y_train = X_train[:, 0] + 0.5 * X_train[:, 1] ** 2 + 0.1 * rng.standard_normal(80)
X_test = rng.standard_normal((20, 4))
y_test = X_test[:, 0] + 0.5 * X_test[:, 1] ** 2 + 0.1 * rng.standard_normal(20)

model = Kernos(
    dim=4,
    mbasis=16,
    abasis=4,
    lk=4,
    ridge=1e-2,
    steps=1,
    seed=42,
).fit(X_train, y_train)

print(f"R² on held-out: {model.score(X_test, y_test):.4f}")
# The score depends on the dataset, seed and training budget.
# </snippet>

# <snippet name="grid">
import numpy as np
from sklearn.model_selection import GridSearchCV
from kernos import Kernos

rng = np.random.default_rng(42)
X_train = rng.standard_normal((40, 4))
y_train = X_train[:, 0] + 0.5 * X_train[:, 1] ** 2 + 0.1 * rng.standard_normal(40)

param_grid = {
    "mbasis": [8],
    "abasis": [2],
    "ridge": [1e-2],
}

search = GridSearchCV(
    Kernos(dim=4, lk=2, seed=42, steps=1),
    param_grid,
    cv=2,
    scoring="r2",
    # Keep the public example deterministic and laptop-friendly; callers can
    # opt into parallelism for their own workload after profiling it.
    n_jobs=1,
)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
# </snippet>

# <snippet name="stream">
from kernos import Buffer, Kernos
import numpy as np

# O(m²) memory mode — same algorithm, no full ΦᵀΦ cache.
rng = np.random.default_rng(0)
X_stream = rng.standard_normal((40, 4))
y_stream = X_stream[:, 0] - 0.25 * X_stream[:, 1]
stream = zip(np.array_split(X_stream, 2), np.array_split(y_stream, 2), strict=True)
X_test = rng.standard_normal((8, 4))

model = Kernos(
    mode=Buffer.STREAM,
    dim=4,
    mbasis=8,
    abasis=2,
    lk=2,
    steps=1,
    seed=0,
)

# partial_fit drives the continuous descent; the refresh
# controller still triggers when the basis drifts.
for x_batch, y_batch in stream:
    model.partial_fit(x_batch, y_batch)

print(model.predict(X_test))
# </snippet>
