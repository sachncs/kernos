"""Dataset source sub-modules.

Each module groups loaders from a single origin:

- :mod:`examples.evaluation.datasets.sources.sklearn_sources` —
  scikit-learn built-ins (``CaliforniaHousing``, ``Diabetes``).
- :mod:`examples.evaluation.datasets.sources.openml_sources` —
  OpenML / UCI tabular (``Protein``, ``Kin8nm``, ``YearPredictionMSD``,
  ``NYCTaxi``).
- :mod:`examples.evaluation.datasets.sources.huggingface_sources` —
  HuggingFace ``inria-soda/tabular-benchmark`` regression configs.
- :mod:`examples.evaluation.datasets.sources.synthetic_sources` —
  synthetic spatial field generator.
"""

from __future__ import annotations
