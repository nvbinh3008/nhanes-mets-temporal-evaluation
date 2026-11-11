"""Survey-aware modeling for NHANES R0–R5 representations."""

from __future__ import annotations

import os

# Avoid nested OpenMP/BLAS oversubscription during nested cross-validation.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "1.0.0"
