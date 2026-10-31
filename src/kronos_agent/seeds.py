"""Seeds derived from the decision hour (SPEC section 3).

Canonical string ``kronos-agent/seed/v1/<stream>/<symbol>/<YYYY-MM-DDTHH:00:00Z>`` (plus
``/<salt>`` when a salt is configured). ``torch_seed`` is the first 8 bytes of its sha256 as a
big-endian integer masked to 63 bits; ``numpy_seed`` is bytes 8..12 (32 bits).

Test vectors (salt empty, stream ``forecast``, symbol ``BTC-USDT``):
``2026-10-29T16:00:00Z`` -> torch 6649841357966433750, numpy 1035604865;
``2026-10-29T17:00:00Z`` -> torch 9104675984868344003, numpy 3824972400.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta

SEED_SCHEME = "kronos-agent/seed/v1"
TORCH_SEED_MASK = 0x7FFF_FFFF_FFFF_FFFF


@dataclass(frozen=True)
class Seeds:
    canonical: str
    digest: str
    torch_seed: int
    numpy_seed: int


def decision_hour_key(ts: datetime) -> str:
    """``YYYY-MM-DDTHH:00:00Z`` for a timezone-aware UTC datetime on an exact hour."""
    if ts.tzinfo is None or ts.utcoffset() != timedelta(0):
        raise ValueError("decision hour must be timezone-aware and in UTC")
    if ts.minute or ts.second or ts.microsecond:
        raise ValueError("decision hour must be on an exact hour boundary")
    return ts.strftime("%Y-%m-%dT%H:00:00Z")


def seed_for_decision(
    decision_hour: datetime, *, stream: str = "forecast", symbol: str = "BTC-USDT", salt: str = ""
) -> Seeds:
    for name, value in (("stream", stream), ("symbol", symbol), ("salt", salt)):
        if "/" in value or "\n" in value:
            raise ValueError(f"{name} must not contain '/' or newlines")
    if not stream or not symbol:
        raise ValueError("stream and symbol must be non-empty")
    canonical = f"{SEED_SCHEME}/{stream}/{symbol}/{decision_hour_key(decision_hour)}"
    if salt:
        canonical += f"/{salt}"
    digest = hashlib.sha256(canonical.encode("ascii")).digest()
    return Seeds(
        canonical=canonical,
        digest=digest.hex(),
        torch_seed=int.from_bytes(digest[:8], "big") & TORCH_SEED_MASK,
        numpy_seed=int.from_bytes(digest[8:12], "big"),
    )


def seed_everything(seeds: Seeds) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` from ``seeds`` (torch is imported lazily)."""
    import random

    import numpy as np
    import torch

    random.seed(seeds.numpy_seed)
    np.random.seed(seeds.numpy_seed)
    torch.manual_seed(seeds.torch_seed)
