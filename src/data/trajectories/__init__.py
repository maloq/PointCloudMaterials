"""Validated readers with distinct temporal and shooting contracts."""

import hashlib

import numpy as np


def array_sha256(values: np.ndarray) -> str:
    """Hash C-order array bytes, keeping trajectory copies bounded to one frame.

    Shape and dtype are recorded separately in trajectory manifests; neither is
    included in this digest. Preserve stored precision and byte order.
    """
    digest = hashlib.sha256()
    if values.ndim <= 1:
        digest.update(np.ascontiguousarray(values).tobytes())
    else:
        for frame in values:
            digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()
