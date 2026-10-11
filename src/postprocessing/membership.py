"""Integer-only cluster membership operations; stable atom and component order."""
import numpy as np
from numba import njit


@njit
def overlap_counts(labels, indices, counts, touched):
    """Count nonzero successors using reusable workspace instead of per-node unique."""
    size = 0
    for atom in indices:
        component = labels[atom]
        if component == 0:
            continue
        if counts[component] == 0:
            touched[size] = component
            size += 1
        counts[component] += 1
    identifiers = np.sort(touched[:size])
    shared = np.empty(size, np.int64)
    for i in range(size):
        component = identifiers[i]
        shared[i] = counts[component]
        counts[component] = 0
    return identifiers, shared
