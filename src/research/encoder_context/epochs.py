"""True shuffled passes, including the final partial batch and exact resume."""
import math
import numpy as np


def batches(indices, batch_size, epochs, seed, start=0):
    indices=np.asarray(indices,dtype=np.int64)
    steps=math.ceil(len(indices)/batch_size)
    for epoch in range(start//steps,epochs):
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(indices)
        for offset in range(0,len(order),batch_size):
            step=epoch*steps+offset//batch_size
            if step>=start:yield step,order[offset:offset+batch_size]


def epoch_weights(source):
    """Uniform permutations weighted to equal-source risk, without oversampling."""
    from src.research.local_predictability.metrics import source_weights
    return len(source)*source_weights(source)
