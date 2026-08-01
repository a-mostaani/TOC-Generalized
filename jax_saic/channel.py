"""Binary symmetric channel. Port of SAIC/bsc_ch.m.

Per PORT_NOTES.md SS0.5/SS3: Phase 1's training loop uses a rate-limited but
PERFECT channel (no bit errors once the bit budget is respected) -- this
module is NOT called anywhere in train.py's main loop. It is ported and
unit-tested standalone here because it was explicitly named in scope and a
later phase may want the noisy-channel model.

Generalization note: MATLAB's bsc_ch.m only ever writes cs(i,1,j) -- it
hardcodes index 1 in the middle ("which other agent") dimension, and its
own comment flags this explicitly: "if noa increases, this line should be
revisited! #noa". That is a known, self-acknowledged 2-agent-only gap in
the original, not a deliberate design choice, and this function is not
wired into any trained pipeline (see above) -- so this port implements the
evidently-intended general behavior (independent bit flips across the full
(noa, noa-1, bits) message tensor) rather than replicating the acknowledged
gap.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def apply(cs_bc: jnp.ndarray, bsc_p: float, rng: jax.Array) -> jnp.ndarray:
    """Flip each bit of cs_bc independently with probability bsc_p.

    cs_bc: (noa, noa-1, bits) array of bits (0/1) about to be sent.
    bsc_p: flip probability.
    rng: PRNGKey.

    MATLAB draws one rand() per (agent, bit) pair -- agent-outer, bit-inner
    -- and compares `rand() <= 1 - bsc_p` for "no flip". This draws one
    uniform sample per element of cs_bc's full shape in one call, which is
    equivalent for i.i.d. draws; exact call-by-call RNG-sequence fidelity
    isn't the target (PORT_NOTES.md's validation criteria are curve-shape/
    convergence, not bit-exact RNG replay).
    """
    no_flip_prob = 1.0 - bsc_p
    draws = jax.random.uniform(rng, shape=cs_bc.shape)
    flip = draws > no_flip_prob
    return jnp.where(flip, 1 - cs_bc, cs_bc)
