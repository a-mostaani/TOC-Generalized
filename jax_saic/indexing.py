"""Shared 0-indexed mixed-radix (de)flattening.

MATLAB is 1-indexed throughout; this port is 0-indexed throughout. The same
little-endian mixed-radix flatten/unflatten pattern shows up three times in
the original MATLAB under three different names, all doing the same thing:

  - centralized.py:  ps_calc / pa_calc / mps_calc  (PORT_NOTES.md SS4.1)
                      flatten a per-agent position or action tuple into one
                      scalar joint-state / joint-action index.
  - train.py/qlearning.py: de2bi / bi2de           (PORT_NOTES.md SS5.3, SS7)
                      encode/decode the communicated bit vector.

MATLAB's ps_calc/pa_calc/mps_calc use base n^2 or 5 and are 1-indexed;
MATLAB's de2bi/bi2de use base 2 (bits) and are 0/1-valued already. Once
everything here is converted to 0-indexed values, both collapse to the
exact same operation, verified against the MATLAB source in PORT_NOTES.md
SS4.1:

    main_ps (MATLAB, 1-indexed) = ps(1) + sum_{k=2}^{noa} (ps(k)-1)*(n^2)^(k-1)
    main_ps0 := main_ps - 1, ps0(k) := ps(k) - 1
             => main_ps0 = sum_{k=1}^{noa} ps0(k) * (n^2)^(k-1)

which is exactly `flatten_mixed_radix(ps0, base=n^2)` below, with agent 0
(first element) as the LEAST-significant digit. MATLAB's `bi2de`/`de2bi`
default to the same convention (`'right-msb'`: leftmost/first element is
least significant), so this same pair of functions also serves as the
bit encode/decode used for inter-agent communication.
"""
from __future__ import annotations

import jax.numpy as jnp


def flatten_mixed_radix(vals0: jnp.ndarray, base: int) -> jnp.ndarray:
    """0-indexed little-endian mixed-radix flatten.

    vals0: (..., k) array of digits in [0, base), vals0[..., 0] least significant.
    Returns (...,) array of flattened scalars in [0, base**k).
    """
    k = vals0.shape[-1]
    powers = base ** jnp.arange(k, dtype=jnp.int32)
    return jnp.sum(vals0 * powers, axis=-1).astype(jnp.int32)


def unflatten_mixed_radix(flat0: jnp.ndarray, base: int, k: int) -> jnp.ndarray:
    """Inverse of flatten_mixed_radix for a Python-static digit count k.

    flat0: (...,) array of scalars in [0, base**k).
    Returns (..., k) array of digits, digit 0 least significant.
    """
    digits = []
    v = flat0
    for _ in range(k):
        digits.append(v % base)
        v = v // base
    return jnp.stack(digits, axis=-1)
