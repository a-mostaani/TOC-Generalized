"""Boundary-gate value certification + refinement for ESAIC.

Additive to jax_saic/'s Phase-1 SAIC/ESAIC port -- see CERTIFICATION.md for
the math and PORT_NOTES.md's project-level context. Nothing here is wired
into jax_saic/'s existing pipeline; certification.py's runtime path is
opt-in via CertificationConfig.cert_enabled (default False) and depends
only on generic quantities (see certification.py's module docstring, G1).
"""
