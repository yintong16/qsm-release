"""Affine point arithmetic: experimental lazy and conservative canonical variants.

Wei25519 is birationally equivalent to Ed25519. The chosen incomplete addition formula
also supports the secp256k1 control profile; exceptional inputs remain a schedule obligation.
Importing this package registers its circuit versions.
"""

from . import curve  # noqa: F401  — Wei25519 constants and the classical group law
from . import affine  # noqa: F401  — experimental lazy point versions
from . import conservative  # noqa: F401  — canonical conservative point versions
