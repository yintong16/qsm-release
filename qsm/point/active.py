"""The curve belonging to whichever field `QSM_FIELD` selected.

`params` picks the prime; this picks the curve over it. They are two separate choices in
principle -- many curves share a prime -- but in this project they are locked together,
because the whole point of the secp256k1 profile is to run OUR construction over THEIR
field, and a Wei25519 point is not a point of the secp256k1 group.

Importing the module rather than copying its constants keeps `curve.add` available for
the reference checks, which is the only place the group law is actually needed.
"""

from __future__ import annotations

from ..params import FIELD

if FIELD == "secp256k1":
    from . import secp256k1 as CURVE
else:
    from . import curve as CURVE

#: Re-exported for the registration blocks, which want two numbers and not a module.
GX = CURVE.GX
GY = CURVE.GY
ORDER = CURVE.ORDER

add = CURVE.add
neg = CURVE.neg
mul = CURVE.mul
is_on_curve = CURVE.is_on_curve
