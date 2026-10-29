"""Layer 1: products and modular multiplication for p = 2^255 - 19.

Importing this package registers its circuit versions (so `python -m qsm.bench` sees
them). Add each new circuit module to the imports below.

Two independent constructions, measured against each other in docs/decisions/07:

    product.py     raw x*y and x^2 into a 2U-bit register (the add/subtract trick)
    fold.py        the pseudo-Mersenne reduction 2^U == F (mod p)
    modmul.py      the modular operations built from those, and their versions

    accumulate.py  the same modular operations with NO product register: reduce on
                   every row. Cheaper and 43% narrower for the general product, and
                   -- once the rows read a free prefix slice of the operand instead of
                   all of it -- the cheapest square we have too, subtract fused in.
                   Doc 07 for the schedule, doc 13 for the triangle.
"""

from . import accumulate  # noqa: F401  — registers mul.*_acc.* versions
from . import fold  # noqa: F401  — reduction primitives, no versions of its own
from . import modmul  # noqa: F401  — registers mul.* versions
from . import product  # noqa: F401  — product cores, registered via modmul
