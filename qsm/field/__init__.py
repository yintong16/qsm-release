"""Field arithmetic for p = 2^255 - 19.

Importing this package registers its circuit versions (so `python -m qsm.bench` sees
them). Add each new circuit module to the imports below.
"""

from . import add  # noqa: F401  — registers field.add_* versions
from . import compare  # noqa: F401  — registers field.cmp_* versions
from . import add_chunked  # noqa: F401  — registers field.add_gidney_chunked_* (needs compare)
from . import constadd  # noqa: F401  — registers field.addconst_* versions
from . import modarith  # noqa: F401  — registers field.mod* versions
from . import sub  # noqa: F401  — registers field.sub_* versions
from . import swap  # noqa: F401  — registers field.cswap_* versions
from . import shift  # noqa: F401  — registers field.rshift1_* versions (needs swap)
