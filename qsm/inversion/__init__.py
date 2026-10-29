"""Dialog-GCD multiplication/division: tuned experimental and canonical conservative.

Multiplication by a nonzero field element and division are inverse arithmetic maps.
The experimental lazy implementations use matched forward/reverse schedules, but
independently measurement-vented approximate circuits are NOT physical quantum
inverses. Their empirical models assume generic/uniform intermediate marginals;
quantum superposition alone does not supply those assumptions.

The conservative composites use full-width GCD with a deterministic iteration bound
and matched exact canonical replay. They accept every canonical second operand,
including zero and constant one: division with numerator one computes an inverse.
The experimental tuned division retains its structured-numerator limitations.

The construction records binary-GCD decisions on (p,x), replays the packed tape
against a field-register pair, then unbuilds the tape to restore x. All returned
register handles are explicit. Importing this package registers both version families.
"""

from . import schedule  # noqa: F401
from . import gcd  # noqa: F401  — registers inv.to_dialog_* versions
from . import replay  # noqa: F401  — registers inv.apply_dialog_* versions
from . import undo  # noqa: F401  — registers inv.un_to_dialog_* versions
from . import unreplay  # noqa: F401  — registers inv.un_apply_dialog_* versions
from . import inplace  # noqa: F401  — registers inv.modmul_inplace, inv.moddiv_inplace
