"""Standing audit tools: measure what the code CLAIMS against what it DOES.

These are not tests. Tests pin behaviour that is already understood; these answer
"is the number in the docstring the number the circuit produces?" over a range of
widths, and print a table you can read.

They exist because three separate claim bugs shipped and were caught by a human
reading a benchmark table rather than by anything automatic:

    a version named `_exact` that carried an error term
    a `_trunc` naming inversion that silently swapped which circuit was the default
    a circuit whose docstring promised a retained carry it could never set

and then a fourth, found by `rates`: every controlled-`+F` truncation rate was
quoted 2x too high, because the `+F` fires only on the reduction branch.

    python -m qsm.audit rates       truncation rates vs their declared formulas
    python -m qsm.audit refs        exhaustive correctness vs from-definition refs
    python -m qsm.audit baselines   exact-vs-approximate against Litinski
    python -m qsm.audit all
"""
