"""Immutable, field-resolved inversion construction and component error model.

Tuned estimates are distribution-conditioned empirical models, not worst-case
quantum error bounds. Arbitrary amplitude distributions need not satisfy their
uniform-marginal assumptions. Conservative configurations make no such model.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from .. import params
from ..field.add import gidney_add_ctrl_threaded
from ..field.add_chunked import gidney_add_ctrl_chunked_threaded
from . import schedule

_DEFAULT = object()


@dataclass(frozen=True)
class InversionConfig:
    n: int
    p: int
    iters: int
    widths: tuple[int, ...]
    arithmetic_family: str
    gcd_cmp_bits: int | None
    replay_cmp_bits: int | None
    replay_chunk: int | None
    replay_chunk_cmp_bits: int | None
    carry_window: int | None
    double_window: int | None

    def __post_init__(self) -> None:
        # Copy even explicitly supplied sequences: frozen must be deeply immutable.
        object.__setattr__(self, "widths", tuple(self.widths))
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 2:
            raise ValueError("inversion width must be an integer at least two")
        if not isinstance(self.p, int) or not 1 < self.p < (1 << self.n) or not self.p & 1:
            raise ValueError("inversion requires an odd modulus fitting n bits")
        if not isinstance(self.iters, int) or isinstance(self.iters, bool) or self.iters < 1:
            raise ValueError("iteration count must be positive")
        if len(self.widths) != self.iters or any(
            not isinstance(w, int) or isinstance(w, bool) or not 1 <= w <= self.n
            for w in self.widths
        ) or any(a < b for a, b in zip(self.widths, self.widths[1:])):
            raise ValueError("widths must be a non-increasing positive iteration schedule")
        if self.arithmetic_family not in ("tuned", "canonical"):
            raise ValueError("unsupported inversion arithmetic family")
        for name in ("gcd_cmp_bits", "replay_cmp_bits", "replay_chunk",
                     "replay_chunk_cmp_bits", "carry_window", "double_window"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.replay_chunk is None and self.replay_chunk_cmp_bits is not None:
            raise ValueError("chunk comparison requires a replay chunk")
        if self.arithmetic_family == "canonical":
            if self.iters < schedule.conservative_iterations_j2(self.n):
                raise ValueError("conservative configuration needs the deterministic iteration bound")
            if self.widths != (self.n,) * self.iters or any(
                getattr(self, name) is not None for name in
                ("gcd_cmp_bits", "replay_cmp_bits", "replay_chunk",
                 "replay_chunk_cmp_bits", "carry_window", "double_window")
            ):
                raise ValueError("conservative configuration must use full-width exact arithmetic")

    @classmethod
    def tuned(cls, n: int = params.U, p: int = params.PRIME, *,
              iters: int | None = None, widths=None, gcd_cmp_bits=_DEFAULT,
              replay_cmp_bits: int | None = params.CMP_BITS,
              replay_chunk: int | None = 43, replay_chunk_cmp_bits: int | None = 32,
              carry_window: int | None = _DEFAULT,
              double_window: int | None = _DEFAULT) -> InversionConfig:
        """Resolve the operating point from this field, with explicit retuning knobs.

        Omitted carry/doubling windows use ERROR_BUDGET + ceil(log2(2**n-p)),
        computed exactly as ERROR_BUDGET + (deficit-1).bit_length(). This is
        independent of the active process profile. Explicit None stays exact.
        """
        deficit = (1 << n) - p
        if deficit < 1:
            raise ValueError("inversion requires an odd modulus fitting n bits")
        default_window = params.ERROR_BUDGET + (deficit - 1).bit_length()
        if carry_window is _DEFAULT:
            carry_window = default_window
        if double_window is _DEFAULT:
            double_window = default_window
        if iters is None:
            iters = schedule.iterations_j2(n)
        if widths is None:
            widths = schedule.widths_j2(n, iters)
        if gcd_cmp_bits is _DEFAULT:
            gcd_cmp_bits = schedule.compare_bits(n)
        return cls(n, p, iters, tuple(widths), "tuned", gcd_cmp_bits,
                   replay_cmp_bits, replay_chunk, replay_chunk_cmp_bits,
                   carry_window, double_window)

    @classmethod
    def conservative(cls, n: int = params.U, p: int = params.PRIME) -> InversionConfig:
        """Full-width GCD, packed tape, canonical replay; no distribution model."""
        iters = schedule.conservative_iterations_j2(n)
        return cls(n, p, iters, (n,) * iters, "canonical", None, None,
                   None, None, None, None)

    def resolved(self, n: int, p: int) -> InversionConfig:
        """Validate the operand field rather than silently reinterpreting a tuning."""
        if (n, p) != (self.n, self.p):
            raise ValueError("inversion configuration does not match operand width/modulus")
        return self

    @property
    def assumptions(self) -> str:
        if self.arithmetic_family == "canonical":
            return "odd prime p; canonical nonzero divisor; canonical second operand including zero"
        return ("experimental distribution-conditioned estimates: uniform/generic nonzero "
                "field operands and modeled intermediate marginals; empirical GCD tails; "
                "not a worst-case quantum error bound")

    def gcd_kwargs(self) -> dict:
        return {"cmp_bits": self.gcd_cmp_bits, "add_ctrl": gidney_add_ctrl_threaded}

    def replay_kwargs(self) -> dict:
        if self.arithmetic_family == "canonical":
            return {}
        add = gidney_add_ctrl_threaded
        if self.replay_chunk is not None:
            add = partial(gidney_add_ctrl_chunked_threaded,
                          chunk=self.replay_chunk, cmp_bits=self.replay_chunk_cmp_bits)
        return {"add_ctrl_fn": add, "double_window": self.double_window,
                "carry_window": self.carry_window, "cmp_bits": self.replay_cmp_bits}

    def phase1_fail_prob(self) -> float:
        """Both GCD passes, using exactly this iteration/width/comparator schedule.

        Only the repository's standard width model and the full-width schedule
        have a supported analytic interpretation. A custom narrowing schedule
        must not silently receive the default clip or Gaussian estimate.
        """
        if self.arithmetic_family == "canonical":
            return 0.0
        full = self.widths == (self.n,) * self.iters
        standard = self.widths == tuple(schedule.widths_j2(self.n, self.iters))
        if not (full or standard):
            raise ValueError("analytic model unavailable for custom inversion width schedule")
        iteration = (0.0 if self.iters >= schedule.conservative_iterations_j2(self.n)
                     else schedule.iteration_fail_prob_j2(self.n, self.iters))
        clip = 0.0 if full else params.gcd_width_clip_fail_prob(n=self.n, jump=2)
        compare = 0.0
        if self.gcd_cmp_bits is not None and self.gcd_cmp_bits < max(self.widths):
            effective = self.gcd_cmp_bits - (0 if full else schedule.width_pad(self.n))
            if effective < 1:
                raise ValueError("analytic comparison model requires positive informative width")
            compare = schedule.compare_fail_prob(effective)
        return 2 * (iteration + clip + compare)

    def replay_fail_prob(self, reverse: bool = False) -> float:
        if self.arithmetic_family == "canonical":
            return 0.0
        # Reject unsupported full configurations even when just a component is asked for.
        self.phase1_fail_prob()
        fn = params.un_replay_j2_fail_prob if reverse else params.replay_j2_fail_prob
        return fn(self.iters, double_window=self.double_window,
                  carry_window=self.carry_window, cmp_bits=self.replay_cmp_bits,
                  u=self.n, f=(1 << self.n) - self.p, chunk=self.replay_chunk,
                  chunk_cmp_bits=self.replay_chunk_cmp_bits)

    def fail_prob(self, reverse: bool = False) -> float:
        """Component union estimate; zero for the conservative domain contract."""
        return self.phase1_fail_prob() + self.replay_fail_prob(reverse=reverse)
