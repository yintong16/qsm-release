"""Exact signed-radix references and isolated, scalar-preserving reversible recoders.

No production registry/default changes. Digits use radix 2**(a+1), interval
[-2**a, 2**a-1], and packed words ``address | (sign << a)``. Address zero
means logical zero for sign=0 and the negative endpoint for sign=1.

The circuit is an XOR oracle, not an ideal/Python semantic operation: it emits
only X/CX/CCX and returns every temporary qubit to zero. It retains the entire
original scalar AND all digit outputs. No streaming/recycled-QFT space claim
is made. The carry variant temporarily retains a full q-bit carry tape.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.circuit import Circuit, Register
from ..field.add import cdkm_add


VARIANTS = ("carry", "bias")


def _positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _parameters(scalar_bits: int, address_bits: int, variant: str) -> None:
    _positive("scalar_bits", scalar_bits)
    _positive("address_bits", address_bits)
    if variant not in VARIANTS:
        raise ValueError("variant must be 'carry' or 'bias'")


def encode_digit(digit: int, address_bits: int) -> tuple[int, int]:
    """Return (sign, address); (0,0) is zero and (1,0) is -2**a."""
    _positive("address_bits", address_bits)
    if not isinstance(digit, int) or isinstance(digit, bool):
        raise TypeError("digit must be an integer")
    bound = 1 << address_bits
    if not -bound <= digit < bound:
        raise ValueError("digit outside signed radix interval")
    return int(digit < 0), abs(digit) % bound


def decode_digit(sign: int, address: int, address_bits: int) -> int:
    """Decode the endpoint-aware sign/address bijection."""
    _positive("address_bits", address_bits)
    if not isinstance(sign, int) or sign not in (0, 1):
        raise ValueError("sign must be a bit")
    if not isinstance(address, int) or isinstance(address, bool):
        raise TypeError("address must be an integer")
    bound = 1 << address_bits
    if not 0 <= address < bound:
        raise ValueError("address does not fit address_bits")
    return (-bound if sign else 0) if address == 0 else (-address if sign else address)


@dataclass(frozen=True)
class SignedRecoding:
    scalar_bits: int
    address_bits: int
    variant: str
    digits: tuple[int, ...]
    terminal_carry: int
    bias: int

    @property
    def encoded(self) -> tuple[tuple[int, int], ...]:
        return tuple(encode_digit(d, self.address_bits) for d in self.digits)

    @property
    def packed_digits(self) -> int:
        w = self.address_bits + 1
        return sum((address | (sign << self.address_bits)) << (i * w)
                   for i, (sign, address) in enumerate(self.encoded))

    def reconstruct(self) -> int:
        w = self.address_bits + 1
        return (self.bias + sum(d << (i * w) for i, d in enumerate(self.digits))
                + (self.terminal_carry << (len(self.digits) * w)))


def recode_scalar(
    scalar: int, scalar_bits: int, address_bits: int, variant: str = "carry",
) -> SignedRecoding:
    """Exact integer identity on the explicit domain 0 <= scalar < 2**L.

    Top chunks are zero-padded. Carry: t=u+c, c'=[t>=B], d=t-2B*c'.
    Bias: d=u-B, with public C=B*sum((2B)**i). No subgroup reduction.
    """
    _parameters(scalar_bits, address_bits, variant)
    if not isinstance(scalar, int) or isinstance(scalar, bool):
        raise TypeError("scalar must be an integer")
    if not 0 <= scalar < 1 << scalar_bits:
        raise ValueError("scalar outside the declared L-bit domain")
    w, bound = address_bits + 1, 1 << address_bits
    q = (scalar_bits + w - 1) // w
    digits, carry, bias = [], 0, 0
    for i in range(q):
        chunk = (scalar >> (i * w)) & ((1 << w) - 1)
        if variant == "carry":
            total = chunk + carry
            carry = int(total >= bound)
            digits.append(total - (carry << w))
        else:
            digits.append(chunk - bound)
            bias += bound << (i * w)
    return SignedRecoding(scalar_bits, address_bits, variant, tuple(digits), carry, bias)


def _disjoint_live(circ: Circuit, *registers: Register) -> None:
    wires = [q for register in registers for q in register]
    if len(wires) != len(set(wires)):
        raise ValueError("registers must contain distinct, disjoint wires")
    if any(q not in circ._live_qids for q in wires):
        raise ValueError("all register wires must be live")


def twos_complement_to_sign_address_inplace(circ: Circuit, word: Register) -> None:
    """Self-inverse conversion of a w-bit digit; sign stays in word[w-1].

    Conditionally negate the low a=w-1 bits modulo 2**a using controlled
    complement followed by the exact CDKM adder. Its materialized padding and
    carry ancilla are really emitted and counted, not free reinterpretation.
    This is phase-free, measurement-free, and valid on every w-bit word.
    """
    if len(word) < 2:
        raise ValueError("a signed word needs at least two bits")
    _disjoint_live(circ, word)
    low, sign = word[:-1], word[-1]
    if len(low) == 1:  # negation modulo two is the identity
        return
    for bit in low:
        circ.cx(sign, bit)
    cdkm_add(circ, Register([sign]), low)


def _subtract_bit(circ: Circuit, bit: int, word: Register) -> None:
    # ~(~word + bit) = word - bit modulo 2**len(word).
    for wire in word:
        circ.x(wire)
    cdkm_add(circ, Register([bit]), word)
    for wire in word:
        circ.x(wire)


def _load_chunk(
    circ: Circuit, chunk: Register, temporary: Register, incoming: int | None,
    *, inverse: bool = False,
) -> None:
    if inverse and incoming is not None:
        _subtract_bit(circ, incoming, temporary)
    for source, target in zip(chunk, temporary):
        circ.cx(source, target)
    if not inverse and incoming is not None:
        cdkm_add(circ, Register([incoming]), temporary)


def signed_recode_inplace(
    circ: Circuit,
    scalar: Register,
    digits: Register,
    address_bits: int,
    *,
    variant: str = "carry",
    terminal: int | None = None,
    inverse: bool = False,
) -> None:
    """XOR the signed representation into ``digits``, preserving ``scalar``.

    ``digits`` has q*(a+1) bits, packed LSB-first by radix slot. For carry,
    ``terminal`` is a separate required bit, XORed with the terminal carry
    (even when that carry is provably zero on this scalar domain). Bias has
    no terminal register. The public bias point correction is NOT performed.

    Zero outputs give the recoding; arbitrary initial destinations give XOR
    with it. The XOR oracle is its own inverse, so inverse=True deliberately
    emits the same circuit and cleans outputs after their last use. Outputs
    must be restored by their consumers before uncomputation. All temporary
    scratch, including the carry tape, is zero and freed on return.

    Workspace is full scalar + full output + transient scratch, NOT streaming.
    This is a correctness prototype, not a claimed optimal recoder.
    """
    _parameters(len(scalar), address_bits, variant)
    if not isinstance(inverse, bool):
        raise TypeError("inverse must be a bool")
    w = address_bits + 1
    q = (len(scalar) + w - 1) // w
    if len(digits) != q * w:
        raise ValueError("digits must have ceil(L/(a+1))*(a+1) bits")
    if variant == "carry" and terminal is None:
        raise ValueError("carry variant requires a terminal bit")
    if variant == "bias" and terminal is not None:
        raise ValueError("bias variant has no terminal bit")
    _disjoint_live(circ, scalar, digits, Register([] if terminal is None else [terminal]))

    carries = circ.alloc(q) if variant == "carry" else None
    temporary = circ.alloc(w + 1 if carries is not None else w)
    for i in range(q):
        chunk = scalar[i*w:min((i+1)*w, len(scalar))]
        incoming = carries[i-1] if carries is not None and i else None
        _load_chunk(circ, chunk, temporary, incoming)
        if carries is not None:
            # t can equal 2B. Its overflow bit is then 1, its sign bit 0:
            # c'=[t>=B] = t[w] XOR t[w-1], NOT just the sign of d.
            circ.cx(temporary[w], carries[i])
            circ.cx(temporary[w-1], carries[i])
        else:
            circ.x(temporary[w-1])  # u-B modulo 2**w
        twos_complement_to_sign_address_inplace(circ, temporary[:w])
        for source, target in zip(temporary[:w], digits[i*w:(i+1)*w]):
            circ.cx(source, target)
        twos_complement_to_sign_address_inplace(circ, temporary[:w])
        if carries is None:
            circ.x(temporary[w-1])
        _load_chunk(circ, chunk, temporary, incoming, inverse=True)

    if carries is not None:
        circ.cx(carries[-1], terminal)
        # Recompute only the carry predicates in reverse order. Original scalar
        # and earlier carries survive until each predicate has been erased.
        for i in reversed(range(q)):
            chunk = scalar[i*w:min((i+1)*w, len(scalar))]
            incoming = carries[i-1] if i else None
            _load_chunk(circ, chunk, temporary, incoming)
            circ.cx(temporary[w], carries[i])
            circ.cx(temporary[w-1], carries[i])
            _load_chunk(circ, chunk, temporary, incoming, inverse=True)
        circ.free(carries)
    circ.free(temporary)
