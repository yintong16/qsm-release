#!/usr/bin/env python3
"""Estimate jump-2 GCD heuristics over p = 2^255 - 19; standard library only."""
from __future__ import annotations

import argparse
import math
import random
import statistics


N = 255
PRIME = (1 << N) - 19


def trace_widths(p: int, x: int, jump: int = 2) -> list[int]:
    """Record bitlen(max(u, v)) before each iteration, stopping when v is zero."""
    if p <= 2 or p % 2 == 0 or not 0 < x < p or jump < 1:
        raise ValueError("require odd p > 2, 0 < x < p, and jump >= 1")
    u, v, i, out = p, x, 0, []
    while v:
        out.append(max(u, v).bit_length())
        for j in range(jump):
            if j == 0 and i > 0:
                v >>= 1
            elif v % 2 == 0 and v:
                v >>= 1
            else:
                break
        if v & 1:
            if u > v:
                u, v = v, u
            v -= u
        i += 1
    return out


def fitted_decrease(widths: list[int]) -> float:
    """OLS decrease for width(i) = intercept - rate*i, with both parameters free."""
    if len(widths) < 2:
        raise ValueError("a slope requires at least two observations")
    mean_i = (len(widths) - 1) / 2
    mean_w = statistics.mean(widths)
    numerator = sum((i - mean_i) * (w - mean_w) for i, w in enumerate(widths))
    denominator = sum((i - mean_i) ** 2 for i in range(len(widths)))
    return -numerator / denominator


def measure(heuristic: str, samples: int = 2000) -> dict[str, float]:
    if heuristic not in ("width", "iterations"):
        raise ValueError("heuristic must be width or iterations")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    rng = random.Random(31337)
    values = []
    for _ in range(samples):
        widths = trace_widths(PRIME, rng.randrange(1, PRIME))
        if heuristic == "width":
            values.append(fitted_decrease([w for w in widths if w > 1]))
        else:
            values.append(len(widths))
    if heuristic == "width":
        return {"width_rate": statistics.mean(values)}
    mean, sd = statistics.mean(values), statistics.pstdev(values)
    return {"mean": mean, "sd": sd, "mean_per_bit": mean / N,
            "sd_per_sqrt_bit": sd / math.sqrt(N)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heuristic", choices=("width", "iterations"))
    parser.add_argument("--samples", type=int, default=2000, help="sample count (default: 2000)")
    args = parser.parse_args(argv)
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    result = measure(args.heuristic, args.samples)
    print(f"Jump-2 GCD; p=2^255-19; n={N}; samples={args.samples}")
    print("Inputs: uniform nonzero field elements, sampled with replacement.")
    if args.heuristic == "width":
        print("Method: record bitlen(max(u,v)) before each iteration; retain widths > 1.")
        print("Fit width(i) = intercept - rate*i by least squares per input (both free); average rates.")
        print(f"Width decrease = {result['width_rate']:.8f} bits/iteration")
    else:
        print("Method: T counts iterations until v=0; mean=sum(T)/N; sd=sqrt(sum((T-mean)^2)/N).")
        print("Here N is the sample count; sd uses the population denominator.")
        print(f"Mean = {result['mean']:.8f}; sd = {result['sd']:.8f} iterations")
        print(f"Mean/n = {result['mean_per_bit']:.8f}")
        print(f"sd/sqrt(n) = {result['sd_per_sqrt_bit']:.8f}")


if __name__ == "__main__":
    main()
