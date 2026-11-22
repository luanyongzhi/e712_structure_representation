#!/usr/bin/env python3
"""Rational matrices for the Lie algebra e_{7 1/2}.

Supplement to Section 3.2 of Yongzhi Luan, "Basic structure and representations
of the 190-dimensional Lie algebra e_{7 1/2}", Shandong University.

The 190 basis matrices act on W = Q + V + Q, where V is the exterior square
of Q^8 plus its dual. The Levi part uses the exterior-power model of E7
(Adams, Lectures on Exceptional Lie Groups, Chapter 12). The last basis
matrix Z represents the central generator xi and has its only entry at (1,58).

To read the matrices, open data/generators.json. Each nonzero entry has the
form [row, column, numerator, denominator]. Columns are images of basis vectors.
The arrays basis_order and coordinate_order give the orders used in the paper.
The xi_12 through xi_78 coordinate labels refer to the dual exterior basis.

In structure_constants.json the fields "i" and "j" give the first and second
basis indices of a bracket. Each [k, n, d] in coefficients contributes n/d
times basis matrix k. All indices in the files start at 1. Omitted matrix
entries are zero. For brackets, only pairs in increasing order are listed,
and a missing pair in this order has zero bracket. Use antisymmetry to reverse it.

To reproduce both files from the formulas using only the Python 3 standard library,
run the following command from the directory containing this script.

    python e712_matrices.py --output reproduced
"""
import argparse
import json
from collections import defaultdict
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path

PAIRS = list(combinations(range(8), 2))
QUARTETS = list(combinations(range(8), 4))
PAIR_INDEX = {p: r for r, p in enumerate(PAIRS)}
N = 58


def sign(indices):
    if len(set(indices)) != len(indices):
        return 0
    return (-1) ** sum(x > y for r, x in enumerate(indices)
                       for y in indices[r + 1:])


def clean(a):
    return {k: F(v) for k, v in a.items() if v}


def linear_combination(terms):
    result = defaultdict(F)
    for scalar, matrix in terms:
        for key, value in matrix.items():
            result[key] += scalar * value
    return clean(result)


def multiply(a, b):
    rows = defaultdict(list)
    for (j, k), value in b.items():
        rows[j].append((k, value))
    result = defaultdict(F)
    for (r, j), value in a.items():
        for k, other in rows[j]:
            result[r, k] += value * other
    return clean(result)


def commutator(a, b):
    return linear_combination([(1, multiply(a, b)), (-1, multiply(b, a))])


def transpose(a):
    return {(j, r): value for (r, j), value in a.items()}


def wedge2(a):
    """Matrix of A on Lambda^2 Q^8; columns are images of basis vectors."""
    result = defaultdict(F)
    for column, pair in enumerate(PAIRS):
        for position in range(2):
            for k in range(8):
                value = a.get((k, pair[position]), 0)
                image = list(pair)
                image[position] = k
                orientation = sign(image)
                if value and orientation:
                    row = PAIR_INDEX[tuple(sorted(image))]
                    result[row, column] += orientation * value
    return clean(result)


def levi(a):
    d = wedge2(a)
    result = {(r + 1, j + 1): v for (r, j), v in d.items()}
    result.update({(j + 29, r + 29): -v for (r, j), v in d.items()})
    return result


def four_form(quartet):
    result = {}
    for r, p in enumerate(PAIRS):
        for j, q in enumerate(PAIRS):
            if set(p + q) == set(quartet) and sign(p + q):
                result[r + 1, j + 29] = F(sign(p + q))
            orientation = sign(quartet + p + q)
            if orientation:
                result[r + 29, j + 1] = F(orientation)
    return result


def heisenberg(k):
    result = {(k + 1, 57): F(1)}
    if k < 28:
        result[0, k + 29] = F(1, 2)
    else:
        result[0, k - 27] = F(-1, 2)
    return result


def basis():
    labels, matrices = [], []
    for r in range(8):
        for j in range(8):
            if r != j:
                labels.append(f"E_{r + 1}_{j + 1}")
                matrices.append(levi({(r, j): 1}))
    for r in range(7):
        labels.append(f"H_{r + 1}")
        matrices.append(levi({(r, r): 1, (r + 1, r + 1): -1}))
    for quartet in QUARTETS:
        labels.append("T_" + "".join(str(r + 1) for r in quartet))
        matrices.append(four_form(quartet))
    for k, pair in enumerate(PAIRS + PAIRS):
        labels.append(("U_" if k < 28 else "Udual_")
                      + "".join(str(r + 1) for r in pair))
        matrices.append(heisenberg(k))
    labels.append("Z")
    matrices.append({(0, 57): F(1)})
    return labels, matrices


def coordinates(matrix):
    """Recover possible basis coefficients; reconstruction checks membership."""
    def d_entry(r, k, j, l):
        if r == k or j == l:
            return F(0)
        row = PAIR_INDEX[tuple(sorted((r, k)))] + 1
        col = PAIR_INDEX[tuple(sorted((j, l)))] + 1
        return sign((r, k)) * sign((j, l)) * matrix.get((row, col), F(0))

    # Partial trace of the wedge-square action equals (8 - 2) A for traceless A.
    a = {(r, j): sum(d_entry(r, k, j, k) for k in range(8)) / 6
         for r in range(8) for j in range(8)}
    values = [a[r, j] for r in range(8) for j in range(8) if r != j]
    values.extend(sum(a[j, j] for j in range(r + 1)) for r in range(7))
    for quartet in QUARTETS:
        p, q = quartet[:2], quartet[2:]
        values.append(matrix.get((PAIR_INDEX[p] + 1, PAIR_INDEX[q] + 29), F(0)))
    values.extend(matrix.get((k + 1, 57), F(0)) for k in range(56))
    values.append(matrix.get((0, 57), F(0)))
    return {r: F(value) for r, value in enumerate(values) if value}


def reconstruct(coefficients, matrices):
    return linear_combination((value, matrices[r]) for r, value in coefficients.items())


def encoded(value):
    return [value.numerator, value.denominator]


def write_json(path, data):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2)


def verify_and_export(output):
    labels, matrices = basis()
    if len(matrices) != 190:
        raise ArithmeticError("Expected 190 basis matrices")

    # Matrix of 2(bc' - cb') + omega(v,v') on W in the published order.
    omega = {(0, 57): F(2), (57, 0): F(-2)}
    for index in range(28):
        omega[index + 1, index + 29] = F(1)
        omega[index + 29, index + 1] = F(-1)

    for index, matrix in enumerate(matrices):
        if coordinates(matrix) != {index: F(1)}:
            raise ArithmeticError(f"Basis coordinate mismatch for {labels[index]}")
        defect = linear_combination([(1, multiply(transpose(matrix), omega)),
                                     (1, multiply(omega, matrix))])
        if defect:
            raise ArithmeticError(f"Symplectic condition failed for {labels[index]}")

    root_indices = list(range(56)) + list(range(63, 133))
    for index in root_indices + list(range(133, 190)):
        if multiply(matrices[index], matrices[index]):
            raise ArithmeticError(f"Nonzero square for {labels[index]}")

    brackets = []
    checked = 0
    for left, right in combinations(range(len(matrices)), 2):
        bracket = commutator(matrices[left], matrices[right])
        checked += 1
        if right == 189 and bracket:
            raise ArithmeticError(f"Central generator failed to commute with {labels[left]}")
        if left >= 133 and right < 189:
            expected = matrices[-1] if right == left + 28 else {}
            if bracket != expected:
                raise ArithmeticError(f"Heisenberg relation failed for {labels[left]}, {labels[right]}")
        if not bracket:
            continue
        coefficients = coordinates(bracket)
        if reconstruct(coefficients, matrices) != bracket:
            raise ArithmeticError(f"Bracket reconstruction failed for {labels[left]}, {labels[right]}")
        brackets.append({
            "i": left + 1,
            "j": right + 1,
            "coefficients": [[key + 1, *encoded(value)]
                             for key, value in sorted(coefficients.items())],
        })
    if (checked, len(brackets)) != (17955, 4249):
        raise ArithmeticError(f"Found {len(brackets)} nonzero brackets in {checked} pairs")

    coordinates_on_w = (["b"] + [f"x_{r+1}{j+1}" for r, j in PAIRS]
                        + [f"xi_{r+1}{j+1}" for r, j in PAIRS] + ["c"])
    generators = []
    for label, matrix in zip(labels, matrices):
        entries = [[row + 1, column + 1, *encoded(value)]
                   for (row, column), value in sorted(matrix.items())]
        generators.append({"name": label, "entries": entries})

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "generators.json", {
        "size": N, "indices": "one-based", "basis_order": labels,
        "coordinate_order": coordinates_on_w, "generators": generators,
    })
    write_json(output / "structure_constants.json", {
        "basis_order": labels, "indices": "one-based",
        "convention": "[basis[i],basis[j]]=sum coefficient[k]*basis[k]; i<j",
        "brackets": brackets,
    })
    print(f"Verified {len(matrices)} independent symplectic {N} x {N} matrices over Q.")
    print(f"Checked {checked} unordered commutators. Exactly {len(brackets)} are nonzero.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data",
                        metavar="DIRECTORY", help="directory for the two JSON files (default: data)")
    verify_and_export(parser.parse_args().output)
