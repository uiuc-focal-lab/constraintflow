#!/usr/bin/env python3
"""
Regenerate .cf programs under this directory from ../compiler_examples
by applying safe parenthesis reductions (rely on operator precedence).

Run from repo root:  python examples/compiler_examples_minparen/build_minparen.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT.parent / "compiler_examples"


def reduce_parens(text: str) -> str:
    s = text

    # --- Long compute_upper / rational expression (standard precedence) ---
    blob = (
        "(((n[u]) / ((n[u]) - (n[l]))) * ((n))) - "
        "(((n[u]) * (n[l])) / ((n[u]) - (n[l]))))"
    )
    s = s.replace(
        blob,
        "n[u] / (n[u] - n[l]) * n - n[u] * n[l] / (n[u] - n[l])",
    )

    blob_u = (
        "(((n[u]) / ((n[u]) - (n[l]))) * ((n[U]))) - "
        "(((n[u]) * (n[l])) / ((n[u]) - (n[l]))))"
    )
    s = s.replace(
        blob_u,
        "n[u] / (n[u] - n[l]) * n[U] - n[u] * n[l] / (n[u] - n[l])",
    )

    reuse_blob = (
        "(((u) / ((u) - (l))) * ((n[U]))) - (((u) * (l)) / ((u) - (l))))"
    )
    s = s.replace(
        reuse_blob,
        "u / (u - l) * n[U] - u * l / (u - l)",
    )

    # Nested ternary still needs a closing ')' for the group opened after the outer ':'.
    # Fixing ends that lost one paren when the big blob was inlined.
    s = s.replace(
        "? 0.0 : n[u] / (n[u] - n[l]) * n - n[u] * n[l] / (n[u] - n[l]);",
        "? 0.0 : n[u] / (n[u] - n[l]) * n - n[u] * n[l] / (n[u] - n[l]));",
    )
    s = s.replace(
        "? 0.0 : n[u] / (n[u] - n[l]) * n[U] - n[u] * n[l] / (n[u] - n[l]);",
        "? 0.0 : n[u] / (n[u] - n[l]) * n[U] - n[u] * n[l] / (n[u] - n[l]));",
    )
    s = s.replace(
        "? 0.0 : u / (u - l) * n[U] - u * l / (u - l);",
        "? 0.0 : u / (u - l) * n[U] - u * l / (u - l));",
    )

    # --- simplify_* and replace_* ternary lines (all variants) ---
    patterns_trivial = [
        (
            "(coeff >= 0.0) ? (n[l] * coeff) : (coeff * n[u])",
            "coeff >= 0.0 ? n[l] * coeff : coeff * n[u]",
        ),
        (
            "(coeff >= 0.0) ? (coeff * n[u]) : (coeff * n[l])",
            "coeff >= 0.0 ? coeff * n[u] : coeff * n[l]",
        ),
        (
            "(coeff >= 0.0) ? (coeff * n[l]) : (coeff * n[u])",
            "coeff >= 0.0 ? coeff * n[l] : coeff * n[u]",
        ),
        (
            "(coeff >= 0.0) ? (coeff * n[L]) : (coeff * n[U])",
            "coeff >= 0.0 ? coeff * n[L] : coeff * n[U]",
        ),
        (
            "(coeff >= 0.0) ? (coeff * n[U]) : (coeff * n[L])",
            "coeff >= 0.0 ? coeff * n[U] : coeff * n[L]",
        ),
        (
            "(coeff >= 0) ? (coeff * n[l]) : (coeff * n[u])",
            "coeff >= 0 ? coeff * n[l] : coeff * n[u]",
        ),
        (
            "(coeff >= 0) ? (coeff * n[u]) : (coeff * n[l])",
            "coeff >= 0 ? coeff * n[u] : coeff * n[l]",
        ),
        (
            "(coeff >= 0) ? (coeff * n[L]) : (coeff * n[U])",
            "coeff >= 0 ? coeff * n[L] : coeff * n[U]",
        ),
        (
            "(coeff >= 0) ? (coeff * n[U]) : (coeff * n[L])",
            "coeff >= 0 ? coeff * n[U] : coeff * n[L]",
        ),
    ]
    for old, new in patterns_trivial:
        s = s.replace(old, new)

    s = s.replace("(c > 0.0) ? ", "c > 0.0 ? ")

    s = s.replace("(n[l] >= n[u]) ? ", "n[l] >= n[u] ? ")

    s = s.replace("(a < b) ? a : b", "a < b ? a : b")

    # zid / polyzono Relu branch
    s = s.replace(
        "((prev[l]) >= 0.0) ? (prev[Z]) : (((prev[u]) <= 0.0) ? 0.0 : x(prev[l], prev[u], prev[Z]))",
        "prev[l] >= 0.0 ? prev[Z] : (prev[u] <= 0.0 ? 0.0 : x(prev[l], prev[u], prev[Z]))",
    )

    s = s.replace(
        "((curr[last_layer] == 1) or (curr[layer] == 1)) ?",
        "curr[last_layer] == 1 or curr[layer] == 1 ?",
    )

    # Note: do not remove the outer `( ... )` around a comma-separated trans_ret when
    # the first expression begins with `(` — the leading LPAREN is parsed as #parentrans.

    # Sigmoid helpers: strip redundant grouping (ternary arms)
    s = s.replace("(sigma(n[l]))", "sigma(n[l])")
    s = s.replace("(sigma(n[u]))", "sigma(n[u])")

    arm = (
        "((sigma(n[l])) + (lambda(n[l], n[u]) * n) - (lambda(n[l], n[u]) * n[l]))"
    )
    s = s.replace(
        arm,
        "sigma(n[l]) + lambda(n[l], n[u]) * n - lambda(n[l], n[u]) * n[l]",
    )

    arm_u = (
        "((sigma(n[u])) + (lambda(n[l], n[u]) * n) - (lambda(n[l], n[u]) * n[u]))"
    )
    s = s.replace(
        arm_u,
        "sigma(n[u]) + lambda(n[l], n[u]) * n - lambda(n[l], n[u]) * n[u]",
    )

    arm_up = (
        "((sigma(n[u])) + (lambda_prime(n[l], n[u]) * n) - "
        "(lambda_prime(n[l], n[u]) * n[u]))"
    )
    s = s.replace(
        arm_up,
        "sigma(n[u]) + lambda_prime(n[l], n[u]) * n - lambda_prime(n[l], n[u]) * n[u]",
    )

    arm_lp = (
        "((sigma(n[l])) + (lambda_prime(n[l], n[u]) * n) - "
        "(lambda_prime(n[l], n[u]) * n[l]))"
    )
    s = s.replace(
        arm_lp,
        "sigma(n[l]) + lambda_prime(n[l], n[u]) * n - lambda_prime(n[l], n[u]) * n[l]",
    )

    # crown.cf Relu: too entangled (nested tuples + ternaries); keep original bracketing
    # from compiler_examples when regenerating.

    # Generic: double parens around simple indexed neurons in tuples
    s = re.sub(r"\(\(prev\[([^\]]+)\]\)\)", r"prev[\1]", s)
    s = re.sub(r"\(\(prev\[l\]\)\)", "prev[l]", s)
    s = re.sub(r"\(\(prev\[u\]\)\)", "prev[u]", s)

    return s


def mirror_tree(src: Path, dst: Path) -> None:
    for path in sorted(src.rglob("*.cf")):
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8")
        out.write_text(reduce_parens(text), encoding="utf-8")
        print(f"Wrote {out.relative_to(dst.parent)}")


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Missing source directory: {SOURCE_DIR}")
    mirror_tree(SOURCE_DIR, ROOT)
    print("Done.")


if __name__ == "__main__":
    main()
