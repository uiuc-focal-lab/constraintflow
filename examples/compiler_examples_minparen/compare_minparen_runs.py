#!/usr/bin/env python3
"""
Run each compiler_examples .cf program alongside its minparen twin and compare CLI output.

Example:
  python examples/compiler_examples_minparen/compare_minparen_runs.py \\
    --network ~/nets/mnist_relu_3_50.onnx \\
    --dataset mnist --eps 0.008 --batch-size 2

Requires a working local install (same environment as `constraintflow run`).

If --network does not exist, only compilation of each pair is checked (no execution).
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def compile_ok(program: Path, out_dir: Path) -> tuple[bool, str]:
    """Compile in a fresh interpreter to avoid deep stacks / global state from batch runs."""
    import textwrap

    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(repo_root())!r})
        from constraintflow.compiler.compile import compile as _c
        _c({str(program)!r}, {str(out_dir)!r})
        """
    ).strip()
    os.makedirs(out_dir, exist_ok=True)
    p = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
    )
    if p.returncode == 0:
        return True, "compile ok"
    msg = (p.stderr or "") + (p.stdout or "")
    return False, msg.strip() or f"exit {p.returncode}"


def run_constraintflow(
    program: Path,
    *,
    network: str,
    dataset: str,
    eps: float,
    batch_size: int,
    device: str,
    out_dir: Path,
) -> tuple[int, str, str]:
    cmd = [
        sys.executable,
        "-m",
        "constraintflow.cli",
        "run",
        str(program),
        "--network",
        network,
        "--dataset",
        dataset,
        "--eps",
        str(eps),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--compile",
        "--output-path",
        str(out_dir),
    ]
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_root()),
        env={**os.environ, "PYTHONPATH": str(repo_root())},
    )
    return p.returncode, p.stdout, p.stderr


def extract_bounds_block(text: str) -> tuple[str | None, str | None]:
    lb = None
    ub = None
    for line in text.splitlines():
        if line.startswith("Lower bounds:"):
            lb = line.split(":", 1)[1].strip()
        if line.startswith("Upper bounds:"):
            ub = line.split(":", 1)[1].strip()
    return lb, ub


def parse_array(s: str):
    """Parse printed numpy array / list from constraintflow stdout."""
    s = s.strip()
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        pass
    inner = re.sub(r"^(array|tensor)\s*\(\s*", "", s)
    inner = re.sub(r"\s*\)\s*$", "", inner)
    try:
        return ast.literal_eval(inner)
    except (SyntaxError, ValueError):
        return None


def bounds_equal(a, b, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if type(a) != type(b):
        return False
    try:
        import numpy as np

        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        return aa.shape == bb.shape and np.allclose(aa, bb, rtol=tol, atol=tol)
    except Exception:
        return str(a) == str(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--orig-dir",
        type=Path,
        default=repo_root() / "examples" / "compiler_examples",
        help="Directory containing original .cf files",
    )
    ap.add_argument(
        "--min-dir",
        type=Path,
        default=repo_root() / "examples" / "compiler_examples_minparen",
        help="Directory containing reduced-parenthesis .cf files",
    )
    ap.add_argument("--network", default=os.path.expanduser("~/nets/mnist_relu_3_50.onnx"))
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--eps", type=float, default=0.008)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N pairs (sorted by path); 0 means all.",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="Only verify both versions compile; skip execution (fast).",
    )
    args = ap.parse_args()

    orig_dir = args.orig_dir.resolve()
    min_dir = args.min_dir.resolve()

    if not orig_dir.is_dir():
        print(f"ERROR: {orig_dir} not found", file=sys.stderr)
        return 1

    network_path = os.path.expanduser(args.network)
    do_run = (not args.compile_only) and os.path.isfile(network_path)
    if not do_run and not args.compile_only:
        print(
            f"Note: network file not found ({network_path}); "
            "will only check compilation for each pair.\n",
            file=sys.stderr,
        )
    if args.compile_only:
        print("Compile-only mode (no execution).\n", file=sys.stderr)

    pairs: list[tuple[Path, Path]] = []
    for ocf in sorted(orig_dir.rglob("*.cf")):
        rel = ocf.relative_to(orig_dir)
        mcf = min_dir / rel
        if mcf.is_file():
            pairs.append((ocf, mcf))

    if not pairs:
        print("No matching .cf pairs found.", file=sys.stderr)
        return 1

    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    compile_failures = []  # one side ok, one side not — regression risk
    compile_both_fail = []  # neither compiles in this project (often pre-existing)
    run_mismatches = []
    run_failures = []

    for ocf, mcf in pairs:
        tag = str(ocf.relative_to(orig_dir))

        odir = tempfile.mkdtemp(prefix="cf_orig_")
        mdir = tempfile.mkdtemp(prefix="cf_min_")

        ok_o, msg_o = compile_ok(ocf, Path(odir))
        ok_m, msg_m = compile_ok(mcf, Path(mdir))
        if ok_o and ok_m:
            pass
        elif not ok_o and not ok_m:
            compile_both_fail.append((tag, msg_o[:200], msg_m[:200]))
            continue
        else:
            compile_failures.append((tag, msg_o, msg_m))
            continue

        if not do_run:
            print(f"[compile-only OK] {tag}")
            continue

        rc_o, out_o, err_o = run_constraintflow(
            ocf,
            network=network_path,
            dataset=args.dataset,
            eps=args.eps,
            batch_size=args.batch_size,
            device=args.device,
            out_dir=Path(tempfile.mkdtemp(prefix="cf_run_orig_")),
        )
        rc_m, out_m, err_m = run_constraintflow(
            mcf,
            network=network_path,
            dataset=args.dataset,
            eps=args.eps,
            batch_size=args.batch_size,
            device=args.device,
            out_dir=Path(tempfile.mkdtemp(prefix="cf_run_min_")),
        )

        if rc_o != 0 or rc_m != 0:
            run_failures.append((tag, rc_o, rc_m, err_o + err_m))
            continue

        lb_o, ub_o = extract_bounds_block(out_o)
        lb_m, ub_m = extract_bounds_block(out_m)

        if not bounds_equal(parse_array(lb_o or ""), parse_array(lb_m or "")) or not bounds_equal(
            parse_array(ub_o or ""), parse_array(ub_m or "")
        ):
            run_mismatches.append((tag, out_o, out_m))

    ret = 0

    if compile_both_fail:
        print(
            f"\nNote: {len(compile_both_fail)} pair(s) fail to compile for "
            "both original and minparen (ignored for regression check).\n"
        )

    if compile_failures:
        ret = 1
        print("\n=== Compile mismatch (one side only) ===")
        for tag, a, b in compile_failures:
            print(f"--- {tag} ---\norig: {a}\nmin:  {b}\n")

    if run_failures:
        ret = 1
        print("\n=== Run failures (non-zero exit) ===")
        for tag, rc_o, rc_m, err in run_failures:
            print(f"--- {tag} rc=({rc_o},{rc_m}) ---\n{err}\n")

    if run_mismatches:
        ret = 1
        print("\n=== Output mismatches ===")
        for tag, o, m in run_mismatches:
            print(f"--- {tag} ---\n<<< original stdout (excerpt):\n{o[:2000]}\n<<< minparen stdout (excerpt):\n{m[:2000]}\n")

    if ret == 0 and do_run:
        ran = len(pairs) - len(compile_both_fail)
        print(f"All {ran} comparable pairs: compile OK; run output bounds match.")
    elif ret == 0 and not do_run:
        okn = len(pairs) - len(compile_both_fail)
        print(f"All {okn} comparable pairs: compile OK (run skipped).")

    return ret


if __name__ == "__main__":
    raise SystemExit(main())
