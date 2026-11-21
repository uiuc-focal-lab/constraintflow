#!/usr/bin/env python3
"""Rerun the paper's cf-gpu cells with the settings in configs.csv and summarize them.

    python reproduce/run.py --nets nets --certifiers ibp,crownibp --configs N1,N16
    python reproduce/run.py --nets nets                  # every cell (many hours)

Run it from the directory that holds the datasets (MNIST, CIFAR-10/100 are downloaded there on
first use; TinyImageNet is read from ./tinyimagenet). Each cell is built with `scepter` (and the
row_chunks rewrite where the paper used it) into <out>/<certifier>/<config>/program, then run with
the paper's protocol; per-batch times, peak memory and bounds go to cf.jsonl. Finished cells are
skipped on a rerun. summary.csv has one row per cell: mean time per image, peak GPU memory and the
number of verified images (every margin lower bound > 0).
"""
import argparse
import csv
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# As in the paper's runs: full fp32 (no TF32), one Inductor compile thread, and Inductor's tiling
# analysis off (a workaround for a PyTorch 2.10 crash that the measurements were taken with).
ENV = {"CF_DISABLE_TF32": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1", "TORCHINDUCTOR_COALESCE_TILING_ANALYSIS": "0"}


def cells(certifiers, configs):
    with open(os.path.join(HERE, "configs.csv")) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if (not certifiers or r["certifier"] in certifiers) and (not configs or r["config"] in configs)]


def option(args, name):
    return args[args.index(name) + 1]


def complete(results, run_args):
    if not os.path.exists(results):
        return False
    with open(results) as f:
        batches = sum(1 for line in f if '"kind": "batch"' in line)
    return batches * int(option(run_args, "--batch-size")) == int(option(run_args, "--num-images"))


def run_cell(cell, nets, out, keep_caches, log):
    d = os.path.join(out, cell["certifier"], cell["config"])
    program, results = os.path.join(d, "program"), os.path.join(d, "cf.jsonl")
    run_args = shlex.split(cell["run_args"])
    if complete(results, run_args):
        return "ok (done earlier)"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    network = os.path.join(nets, cell["network"])
    dataset = cell["dataset"].replace("{nets}", os.path.abspath(nets))
    cf = [sys.executable, "-m", "constraintflow.cli"]
    source = os.path.join(ROOT, cell["program"])
    steps = [("scepter", cf + ["scepter", source, "--network", network, "--dataset", dataset]
              + shlex.split(cell["jit_args"]) + ["--output-path", program + "/"])]
    if cell["row_chunks_args"]:
        steps.append(("row_chunks", [sys.executable, "-m", "constraintflow.compiler.optimizations.row_chunks", program,
                                     "--network", network, "--dataset", dataset] + shlex.split(cell["row_chunks_args"])))
    steps.append(("run", cf + ["run", source, "--network", network, "--dataset", dataset] + run_args
                  + ["--results-file", results, "--output-path", program]))
    # Fresh Inductor/Triton caches per cell, as in the paper: compilation happens in the untimed warmup.
    env = dict(os.environ, **ENV, PYTHONPATH=os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")])))
    if not keep_caches:
        env.update(TORCHINDUCTOR_CACHE_DIR=os.path.join(d, "inductor"), TRITON_CACHE_DIR=os.path.join(d, "triton"))
    for name, cmd in steps:
        start = time.perf_counter()
        with open(os.path.join(d, f"{name}.log"), "w") as f:
            f.write(shlex.join(cmd) + "\n\n")
            f.flush()
            proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        log(f"  {name}: {time.perf_counter() - start:.0f} s")
        if proc.returncode:
            return f"{name} failed (exit {proc.returncode}), see {d}/{name}.log"
    if not keep_caches:
        for c in ("inductor", "triton"):
            shutil.rmtree(os.path.join(d, c), ignore_errors=True)
    return "ok"


def summarize(cell, out, status):
    row = {"certifier": cell["certifier"], "config": cell["config"], "status": status}
    results = os.path.join(out, cell["certifier"], cell["config"], "cf.jsonl")
    if not status.startswith("ok") or not os.path.exists(results):
        return row
    with open(results) as f:
        batches = [r for r in map(json.loads, f) if r["kind"] == "batch"]
    batch_size = int(option(shlex.split(cell["run_args"]), "--batch-size"))
    verified = sum(all(v > 0 for v in margins) for b in batches for margins in b["lower"])
    row.update(images=len(batches) * batch_size,
               time_per_image_s=statistics.mean(b["time_s"] for b in batches) / batch_size,
               peak_gpu_bytes=max(b["peak_bytes"] for b in batches), verified=verified)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nets", default="nets", help="networks folder written by download.py (default: ./nets)")
    p.add_argument("--out", default="reproduce_results", help="output folder (default: ./reproduce_results)")
    p.add_argument("--certifiers", default="", help="comma list (default: all), e.g. deeppoly_add,ibp")
    p.add_argument("--configs", default="", help="comma list of network IDs (default: all), e.g. N1,N33")
    p.add_argument("--keep-caches", action="store_true", help="reuse Inductor's default caches instead of fresh ones")
    args = p.parse_args()
    todo = cells(set(filter(None, args.certifiers.split(","))), set(filter(None, args.configs.split(","))))
    if not todo:
        p.error("no cell matches --certifiers/--configs")
    rows = []
    for i, cell in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {cell['certifier']} {cell['config']}", flush=True)
        status = run_cell(cell, args.nets, args.out, args.keep_caches, lambda m: print(m, flush=True))
        rows.append(summarize(cell, args.out, status))
        print(f"  {status}", flush=True)
    fields = ["certifier", "config", "status", "images", "time_per_image_s", "peak_gpu_bytes", "verified"]
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.join(args.out, 'summary.csv')}")


if __name__ == "__main__":
    main()
