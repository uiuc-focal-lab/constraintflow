import csv
import datetime
import os
import random
import re
import shlex
import subprocess
import sys
import ast
from typing import Tuple

import torch
import typer

# Shared with the experiment drivers, so it lives outside this repo (/home/db50/bench).
_BENCH_DIR = os.environ.get(
    "CF_BENCH_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "bench"))
sys.path.insert(0, os.path.dirname(os.path.abspath(_BENCH_DIR)))
from bench import configs as bench_configs

app = typer.Typer(help="ConstraintFlow JIT test")

# Logs live next to this file (not under the cwd) so they land in the same place
# no matter which directory the test was launched from.
LOG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
# Set by _init_logging() to logs/<label>_<timestamp>/, so each invocation gets its own
# folder. None means logging is off, which keeps the CLI helpers usable if imported.
CURRENT_LOG_DIR = None
_LOG_CONTEXT = "cli"
_LOG_SEQ = 0

# The terminal command that started this test run, recorded in every log it writes.
INVOCATION_CMD = shlex.join([sys.executable, *sys.argv])

# One identifier per invocation, stamped into the log folder name, the sweep CSV filename,
# and every CSV row's `run_id` column, so a result row can always be traced back to the
# logs that produced it. Timestamp sorts; the name half is what's memorable.
_RUN_ADJECTIVES = ["amber", "basalt", "cobalt", "dusky", "ember", "flinty", "gilded",
                   "hazel", "indigo", "jasper", "kelpy", "lucent", "murky", "nimbus",
                   "opal", "quartz"]
_RUN_NOUNS = ["lynx", "heron", "otter", "falcon", "marten", "kestrel", "badger", "ibis",
              "shrike", "tapir", "gannet", "civet", "osprey", "serval", "curlew", "vireo"]


def _new_run_id() -> str:
    return (f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{random.choice(_RUN_ADJECTIVES)}-{random.choice(_RUN_NOUNS)}")


RUN_ID = _new_run_id()


def _init_logging(label: str, csv_path: str = "", *, append_run_id: bool = True) -> None:
    """Open a fresh log folder for this invocation and record how it was launched."""
    global CURRENT_LOG_DIR
    folder = f"{label}_{RUN_ID}" if append_run_id else label
    CURRENT_LOG_DIR = os.path.join(LOG_ROOT, folder)
    if not append_run_id:
        problems = []
        if os.path.exists(CURRENT_LOG_DIR):
            problems.append(f"log dir already exists: {CURRENT_LOG_DIR}")
        if csv_path and os.path.exists(csv_path):
            problems.append(f"CSV already exists: {csv_path}")
        if problems:
            raise SystemExit(
                "Refusing to overwrite output from a previous run with the same name:\n  "
                + "\n  ".join(problems) +
                "\nPick a different --csv-path, or move/delete the existing file(s)/folder.")
    os.makedirs(CURRENT_LOG_DIR, exist_ok=True)
    with open(os.path.join(CURRENT_LOG_DIR, "invocation.txt"), "w") as f:
        f.write(f"# run id: {RUN_ID}\n"
                f"# terminal command: {INVOCATION_CMD}\n"
                f"# cwd: {os.getcwd()}\n"
                f"# started: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                + (f"# csv: {csv_path}\n" if csv_path else ""))
    typer.echo(f"Run id: {RUN_ID}")
    typer.echo(f"Logging this run to {CURRENT_LOG_DIR}")


def _set_log_context(context: str) -> None:
    """Tag subsequent log filenames with the config/program they belong to."""
    global _LOG_CONTEXT
    _LOG_CONTEXT = re.sub(r"[^A-Za-z0-9._-]+", "_", context).strip("_") or "cli"


def _save_log(cmd: list[str], result: subprocess.CompletedProcess) -> None:
    """Write one subprocess's stdout/stderr, each headed by the command that produced it.

    `subprocess command` is copy-pasteable after `cd`-ing to `cwd`; `terminal command`
    is how this test script itself was launched. Called before the returncode check so
    failed runs are logged too.
    """
    if CURRENT_LOG_DIR is None:
        return
    global _LOG_SEQ
    _LOG_SEQ += 1
    # cli.py's subcommand ('run'/'jit'/'compile') is the argument right after the script.
    phase = cmd[2] if len(cmd) > 2 else "cli"
    base = os.path.join(CURRENT_LOG_DIR, f"{_LOG_SEQ:03d}_{_LOG_CONTEXT}_{phase}")
    header = (
        f"# run id: {RUN_ID}\n"
        f"# terminal command: {INVOCATION_CMD}\n"
        f"# subprocess command: {shlex.join(cmd)}\n"
        f"# cwd: {os.getcwd()}\n"
        f"# exit code: {result.returncode}\n"
        "# " + "-" * 70 + "\n\n"
    )
    with open(f"{base}_stdout.txt", "w") as f:
        f.write(header + result.stdout)
    with open(f"{base}_stderr.txt", "w") as f:
        f.write(header + result.stderr)

def _extract_bound(output: str, label: str) -> str:
    # Bounds can span multiple lines, so use DOTALL and non-greedy capture.
    match = re.search(rf"{label}:\s*(tensor\(.*?\))", output, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find '{label}' in CLI output.")
    return match.group(1)


def _parse_tensor(tensor_text: str) -> torch.Tensor:
    # Parse CLI output like: tensor([[...]], dtype=torch.float64) without eval.
    if not tensor_text.startswith("tensor(") or not tensor_text.endswith(")"):
        raise ValueError(f"Unexpected tensor format: {tensor_text}")

    inner = tensor_text[len("tensor("):-1]
    dtype_match = re.search(r",\s*dtype=torch\.([a-zA-Z0-9_]+)\s*$", inner, re.DOTALL)

    dtype = None
    if dtype_match:
        dtype_name = dtype_match.group(1)
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported dtype in output: torch.{dtype_name}")
        data_text = inner[:dtype_match.start()]
    else:
        data_text = inner

    data = ast.literal_eval(data_text.strip())
    return torch.tensor(data, dtype=dtype) if dtype is not None else torch.tensor(data)


def _extract_peak_memory(output: str):
    # cli.py prints "Peak GPU memory: <int> bytes" on CUDA and
    # "Peak CPU memory: <int> bytes" (process peak RSS) on CPU. The compile-only
    # paths (`compile`, `jit`) print "Peak CPU memory" (host RSS) too.
    match = re.search(r"Peak (?:GPU|CPU) memory:\s*([0-9]+)\s*bytes", output)
    if not match:
        return None
    return int(match.group(1))


def _extract_total_time(output: str):
    # cli.py prints "Total time: <float> seconds" from run, compile, and jit.
    match = re.search(r"Total time:\s*([0-9.]+)\s*seconds", output)
    if not match:
        return None
    return float(match.group(1))


def _fail(cmd: list[str], result: subprocess.CompletedProcess) -> RuntimeError:
    return RuntimeError(
        "CLI command failed.\n"
        f"Command: {' '.join(cmd)}\n"
        f"Exit code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def _run_cli(program_file: str, network: str, dataset: str, extra_args: list[str], compile=True, device: str = "cpu"):
    cmd = [
        sys.executable,
        "constraintflow/cli.py",
        "run",
        program_file,
        "--network",
        network,
        "--dataset",
        dataset,
        "--device",
        device,
        *extra_args,
    ]

    if compile:
        cmd.append("--compile")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    _save_log(cmd, result)
    if result.returncode != 0:
        raise _fail(cmd, result)
    print(result.stdout)
    peak_mem = _extract_peak_memory(result.stdout)
    lower = _parse_tensor(_extract_bound(result.stdout, "Lower bounds"))
    upper = _parse_tensor(_extract_bound(result.stdout, "Upper bounds"))
    return lower, upper, peak_mem


def _run_jit(program_file: str, network: str, dataset: str, extra_args: list[str], device: str = "cpu"):
    """Run the whole simulacrum+reuse pipeline in one compile pass via `jit`.
    """
    cmd = [
        sys.executable,
        "constraintflow/cli.py",
        "jit",
        program_file,
        "--network",
        network,
        "--dataset",
        dataset,
        "--device",
        device,
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    _save_log(cmd, result)
    if result.returncode != 0:
        raise _fail(cmd, result)
    print(result.stdout)


def _run_and_capture(cli_args: list[str]) -> str:
    """Invoke `constraintflow/cli.py <cli_args>` and return its stdout.

    Used by profiling to drive the `compile`, `jit`, and `run` subcommands and
    parse the uniform 'Total time' / 'Peak ... memory' lines each of them prints.
    """
    cmd = [sys.executable, "constraintflow/cli.py", *cli_args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    _save_log(cmd, result)
    if result.returncode != 0:
        raise _fail(cmd, result)
    print(result.stdout)
    return result.stdout


def _assert_bounds_close(lhs: torch.Tensor, rhs: torch.Tensor, name: str) -> None:
    if lhs.shape != rhs.shape:
        raise AssertionError(f"{name} shape mismatch: {lhs.shape} != {rhs.shape}")
    if not torch.allclose(lhs, rhs, atol=1e-3, rtol=0):
        raise AssertionError(f"{name} mismatch.\nLeft:\n{lhs}\nRight:\n{rhs}")


def _bounds_match(lb_a, ub_a, lb_b, ub_b) -> bool:
    try:
        _assert_bounds_close(lb_a, lb_b, "Lower bounds")
        _assert_bounds_close(ub_a, ub_b, "Upper bounds")
        return True
    except AssertionError:
        return False


def _format_mem(mem) -> str:
    if mem is None:
        return "N/A"
    return f"{mem / (1024 * 1024):.2f} MB"


def _format_time(t) -> str:
    if t is None:
        return "N/A"
    return f"{t:.2f} s"


def _format_ratio(num, den) -> str:
    if num is None or den is None or den == 0:
        return "N/A"
    return f"{num / den:.2f}x"


def _avg(values):
    """Mean of the non-None measurements, or None if there are none."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _format_bounds_for_csv(t: torch.Tensor) -> str:
    # .tolist() first: torch's own repr/str truncates large tensors (summarizes
    # with "..."), which would silently drop the data we're trying to preserve.
    return str(t.tolist())


def _kernel_paths(program_path: str, network_path: str, batch_size: int, config_id: str = None):
    """The two prebuilt kernels a profiling row runs: Normal `compile` vs `jit`."""
    only = f" --configs {config_id}" if config_id else \
        f" --network {network_path} --batch-size {batch_size}"
    return (bench_configs.require_kernel(
                bench_configs.kernel_dir("normal", program_path),
                "python3 bench/build_kernels.py --tool normal"),
            bench_configs.require_kernel(
                bench_configs.kernel_dir("jit", program_path, network_path, batch_size),
                f"python3 bench/build_kernels.py --tool jit{only}"))


def _profile_one_config(program_file: str, program_path: str, config_id: str, network_name: str,
                         network_path: str, dataset: str, eps: float, batch_size: int, device: str,
                         repeat: int, in_memory: bool = False, use_cache: bool = False) -> list[dict]:
    """Run one (program, network config) through Normal vs JIT compile+run `repeat` times and
    return one row per phase (Compile, Run) with raw time/memory measurements, for CSV export.
    """
    jit_extra = []  # jit always keeps captures in memory
    normal_out, jit_out = _kernel_paths(program_path, network_path, batch_size, config_id) \
        if use_cache else (None, None)
    run_args = [
        program_path,
        "--network", network_path,
        "--dataset", dataset,
        "--device", device,
        "--eps", str(eps),
        "--batch-size", str(batch_size),
    ]
    normal_run = ["run", *run_args] + (["--output-path", normal_out] if normal_out else [])
    jit_run = ["run", *run_args] + (["--output-path", jit_out] if jit_out else [])

    nc_t, nc_m, nr_t, nr_m = [], [], [], []
    jc_t, jc_m, jr_t, jr_m = [], [], [], []
    match = None
    for _ in range(repeat):
        # --- Normal path: compile (device-agnostic codegen), then run
        if not use_cache:
            out = _run_and_capture(["compile", program_path, "--output-path", "output/"])
            nc_t.append(_extract_total_time(out)); nc_m.append(_extract_peak_memory(out))

        out = _run_and_capture(normal_run)
        nr_t.append(_extract_total_time(out)); nr_m.append(_extract_peak_memory(out))
        n_lb = _parse_tensor(_extract_bound(out, "Lower bounds"))
        n_ub = _parse_tensor(_extract_bound(out, "Upper bounds"))

        # --- JIT path: one compile pass (probe + reuse compile), then run
        if not use_cache:
            out = _run_and_capture(["jit", *run_args, *jit_extra])
            jc_t.append(_extract_total_time(out)); jc_m.append(_extract_peak_memory(out))

        out = _run_and_capture(jit_run)
        jr_t.append(_extract_total_time(out)); jr_m.append(_extract_peak_memory(out))
        j_lb = _parse_tensor(_extract_bound(out, "Lower bounds"))
        j_ub = _parse_tensor(_extract_bound(out, "Upper bounds"))

        # Bounds are deterministic, so decide the match once.
        if match is None:
            match = "Yes" if _bounds_match(n_lb, n_ub, j_lb, j_ub) else "No"

    base = {
        "run_id": RUN_ID,
        "config_id": config_id, "program": program_file, "network": network_name,
        "dataset": dataset, "batch": batch_size, "eps": eps, "device": device, "repeats": repeat,
    }
    n_compile_time, j_compile_time = _avg(nc_t), _avg(jc_t)
    n_compile_mem, j_compile_mem = _avg(nc_m), _avg(jc_m)
    n_run_time, j_run_time = _avg(nr_t), _avg(jr_t)
    n_run_mem, j_run_mem = _avg(nr_m), _avg(jr_m)

    empty_bounds = {"n_lower_bounds": "", "n_upper_bounds": "", "j_lower_bounds": "", "j_upper_bounds": ""}
    run_bounds = empty_bounds
    if match == "No":
        run_bounds = {
            "n_lower_bounds": _format_bounds_for_csv(n_lb),
            "n_upper_bounds": _format_bounds_for_csv(n_ub),
            "j_lower_bounds": _format_bounds_for_csv(j_lb),
            "j_upper_bounds": _format_bounds_for_csv(j_ub),
        }

    return [
        {**base, "phase": "Compile",
         "n_time": n_compile_time, "j_time": j_compile_time,
         "n_mem": n_compile_mem, "j_mem": j_compile_mem,
         "speedup_normal_over_jit": _ratio(n_compile_time, j_compile_time),
         "mem_overhead_jit_over_normal": _ratio(j_compile_mem, n_compile_mem),
         "match": "", **empty_bounds},
        {**base, "phase": "Run",
         "n_time": n_run_time, "j_time": j_run_time,
         "n_mem": n_run_mem, "j_mem": j_run_mem,
         "speedup_normal_over_jit": _ratio(n_run_time, j_run_time),
         "mem_overhead_jit_over_normal": _ratio(j_run_mem, n_run_mem),
         "match": match, **run_bounds},
    ]


def _export_profile_csv(rows: list[dict], csv_path: str) -> None:
    """Append rows to csv_path, writing the header only if the file doesn't exist yet.

    Opens, writes, and closes on every call (rather than holding the file handle open
    across a whole sweep) so each call's rows are flushed to disk immediately -- callers
    can call this once per config to make a long sweep incrementally durable/tailable.
    """
    if not rows:
        typer.echo("No results to write (every config was skipped or failed).")
        return
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    typer.echo(f"  -> {len(rows)} row(s) appended to {csv_path}")


def _print_profile_table(rows: list[dict], device: str, repeat: int) -> None:
    headers = [
        "Program", "Batch", "Eps", "Phase",
        "Normal Time", "JIT Time", "Speedup (N/J)",
        "Normal Mem", "JIT Mem", "Mem Ovhd (J/N)",
        "Match?",
    ]
    table_rows = [
        [
            r["program"],
            str(r["batch"]),
            str(r["eps"]),
            r["phase"],
            _format_time(r["n_time"]),
            _format_time(r["j_time"]),
            _format_ratio(r["n_time"], r["j_time"]),
            _format_mem(r["n_mem"]),
            _format_mem(r["j_mem"]),
            _format_ratio(r["j_mem"], r["n_mem"]),
            r["match"],
        ]
        for r in rows
    ]

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    sep = "-+-".join("-" * w for w in widths)
    print()
    print(f"Profiling: Normal vs JIT compile & run (time and peak memory), device={device}, averaged over {repeat} run(s)")
    print("Note: run memory is the device peak (CPU process RSS or GPU CUDA); compile memory is host process RSS.")
    print(fmt_row(headers))
    print(sep)
    for row in table_rows:
        print(fmt_row(row))
    print()


def _run_profile_mode(program_files: str, network: str, dataset: str, device: str, repeat: int,
                       in_memory: bool = False, use_cache: bool = False) -> None:
    jit_extra = []  # jit always keeps captures in memory
    epss = [0]
    batch_sizes = [10]
    rows = []
    for program_file in program_files.split(","):
        _set_log_context(os.path.splitext(program_file)[0])
        program_path = "examples/compiler_examples/" + program_file
        print(f"Profiling program file: {program_path} with network: {network}, dataset: {dataset}, device: {device}, repeat: {repeat}")
        for batch_size in batch_sizes:
            normal_out, jit_out = _kernel_paths(program_path, network, batch_size) \
                if use_cache else (None, None)
            for eps in epss:
                run_args = [
                    program_path,
                    "--network", network,
                    "--dataset", dataset,
                    "--device", device,
                    "--eps", str(eps),
                    "--batch-size", str(batch_size),
                ]
                normal_run = ["run", *run_args] + (["--output-path", normal_out] if normal_out else [])
                jit_run = ["run", *run_args] + (["--output-path", jit_out] if jit_out else [])

                nc_t, nc_m, nr_t, nr_m = [], [], [], []
                jc_t, jc_m, jr_t, jr_m = [], [], [], []
                match = None
                for _ in range(repeat):
                    # --- Normal path: compile (device-agnostic codegen), then run
                    if not use_cache:
                        out = _run_and_capture(["compile", program_path, "--output-path", "output/"])
                        nc_t.append(_extract_total_time(out)); nc_m.append(_extract_peak_memory(out))

                    out = _run_and_capture(normal_run)
                    nr_t.append(_extract_total_time(out)); nr_m.append(_extract_peak_memory(out))
                    n_lb = _parse_tensor(_extract_bound(out, "Lower bounds"))
                    n_ub = _parse_tensor(_extract_bound(out, "Upper bounds"))

                    # --- JIT path: one compile pass (probe + reuse compile), then run
                    if not use_cache:
                        out = _run_and_capture(["jit", *run_args, *jit_extra])
                        jc_t.append(_extract_total_time(out)); jc_m.append(_extract_peak_memory(out))

                    out = _run_and_capture(jit_run)
                    jr_t.append(_extract_total_time(out)); jr_m.append(_extract_peak_memory(out))
                    j_lb = _parse_tensor(_extract_bound(out, "Lower bounds"))
                    j_ub = _parse_tensor(_extract_bound(out, "Upper bounds"))

                    # Bounds are deterministic, so decide the match once.
                    if match is None:
                        match = "Yes" if _bounds_match(n_lb, n_ub, j_lb, j_ub) else "No"

                base = {"program": program_file, "batch": batch_size, "eps": eps}
                # One row per phase, Normal and JIT side by side; match applies to Run only.
                rows.append({**base, "phase": "Compile",
                             "n_time": _avg(nc_t), "j_time": _avg(jc_t),
                             "n_mem": _avg(nc_m), "j_mem": _avg(jc_m), "match": ""})
                rows.append({**base, "phase": "Run",
                             "n_time": _avg(nr_t), "j_time": _avg(jr_t),
                             "n_mem": _avg(nr_m), "j_mem": _avg(jr_m), "match": match})

    _print_profile_table(rows, device, repeat)


def _echo_context(program_file: str, network: str, dataset: str, device: str, batch_size: int, eps: float) -> None:
    typer.echo(f"Program file: {program_file}")
    typer.echo(f"Network: {network}")
    typer.echo(f"Dataset: {dataset}")
    typer.echo(f"Device: {device}")
    typer.echo(f"Batch size: {batch_size}")
    typer.echo(f"Eps: {eps}")


@app.command()
def test(
    program_files: str = typer.Argument(..., help="Comma-separated list of ConstraintFlow program files to test"),
    network: str = typer.Argument(..., help="Network path/name"),
    dataset: str = "mnist",
    device: str = typer.Option("gpu", help="Device to run on: cpu, gpu (CUDA), or gpumac (Apple MPS)."),
    profile: bool = typer.Option(False, "--profile", help="Profile compile & run time and peak memory for the Normal and JIT paths and print a table."),
    repeat: int = typer.Option(1, help="In --profile mode, run each configuration this many times and average the time and memory."),
    in_memory: bool = typer.Option(False, "--in-memory", help="Run the jit compile with --in-memory (keep captures in a process-local dict instead of on disk)."),
    use_cache: bool = typer.Option(False, "--use-cache", help="In --profile mode, run kernels prebuilt by bench/build_kernels.py instead of compiling. The Compile row is then blank."),
):
    if profile:
        _init_logging(f"profile_{dataset}_{device}")
        _run_profile_mode(program_files, network, dataset, device, repeat, in_memory, use_cache)
        return

    # `test` still compiles in place on purpose: it compares a fresh normal compile
    # against a fresh jit compile.
    _init_logging(f"test_{dataset}_{device}")
    jit_extra = []  # jit always keeps captures in memory

    for program_file in program_files.split(","):
        _set_log_context(os.path.splitext(program_file)[0])
        program_file = "examples/compiler_examples/" + program_file
        print(f"Testing program file: {program_file} with network: {network}, dataset: {dataset}, device: {device}")
        epss = [0.005]
        batch_sizes = [100]
        for batch_size in batch_sizes:
            batch_args = ["--batch-size", str(batch_size)]

            # --- Baseline: normal compile + run for each eps. ---
            baseline_lbs, baseline_ubs = [], []
            for eps in epss:
                try:
                    baseline_lb, baseline_ub, _ = _run_cli(
                        program_file, network, dataset, ["--eps", str(eps), *batch_args], device=device
                    )
                except (AssertionError, RuntimeError) as e:
                    typer.echo("Baseline run failed")
                    _echo_context(program_file, network, dataset, device, batch_size, eps)
                    raise e
                baseline_lbs.append(baseline_lb)
                baseline_ubs.append(baseline_ub)

            # --- Simulacrum+reuse: ONE compile pass. 
            try:
                _run_jit(program_file, network, dataset, ["--eps", str(epss[0]), *batch_args, *jit_extra], device=device)
            except (AssertionError, RuntimeError) as e:
                typer.echo("Simulacrum (jit) compile pass failed")
                _echo_context(program_file, network, dataset, device, batch_size, epss[0])
                raise e

            reuse_lbs, reuse_ubs = [], []
            for eps in epss:
                try:
                    reuse_lb, reuse_ub, _ = _run_cli(
                        program_file, network, dataset, ["--eps", str(eps), *batch_args], compile=False, device=device
                    )
                except (AssertionError, RuntimeError) as e:
                    typer.echo("Reuse run failed")
                    _echo_context(program_file, network, dataset, device, batch_size, eps)
                    raise e
                reuse_lbs.append(reuse_lb)
                reuse_ubs.append(reuse_ub)

            # --- Compare baseline vs reuse bounds for each eps. ---
            for eps, baseline_lb, baseline_ub, reuse_lb, reuse_ub in zip(
                epss, baseline_lbs, baseline_ubs, reuse_lbs, reuse_ubs
            ):
                try:
                    _assert_bounds_close(baseline_lb, reuse_lb, "Lower bounds baseline vs reuse")
                    _assert_bounds_close(baseline_ub, reuse_ub, "Upper bounds baseline vs reuse")
                except AssertionError as e:
                    typer.echo(f"AssertionError: {e}")
                    typer.echo(f"Baseline LB: {baseline_lb}")
                    typer.echo(f"Reuse LB: {reuse_lb}")
                    typer.echo(f"Baseline UB: {baseline_ub}")
                    typer.echo(f"Reuse UB: {reuse_ub}")
                    _echo_context(program_file, network, dataset, device, batch_size, eps)
                    raise e

                typer.echo(f"JIT test passed: baseline and jit modes bounds match for eps={eps} and batch_size={batch_size} on {device}.")

    print("JIT test passed: all eps and batch sizes passed.")


@app.command()
def sweep(
    program_files: str = typer.Argument(..., help="Comma-separated list of ConstraintFlow program files to profile"),
    dataset: str = typer.Option("all", help="Restrict the sweep to networks from this dataset: mnist, cifar, or all."),
    config_ids: str = typer.Option(None, "--configs", help="Comma-separated config ids to run, e.g. N1,N12 (default: all)."),
    device: str = typer.Option("gpu", help="Device to run on: cpu, gpu (CUDA), or gpumac (Apple MPS)."),
    repeat: int = typer.Option(1, help="Run each configuration this many times and average the time and memory."),
    in_memory: bool = typer.Option(False, "--in-memory", help="Run the jit compile with --in-memory (keep captures in a process-local dict instead of on disk)."),
    use_cache: bool = typer.Option(False, "--use-cache", help="Run kernels prebuilt by bench/build_kernels.py instead of compiling. The Compile rows are then blank."),
    csv_path: str = typer.Option("simulacrum_sweep_results.csv", help="Path to write the CSV results to. Its basename also names the paired log folder; re-running with a path that already has output errors out rather than overwriting it."),
):
    """
    Profile Normal vs JIT compile & run (time and peak memory) across the full model
    config table (network, dataset, eps, batch_size per config, shared with the
    experiment drivers via bench/configs.py). Results are appended to a CSV instead of
    being printed.
    """
    label = f"sweep_{dataset}_{device}_{os.path.splitext(os.path.basename(csv_path))[0]}"
    _init_logging(label, csv_path, append_run_id=False)
    only = [c.strip() for c in config_ids.split(",")] if config_ids else None
    total_rows = 0
    for program_file in program_files.split(","):
        program_path = "examples/compiler_examples/" + program_file
        for (config_id, network_name, config_dataset, eps, batch_size, _method,
             network_path) in bench_configs.iter_configs(dataset, only):
            _set_log_context(f"{config_id}_{os.path.splitext(program_file)[0]}")
            typer.echo(f"Profiling {config_id} ({network_name} on {config_dataset}, device={device}, "
                        f"eps={eps}, batch_size={batch_size}, {repeat} repeat(s))")
            try:
                config_rows = _profile_one_config(
                    program_file, program_path, config_id, network_name, network_path,
                    config_dataset, eps, batch_size, device, repeat, in_memory, use_cache,
                )
            except (AssertionError, RuntimeError) as e:
                typer.echo(f"  [{config_id}] profiling failed, skipping: {e}")
                continue

            # Write this config's rows immediately (rather than batching until the
            # whole sweep finishes) so results survive a crash/interrupt partway
            # through and can be tailed live while the sweep is still running.
            _export_profile_csv(config_rows, csv_path)
            total_rows += len(config_rows)

    if total_rows == 0:
        typer.echo("No results were written (every config was skipped or failed).")
    else:
        typer.echo(f"Done. {total_rows} row(s) written to {csv_path} "
                    f"(run id {RUN_ID}, logs in {CURRENT_LOG_DIR})")


if __name__ == "__main__":
    app()
