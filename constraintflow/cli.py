import os
import gc
import sys
import faulthandler

DISABLE_TF32 = os.environ.get("CF_DISABLE_TF32", "0") not in ("", "0")
if DISABLE_TF32:
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import torch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

#FEAT: --dtype float64 (fp64 precision oracle) must be applied before modules that read torch.float at import.
from constraintflow.lib.precision import dtype_from_argv, use_float64
DTYPE = dtype_from_argv(sys.argv[1:])
if DTYPE not in ("float32", "float64"):
    sys.exit(f"Error: --dtype must be float32 or float64, got {DTYPE!r}")
if DTYPE == "float64":
    use_float64()

import constraintflow.lib.globals as globals

os.environ.setdefault("TORCHINDUCTOR_VEC_ISA_OK", "1")

argv = sys.argv[1:]
globals.dummy_mode.reset_flag()
globals.reuse_mode.reset_flag()
globals.dense_default_mode.set_flag() if "--dense" in argv else globals.dense_default_mode.reset_flag()
_command = argv[0] if argv else ""
globals.inductor_mode.set_flag() if _command in ("jit", "run") and "--no-inductor" not in argv else globals.inductor_mode.reset_flag()
globals.sroa.set_flag()

print(f'dummy_mode in cli: {globals.dummy_mode}')
print(f'reuse_mode in cli: {globals.reuse_mode}')
print(f'inductor_mode in cli: {globals.inductor_mode}')
print(f'tf32 in cli: {"disabled (CF_DISABLE_TF32)" if DISABLE_TF32 else "PyTorch default"}')
#FEAT: report the run dtype.
print(f'dtype in cli: {DTYPE}')


import os
import shutil
import sys
import torch
import typer
import time
import csv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from constraintflow.gbcsr.tensor_ops import binary
from constraintflow.lib.abs_elem import Abs_elem_sparse
from constraintflow.lib.llist import Llist
from constraintflow.lib.flow_sparse import Flow
from constraintflow.lib.globals import *
from constraintflow.lib.globals import dummy_mode, reuse_mode
import constraintflow.lib.globals 
from constraintflow.compiler.compile import compile as _compile
from constraintflow.compiler.optimizations.row_refinement import warmup_row_kernels
from constraintflow.verifier.provesound import provesound as _provesound
from constraintflow.lib.spec import get_network_and_input_spec, vnnlib_path

app = typer.Typer(help="ConstraintFlow CLI for verification and compilation of DSL programs.")

# --------------------------
# Utility Functions
# --------------------------

def get_program(program_file: str) -> str:
    return program_file

def _require_network(network: str) -> None:
    if not os.path.isfile(network):
        typer.echo(f"Error: --network must be the path to an ONNX file; '{network}' does not exist.")
        raise typer.Exit(code=1)


def get_network(network: str, network_format: str, dataset: str) -> str:
    if dataset not in ["mnist", "cifar"]:
        return network
    return network


def _configure_cuda_cpu_threads() -> None:
    """Cap eager ATen CPU parallelism before the first compiled CUDA run."""
    raw_limit = os.environ.get("CF_CPU_THREADS", "8")
    try:
        limit = int(raw_limit)
    except ValueError:
        typer.echo(f"Error: CF_CPU_THREADS must be a positive integer, got {raw_limit!r}.")
        raise typer.Exit(code=1)
    if limit < 1:
        typer.echo(f"Error: CF_CPU_THREADS must be a positive integer, got {raw_limit!r}.")
        raise typer.Exit(code=1)

    torch.set_num_threads(min(torch.get_num_threads(), limit))


def get_dataset(
    batch_size: int,
    dataset: str,
    train: bool = False,
    device: str | torch.device | None = None,
):
    #FEAT: ACAS Xu has no dataset; trace on zero inputs (the box comes from input_box at run time).
    if dataset == "acasxu":
        from constraintflow.lib.spec import ACAS_INPUT_SHAPE
        return (torch.zeros(batch_size, *ACAS_INPUT_SHAPE, device=device),
                torch.zeros(batch_size, dtype=torch.long, device=device))
    #FEAT: vnnlib specs: image i is spec i mod #specs; its label is that spec's index. The image is a placeholder
    # [lo, hi] box: each run parses its specs (lib/spec.py read_vnnlib_batch).
    from constraintflow.lib.spec import vnnlib_specs, vnnlib_path
    if vnnlib_path(dataset):
        specs = vnnlib_specs()
        idx = torch.arange(batch_size) % len(specs.spec_files(vnnlib_path(dataset)))
        image = torch.zeros(batch_size, 2, *specs.input_shape(vnnlib_path(dataset)))
        return image.to(device), idx.to(device)
    #FEAT: box specs likewise trace on zero inputs of the shape named in the dataset.
    from constraintflow.lib.spec import parse_box_dataset
    box = parse_box_dataset(dataset)
    if box is not None:
        return (torch.zeros(batch_size, *box[0], device=device),
                torch.zeros(batch_size, dtype=torch.long, device=device))
    if dataset == "mnist":
        transform = transforms.ToTensor()  # keep 28x28
        data = datasets.MNIST(root=".", train=train, download=True, transform=transform)
    elif dataset == "cifar10" or dataset == "cifar":
        transform = transforms.ToTensor()  # keep 32x32
        data = datasets.CIFAR10(root=".", train=train, download=True, transform=transform)
    #FEAT: CIFAR-100 test set (CIFAR100_TinyImageNet_ResNet benchmark).
    elif dataset == "cifar100":
        data = datasets.CIFAR100(root=".", train=train, download=True, transform=transforms.ToTensor())
    elif dataset == "tinyimagenet":
        #FIX FOR: TinyImageNet came from the unlabeled-order train split with a top-left crop; the benchmark verifies the labeled val split, center-cropped.
        from constraintflow.lib.tinyimagenet import TinyImageNetValidation
        data = TinyImageNetValidation(os.environ.get("CF_TINYIMAGENET_ROOT", "tinyimagenet/tiny-imagenet-200"))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    target_device = torch.device(device) if device is not None else None
    pin_memory = target_device is not None and target_device.type == "cuda"
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    image, label = next(iter(dataloader))
    # ensure labels are a tensor
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    if target_device is not None:
        image = image.to(target_device, non_blocking=pin_memory)
        label = label.to(target_device, non_blocking=pin_memory)
    return image, label



def get_precision(lb):
    verified = (lb >= 0).all(dim=1)
    precision = verified.sum() / verified.shape[0]
    return precision


# --------------------------
# CLI Commands
# --------------------------

@app.command()
def provesound(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    nprev: int = typer.Option(1, help="Number of previous states"),
    nsymb: int = typer.Option(1, help="Number of symbols"),
):
    """
    Prove soundness of a ConstraintFlow program.
    """
    program = get_program(program_file)
    res = _provesound(program, nprev=nprev, nsymb=nsymb)
    typer.echo(f"Provesound result: {res}")


def compile_code(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    output_path: str = typer.Option("output/", help="Output path for generated code"),
):
    """
    Compile a ConstraintFlow program into Python.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        typer.echo(f"Error creating folder '{output_path}': {e}")
        raise typer.Exit(code=1)


    program = get_program(program_file)
    res = _compile(program, output_path)
    if res:
        typer.echo("Compilation successful ✅")
    else:
        typer.echo("Compilation failed ❌")
        raise typer.Exit(code=1)

@app.command()
def compile(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    output_path: str = typer.Option("output/", help="Output path for generated code"),
    fuse_affine_subst: bool = typer.Option(False, "--fuse-affine-subst", help="Two optimizations gated by one flag: (1) skip both concretizing traversals at any Affine layer that feeds only further Affine layers (always sound; single_bound.py). (2) Assert every Affine op's L and U outputs are identical (true for all deeppoly*/crown specs here) and drop the redundant sign-split when a traverse() substitution step crosses an Affine layer -- only affects a jit reuse compile (no-op on plain compile, which never runs tensor_to_block); unsound if the assertion doesn't hold."),
    center_radius: bool = typer.Option(True, "--center-radius/--no-center-radius", help="Concretize sign-selected bounds (c >= 0 ? c*u : c*l, and the final input-box concretization) as c*mid + |c|*rad, as auto_LiRPA does, and fold conv biases into the center conv. --no-center-radius restores original constraintflow's sign split c+*u + c-*l, which loses interval width in fp32 on narrow boxes."),
    #FEAT: torch.compile without specialization (the generic build's Flow.flow), for the specialization ablation.
    compile_flow: bool = typer.Option(False, "--compile-flow", help="Wrap Flow.flow in torch.compile (Inductor, fullgraph=False, the jit build's options) in the generated main.py; transformers.py is unchanged. Run with --inductor."),
):
    start_time = time.perf_counter()
    globals.compile_flow.set_flag() if compile_flow else globals.compile_flow.reset_flag()
    globals.fuse_affine_subst.set_flag() if fuse_affine_subst else globals.fuse_affine_subst.reset_flag()
    globals.center_radius.set_flag() if center_radius else globals.center_radius.reset_flag()
    compile_code(program_file, output_path)
    total_time = time.perf_counter() - start_time
    typer.echo(f"Total time: {total_time:.6f} seconds")
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = maxrss if sys.platform == "darwin" else maxrss * 1024
    typer.echo(f"Peak CPU memory: {peak_bytes} bytes")


@app.command(name="jit")
def simulacrum_compile(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    network: str = typer.Option(..., help="Path to the ONNX network"),
    network_format: str = typer.Option("onnx", help="Network format"),
    dataset: str = typer.Option("mnist", help="Dataset (mnist or cifar)"),
    batch_size: int = typer.Option(1, help="Batch size"),
    eps: float = typer.Option(0, help="Epsilon"),
    train: bool = typer.Option(False, help="Trace on training dataset"),
    no_sparsity: bool = typer.Option(False, help="Disable sparsity optimizations"),
    device: str = typer.Option("cpu", help="Device mode: cpu, gpu (CUDA), or gpumac (Apple MPS)"),
    output_path: str = typer.Option("output/", help="Output path for generated code"),
    dense: bool = typer.Option(False, help="Use dense blocks by default"),
    inductor: bool = typer.Option(True, "--inductor/--no-inductor", help="Compile flow() with torch.compile (Inductor). --no-inductor emits an eager flow(), as the row_chunks rewrite needs. `run` must be given the same choice."),
    fuse_affine_subst: bool = typer.Option(False, "--fuse-affine-subst/--no-fuse-affine-subst", help="Two optimizations gated by one flag: (1) skip both concretizing traversals at any Affine layer that feeds only further Affine layers (always sound; single_bound.py). (2) Assert every Affine op's L and U outputs are identical (true for all deeppoly*/crown specs here) and drop the redundant sign-split when a traverse() substitution step crosses an Affine layer -- unsound if the assertion doesn't hold. Both take effect on the simulacrum and reuse compile passes below. The functional SROA pass also proves and removes matching sign-split convolution pairs across residual branches."),
    #FEAT: block sizes and batch rule for rows selected by row-invariant traversal stops.
    traversal_stop_chunk: str = typer.Option("128", "--traversal-stop-chunk", help="Rows per compiled kernel for traversal-stop selection; a comma list (e.g. 8,16,32,64,128,256,512) compiles one kernel per size and uses the smallest that fits."),
    traversal_stop_per_example: bool = typer.Option(False, "--traversal-stop-per-example", help="Keep each example's own stop decision, as the program states. Default: a row refined for any example takes the refined values for every example (auto_LiRPA's sparse rule)."),
    flow_segment_mb: float = typer.Option(8192.0, "--flow-segment-mb", help="Split the flow into regions with approximately this many MB of named allocations; Inductor compiles each region separately. This is a compilation-region budget, not a bound on total GPU memory. 0 keeps one region (needed by the row_chunks rewrite)."),
    cudagraphs: bool = typer.Option(False, "--cudagraphs/--no-cudagraphs", help="Compile flow() with mode='reduce-overhead' so a run replays one CUDA graph instead of launching each kernel. Cuts per-run launch overhead on small networks. Replays allocate from a private pool set up during warmup, so `run`'s per-run peak (max_memory_allocated) no longer counts flow intermediates. Requires --inductor."),
    center_radius: bool = typer.Option(True, "--center-radius/--no-center-radius", help="Concretize sign-selected bounds (c >= 0 ? c*u : c*l, and the final input-box concretization) as c*mid + |c|*rad, as auto_LiRPA does, and fold conv biases into the center conv. --no-center-radius restores original constraintflow's sign split c+*u + c-*l, which loses interval width in fp32 on narrow boxes."),
    memory_budget_mb: float = typer.Option(-1.0, "--memory-budget-mb", help="When the flow's estimated peak tensor memory exceeds this many MB, reorder it group by group and recompute large values instead of keeping them live (mem_plan). -1 (default) uses 70% of the CUDA device's memory; 0 disables."),
    #FEAT: --dtype on jit, so typer accepts it (applied from argv above).
    dtype: str = typer.Option("float32", "--dtype", help="float32, or float64 to compute every float tensor in fp64 (a precision oracle for bound mismatches; compile and run must agree). Read from argv before the imports; listed here so typer accepts it."),
    #FEAT: certified-training build (constraintflow/lib/training.py).
    training: bool = typer.Option(False, "--training", help="Build for certified training: importable as a package, NaN-free division gradients, margin spec folded into the last Linear."),
):
    """
    Compile a ConstraintFlow program through the whole simulacrum+reuse pipeline
    in one shot.
    """
    start_time = time.perf_counter()
    globals.fuse_affine_subst.set_flag() if fuse_affine_subst else globals.fuse_affine_subst.reset_flag()
    globals.compact_patches.set_flag()
    globals.fused_flow.set_flag()
    globals.sroa.set_flag()
    globals.center_radius.set_flag() if center_radius else globals.center_radius.reset_flag()  # simulacrum and reuse passes alike
    if cudagraphs and not inductor:
        raise typer.BadParameter('--cudagraphs requires --inductor')
    if flow_segment_mb < 0:
        raise typer.BadParameter('--flow-segment-mb must be nonnegative')
    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        typer.echo(f"Error creating folder '{output_path}': {e}")
        raise typer.Exit(code=1)

    _require_network(network)

    globals.dense_default_mode.set_flag() if dense else globals.dense_default_mode.reset_flag()

    valid_devices = {"cpu", "gpu", "gpumac"}
    if device not in valid_devices:
        typer.echo(f"Error: unknown device '{device}'. Choose from: {sorted(valid_devices)}")
        raise typer.Exit(code=1)
    if device == "gpu" and not torch.cuda.is_available():
        typer.echo("Error: device='gpu' requested but CUDA is not available.")
        raise typer.Exit(code=1)
    if device == "gpumac" and not torch.backends.mps.is_available():
        typer.echo("Error: device='gpumac' requested but MPS is not available.")
        raise typer.Exit(code=1)
    device_mode.set_mode(device)

    is_cuda = device_mode.get_device() == "cuda"

    globals.jit_store_clear()

    globals.training.set_flag() if training else globals.training.reset_flag()
    # Simulacrum.
    globals.inductor_mode.reset_flag()
    globals.dummy_mode.set_flag()
    globals.reuse_mode.reset_flag()
    compile_code(program_file, output_path)

    sys.path.insert(0, os.path.abspath(output_path))
    network_file = get_network(network, network_format, dataset)

    from main import run as _probe_run  # probe build provides this

    dataset_device = device_mode.get_device() if is_cuda else None
    X, y = get_dataset(batch_size, dataset, train=train, device=dataset_device)
    _probe_run(
        network_file,
        batch_size,
        eps,
        X,
        y,
        dataset=dataset,
        train=train,
        print_intermediate_results=False,
        no_sparsity=no_sparsity,
    )

    if not globals.capture_exists("jit_layers/layers.json"):
        typer.echo(
            "Error: simulacrum pass did not produce a trace in the in-memory store. "
            "Cannot proceed to the reuse compile."
        )
        raise typer.Exit(code=1)

    # Reuse
    globals.dummy_mode.reset_flag()
    globals.reuse_mode.set_flag()
    if inductor:
        globals.inductor_mode.set_flag()
    globals.early_reductions.set_flag()
    globals.fuse_sign_convs.set_flag()
    globals.cudagraphs.set_flag() if cudagraphs else globals.cudagraphs.reset_flag()
    globals.product_grids.set_flag()
    globals.flow_segment_mb.set_value(flow_segment_mb)
    globals.memory_budget_mb.set_value(memory_budget_mb)
    globals.set_network_path(network_file)
    stop_chunk = [int(c) for c in traversal_stop_chunk.split(',')]
    globals.traversal_stop_config = (stop_chunk[0] if len(stop_chunk) == 1 else tuple(stop_chunk), not traversal_stop_per_example)
    try:
        compile_code(program_file, output_path)
    finally:
        globals.reuse_mode.reset_flag()
        globals.inductor_mode.reset_flag()
        globals.fused_flow.reset_flag()
        globals.compact_patches.reset_flag()
        globals.early_reductions.reset_flag()
        globals.fuse_sign_convs.reset_flag()
        globals.cudagraphs.reset_flag()
        globals.training.reset_flag()
        globals.product_grids.reset_flag()
        globals.center_radius.set_flag()   # back to the default
        globals.flow_segment_mb.set_value(0.0)
        globals.memory_budget_mb.reset()
        globals.set_network_path(None)
        globals.traversal_stop_config = (128, True)
        globals.jit_store_clear()

    typer.echo("Simulacrum+reuse compile complete ✅")
    typer.echo(f"Optimized code written to: {os.path.abspath(output_path)}")
    if is_cuda:
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    typer.echo(f"Total time: {total_time:.6f} seconds")
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = maxrss if sys.platform == "darwin" else maxrss * 1024
    typer.echo(f"Peak CPU memory: {peak_bytes} bytes")


@app.command()
def run(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    network: str = typer.Option(..., help="Path to the ONNX network"),
    network_format: str = typer.Option("onnx", help="Network format"),
    dataset: str = typer.Option("mnist", help="Dataset (mnist or cifar)"),
    batch_size: int = typer.Option(1, help="Batch size"),
    eps: float = typer.Option(0.01, help="Epsilon"),
    train: bool = typer.Option(False, help="Run on training dataset"),
    print_intermediate_results: bool = False,
    no_sparsity: bool = typer.Option(False, help="Disable sparsity optimizations"),
    device: str = typer.Option("cpu", help="Device mode: cpu, gpu (CUDA), or gpumac (Apple MPS)"),
    output_path: str = typer.Option("output/", help="Path where compiled program is stored"),
    compile: bool = typer.Option(False, help="Run compilation before execution"),
    dense: bool = typer.Option(False, help="Use dense blocks by default"),
    inductor: bool = typer.Option(True, "--inductor/--no-inductor", help="Run with PyTorch Inductor; must match how the program was built (jit --inductor/--no-inductor; the generic `compile` build runs with --no-inductor unless built with --compile-flow)."),
    warmup: int = typer.Option(0, help="Number of warmup runs on different data before the timed run"),
    repeat: int = typer.Option(1, "--repeat", help="Number of timed runs, each reported separately. Independent of --warmup: the warmup runs (if any) still happen once, before the first timed run."),
    num_images: int = typer.Option(0, "--num-images", help="Verify this many test images, batch-size at a time, one timed run per batch (batch i = images [i*batch_size, (i+1)*batch_size)); warmup j uses the images right after them. Requires --results-file; replaces --repeat. 0 (default): the --repeat protocol, every timed run on the first batch."),
    results_file: str = typer.Option(None, "--results-file", help="With --num-images: write one JSON line per warmup and per timed batch (time, peak/resident memory, bounds) here, outside the timer."),
    #FEAT: --dtype on run, so typer accepts it (applied from argv above).
    dtype: str = typer.Option("float32", "--dtype", help="float32, or float64 to compute every float tensor in fp64 (a precision oracle for bound mismatches; compile and run must agree). Read from argv before the imports; listed here so typer accepts it."),
    #FEAT: --profile-comm records CPU-GPU communication per timed batch (lib/comm_profile.py).
    profile_comm: bool = typer.Option(False, "--profile-comm", help="With --num-images: profile each timed batch with torch.profiler and add syncs, alloc_syncs, gpu_idle_pct and sync_calls to its results line; its time goes to profiled_time_s instead of time_s. Warmups are not profiled."),
    #FEAT: per-batch time limit.
    batch_timeout: float = typer.Option(0.0, "--batch-timeout", min=0, help="With --num-images: end the process if one timed batch runs longer than this many seconds (faulthandler prints 'Timeout' and every thread's stack). Warmups: --warmup-timeout. 0: no limit."),
    #FEAT: per-warmup time limit.
    warmup_timeout: float = typer.Option(0.0, "--warmup-timeout", min=0, help="With --num-images: end the process if one warmup run takes longer than this many seconds (faulthandler prints 'Timeout' and every thread's stack). 0: no limit."),
):
    """
    Run a compiled ConstraintFlow program.
    """
    _require_network(network)

    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        typer.echo(f"Error creating folder '{output_path}': {e}")
        raise typer.Exit(code=1)

    if compile:
        compile_code(program_file, output_path)

    valid_devices = {"cpu", "gpu", "gpumac"}
    if device not in valid_devices:
        typer.echo(f"Error: unknown device '{device}'. Choose from: {sorted(valid_devices)}")
        raise typer.Exit(code=1)
    if device == "gpu" and not torch.cuda.is_available():
        typer.echo("Error: device='gpu' requested but CUDA is not available.")
        raise typer.Exit(code=1)
    if device == "gpumac" and not torch.backends.mps.is_available():
        typer.echo("Error: device='gpumac' requested but MPS is not available.")
        raise typer.Exit(code=1)
    device_mode.set_mode(device)
    if device == "gpu":
        _configure_cuda_cpu_threads()

    if repeat < 1:
        typer.echo("Error: --repeat must be >= 1.")
        raise typer.Exit(code=1)
    if num_images:
        if num_images < 0 or num_images % batch_size:
            typer.echo(f"Error: --num-images ({num_images}) must be a positive multiple of --batch-size ({batch_size}).")
            raise typer.Exit(code=1)
        if not results_file:
            typer.echo("Error: --num-images requires --results-file.")
            raise typer.Exit(code=1)
    elif profile_comm:
        typer.echo("Error: --profile-comm requires --num-images.")
        raise typer.Exit(code=1)

    sys.path.insert(0, os.path.abspath(output_path))
    from main import run  # compiled code provides this

    network_file = get_network(network, network_format, dataset)
    dataset_device = device_mode.get_device() if device == "gpu" else None
    # Images stay on the host; each run moves only its own batch, as the baselines do.
    X, y = get_dataset(
        num_images + warmup * batch_size if num_images else batch_size * (warmup + 1),
        dataset,
        train=train,
        device=None,
    )

    def _batch(lo):
        xs, ys = X[lo : lo + batch_size], y[lo : lo + batch_size]
        if dataset_device is None:
            return xs, ys
        #FIX FOR: VNN-LIB labels (spec indices) went to the GPU only to be read back by y.tolist() (1 H2D + 1 D2H per batch).
        if vnnlib_path(dataset):
            return xs.to(dataset_device), ys
        return xs.to(dataset_device), ys.to(dataset_device)

    is_cuda = device_mode.get_device() == "cuda"
    print(f"WARMMMMMM")
    torch._dynamo.reset()

    if num_images:
        _run_image_batches(run, _batch, network_file, batch_size, eps, num_images, warmup,
                           results_file, is_cuda, dataset, train,
                           print_intermediate_results, no_sparsity, profile_comm, batch_timeout, warmup_timeout)
        return

    for i in range(warmup):
        warmup_start = time.perf_counter()
        warmup_row_kernels(run,
            network_file,
            batch_size,
            eps,
            *_batch((i + 1) * batch_size),
            dataset=dataset,
            train=train,
            print_intermediate_results=False,
            no_sparsity=no_sparsity,
        )
        if is_cuda:
            torch.cuda.synchronize()
        typer.echo(f"Warmup run {i + 1}/{warmup}: {time.perf_counter() - warmup_start:.6f} s")
        ### Free Memory ####
        gc.collect()
        # Not needed for the per-run peak (max_memory_allocated counts live tensors only);
        # it only made the next run re-cudaMalloc its whole working set.
        # if is_cuda:
        #     torch.cuda.empty_cache()

    # gc.freeze() after warmup, as every tool does: the untimed gc.collect() between runs then scans
    # only per-run garbage instead of the whole torch/inductor heap.
    gc.collect()
    gc.freeze()

    mem_label = "Peak GPU memory" if is_cuda else "Peak CPU memory"

    def _peak_bytes():
        if is_cuda:
            return torch.cuda.max_memory_allocated()
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return maxrss if sys.platform == "darwin" else maxrss * 1024

    repeat_times = []
    repeat_peaks = []
    print("REPEAT")
    for i in range(repeat):
        resident = 0
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()

        start_time = time.perf_counter()
        lb, ub = run(
            network_file,
            batch_size,
            eps,
            *_batch(0),
            dataset=dataset,
            train=train,
            print_intermediate_results=print_intermediate_results,
            no_sparsity=no_sparsity,
        )
        if is_cuda:
            torch.cuda.synchronize()
        run_time = time.perf_counter() - start_time
        run_peak = _peak_bytes()
        repeat_times.append(run_time)
        repeat_peaks.append(run_peak)
        if repeat > 1:
            resident_note = f", Resident GPU memory: {resident} bytes" if is_cuda else ""
            typer.echo(f"Run {i + 1}/{repeat}: {run_time:.6f} s, {mem_label}: {run_peak} bytes{resident_note}")

        ### Free Memory ####
        lb, ub = lb.detach().cpu(), ub.detach().cpu()
        gc.collect()
        # Not needed for the per-run peak (max_memory_allocated counts live tensors only);
        # it only made the next run re-cudaMalloc its whole working set.
        # if is_cuda:
        #     torch.cuda.empty_cache()

    total_time = sum(repeat_times)
    peak_bytes = max(repeat_peaks)

    # Full float32 precision (9 significant digits) and no '...' elision, so the harness
    # can parse the exact bounds into its CSV.
    torch.set_printoptions(precision=8, sci_mode=True, threshold=sys.maxsize)
    typer.echo(f"Lower bounds: {lb}")
    typer.echo(f"Upper bounds: {ub}")
    torch.set_printoptions(profile="default")
    typer.echo(f"Total time: {total_time:.6f} seconds")
    typer.echo(f"{mem_label}: {peak_bytes} bytes")
def _run_image_batches(run, _batch, network_file, batch_size, eps, num_images, warmup,
                       results_file, is_cuda, dataset, train,
                       print_intermediate_results, no_sparsity, profile_comm=False, batch_timeout=0.0,
                       warmup_timeout=0.0):
    """The --num-images protocol: warmups on the images after the evaluated ones, then one
    timed run per batch. The timed region is the --repeat protocol's (network load and
    build, box, spec, host-to-device copy, bounds, sync); the results line is written after
    the timer stops. No error handling: a failing batch kills the process, and the
    batches already written stay in the results file."""
    import json as _json
    #FEAT: --profile-comm profiles each timed region.
    from constraintflow.lib.comm_profile import CommProfile
    from constraintflow.lib import cpu_memory
    #FEAT: Dynamo compile counts per warmup and batch, read outside the timer (--inductor runs; lib/dynamo_stats.py).
    from constraintflow.lib import dynamo_stats
    dynamo = dynamo_stats if globals.inductor_mode.get_flag() else None

    def _write(f, record):
        f.write(_json.dumps(record) + "\n")
        f.flush()

    with open(results_file, "w") as out:
        for j in range(warmup):
            lo = num_images + j * batch_size
            if warmup_timeout:
                faulthandler.dump_traceback_later(warmup_timeout, exit=True)
            dyn_before = dynamo.snapshot() if dynamo else None
            warmup_start = time.perf_counter()
            warmup_row_kernels(run, network_file, batch_size, eps, *_batch(lo), dataset=dataset, train=train,
                print_intermediate_results=False, no_sparsity=no_sparsity)
            if is_cuda:
                torch.cuda.synchronize()
            warmup_time = time.perf_counter() - warmup_start
            faulthandler.cancel_dump_traceback_later()
            typer.echo(f"Warmup run {j + 1}/{warmup}: {warmup_time:.6f} s")
            _write(out, {"kind": "warmup", "index": j, "images": [lo, lo + batch_size],
                         "time_s": warmup_time, **({"dynamo": dynamo.since(dyn_before)} if dynamo else {})})
            gc.collect()

        gc.collect()
        gc.freeze()

        for i in range(num_images // batch_size):
            lo = i * batch_size
            typer.echo(f"=== batch {i} start ===")
            resident = None
            #FEAT: profiler on around the timer; comm.region() is exactly the timed region.
            comm = CommProfile(profile_comm)
            with comm:
                if is_cuda:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    resident = torch.cuda.memory_allocated()
                else:
                    #FEAT: per-run CPU memory: resident set at the start, high-water mark during the run.
                    resident = cpu_memory.start()

                dyn_before = dynamo.snapshot() if dynamo else None
                if batch_timeout:
                    faulthandler.dump_traceback_later(batch_timeout, exit=True)
                start_time = time.perf_counter()
                with comm.region():
                    lb, ub = run(network_file, batch_size, eps, *_batch(lo), dataset=dataset,
                                 train=train, print_intermediate_results=print_intermediate_results,
                                 no_sparsity=no_sparsity)
                    if is_cuda:
                        torch.cuda.synchronize()
                run_time = time.perf_counter() - start_time
                faulthandler.cancel_dump_traceback_later()
            peak = torch.cuda.max_memory_allocated() if is_cuda else cpu_memory.peak()

            _write(out, {"kind": "batch", "batch": i, "images": [lo, lo + batch_size],
                         comm.time_key: run_time, "peak_bytes": peak, "resident_bytes": resident,
                         "lower": lb.detach().cpu().tolist(), "upper": ub.detach().cpu().tolist(),
                         **comm.metrics(), **({"dynamo": dynamo.since(dyn_before)} if dynamo else {})})
            comm.save_trace(results_file, i)
            typer.echo(f"Run {i + 1}/{num_images // batch_size}: {run_time:.6f} s, "
                       f"Peak memory: {peak} bytes, Resident memory: {resident} bytes")
            typer.echo(f"=== batch {i} end ===")
            del lb, ub
            gc.collect()


def main():
    app()


if __name__ == "__main__":
    main()
