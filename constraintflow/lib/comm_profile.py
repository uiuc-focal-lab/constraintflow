"""CPU-GPU communication of one timed batch (--profile-comm), shared by every tool.

Each tool keeps a byte-identical copy of this file next to its timed loop
(experiments/test_comm_profile.py checks the copies). Works on torch 1.12 (IVAN) and 2.x.

How a tool uses it:

    comm = CommProfile(enabled)
    with comm:                      # profiler on (outside the timer)
        ...
        t0 = time.perf_counter()
        with comm.region():         # exactly the timed region
            lb, ub = run(...)
            torch.cuda.synchronize()
        run_time = time.perf_counter() - t0
    record = {comm.time_key: run_time, ..., **comm.metrics()}   # metrics() is {} when disabled
    comm.save_trace(results_file, batch)    # the full trace, for the batches in TRACE_BATCHES

What metrics() reports for the region:

    syncs          calls that make the CPU wait until the GPU has finished its queued work:
                   stream/device/event synchronize and blocking cudaMemcpy (.item(), .tolist(),
                   .cpu(), the final synchronize, ...). A call returns at once if the GPU is
                   already done, so this counts possible waits, not waited time. Not a
                   transfer count either: the final synchronize moves no data, and a pageable
                   CPU-to-GPU copy blocks the CPU without any of these calls.
    alloc_syncs    CUDA allocator calls that may wait for the GPU: cudaMalloc and cudaFree
    h2d_copies     CPU-to-GPU memory copies the GPU ran (weights and image uploads, ...)
    d2h_copies     GPU-to-CPU memory copies the GPU ran (.item(), .tolist(), .cpu(), ...)
    gpu_idle_pct   share of the region in which no kernel, copy or memset runs on the GPU
    h2d_d2h_copy_time_pct
                   share of the region in which the GPU ran a CPU-to-GPU or GPU-to-CPU copy
                   (overlapping copies counted once). GPU copy time, not CPU wait time: the
                   copies are small, the cost of a sync is the waiting around it
    sync_calls     count per CUDA call name behind syncs and alloc_syncs
"""

import contextlib
import gzip
import os
import shutil
import tempfile

import torch

# Name of the profiler range that marks the timed region.
REGION = "timed_batch"

# CPU-side CUDA calls we count.
SYNC_CALLS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy")
ALLOC_CALLS = ("cudaMalloc", "cudaFree")

# GPU copy events are named like "Memcpy HtoD (Pageable -> Device)" on torch 1.12 and 2.x.
H2D = "Memcpy HtoD"
D2H = "Memcpy DtoH"
# Copies that stay on one side (GPU-GPU, peer-to-peer, host-host): known, but not counted.
OTHER_COPIES = ("Memcpy DtoD", "Memcpy PtoP", "Memcpy HtoH")

# Batches whose full Chrome trace is saved next to the results file. Syncs and copies are the
# same every batch, so one shows the pattern; every batch would be too many, too large files.
TRACE_BATCHES = (0,)


# --------------------------------------------------------------------------
# Small helpers on profiler events
# --------------------------------------------------------------------------

def _on_gpu(event):
    return str(event.device_type).endswith("CUDA")


def _on_cpu(event):
    return str(event.device_type).endswith("CPU")


def _merged_length(intervals):
    """Total length covered by (start, end) intervals, overlaps counted once."""
    total = 0.0
    covered_until = None
    for start, end in sorted(intervals):
        if covered_until is None or start > covered_until:
            # a gap before this interval: count all of it
            total += end - start
            covered_until = end
        elif end > covered_until:
            # overlaps what we counted: count only the part sticking out
            total += end - covered_until
            covered_until = end
    return total


# --------------------------------------------------------------------------
# The measurements
# --------------------------------------------------------------------------

def _find_region(events):
    """(start, end) of the timed region, from its CPU-side range."""
    regions = [e for e in events if e.name == REGION and _on_cpu(e)]
    if len(regions) != 1:
        raise RuntimeError(f"comm profile: expected one {REGION} range, found {len(regions)}")
    return regions[0].time_range.start, regions[0].time_range.end


def _count_cuda_calls(events, start, end):
    """{call name: how many times the CPU made it inside the region}."""
    calls = {name: 0 for name in SYNC_CALLS + ALLOC_CALLS}
    for e in events:
        inside = start <= e.time_range.start and e.time_range.end <= end
        if not _on_gpu(e) and e.name in calls and inside:
            calls[e.name] += 1
    return calls


def _gpu_work_in_region(events, start, end):
    """GPU events overlapping the region, as (event, start, end) clipped to the region.
    The closing synchronize means no GPU work runs past the region's end."""
    work = []
    for e in events:
        if e.name == REGION or not _on_gpu(e):
            continue
        if e.time_range.end > start and e.time_range.start < end:
            work.append((e, max(e.time_range.start, start), min(e.time_range.end, end)))
    if not work:
        # every batch runs GPU work, so nothing at all means the profiler saw none (no CUPTI)
        raise RuntimeError("comm profile: no GPU activity recorded in the timed region; "
                           "CUDA activity tracing (CUPTI) is unavailable")
    return work


def _count_copies(gpu_work):
    """(CPU-to-GPU copies, GPU-to-CPU copies) among the GPU work."""
    h2d = d2h = 0
    for e, _start, _end in gpu_work:
        if e.name.startswith(H2D):
            h2d += 1
        elif e.name.startswith(D2H):
            d2h += 1
        elif e.name.startswith("Memcpy") and not e.name.startswith(OTHER_COPIES):
            # a renamed copy event would otherwise read as 0 copies
            raise RuntimeError(f"comm profile: unrecognized copy event {e.name!r}")
    return h2d, d2h


def _copy_time_pct(gpu_work, start, end):
    """Share of the region in which the GPU ran a CPU-to-GPU or GPU-to-CPU copy."""
    length = end - start
    if length <= 0:
        return None
    copies = [(s, t) for e, s, t in gpu_work if e.name.startswith((H2D, D2H))]
    return 100.0 * _merged_length(copies) / length


def _idle_pct(gpu_work, start, end):
    """Share of the region in which the GPU ran nothing."""
    length = end - start
    if length <= 0:
        return None
    busy = _merged_length([(s, t) for _e, s, t in gpu_work])
    return 100.0 * (length - busy) / length


# --------------------------------------------------------------------------
# What the tools use
# --------------------------------------------------------------------------

class CommProfile:
    def __init__(self, enabled):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self._prof = None

    def __enter__(self):
        """Start the profiler (if enabled)."""
        if self.enabled:
            from torch.profiler import profile, ProfilerActivity
            self._prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            self._prof.__enter__()
        return self

    def __exit__(self, *exc):
        """Stop the profiler."""
        if self._prof is not None:
            self._prof.__exit__(*exc)
        return False

    @property
    def time_key(self):
        """A profiled run's time is slowed by the profiler, so it never lands under time_s."""
        return "profiled_time_s" if self.enabled else "time_s"

    def region(self):
        """Marks the timed region inside the profile (does nothing when disabled)."""
        if not self.enabled:
            return contextlib.nullcontext()
        from torch.profiler import record_function
        return record_function(REGION)

    def metrics(self):
        """The region's numbers (see the module docstring); {} when disabled."""
        if self._prof is None:
            return {}
        events = self._prof.events()
        start, end = _find_region(events)
        calls = _count_cuda_calls(events, start, end)
        gpu_work = _gpu_work_in_region(events, start, end)
        h2d, d2h = _count_copies(gpu_work)
        return {
            "syncs": sum(calls[name] for name in SYNC_CALLS),
            "alloc_syncs": sum(calls[name] for name in ALLOC_CALLS),
            "h2d_copies": h2d,
            "d2h_copies": d2h,
            "gpu_idle_pct": _idle_pct(gpu_work, start, end),
            "h2d_d2h_copy_time_pct": _copy_time_pct(gpu_work, start, end),
            "sync_calls": {name: n for name, n in calls.items() if n},
        }

    def save_trace(self, results_file, batch):
        """For a batch in TRACE_BATCHES, gzip the full Chrome trace next to results_file as
        trace_batch<N>.json.gz (open it in ui.perfetto.dev). Call after the profiler stops."""
        if self._prof is None or batch not in TRACE_BATCHES:
            return
        out = os.path.join(os.path.dirname(os.path.abspath(results_file)), f"trace_batch{batch}.json.gz")
        fd, tmp = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(out))
        os.close(fd)
        try:
            self._prof.export_chrome_trace(tmp)
            with open(tmp, "rb") as src, gzip.open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
        finally:
            os.remove(tmp)
