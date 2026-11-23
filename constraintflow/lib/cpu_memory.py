"""Per-run CPU memory for the cf-harness CPU sweeps (the same file sits in every tool).

start() releases freed memory, resets the process's high-water mark and returns its resident
set (VmRSS); peak() after the run returns the high-water mark since (VmHWM). Linux resets
VmHWM to the current RSS when "5" is written to /proc/self/clear_refs, so the peak covers the
run alone. The kernel keeps it, so nothing runs inside the timed region, and it counts every
allocation: Python, torch and C libraries alike.
"""

import ctypes
import gc


def _status_bytes(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024  # reported in kB
    raise RuntimeError(f"{field} missing from /proc/self/status")


def start():
    """Collect garbage, hand glibc's freed heap back to the OS, reset the high-water mark;
    returns the resident bytes the run starts from."""
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")
    return _status_bytes("VmRSS")


def peak():
    """The most resident bytes since start()."""
    return _status_bytes("VmHWM")
