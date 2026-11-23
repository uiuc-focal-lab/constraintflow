#FEAT: --dtype float64 support (fp64 precision oracle for bound mismatches).
import torch


def use_float64():
    """--dtype float64: compute every float tensor in fp64, the precision oracle for bound
    mismatches. Float tensors are spelled `.float()` / `torch.float` throughout (generated
    code included), so those spellings are re-pointed at fp64 rather than rewritten. Call it
    before importing modules that read torch.float at import time."""
    torch.set_default_dtype(torch.float64)
    torch.float = torch.float64
    torch.Tensor.float = torch.Tensor.double


def dtype_from_argv(argv):
    """The --dtype value in argv (float32 if absent); the CLIs need it before their imports."""
    for i, arg in enumerate(argv):
        if arg == "--dtype" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--dtype="):
            return arg.split("=", 1)[1]
    return "float32"
