# Optimizations in `constraintflow jit`

## How `jit` works

`jit` compiles the certifier twice.

1. **First compile (the probe run).** ConstraintFlow builds a version of the certifier and runs it once on the network. While it runs, it records facts about the network: which layers exist, which branch of each `if` is taken, and how many times each loop runs.
2. **Second compile (the reuse compile).** ConstraintFlow compiles the certifier again using those recorded facts. Because the loop counts and branches are now known, it can remove the loops and `if`s and produce one long, straight list of tensor operations, called `flow()`. Every optimization below works on this second compile.

The passes are listed in the order they run. The code that runs them is in [compile.py](../compile.py) and in `emit_sroa_flow` in [codeGen.py](../codeGen.py).

## Two terms used below

- **Each variable is assigned once.** After the second compile, the generated code is a straight list of statements in which no variable is ever reassigned. Many passes depend on this.
- **No in-place writes.** The code never changes part of an existing tensor (for example `x[0:5] = y`); it only creates new tensors. Most passes after step 2 run only when this holds. Otherwise they are skipped.

## Switches

Passes marked **"always on"** run in every `jit` build and have no switch. The few that do have one name it below.

---

## Step 1: Turn the certifier into one straight list of operations

**tensor_to_block** (`tensor_to_block.py`), always runs
Makes one copy of each layer's transformer code. In each copy it replaces every loop with the recorded number of repeats and keeps only the recorded side of each `if`. The result is one straight block of code per layer.

**fuse_affine_subst** (`fuse_affine_subst.py`), off by default, `--fuse-affine-subst`
Two shortcuts for Linear/Conv layers:
- If a Linear/Conv layer only feeds other Linear/Conv layers, it skips computing that layer's concrete bounds, because nothing needs them.
- During back-substitution through a Linear/Conv layer, it computes the lower and upper versions once instead of twice. **This is only correct if the layer's lower and upper outputs really are identical**, which holds for the DeepPoly/CROWN programs here.

**splice_flow** (`tensor_to_block.py`), always on
Joins the per-layer blocks into one function, `flow()`. Each layer's local variable names get a suffix (`_L3`, ...) so names from different layers don't clash.

**SROA** (`sroa.py`), always on
The code passes around helper objects (an object that holds several tensors plus their shapes). SROA removes these objects and replaces each one with the tensors it held, so `flow()` contains only plain tensor operations. It also removes useless clones, casts and assignments whose values are never used.

**Purity check** (`sroa.py`), always on
After SROA, it checks the generated code for in-place writes. If there are none, the later passes that need that are allowed to run.

---

## Step 2: General cleanup

These run only if `flow()` has no in-place writes.

**Common subexpression elimination** (`ssa_cse.py`)
If the same expression is computed twice, compute it once and reuse the result.

**Copy propagation** (`copyPropagation.py`)
Removes `x = y` and uses `y` everywhere `x` was used.

**Dead code elimination** (`dce.py`)
Removes statements whose result is never used.

**Inlining** (`subexp_inlining.py`)
If a temporary variable is used exactly once, put its expression directly where it is used. Python frees a local variable only when the function returns, so a temporary that is no longer a named variable can be freed sooner.

---

## Step 3: Math simplifications

**sign_convs** (`sign_convs.py`), always on
`conv(max(x,0), w) + conv(min(x,0), w)` becomes `conv(x, w)`: one convolution instead of two. It only does this when it can confirm the weights and conv settings are the same. Results can differ slightly because of floating-point rounding.

**sign_reductions** (`sign_reductions.py`), always on
The same idea for sums: `sum(b * positive_part(A)) + sum(b * negative_part(A))` becomes `sum(b * A)`.

**box_concretize** (`box_concretize.py`), on by default, `--center-radius` / `--no-center-radius`
When computing a bound over the input box `[l, u]`, it uses
`A · center ± |A| · radius` (center = (l+u)/2, radius = (u−l)/2) instead of `positive_part(A) · u + negative_part(A) · l`.
The second form subtracts two large, nearly equal numbers, which loses accuracy in fp32 when the box is small. The first form is what auto_LiRPA uses.

**fold_conv_bias** (`box_concretize.py`), on with `--center-radius`
Adds a convolution's bias inside the center convolution, `conv(center, W, bias)`, instead of adding it afterwards. auto_LiRPA does it this way, so the two tools round the same way.

**Reusing variable names** (`subexp_inlining.py`), only when `flow()` *has* in-place writes and is not segmented (early reductions, which are always on, otherwise rule it out)
Reuses the name of a temporary once it is no longer needed, so the old tensor can be freed.

---

## Step 4: Passes run while writing out the generated code

Before these passes, ConstraintFlow runs `flow()` on fake tensors that have shapes but no data (`flow_shapes.py`), which gives the exact size of every tensor without using any memory. The passes below use those sizes. The sizes are measured again after each pass that changes the code.

**1. box_patch_reductions** (`box_concretize.py`), always on
Applies the center/radius rule from step 3 again on the final code, this time also to convolution-patch sums and constant terms. It merges the center and radius parts into one sum instead of two.

**2. expression_limits** (`expression_limits.py`), always on
If an expression is nested more than 64 levels deep, it is split into several temporary variables. Without this, Python can fail to compile the generated file. It does not change any math.

**3. patch_factor** (`patch_factor.py`), always on
Rewrites `A @ (diag(s) @ W)` as `(A @ diag(s)) @ W`. The old code expanded every entry of `diag(s) @ W` into patches. The new code expands only the small scale vector `s` and then runs one transposed convolution with `W`.

**4. conv_chunks** (`conv_chunks.py`), always on, only after patch_factor changed something
If the code splits a convolution's output channels into pieces and then glues the pieces back together, it replaces this with the single convolution.

**5. conv_sum** (`conv_sum.py`), always on, only after patch_factor changed something
`conv(x, W) + conv(y, W)` becomes `conv(x + y, W)`: one convolution instead of two when the weight is the same.

**6. product_grids** (`product_grids.py`), always on
Interval arithmetic computes all four combinations of `{W+, W−} × {lower, upper}`. Instead of four separate convolutions (or matrix products), it stacks the inputs and the weights, runs one bigger convolution, and slices out the four results. One large GPU call is faster than four small ones.
It does the same for transposed convolutions, and counts the stacked inputs when checking that the combined call fits in memory.

**7. early_reductions** (`early_reductions.py`), always on
Computes each sum as soon as its inputs exist instead of at the end. This lets large coefficient tensors be freed sooner. Only the order of operations changes.

**8. conv_partials** (`conv_partials.py`), always runs when there are no in-place writes
Stores partial sums of convolutions in variables. Otherwise the compiler (`torch.compile`) merges a long chain of additions into one kernel and keeps every convolution result in memory until that kernel runs.

**9. pad_inputs** (`pad_inputs.py`), runs with `--inductor`
Stores the input of a `pad` in a variable when that `pad` feeds an `unfold`. This works around a `torch.compile` bug that produced wrong code for that pattern.

**10. view_aliases** (`view_aliases.py`), always on
A view (for example `x.view(...)` or `x[2:]`) shares memory with `x`. If both are passed between parts of the code, both stay alive. This pass recreates the cheap view where it is used, so only `x` needs to be passed.

**11. fused_matvec** (`fused_matvec.py`), always on
For a matrix-vector product where the matrix is computed element-wise (for example `clamp(A) @ v`), it writes the product as an element-wise multiply plus a sum over each row. `torch.compile` can then do the clamp, multiply and sum in one kernel without writing the clamped matrix to memory.

**12. mem_plan** (`mem_plan.py`), `--memory-budget-mb` (default: 70% of GPU memory; 0 turns it off)
Runs only if the estimated peak memory is over the budget. It tries, in order:
- Put an element-wise value directly inside the sums that use it, so it is never stored as its own tensor.
- Recreate tensors of all ones or zeros where they are used, instead of keeping one copy alive.
- Reorder statements so that each tensor's last use comes sooner and it can be freed.
- If still over budget, compute a large tensor again later instead of keeping it in memory.
None of these change the results.

**13. flow_split** (`flow_split.py`), `--flow-segment-mb N` (default 8192; 0 keeps one function)
Splits `flow()` into several smaller functions, each allocating about N MB. `torch.compile` then compiles each function separately. The `row_chunks` rewrite needs one function, so row-chunked builds use 0.

**14. region_reuse** (`region_reuse.py`), runs with flow_split
If two of those smaller functions have the same code (apart from variable names) and the same input shapes and types, it emits the function once and reuses it, so it is compiled once.

---

## Step 5: Passes inside `torch.compile` (with `--inductor`)

These are passed as options to `torch.compile`, so they run when PyTorch compiles `flow()`.

**shared_lhs_mm** (`shared_lhs_mm.py`), always on
`x @ W` and `x @ b` become one product `x @ [W | b]`, which is then split, so `x` is read from memory once instead of twice. (This happens in back-substitution, where the same coefficient matrix is multiplied by a layer's weight and by its bias.)

**pre_grad_cse** (`../../lib/pre_grad_cse.py`), only with `--training`
Removes duplicate operations before PyTorch builds the backward pass, so a convolution that appears twice gets one backward computation instead of two.

**Channels-last layout disabled**, always
Stops `torch.compile` from changing the memory layout of convolution inputs to channels-last. On these networks (large batches, few channels) cuDNN is several times slower in that layout.

**CUDA graphs**, off by default, `--cudagraphs`
Records all the GPU kernels of `flow()` once and replays the recording on each run. This saves the time spent launching kernels one by one, which matters on small networks.

---

## Earlier passes (both compiles)

These run on the program before the probe run and before the reuse compile, so strictly speaking they are not "after jit". They are the usual compiler cleanups: common subexpression elimination, copy propagation, dead code elimination, moving code that does not change inside a loop to before the loop, plus `polyOpt.py`, `rewrite.py` and `symexpCount.py` (see `optimizations_rewrite` in [compile.py](../compile.py)).

---

## Transposed convolutions

Every transposed convolution in `flow()` (for example the ones back-substitution uses to go backwards through a conv layer) calls one custom op, `cf::conv_transpose2d`, in [sparse_block.py](../../gbcsr/sparse_block.py). The op has several ways to compute the same result. cuDNN's (Nvidia's kernel library) speed varies a lot between shapes, so on the **first call for each input shape (essentially in the first warmup)** the op times every way that applies and keeps the fastest one for all later calls. We use different data between warmup and the actual runs, hence this is not cheating. A way is only chosen over `chunked` if it is at least 5% faster.

| Option | When it is tried | What it does |
|---|---|---|
| `chunked` (default) | always | Plain `conv_transpose2d`, but with the batch split into pieces of at most 32768. Some cuDNN kernels become about 15× slower once the batch reaches 65536. |
| `plain` | always | Plain `conv_transpose2d` on the whole batch. |
| `forward`, `forward_chunked` | stride 1 | A stride-1 transposed convolution is the same as a normal convolution with the kernel flipped and the input and output channels swapped. cuDNN's normal convolutions are often several times faster. |
| `packed2`, `packed4`, `packed8` | stride 1, small images | Tiles 2×2, 4×4 or 8×8 small images into one larger image, with enough zero padding between them that they don't affect each other. Runs one convolution on the larger images, then cuts the results apart. cuDNN is much more efficient on larger images. |
| `subpixel`, `subpixel_chunked` | stride > 1 | Writes a stride-s transposed convolution as one normal stride-1 convolution that computes the s×s interleaved parts of the output (as extra output channels), then interleaves them. Same amount of arithmetic, but it uses the faster normal-convolution kernels. |
| `phases` | stride > 1 and kernel size not a multiple of the stride | Like `subpixel`, but one normal convolution per interleaved part, each using only its own kernel entries (no zero padding of the kernel). |
| `toeplitz` | the convolution's full matrix is small (GPU: at most 2²⁴ entries and at most 16× the convolution's multiply-adds; CPU: 2²⁶ and 64×) | Builds the convolution's full matrix (a Toeplitz matrix) by passing an identity matrix through the convolution, then computes the transposed convolution as one matrix multiply, `x @ matrix`. It does more arithmetic, but on these shapes (huge batch, small images) matrix multiply runs much faster than cuDNN. |

On the CPU there are two extra options: `forward_native` (the `forward` option with oneDNN turned off) and `subpixel_cl` (`subpixel` with the input in channels-last layout). The backward pass of this op (used in certified training) is chosen the same way, between cuDNN's backward and a patch-based matrix multiply.
