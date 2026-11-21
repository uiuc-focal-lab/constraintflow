# ConstraintFlow-GPU

**ConstraintFlow** is a domain-specific language for writing neural network certifiers (DeepPoly, CROWN, IBP,
DeepZ, ...) as short, declarative `.cf` programs. This repository compiles those programs into fast GPU code:
`jit` specializes a certifier to one network and batch size, removes all interpretation overhead, and compiles
the resulting tensor program with PyTorch Inductor.

The certifiers used in our experiments are in [`examples/compiler_examples/`](examples/compiler_examples).

## Installation

Python 3.10 or newer and an NVIDIA GPU with CUDA. The pinned versions are the ones our results were measured
with (PyTorch 2.10.0, CUDA 12.8).

```bash
git clone https://github.com/ADAPT-uiuc/constraintflow-gpu.git
cd constraintflow-gpu
pip install -e .
```

## Quick start

Download a network (or use any ONNX file), then build and run a certifier:

```bash
python reproduce/download.py --nets nets --only N1

# Build DeepPoly for this network, batch size 100 (writes the generated code to output/)
constraintflow jit examples/compiler_examples/deeppoly_priority_add.cf \
    --network nets/mnist/mnist_relu_3_50.onnx --dataset mnist --eps 0.004 --batch-size 100 \
    --device gpu --fuse-affine-subst --output-path output/

# Certify the first 1000 MNIST test images
constraintflow run examples/compiler_examples/deeppoly_priority_add.cf \
    --network nets/mnist/mnist_relu_3_50.onnx --dataset mnist --eps 0.004 --batch-size 100 \
    --device gpu --num-images 1000 --results-file results.jsonl --output-path output/
```

`run` warms up once (the first run compiles the program), then times one run per batch of 100 images.
`results.jsonl` has one line per batch with its time, peak GPU memory and the lower and upper bounds of every
margin `f_y - f_j`; an image is verified when all its margin lower bounds are positive.

Datasets are read from the working directory: MNIST, CIFAR-10 and CIFAR-100 are downloaded there by torchvision
on first use; TinyImageNet is read from `./tinyimagenet` (`python reproduce/download.py --tinyimagenet`).

## Commands

| Command | What it does |
|---|---|
| `jit` | Builds a certifier specialized to one network, dataset and batch size (the fast path). |
| `run` | Runs a built program on test images and reports time, memory and bounds. |
| `compile` | Builds the general, unspecialized interpreter of a program, as the original ConstraintFlow does. Run it with `run --no-inductor`, or with `run --compile` to compile and run in one step. |
| `provesound` | Checks a certifier's soundness automatically. |

Run `constraintflow <command> --help` for every option. The ones you are most likely to need:

**`jit`**

- `--network`, `--dataset`, `--eps`, `--batch-size`, `--device gpu`: the network and the property being
  certified. `--dataset` is `mnist`, `cifar`, `cifar100`, `tinyimagenet`, `vnnlib:<benchmark folder>` (VNN-COMP
  format: `instances.csv`, `onnx/`, `vnnlib/`), or `box-<input shape>-<outputs>-<spec rows>`.
- `--fuse-affine-subst`: skips redundant work at Linear/Conv layers. It is correct only for programs whose
  Affine rule gives identical lower and upper expressions, which holds for the DeepPoly/CROWN programs here, so
  it is off by default. Our DeepPoly-family results use it.
- `--center-radius` / `--no-center-radius`: how bounds over a box are concretized, as `c·mid + |c|·rad` (the
  default; more precise in fp32) or as ConstraintFlow's original sign split `c⁺·u + c⁻·l`.
- `--traversal-stop-chunk`: block sizes for programs whose traversals stop early per neuron (e.g.
  `deeppoly_traversal_stop.cf`); a comma list such as `8,16,32,64,128,256,512` compiles one kernel per size.
  By default a neuron refined for any image of the batch is refined for every image, as auto_LiRPA's sparse
  mode does; `--traversal-stop-per-example` keeps each image's own decision instead.
- `--flow-segment-mb` (default 8192): splits the generated program into regions that Inductor compiles
  separately. Smaller regions can lower peak memory; 0 keeps one region.
- `--no-inductor`: emits eager PyTorch code instead of compiling it. `run` must be given the same choice.
- `--verbose`: prints what each compiler pass did.

**`run`**

- `--num-images N --results-file F`: certify the first `N` test images, one timed run per batch.
- `--warmup` (default 1), `--repeat`: untimed warmup runs, and the number of timed runs on the first batch
  when `--num-images` is not given.
- `--batch-timeout`, `--warmup-timeout`: stop if one batch or one warmup takes longer than this many seconds.
- `--dtype float64`: computes everything in fp64 (pass it to `jit` too), to check fp32 bounds.
- `--profile-comm`: profiles each timed batch and records CPU–GPU synchronizations and copies.

Set `CF_DISABLE_TF32=1` to keep cuDNN and matmuls in full fp32 instead of PyTorch's default TF32 convolutions.

## Very large networks

For networks whose back-substitution does not fit in GPU memory (ResNet50, VGG16), the `row_chunks` rewrite
processes the rows of each large back-substitution in chunks that fit a memory budget:

```bash
constraintflow jit <program> --network <onnx> ... --no-inductor --flow-segment-mb 0 --output-path build/
python -m constraintflow.compiler.optimizations.row_chunks build/ --network <onnx> --dataset <dataset> \
    --budget-gb 40 --compile-bodies
constraintflow run <program> --network <onnx> ... --no-inductor --output-path build/
```

## Certified training

`constraintflow.lib.training.TrainingBounds` gives differentiable margin bounds of a PyTorch model, like
auto_LiRPA's `BoundedModule`, using compiled ConstraintFlow programs (`IBP`, `CROWN-IBP`, `CROWN`, `zid`, ...):

```python
from constraintflow.lib.training import TrainingBounds

bounds = TrainingBounds(model, dummy_input, "mnist", ["CROWN-IBP"], [batch_size], cache_dir="cf_cache")
for x, y in loader:
    bounds.begin_step()
    lower, upper = bounds.compute_bounds("CROWN-IBP", (x - eps).clamp(0, 1), (x + eps).clamp(0, 1), y)
    loss = robust_loss(lower)  # lower bounds of f_y - f_j, differentiable in the model's parameters
```

Each method and batch size is compiled once and cached in `cache_dir`.

## Reproducing our results

[`reproduce/`](reproduce) downloads every network we used and reruns our cf-gpu measurements with the exact
settings of each experiment. See [`reproduce/README.md`](reproduce/README.md).

## How the compiler works

[`constraintflow/compiler/optimizations/README.md`](constraintflow/compiler/optimizations/README.md) describes
the two-pass `jit` build and every optimization pass.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Some tests need a CUDA GPU and are skipped without one.

## Citation

If you use ConstraintFlow in your research, please cite:

```bibtex
@InProceedings{constraintflow,
  author = {Avaljot Singh and Yasmin Sarita and Charith Mendis and Gagandeep Singh},
  title = {ConstraintFlow: A DSL for Specification and Verification of Neural Network Analyses},
  booktitle = {Static Analysis},
  year = {2024},
  publisher = {Springer Nature Switzerland},
}

@InProceedings{provesound,
  author = {Avaljot Singh and Yasmin Sarita and Charith Mendis and Gagandeep Singh},
  title = {Automated Verification of Soundness of DNN Certifiers},
  booktitle = {OOPSLA},
  year = {2025},
}

@Article{compiler,
  author = {Avaljot Singh and Yasmin Sarita and Aditya Mishra and Ishaan Goyal and Gagandeep Singh and Charith Mendis},
  title = {A Tensor-Based Compiler and Runtime for Neuron-Level DNN Certifier Specifications},
  journal = {arXiv},
  year = {2025},
}
```

## License

MIT License. See [LICENSE](LICENSE).
