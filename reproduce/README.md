# Reproducing the paper's cf-gpu results

These scripts rerun the paper's cf-gpu measurements: 8 certifiers on 44 networks, 352 cells, each with the exact
build and run settings the paper used.

| Paper name | `--certifiers` value | Program |
|---|---|---|
| DeepPoly | `deeppoly_add` | `examples/compiler_examples/deeppoly_priority_add.cf` |
| BoxPoly | `deeppoly_traversal_stop` | `deeppoly_traversal_stop.cf` |
| SkipPoly | `deeppoly_efficient_topo` | `deeppoly_efficient_topo.cf` |
| CROWN-IBP | `crownibp` | `crownibp.cf` |
| IBP | `ibp` | `ibp.cf` |
| DeepZ | `zono` | `zono.cf` |
| ReuseCert | `reuse_topo` | `reuse_topo.cf` |
| ZID | `zid_topo` | `zid_topo.cf` |

The comparisons with other tools (auto_LiRPA, IVAN, GPUPoly, the original ConstraintFlow) need those tools and are
not part of this repository.

## Setup

The paper's numbers were measured on an NVIDIA A100 80GB with PyTorch 2.10.0 (CUDA 12.8, cuDNN 9.10).

```bash
pip install -e .
python reproduce/download.py --nets nets --tinyimagenet
```

`download.py` fetches every network from its original source (ERAN, the VNN-COMP CIFAR-100/TinyImageNet ResNets,
the ONNX model zoo) and checks each file against the checksum in `networks.csv`. MNIST, CIFAR-10 and CIFAR-100
are downloaded by torchvision into the working directory on first use. Two networks have no ready-made file:

- **N40 (ResNet50):** `--resnet50` downloads the MadryLab robustness checkpoint (`cifar_linf_8.pt`, about 190 MB)
  and exports it with `export_resnet50.py`. The export is deterministic and must match the checksum.
- **N54 (VGG16):** the network downloads normally. Its 10 whole-image L-inf specifications (164 MB) are generated
  with VNN-COMP's `vggnet16_2022` generator plus `vggnet16_linf.patch`, which perturbs every input by ±7.5e-5.
  Pass `--vgg16 --mxnet-python <python>` with a Python that has `mxnet==1.9.1`, `numpy==1.23.5`, `onnx` and
  `onnxruntime`. The output must match `vggnet16_linf_specs.md5`.

## Running

Run from the directory that holds the datasets:

```bash
python reproduce/run.py --nets nets --certifiers ibp,deeppoly_add --configs N1,N16
python reproduce/run.py --nets nets          # every cell; takes many hours
```

Each cell is compiled with `scepter` (plus the `row_chunks` rewrite for DeepPoly and SkipPoly on N40 and N54), then
run with the paper's protocol: untimed warmups, then one timed run per batch. Results go to
`reproduce_results/<certifier>/<config>/`: `cf.jsonl` has every batch's time, peak GPU memory and bounds.
`reproduce_results/summary.csv` has one row per cell with the mean time per image, peak GPU memory and the number
of verified images (every margin lower bound > 0). Finished cells are skipped when you rerun.

`configs.csv` lists the settings of every cell. They differ by certifier and network in a few places:

- `--fuse-affine-subst` is on for the DeepPoly family (DeepPoly, BoxPoly, SkipPoly), as in the paper. It is off
  by default in the CLI because it is not sound for every program.
- `--center-radius` is on only for BoxPoly; the other certifiers ran with `--no-center-radius`.
- BoxPoly's `--traversal-stop-chunk` block sizes depend on the network.
- N40 and N54 (DeepPoly, SkipPoly) are built with `--no-inductor --flow-segment-mb 0` and row-chunked with a
  40 GB budget.
- Warmups: 5 for most networks, 1 for the large ones. Some cells have per-batch or per-warmup time limits.

`run.py` sets the environment the paper's runs used: `CF_DISABLE_TF32=1` (full fp32), one Inductor compile
thread, and Inductor's coalesce-tiling analysis off. Each cell starts with empty Inductor and Triton caches, so
compilation happens in the untimed warmup.

On the small networks a run takes a few milliseconds per batch, so timings vary by a few percent between runs.
Batch-size-1 networks such as N33 spend most of each image loading the ONNX file on the host, which makes their
times sensitive to other load on the machine.
