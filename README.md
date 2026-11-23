# ConstraintFlow

**ConstraintFlow** is a domain-specific language (DSL) and toolchain for specifying, verifying, and compiling neural network certifiers. It bridges the gap between high-level formal specifications and efficient tensor-based runtimes, enabling precise and verifiable DNN analysis.

---

## 📚 Features

ConstraintFlow allows you to:

* **Specify** certifiers declaratively using `.cf` files.
* **Verify** certifiers automatically for soundness.
* **Compile** high-level specifications into optimized tensor-based code.
* **Execute** compiled certifiers on neural network models.

---

## 🚀 Quick Start

### Installation

Pip installation ([Pypi](https://pypi.org/project/constraintflow/0.1.1/))

```bash
pip install constraintflow==0.1.1
```

Or install from source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-username/constraintflow.git
cd constraintflow
pip install -e .
```

### Prepare Models

Create a directory for neural networks:

```bash
mkdir nets/
```

Download pretrained DNNs from [ERAN](https://github.com/eth-sri/eran) and place them inside `nets/`.


---

### CLI Usage

ConstraintFlow provides the following commands with optional flags:

#### `provesound`

```bash
constraintflow provesound example.cf [OPTIONS]
```

**Options:**

| Flag      | Description                           | Default |
| --------- | ------------------------------------- | ------- |
| `--nprev` | Number of previous states to consider | 1       |
| `--nsymb` | Number of symbols to track            | 1       |

#### `compile`

```bash
constraintflow compile example.cf [OPTIONS]
```

**Options:**

| Flag            | Description                  | Default   |
| --------------- | ---------------------------- | --------- |
| `--output-path` | Directory for generated code | `output/` |

#### `scepter`

Specializes a certifier to one network, dataset and batch size.

```bash
constraintflow scepter example.cf --network <onnx> [OPTIONS]
```

**Options:**

| Flag                                     | Description                                                                                                  | Default   |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------- |
| `--network`                              | Path to the ONNX network (required)                                                                          |           |
| `--network-format`                       | Format of the network file                                                                                   | `onnx`    |
| `--dataset`                              | `mnist`, `cifar`, `cifar100`, `tinyimagenet`, `vnnlib:<benchmark folder>`, or `box-<input shape>-<outputs>-<spec rows>` | `mnist`   |
| `--batch-size`                           | Batch size the program is specialized to                                                                     | 1         |
| `--eps`                                  | Epsilon                                                                                                      | 0         |
| `--device`                               | `cpu` or `gpu` (CUDA)                                                                  | `cpu`     |
| `--output-path`                          | Directory for the specialized code                                                                           | `output/` |
| `--inductor` / `--no-inductor`           | Compile the program with Inductor, or emit eager PyTorch code; `run` must be given the same choice           | on        |
| `--traversal-stop-chunk`                 | Rows per compiled kernel for traversals that stop early per neuron; a comma list compiles one kernel per size | `128`     |
| `--flow-segment-mb`                      | Split the program into regions of about this many MB that Inductor compiles separately; 0 keeps one region   | 8192      |
| `--memory-budget-mb`                     | Above this estimated peak memory, reorder the program and recompute large values; -1 = 70% of the GPU, 0 = off | -1        |
| `--cudagraphs`                           | Replay one CUDA graph per run (needs `--inductor`)                                                           | False     |
| `--dtype`                                | `float32` or `float64`; `run` must be given the same                                                         | `float32` |
| `--tf32`                                 | Use PyTorch's default TF32 convolutions (faster, less precise); `run` should be given the same              | False     |

Run a `scepter` build with `run` and the same `--network`, `--dataset`, `--batch-size` and `--output-path`.

> **Note:** with `--inductor` (the default), PyTorch compiles the program during the first run. Pass at least one
> warmup to `run` (`--warmup 1`, the default) so that compilation happens before the timed runs.

#### `run`

```bash
constraintflow run example.cf [OPTIONS]
```

**Options:**

| Flag                           | Description                                 | Default           |
| ------------------------------ | ------------------------------------------- | ----------------- |
| `--network`                    | Path to the ONNX network (required)         |                   |
| `--network-format`             | Format of the network file                  | `onnx`            |
| `--dataset`                    | `mnist`, `cifar`, `cifar100`, `tinyimagenet`, `vnnlib:<benchmark folder>`, or `box-<input shape>-<outputs>-<spec rows>` | `mnist` |
| `--batch-size`                 | Batch size                                  | 1                 |
| `--eps`                        | Epsilon                                     | 0.01              |
| `--print-intermediate-results` | Print intermediate results during execution | False             |
| `--output-path`                | Path where compiled program is stored       | `output/`         |
| `--compile`                    | Compile the program before running          | False             |
| `--device`                     | `cpu` or `gpu` (CUDA) | `cpu`             |
| `--inductor` / `--no-inductor` | Run with Inductor; must match the `scepter` build (a `compile` build runs with `--no-inductor`) | on |
| `--warmup`                     | Untimed warmup runs before the timed runs (the first one compiles) | 1  |
| `--repeat`                     | Number of timed runs on the first batch     | 1                 |
| `--num-images`                 | Certify this many test images, one timed run per batch (needs `--results-file`) | 0 |
| `--results-file`               | With `--num-images`: one JSON line per warmup and per batch (time, memory, bounds) |  |
| `--save-bounds` / `--no-save-bounds` | Save each timed run's bounds to `--bounds-file`; with more than one run they are saved instead of printed | on |
| `--bounds-file`                | Where `--save-bounds` writes                | `<output-path>/bounds.jsonl` |
| `--batch-timeout`              | With `--num-images`: stop if one batch takes longer than this many seconds (0 = no limit) | 0 |
| `--warmup-timeout`             | With `--num-images`: stop if one warmup takes longer than this many seconds (0 = no limit) | 0 |
| `--profile-comm`               | With `--num-images`: record CPU–GPU synchronizations and copies per batch | False |
| `--dtype`                      | `float32` or `float64`; must match the build | `float32`        |
| `--tf32`                       | Use PyTorch's default TF32 convolutions; should match the build | False |


---

## 📄 Citations

If you use ConstraintFlow in your research, please cite the following papers:

<p>
    <a href="https://arxiv.org/abs/2501.01234"><img src="https://img.shields.io/badge/Paper-arXiv-blue"></a>
    <a href="https://dl.acm.org/doi/10.1145/OOPSLA2025"><img src="https://img.shields.io/badge/Paper-OOPSLA2025-blue"></a>
    <a href="https://example.com/dsl-paper"><img src="https://img.shields.io/badge/Paper-SAS2024-blue"></a>
</p>

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

---

## 🛠 Development

### Requirements

* Python 3.9+
* `antlr4-python3-runtime==4.9.2`

### Running Locally

Install the requirements:

```bash
pip install -r requirements.txt
```

You can then run, compile, or verify any `.cf` file using the CLI.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
