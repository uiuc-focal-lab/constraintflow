# ConstraintFlow

**ConstraintFlow** is a domain-specific language (DSL) and toolchain for specifying, verifying, compiling, and specializing neural network certifiers. It bridges the gap between high-level formal specifications and efficient tensor-based runtimes, enabling precise and verifiable DNN analysis.

---

## 📚 Features

ConstraintFlow allows you to:

* **Specify** certifiers declaratively using `.cf` files.
* **Verify** certifiers automatically for soundness.
* **Compile** high-level specifications into optimized tensor-based code.
* **Specialize** a certifier to one network and batch size, compiling it into fast tensor code.
* **Execute** compiled certifiers on neural network models.

---

## 🚀 Quick Start

### Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/uiuc-focal-lab/constraintflow.git
cd constraintflow
pip install -e .
```

### Prepare Models

Create a directory for neural networks:

```bash
mkdir nets/
```

Put the ONNX networks in `nets/`:

| Networks | Source |
| -------- | ------ |
| MNIST, CIFAR-10 | [ERAN](https://github.com/eth-sri/eran) |
| CIFAR-100, TinyImageNet ResNets | [VNN-COMP](https://github.com/Lucas110550/CIFAR100_TinyImageNet_ResNet), with the [TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) dataset |
| CIFAR-10 ResNet50 | [ONNX export](https://drive.google.com/file/d/13CtXfo88oANMolEk0Q3HHp-hP76xUFPW/view?usp=sharing) of [MadryLab robustness](https://github.com/MadryLab/robustness) `cifar_linf_8` |
| VGG16 | [ONNX models](https://github.com/onnx/models/tree/main/validated/vision/classification/vgg), with properties from [VNN-COMP](https://github.com/stanleybak/vggnet16_benchmark2022) (`--dataset vnnlib:<folder>`) |

MNIST and CIFAR datasets download on first use. ResNet50 and VGG16 with back-substitution certifiers need [row chunking](#large-networks-row-chunking).


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
| `--dataset`                              | `mnist`, `cifar`, `cifar100`, `tinyimagenet`, `vnnlib:<benchmark folder>`, or `box-<input shape>-<outputs>-<spec rows>` | `mnist`   |
| `--batch-size`                           | Batch size the program is specialized to                                                                     | 1         |
| `--eps`                                  | Epsilon                                                                                                      | 0         |
| `--device`                               | `cpu` or `gpu` (CUDA)                                                                  | `cpu`     |
| `--output-path`                          | Directory for the specialized code                                                                           | `output/` |
| `--inductor` / `--no-inductor`           | Compile the program with Inductor, or emit eager PyTorch code                                                | on        |
| `--flow-segment-mb`                      | Split the program into regions of about this many MB that Inductor compiles separately; 0 keeps one region   | 8192      |
| `--row-chunks-gb`                      | Fit each back-substitution into this many GB by running it over chunks of neuron rows (large networks); 0 = off | 0 |
| `--dtype`                                | `float32` or `float64`; `run` must be given the same                                                         | `float32` |
| `--tf32`                                 | Use PyTorch's default TF32 convolutions (faster, less precise); `run` should be given the same              | False     |

More options: `constraintflow scepter --help`. Run a `scepter` build with `run` and the same `--network`, `--dataset`, `--batch-size` and `--output-path`.

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
| `--dataset`                    | `mnist`, `cifar`, `cifar100`, `tinyimagenet`, `vnnlib:<benchmark folder>`, or `box-<input shape>-<outputs>-<spec rows>` | `mnist` |
| `--batch-size`                 | Batch size                                  | 1                 |
| `--eps`                        | Epsilon                                     | 0.01              |
| `--print-intermediate-results` | Print intermediate results during execution | False             |
| `--output-path`                | Path where compiled program is stored       | `output/`         |
| `--compile`                    | Compile the program before running          | False             |
| `--device`                     | `cpu` or `gpu` (CUDA) | `cpu`             |
| `--inductor` / `--no-inductor` | Pass `--no-inductor` for a `compile` build (it turns the library's checks and counters back on); a `scepter` build runs either way | on |
| `--warmup`                     | Untimed warmup runs before the timed runs (the first one compiles) | 1  |
| `--num-images`                 | Certify this many test images, one timed run per batch (needs `--results-file`); a warmup over 2500 s or a batch over 1000 s stops the run | 0 |
| `--results-file`               | With `--num-images`: one JSON line per warmup and per batch (time, memory, bounds) |  |
| `--save-bounds` / `--no-save-bounds` | Save each timed run's bounds to `<output-path>/bounds.jsonl`; with more than one run they are saved instead of printed | on |
| `--tf32`                       | Use PyTorch's default TF32 convolutions; should match the build | False |

More options (timeouts, fp64, profiling): `constraintflow run --help`.

#### Large networks: row chunking

Row chunking helps certifiers that back-substitute (e.g. DeepPoly, CROWN), whose coefficient matrices grow with the
size of each layer. On networks whose back-substitution does not fit in GPU memory (e.g. ResNet50, VGG16), `scepter --row-chunks-gb <GB>`
runs each back-substitution over that budget as a loop over chunks of neuron rows. This pass keeps the same bounds while dropping peak
memory. `run` the build as usual:

```bash
constraintflow scepter example.cf --network net.onnx --dataset cifar --eps 4e-05 --batch-size 1 --device gpu --row-chunks-gb 40 --output-path out/
constraintflow run example.cf --network net.onnx --dataset cifar --eps 4e-05 --batch-size 1 --device gpu --output-path out/
```

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
