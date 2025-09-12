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

#### `run`

```bash
constraintflow run example.cf [OPTIONS]
```

**Options:**

| Flag                           | Description                                 | Default           |
| ------------------------------ | ------------------------------------------- | ----------------- |
| `--network`                    | Network name                                | `mnist_relu_3_50` |
| `--network-format`             | Format of the network file                  | `onnx`            |
| `--dataset`                    | Dataset to use (`mnist` or `cifar`)         | `mnist`           |
| `--batch-size`                 | Batch size                                  | 1                 |
| `--eps`                        | Epsilon                                     | 0.01              |
| `--train`                      | Use training dataset                        | False             |
| `--print-intermediate-results` | Print intermediate results during execution | False             |
| `--no-sparsity`                | Disable sparsity optimizations              | False             |
| `--output-path`                | Path where compiled program is stored       | `output/`         |
| `--compile`                    | Compile the program before running          | False             |


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
