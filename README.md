# ConstraintFlow

###  DNN Certifier Specifications
The DNN certifier specifications can be found in:
 - verification/data/test_cases_correct


### Set up environment
```
sudo apt update
sudo apt install python
pip install -r requirements.txt
```

### To re-generate the lexer and parser

```
sudo apt install antlr4
cd ast_cflow
antlr4 -Dlanguage=Python3 -visitor dsl.g4
```

## ConstraintFlow Verification
ConstraintFlow verification procedure can be used to automatically check the soundness of the DNN certifiers.

### Run ConstraintFlow Verification procedure for checking the soundness of a DNN certifier

```
python -m verification.experiments.experiments_correct certifier_file [n_prev] [n_sym]
```

Example Usage:
```
python -m verification.experiments.experiments_correct verification/data/test_cases_correct/deeppoly_relu 1 1
```


### To randomly introduce bugs to a DNN certifier and check for soundness

```
python -m verification.experiments.experiments_incorrect certifier_file [n_prev] [n_sym]
```

Example:
```
python -m verification.experiments.experiments_incorrect verification/data/test_cases_correct/deeppoly_relu 1 1
```


### To run all experiments presented in the paper
```
python -m verification.experiments.experiments_full 
```


## ConstraintFlow Compiler
ConstraintFlow Compiler generates a PyTorch based executable for a DNN certifier.

### Compile a DNN certifier in ConstraintFlow

```
python -m compile certifier_file 
```

Example Usage:
```
python -m compile certifier/testcases/deeppoly
```

### Run the compiled certifier

```
python -m run_certifier onnx_network_file batch_size 
```

Example Usage:
```
python -m run_certifier certifier/nets/mnist_relu_3_50.onnx 1
```


## 📜 Citation
<p>
    <a href="https://arxiv.org/abs/2403.18729"><img src="https://img.shields.io/badge/Paper-arXiv-blue"></a>
</p>

```
@InProceedings{constraintflow,
    author={Avaljot Singh and Yasmin Sarita and Charith Mendis and Gagandeep Singh},
    title={ConstraintFlow: A DSL for Specification and Verification of Neural Network Analyses},
    booktitle="Static Analysis",
    year="2024",
    publisher="Springer Nature Switzerland",
}

```
