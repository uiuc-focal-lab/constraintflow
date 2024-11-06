# ConstraintFlow

###  DNN Certifier Specifications
The DNN certifier specifications can be found in:
 - verification/data/test_cases_correct

### Directory Structure

```
constraintflow/
├── ast_cflow/
├── certifier/
├── compiler/
├── interpreter/
├── verification/
│   ├── data/
|   ├── experiments/
│   └── src/
├── README.md
└── requirements.txt
```

### Set up environment
```
sudo apt update
sudo apt install python
pip install -r requirements.txt
```

### To run all experiments
```
python3 -m verification.experiments.experiments_full 
```

### To run ConstraintFlow for a sound DNN certifier

```
python3 -m verification.experiments.experiments_correct certifier_file [n_prev] [n_sym]
```

Example:
```
python3 -m verification.experiments.experiments_correct verification/data/test_cases_correct/deeppoly_relu 1 1
```

### To run ConstraintFlow for an unsound DNN certifier

```
python3 -m verification.experiments.experiments_incorrect certifier_file [n_prev] [n_sym]
```

Example:
```
python3 -m verification.experiments.experiments_incorrect verification/data/test_cases_correct/deeppoly_relu 1 1
```

### To re-generate the lexer and parser

```
sudo apt install antlr4
cd ast_cflow
antlr4 -Dlanguage=Python3 -visitor dsl.g4
```