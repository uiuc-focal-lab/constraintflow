# ConstraintFlow

## Directory Structure
```
oopsla_artifact/
├── constraintflow/              # Implements ConstraintDlow DSL
├── dnn_certifiers/              # Contains DNN certifiers 
├── experiments/                 # Experiments for the artifact
├── provesound/                  # Implements ProveSound
│   ├── lib                      # Libraries used in provesound
│   │   ├── globals.py           # Global values to track time
│   │   ├── optGraph.py          # Optimizations for solver
│   │   ├── optSolver.py         # z3-based solver 
│   │   └── utils.py             # General utility functions
│   └── src                      # Source code for provesound
│       ├── symbolicDNN.py       # Symbolic DNN expansion
│       ├── symbolicSemantics.py # Symbolic Semantics
│       ├── value.py             # Symbolic values 
│       └── verify.py            # Verification logic
├── README.md                    # README
└── requirements.txt             # Dependencies
```

## Set up environment for docker
```
cd oopsla_artifact/
sudo docker build -t oopsla_artifact .
sudo docker run -it oopsla_artifact
```

## Set up instructions to build from scratch
```
cd oopsla_artifact/
pip install -r requirements.txt
```

## ProveSound
ProveSound verification procedure can be used to automatically check the soundness of the DNN certifiers. The directory `constraintflow/` contains the grammar, lexer, parser and type-checking for the ConstraintFlow DSL. The directory `provesound/` contains the code for 
ProveSound algorithm. The directory `dnn_certifiers/` contains the constraintflow specifications for the DNN certifiers discussed throughout the main paper and Appendix K. The directory `experiments/` contains the various experiments that can be run to test the verification algorithm on different DNN certifiers specified in ConstraintFlow. 

The steps to run these experiments are described below. In all these experiments, we show the times taken by different parts of the algorithm - the query generation time (G), verification time (V) for correct implementation, and bug-finding time
for randomly introduced bugs (B) in seconds for DNN certifiers. Note that these times depend on the machine specifications and thus, may not match the reported times exactly. Further, for the experiments involving bug-finding time, random bugs are introduced each time the experiment is run. So, the bug-finding time can be different each time.  

In the Evaluation Section (§ 6) of the paper, the results are summarized in Tables 1a, 1b, 2a, 2b, and Fig. 17. 
The instructions to run these experiments are provided below.

### Table 1a
Table 1a shows G, V, B for the new DNN certifiers introduced in § 6.1. These are BALANCE Cert and REUSE Cert.
Run the following command to execute these experiments and generate table 1a. 
(Approx. runtime ~20 s)
```
python3 -m experiments.table1a
```

### Table 1b
Table 1a shows G, V, B for the DNN certifiers and new DNN operations discussed in § 6.2. These DNN operations are Relu6, Abs, HardSigmoid, HardTanh, and HardSwish.
Run the following command to execute these experiments and generate table 1b. 
(Approx. runtime ~45 s)
```
python3 -m experiments.table1b
```

### Table 2a
Table 1a shows G, V, B for existing DNN certifiers and primitive DNN operations discussed in § 6.4. 
Run the following command to execute these experiments and generate table 2a. 
(Approx. runtime ~30 s)
```
python3 -m experiments.table2a
```

### Table 2b
Table 1a shows G, V, B for existing DNN certifiers and composite DNN operations discussed in § 6.4. 
Run the following command to execute these experiments and generate table 2b. This experiments takes a lot of time to run (approx 5-7 hours). So, we also give a version of these experiments with reduced parameter values (approx 30s).  
Run the reduced version:
```
python3 -m experiments.table2b_reduced
```
Run the complete version
```
python3 -m experiments.table2b
```



### Scalability Graph in Fig. 17
The graph shows the scalability of the verification algorithm with increasing parameter values. In order to run all the experiments to generate the data points in the graph, it takes around 50 hours. So, we also provide the code which generates the graph from pre computed data points.
The graph is not plotted when run inside the docker. So, to visualize the graph without plotting, we print the the graph data in a table. To view the graph, run the code outside of docker.

To generate graph (and table) from pre computed data points: 
```
python3 -m experiments.graph_precomputed
```
To generate the graph (and table) from scratch:
```
python3 -m experiments.graph
```

### New DNN certifier
To run the ConstraintFlow verification procedure to check the soundness of a single DNN certifier from the `dnn_certifiers/` folder, use the following command.

```
python3 -m experiments.experiments_correct certifier_file [n_prev] [n_sym]
```

Example Usage:
```
python3 -m experiments.experiments_correct dnn_certifiers/deeppoly 1 1
```


Further, to randomly introduce bugs to the DNN certifier and check for unsoundness, run the following command.

```
python3 -m experiments.experiments_incorrect certifier_file [n_prev] [n_sym]
```

Example:
```
python3 -m experiments.experiments_incorrect dnn_certifiers/deeppoly 1 1
```
