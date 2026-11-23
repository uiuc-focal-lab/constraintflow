# Min-parenthesis compiler examples

These `.cf` files are generated from `examples/compiler_examples` with redundant
parentheses removed (relying on the language’s operator precedence).

**Regenerate** after changing either the originals or the reduction rules:

```bash
python examples/compiler_examples_minparen/build_minparen.py
```

**Compare** each original with its minparen twin (compilation always; full `run`
requires your ONNX path):

```bash
PYTHONPATH=. python examples/compiler_examples_minparen/compare_minparen_runs.py \
  --network ~/nets/mnist_relu_3_50.onnx \
  --dataset mnist --eps 0.008 --batch-size 2
```

Programs `deeppoly2.cf`, `deeppolyh.cf`, and `hybrid_zono.cf` do not compile in
this tree for **either** the original or minparen copy; the compare script skips
them when both fail the same way.

