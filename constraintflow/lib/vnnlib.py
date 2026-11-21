"""VNN-LIB benchmark folders (instances.csv, onnx/, vnnlib/) for --dataset vnnlib:<folder>.

Image i is spec i mod #specs. Outside the timer `run` reads the spec list and the network's input shape;
inside, read_batch parses the batch's spec files with the copied α,β-CROWN parser (abcrown_read_vnnlib.py),
as α,β-CROWN does inside its own timer.
"""

import csv
import functools
import os

import numpy as np

from constraintflow.lib.abcrown_read_vnnlib import read_box_and_margins


@functools.lru_cache(maxsize=None)
def spec_files(folder):
    """The benchmark's spec files in instances.csv order (a list of names, not their contents)."""
    with open(os.path.join(folder, "instances.csv")) as f:
        return [os.path.join(folder, row[1].strip()) for row in csv.reader(f) if row]


def input_shape(folder):
    """The input shape, without the batch dimension, of instances.csv's first network."""
    import onnx
    with open(os.path.join(folder, "instances.csv")) as f:
        network = os.path.join(folder, next(csv.reader(f))[0].strip())
    graph = onnx.load(network).graph
    weights = {w.name for w in graph.initializer}
    (x,) = [i for i in graph.input if i.name not in weights]
    return [d.dim_value for d in x.type.tensor_type.shape.dim][1:]


def read_batch(folder, idx):
    """Parse specs idx: lo, hi [n, inputs], C [n, rows, outputs], d [n, rows] (float64); verified iff C y + d > 0."""
    files = spec_files(folder)
    parts = [read_box_and_margins(files[i]) for i in idx]
    return tuple(np.stack(p) for p in zip(*parts))
