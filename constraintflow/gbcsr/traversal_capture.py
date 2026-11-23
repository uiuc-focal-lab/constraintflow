"""Host-only validation of sparse support entering a captured substitution."""
from constraintflow.lib import globals as G


def selection_path(layer_index, while_number, iteration):
    return f'jit_selection/selection_{layer_index}_{while_number}_{iteration}.json'


def _contained(lo, hi, ranges):
    """Containment in a union, including adjacent intervals and priority ties."""
    for start, end in sorted(ranges):
        if end <= lo:
            continue
        if start > lo:
            return False
        lo = max(lo, end)
        if lo >= hi:
            return True
    return lo >= hi


def validate_substitution(mat, layer_index, counter, while_number, iteration):
    from constraintflow.gbcsr.sparse_block import ConstBlock

    path = selection_path(layer_index, while_number, iteration)
    context = f'layer={layer_index}, while={while_number}, iteration={iteration}, counter={counter}'
    if not G.capture_exists(path):
        raise ValueError(f'Missing priority selection for substitution ({context}); rebuild captures.')
    selection = G.load_capture(path)
    # Runtime-valued blocks are structurally live even if the sample payload is
    # zero. Never specialize traversal validity to captured weights or inputs.
    if mat.dense_const != 0:
        raise ValueError(f'Nonzero implicit coefficient support in substitution ({context}).')
    ranges = []
    for start, end, block in zip(mat.start_indices, mat.end_indices, mat.blocks):
        if isinstance(block, ConstBlock) and block.block == 0:
            continue
        if any(int(a) >= int(b) for a, b in zip(start, end)):
            continue
        lo, hi = int(start[-1]), int(end[-1])
        if not _contained(lo, hi, selection['ranges']):
            raise ValueError(f'Substitution outside selected priority set ({context}): '
                             f'coefficient range [{lo}, {hi}), selected ranges {selection["ranges"]}.')
        ranges.append((lo, hi))
    layers = [ident for ident, start, end in G.capture_layers
              if any(start < hi and lo < end for lo, hi in ranges)]
    if not set(layers).issubset(selection['layers']):
        raise ValueError(f'Substitution layer IDs outside selected priority set ({context}): '
                         f'actual {layers}, selected {selection["layers"]}.')
    checks = selection.setdefault('substitutions', {})
    checks[str(counter)] = {'layers': layers, 'ranges': ranges}
    G.save_capture(path, selection)
