"""Constants for walks over the IR tree.

Analysis back-edges (def-use chains, parent links) can lead from one statement
to the rest of the program. Tree walks that skip them visit only the code
actually reachable through statements, children and operator attributes.
"""

ANALYSIS_EDGES = frozenset(('uses', 'parents', 'defs'))
ATOMS = frozenset((int, float, str, bool, type(None)))
# The metadata on every expression holds no IR to visit.
TREE_SKIP = ANALYSIS_EDGES | {'irMetadata'}
