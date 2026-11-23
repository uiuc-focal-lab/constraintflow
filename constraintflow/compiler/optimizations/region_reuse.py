"""Deduplicate alpha-equivalent tensor regions with matching input signatures."""
import ast
import textwrap


def canonical_body(source, parameters):
    tree = ast.parse(textwrap.dedent(source).strip())
    mapping = {name: 'arg_' + str(i) for i, name in enumerate(parameters)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            mapping.setdefault(node.id, 'local_' + str(len(mapping)))
    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id=mapping.get(node.id, node.id), ctx=node.ctx), node)
    tree = Rename().visit(tree)
    return ast.dump(tree, include_attributes=False)


def key(source, parameters, shapes):
    if any(name not in shapes or
           any(field not in shapes[name] for field in ('shape', 'dtype', 'stride', 'device'))
           for name in parameters):
        return None  # Never share a dynamic/unknown signature blindly.
    signature = tuple((tuple(shapes[n]['shape']), shapes[n]['dtype'],
                       tuple(shapes[n]['stride']), shapes[n]['device'])
                      for n in parameters)
    return canonical_body(source, parameters), signature
