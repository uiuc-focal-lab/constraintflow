import operator

comparison_ops = [operator.lt, operator.le, operator.eq, operator.ne, operator.gt, operator.ge, operator.and_, operator.or_]
all_ops = [operator.lt, operator.le, operator.eq, operator.ne, operator.gt, operator.ge, operator.and_, operator.or_, operator.add, operator.sub, operator.mul, operator.truediv]

disjunction_ops = [operator.lt, operator.le, operator.eq, operator.ne, operator.gt, operator.ge, operator.or_, operator.add, operator.sub]
conjunction_ops = [operator.and_, operator.mul, operator.truediv]

commutative_ops = [operator.add, operator.mul, operator.and_, operator.or_]

def identity_element(op):
    if op == operator.add:
        return 0
    elif op == operator.mul:
        return 1
    elif op == operator.truediv:
        return 1
    elif op == operator.sub:
        return 0
    elif op == operator.and_:
        return True
    elif op == operator.or_:
        return False
    else:
        return None

def annihilator_element(op):
    if op == operator.mul:
        return 0
    elif op == operator.and_:
        return False
    elif op == operator.or_:
        return True
    else:
        return None

def _record_lambda(op, json_list):
    if json_list is not None:
        json_list.append({
            "method": "lambda",
            "op": op.__name__,
            "output": len(json_list),
        })

def binary_to_identity_unary(op, json_list=None):
    if op == operator.add:
        _record_lambda(op, json_list)
        if json_list is not None:
            return lambda x: x, len(json_list) - 1
        return lambda x: x
    elif op == operator.mul:
        _record_lambda(op, json_list)
        if json_list is not None:
            return lambda x: x, len(json_list) - 1
        return lambda x: x
    elif op == operator.sub:
        _record_lambda(op, json_list)
        if json_list is not None:
            return lambda x: operator.neg(x), len(json_list) - 1
        return lambda x: operator.neg(x)
    elif op == operator.and_:
        _record_lambda(op, json_list)
        if json_list is not None:
            return lambda x: x, len(json_list) - 1
        return lambda x: x
    elif op == operator.or_:
        _record_lambda(op, json_list)
        if json_list is not None:
            return lambda x: x, len(json_list) - 1
        return lambda x: x
    else:
        assert(False)
