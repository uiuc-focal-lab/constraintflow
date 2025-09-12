import operator

comparison_ops = [operator.lt, operator.le, operator.eq, operator.ne, operator.gt, operator.ge, operator.and_, operator.or_]
all_ops = [operator.lt, operator.le, operator.eq, operator.ne, operator.gt, operator.ge, operator.and_, operator.or_, operator.add, operator.sub, operator.mul, operator.truediv]

disjunction_ops = [operator.lt, operator.le, operator.eq, operator.ne, operator.gt, operator.ge, operator.or_, operator.add, operator.sub]
conjunction_ops = [operator.and_, operator.mul, operator.truediv]

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

def binary_to_identity_unary(op):
    if op == operator.add:
        return lambda x: x
    elif op == operator.mul:
        return lambda x: x
    elif op == operator.sub:
        return lambda x: x.unary(operator.neg)
    elif op == operator.and_:
        return lambda x: x
    elif op == operator.or_:
        return lambda x: x
    else:
        assert(False)
