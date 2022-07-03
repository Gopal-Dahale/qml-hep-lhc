import cirq
from sympy.functions.elementary.trigonometric import TrigonometricFunction, InverseTrigonometricFunction, HyperbolicFunction
import sympy
from tensorflow import sin, cos, tan, asin, acos, atan, sinh, cosh, tanh
import tensorflow as tf
from sympy.core import numbers as sympy_numbers
import numbers
import re
import numpy as np


def get_count_of_qubits(feature_map, input_dim):
    if feature_map == 'AmplitudeMap':
        return int(np.ceil(np.log2(np.prod(input_dim))))
    return np.prod(input_dim)


def get_num_in_symbols(feature_map, input_dim):
    if feature_map == 'AmplitudeMap':
        return 2**int(np.ceil(np.log2(np.prod(input_dim))))
    if feature_map == "DoubleAngleMap":
        return 2 * np.prod(input_dim)
    return np.prod(input_dim)


def one_qubit_unitary(bit, symbols):
    """Make a Cirq circuit enacting a rotation of the bloch sphere about the X,
    Y and Z axis, that depends on the values in `symbols`.
    """
    return cirq.Circuit(
        cirq.X(bit)**symbols[0],
        cirq.Y(bit)**symbols[1],
        cirq.Z(bit)**symbols[2])


############################### RESOLVERS ######################################

tf_ops_map = {
    sympy.sin: sin,
    sympy.cos: cos,
    sympy.tan: tan,
    sympy.asin: asin,
    sympy.acos: acos,
    sympy.atan: atan,
    sympy.tanh: tanh,
    sympy.sinh: sinh,
    sympy.cosh: cosh,
}


def stack(func, lambda_set, intermediate=None):
    if intermediate is None:
        return stack(func, lambda_set[1:], lambda_set[0])
    if len(lambda_set) > 0:
        new_lambda = lambda x: func(intermediate(x), lambda_set[0](x))
        return stack(func, lambda_set[1:], new_lambda)
    else:
        return intermediate


def resolve_formulas(formulas, symbols):
    lambda_set = [resolve_formula(f, symbols) for f in formulas]
    stacked_ops = stack(lambda x, y: tf.concat((x, y), 0), lambda_set)
    n_formula = tf.constant([len(formulas)])
    transposed_x = lambda x: tf.transpose(
        x, perm=tf.roll(tf.range(tf.rank(x)), shift=1, axis=0))
    resolved_x = lambda x: stacked_ops(transposed_x(x))
    reshaped_x = lambda x: tf.reshape(
        resolved_x(x),
        tf.concat(
            (n_formula, tf.strided_slice(tf.shape(x), begin=[0], end=[-1])),
            axis=0))
    transformed_x = lambda x: tf.transpose(
        reshaped_x(x), perm=tf.roll(tf.range(tf.rank(x)), shift=-1, axis=0))
    return transformed_x


def resolve_value(val):
    if isinstance(val, numbers.Number):
        return tf.constant(float(val), dtype=tf.float32)
    elif isinstance(val, (
            sympy_numbers.IntegerConstant,
            sympy_numbers.Integer,
    )):
        return tf.constant(float(val.p), dtype=tf.float32)
    elif isinstance(val,
                    (sympy_numbers.RationalConstant, sympy_numbers.Rational)):
        return tf.divide(tf.constant(val.p, dtype=tf.float32),
                         tf.constant(val.q, dtype=tf.float32))
    elif val == sympy.pi:
        return tf.constant(np.pi, dtype=tf.float32)

    else:
        return NotImplemented


def resolve_formula(formula, symbols):

    # Input is a pass through type, no resolution needed: return early
    value = resolve_value(formula)
    if value is not NotImplemented:
        return lambda x: value

    # Handles 2 cases:
    # formula is a string and maps to a number in the dictionary
    # formula is a symbol and maps to a number in the dictionary
    # in both cases, return it directly.
    if formula in symbols:
        index = symbols.index(formula)
        return lambda x: x[index]

    # formula is a symbol (sympy.Symbol('a')) and its string maps to a number
    # in the dictionary ({'a': 1.0}).  Return it.
    if isinstance(formula, sympy.Symbol) and formula.name in symbols:
        index = symbols.index(formula.name)
        return lambda x: x[index]

    if isinstance(formula, sympy.Abs):
        arg = resolve_formula(formula.args[0], symbols)
        return lambda x: tf.abs(arg(x))

    # the following resolves common sympy expressions
    if isinstance(formula, sympy.Add):
        addents = [resolve_formula(arg, symbols) for arg in formula.args]
        return stack(tf.add, addents)

    if isinstance(formula, sympy.Mul):
        factors = [resolve_formula(arg, symbols) for arg in formula.args]
        return stack(tf.multiply, factors)

    if isinstance(formula, sympy.Pow) and len(formula.args) == 2:
        base = resolve_formula(formula.args[0], symbols)
        exponent = resolve_formula(formula.args[1], symbols)
        return lambda x: tf.pow(base(x), exponent(x))

    if isinstance(formula, sympy.Pow):
        base = resolve_formula(formula.args[0], symbols)
        exponent = resolve_formula(formula.args[1], symbols)
        return lambda x: tf.pow(base(x), exponent(x))

    if isinstance(formula, (TrigonometricFunction, InverseTrigonometricFunction,
                            HyperbolicFunction)):
        ops = tf_ops_map.get(type(formula), None)
        if ops is None:
            raise ValueError("unsupported sympy operation: {}".format(
                type(formula)))
        arg = resolve_formula(formula.args[0], symbols)
        return lambda x: ops(arg(x))


def natural_key(symbol):
    '''Keys for human sorting
    Reference:
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(s) for s in re.split(r'(\d+)', symbol.name)]


def symbols_in_expr_map(expr_map, to_str=False, sort_key=natural_key):
    """Returns the set of symbols in an expression map
    
    Arguments:
        expr_map: cirq.ExpressionMap
            The expression map to find the set of symbols in
        to_str: boolean, default=False
            Whether to convert symbol to strings
        sort_key: 
            Sort key for the list of symbols
    Returns:
        Set of symbols in the experssion map
    """
    all_symbols = set()
    for expr in expr_map:
        if isinstance(expr, sympy.Basic):
            all_symbols |= expr.free_symbols
    sorted_symbols = sorted(list(all_symbols), key=sort_key)
    if to_str:
        return [str(x) for x in sorted_symbols]
    return sorted_symbols


def symbols_in_op(op):
    """Returns the set of symbols associated with a parameterized gate operation.
    
    Arguments:
        op: cirq.Gate
            The parameterised gate operation to find the set of symbols associated with
    
    Returns:
        Set of symbols associated with the parameterized gate operation
    """
    if isinstance(op, cirq.EigenGate):
        return op.exponent.free_symbols

    if isinstance(op, cirq.FSimGate):
        ret = set()
        if isinstance(op.theta, sympy.Basic):
            ret |= op.theta.free_symbols
        if isinstance(op.phi, sympy.Basic):
            ret |= op.phi.free_symbols
        return ret

    if isinstance(op, cirq.PhasedXPowGate):
        ret = set()
        if isinstance(op.exponent, sympy.Basic):
            ret |= op.exponent.free_symbols
        if isinstance(op.phase_exponent, sympy.Basic):
            ret |= op.phase_exponent.free_symbols
        return ret

    raise ValueError("Attempted to scan for symbols in circuit with unsupported"
                     " ops inside. Expected op found in tfq.get_supported_gates"
                     " but found: ".format(str(op)))


def atoi(symbol):
    return int(symbol) if symbol.isdigit() else symbol