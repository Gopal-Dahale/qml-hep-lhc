from tensorflow.keras import Model, Input
import sympy
import tensorflow_quantum as tfq
import cirq
from qml_hep_lhc.models.base_model import BaseModel
from tensorflow.keras.layers import Layer, Flatten
import numpy as np
from tensorflow import random_uniform_initializer, Variable, constant, shape, repeat, tile, concat, gather
from qml_hep_lhc.utils import _import_class
import re
from sympy.core import numbers as sympy_numbers
import numbers
import tensorflow as tf
from sympy.functions.elementary.trigonometric import TrigonometricFunction
import sympy as sp


class QLinear(Layer):

    def __init__(self, input_dim, fm_class):
        super(QLinear, self).__init__()

        self.dim = np.prod(input_dim)
        self.fm_class = fm_class

        # Prepare qubits
        self.n_qubits = self.dim

        if self.fm_class == 'AmplitudeMap':
            self.n_qubits = int(np.ceil(np.log2(self.dim)))

        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)
        self.readout = cirq.GridQubit(-1, -1)
        self.observables = [cirq.Z(self.readout)]

        var_symbols = sympy.symbols(f'θ0:{3*self.n_qubits}')
        self.var_symbols = np.asarray(var_symbols).reshape((self.n_qubits, 3))

        in_symbols = sympy.symbols(f'x0:{self.dim}')
        self.in_symbols = np.asarray(in_symbols).reshape((self.dim))

    def build(self, input_shape):
        data_circuit = cirq.Circuit()
        model_circuit = cirq.Circuit()

        # Prepare the readout qubit
        model_circuit.append(cirq.X(self.readout))
        model_circuit.append(cirq.H(self.readout))

        self.fm = _import_class(f"qml_hep_lhc.encodings.{self.fm_class}")()
        data_circuit += self.fm.build(self.qubits, self.in_symbols)

        for i, qubit in enumerate(self.qubits):
            model_circuit.append(
                cirq.XX(qubit, self.readout)**self.var_symbols[i, 0])
        for i, qubit in enumerate(self.qubits):
            model_circuit.append(
                cirq.ZZ(qubit, self.readout)**self.var_symbols[i, 1])
        for i, qubit in enumerate(self.qubits):
            model_circuit.append(
                cirq.XX(qubit, self.readout)**self.var_symbols[i, 2])

        # Finally, prepare the readout qubit.
        model_circuit.append(cirq.H(self.readout))

        self.var_symbols = list(self.var_symbols.flat)
        self.in_symbols = list(self.in_symbols.flat)

        var_init = random_uniform_initializer(minval=-np.pi / 2,
                                              maxval=np.pi / 2)
        self.theta = Variable(initial_value=var_init(
            shape=(1, len(self.var_symbols)), dtype="float32"),
                              trainable=True,
                              name="thetas")

        data_circuit_flattened, expr_map = cirq.flatten(data_circuit)
        data_circuit = data_circuit_flattened
        self.raw_symbols = symbols_in_expr_map(expr_map)
        self.expr = list(expr_map)

        # Define explicit symbol order
        symbols = [str(symb) for symb in self.var_symbols + self.raw_symbols]
        self.indices = constant([symbols.index(a) for a in sorted(symbols)])

        self.input_resolver = resolve_formulas(self.expr, self.raw_symbols)

        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.circuit = data_circuit + model_circuit
        self.computation_layer = tfq.layers.ControlledPQC(
            self.circuit, self.observables)

    def call(self, input_tensor):
        resolved_inputs = self.input_resolver(input_tensor)
        batch_dim = shape(input_tensor)[0]
        tiled_up_circuits = repeat(self.empty_circuit,
                                   repeats=batch_dim,
                                   name="tiled_up_circuits")
        tiled_up_thetas = tile(self.theta,
                               multiples=[batch_dim, 1],
                               name="tiled_up_thetas")
        joined_vars = concat([tiled_up_thetas, resolved_inputs], axis=1)
        joined_vars = gather(joined_vars,
                             self.indices,
                             axis=1,
                             name='joined_vars')
        out = self.computation_layer([tiled_up_circuits, joined_vars])
        return out


class QNN(BaseModel):
    """
    Quantum Neural Network.
    This implementation is based on https://www.tensorflow.org/quantum/tutorials/mnist
    """

    def __init__(self, data_config, args=None):
        super().__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]
        self.n_qubits = np.prod(self.input_dim)
        self.fm_class = self.args.get("feature_map")
        if self.fm_class is None:
            self.fm_class = "AngleMap"
        self.qlinear = QLinear(self.input_dim, self.fm_class)

    def call(self, input_tensor):
        """
        The function takes in an input tensor and returns the expectation of the input tensor
        
        Args:
          input_tensor: The input tensor to the layer.
        
        Returns:
          The expectation of the input tensor.
        """
        x = Flatten()(input_tensor)
        out = self.qlinear(x)
        return out

    def build_graph(self):
        # x = Input(shape=(), dtype=string)
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="QNN")

    def get_circuit(self):
        return self.qlinear.circuit

    @staticmethod
    def add_to_argparse(parser):
        return parser


############################### HELPERS ######################################

tf_ops_map = {
    sympy.sin: tf.sin,
    sympy.cos: tf.cos,
    sympy.tan: tf.tan,
    sympy.asin: tf.asin,
    sympy.acos: tf.acos,
    sympy.atan: tf.atan,
    sympy.atan2: tf.atan2,
    sympy.cosh: tf.cosh,
    sympy.tanh: tf.tanh,
    sympy.sinh: tf.sinh
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
    if isinstance(val, numbers.Number) and not isinstance(val, sympy.Basic):
        return tf.constant(float(val), dtype=tf.float32)
    elif isinstance(val,
                    (sympy_numbers.IntegerConstant, sympy_numbers.Integer)):
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
    # print(formula, symbols)
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

    # if isinstance(formula, sympy.)

    if isinstance(formula, sympy.Pow) and len(formula.args) == 2:
        base = resolve_formula(formula.args[0], symbols)
        exponent = resolve_formula(formula.args[1], symbols)
        return lambda x: tf.pow(base(x), exponent(x))

    if isinstance(formula, sympy.Pow):
        base = resolve_formula(formula.args[0], symbols)
        exponent = resolve_formula(formula.args[1], symbols)
        return lambda x: tf.pow(base(x), exponent(x))

    if isinstance(formula, TrigonometricFunction):
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
        if isinstance(expr, sp.Basic):
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