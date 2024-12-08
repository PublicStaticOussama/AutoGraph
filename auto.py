from typing import Union
import numpy as np
from autodiff_graph import OpNode

TYPES = Union[
    OpNode, int, float, np.bool_, np.int8,
    np.int16, np.int32, np.int64, np.uint8,
    np.uint16, np.uint32, np.uint64, np.float16,
    np.float32, np.float64, np.complex64, np.complex128,
    np.ndarray
]

def var(initial_val):
    x = OpNode(op="var")
    if initial_val is not None:
        x.substitute(initial_val)
    return x

def const(val):
    return OpNode(op="const", constant=val)

def relu(X: TYPES, axis=-1):
    if type(X) != OpNode:
        X = var(initial_val=X)
    return OpNode(op="max", axis=axis)(0, X).set_graph_inputs(inputs=[X])

def softmax(X: TYPES, axis=-1):
    if type(X) != OpNode:
        X = var(initial_val=X)
    exp_x = OpNode(op="exp")(X - OpNode(op="max", axis=axis, keepdims=True)(X))
    soft_max = exp_x / OpNode(op="max", axis=axis, keepdims=True)(exp_x)
    return soft_max.set_graph_inputs(inputs=[X])