import uuid
import numpy as np
from typing import Union, List, Tuple, Any, Dict
from autograph.auto.utils.dim_utils import (
    get_broadcast_shape,
    eval_reduce_semi_broadcast_shape,
    infer_matmul_broadcast_shape,
    matmul_batch_axes_match,
    infer_reduction_axis,
    
)
from autograph.auto.utils.tensor_utils import (
    im2col,
    col2im,
    im2col_pool,
    col2im_pool,
    reverse_pad
)
from auto.utils.graph_utils import (
    topological_DFS
)


# any functionality that can be represented as an Direct Acyclic Graph can be constructed into a operation/computation graph using OpNode
# there are cases where u can also represent cyclic features, like having running stats that are updated each iteration, thats only possible if u use self.cycle to define that specific cycle
# and there is a special merge operation node (OpNode(op="merge")) where u can define two sub graphs and merge them by a condition

class OpNode:
    NUMERIC_TYPES = [
        int, float, np.bool_, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16,
        np.float32, np.float64, np.complex64, np.complex128,
    ]
    CONSTANT_TYPES = NUMERIC_TYPES + [np.ndarray]

    def __init__(self, op, prefix="", constant=None, is_param=False, **kwargs):
        if op == "const" and constant is None:
            raise Exception("Const node needs to be assigned a value !!")
        self._id = str(uuid.uuid4()) # never to be changed
        self.name = f"{op}_{self._id[:5]}"
        self.prefix = prefix
        self.op = op
        self.vars = []
        self.leafs = []
        self.val = constant
        self.adjoint = None if op != "const" else 0
        self.kwargs = kwargs
        self.is_param = is_param
        self.inputs = []
        self.outputs = []
        self.frozen = False
        self._cycle = None
        if "axis" in self.kwargs and type(self.kwargs["axis"]) not in [list, tuple]:
            self.kwargs["axis"] = (self.kwargs["axis"])

    def set_name(self, name):
        self.name = name
        return self

    def set_graph_inputs(self, inputs: list):
        self.inputs = inputs
        return self
    
    def append_graph_inputs(self, inputs: list):
        self.inputs = self.inputs + inputs
        return self
    
    def set_graph_outputs(self, outputs: list):
        self.outputs = outputs
        return self

    def substitute(self, val):
        if self.op == "const": raise Exception("You cannot assign a new value to constant node")
        self.val = val
        return self

    def __call__(self, *args):
        if len(args):
            self.vars = []
            for i, arg in enumerate(args):
                op_arg = arg
                if type(arg) in OpNode.CONSTANT_TYPES:
                    op_arg = OpNode("const", constant=arg)
                self.vars.append(op_arg)
                if self not in op_arg.leafs: op_arg.leafs.append(self)
        return self
    
    @property
    def cycle(self): return self._cycle

    @cycle.setter
    def cycle(self, cycle):
        if cycle.op != "var" or cycle.is_param: raise Exception("Only none parameter variables can be used as a cycle nodes")
        self._cycle = cycle

    @staticmethod
    def _add(node1, node2):
        op_node1 = node1
        if type(node1) in OpNode.CONSTANT_TYPES:
            op_node1 = OpNode("const", constant=node1)
        op_node2 = node2
        if type(node2) in OpNode.CONSTANT_TYPES:
            op_node2 = OpNode("const", constant=node2)
        if op_node1 == op_node2 and node1 == op_node2:
            return OpNode("*")(OpNode("const", constant=2), op_node2)
        res = OpNode("+")(op_node1, op_node2)
        return res
    
    def __add__(self, other): return OpNode._add(self, other)
    def __radd__(self, other): return OpNode._add(other, self)

    @staticmethod
    def _mul(node1, node2):
        op_node1 = node1
        if type(node1) in OpNode.CONSTANT_TYPES:
            op_node1 = OpNode("const", constant=node1)
        op_node2 = node2
        if type(node2) in OpNode.CONSTANT_TYPES:
            op_node2 = OpNode("const", constant=node2)
        if op_node1 == op_node2 and node1 == op_node2:
            return OpNode("**")(op_node2, OpNode("const", constant=2))
        res = OpNode("*")(op_node1, op_node2)
        return res
    
    def __mul__(self, other): return OpNode._mul(self, other)
    def __rmul__(self, other): return OpNode._mul(other, self)

    @staticmethod
    def _matmul(node1, node2):
        op_node1 = node1
        if type(node1) in OpNode.CONSTANT_TYPES:
            op_node1 = OpNode("const", constant=node1)
        op_node2 = node2
        if type(node2) in OpNode.CONSTANT_TYPES:
            op_node2 = OpNode("const", constant=node2)
        res = OpNode("@")(op_node1, op_node2)
        return res
    
    def __matmul__(self, other): return OpNode._matmul(self, other)
    def __rmatmul__(self, other): return OpNode._matmul(other, self)

    @staticmethod
    def _sub(node1, node2):
        op_node1 = node1
        if type(node1) in OpNode.CONSTANT_TYPES:
            op_node1 = OpNode("const", constant=node1)
        op_node2 = node2
        if type(node2) in OpNode.CONSTANT_TYPES:
            op_node2 = OpNode("const", constant=node2)
        if op_node1 == op_node2 and node1 == op_node2:
            return OpNode("const", constant=0)
        res = OpNode("-")(op_node1, op_node2)
        return res
    
    def __sub__(self, other): return OpNode._sub(self, other)
    def __rsub__(self, other): return OpNode._sub(other, self)
    
    @staticmethod
    def _truediv(node1, node2):
        op_node1 = node1
        if type(node1) in OpNode.CONSTANT_TYPES:
            op_node1 = OpNode("const", constant=node1)
        op_node2 = node2
        if type(node2) in OpNode.CONSTANT_TYPES:
            op_node2 = OpNode("const", constant=node2)
        if op_node1 == op_node2 and node1 == op_node2:
            return OpNode("const", constant=1)
        res = OpNode("/")(op_node1, op_node2)
        return res
    
    def __truediv__(self, other): return OpNode._truediv(self, other)
    def __rtruediv__(self, other): return OpNode._truediv(other, self)
    
    @staticmethod
    def _pow(node1, node2):
        op_node1 = node1
        if type(node1) in OpNode.CONSTANT_TYPES:
            op_node1 = OpNode("const", constant=node1)
        op_node2 = node2
        if type(node2) in OpNode.CONSTANT_TYPES:
            op_node2 = OpNode("const", constant=node2)
        res = OpNode("**")(op_node1, op_node2)
        return res

    def __pow__(self, other): return OpNode._pow(self, other)
    def __rpow__(self, other): return OpNode._pow(other, self)

    def __neg__(self):
        return OpNode("neg")(self)

    def equals(self, other, **kwargs): return OpNode(op="==", **kwargs)(self, other)
    def sum(self, axis, keepdims, **kwargs): return OpNode(op="+", axis=axis, keepdims=keepdims, **kwargs)(self)
    def prod(self, axis, keepdims, **kwargs): return OpNode(op="*", axis=axis, keepdims=keepdims, **kwargs)(self)
    def mean(self, axis, keepdims, **kwargs): return OpNode(op="mean", axis=axis, keepdims=keepdims, **kwargs)(self)
    def max(self, axis, keepdims, **kwargs): return OpNode(op="max", axis=axis, keepdims=keepdims, **kwargs)(self)
    def min(self, axis, keepdims, **kwargs): return OpNode(op="min", axis=axis, keepdims=keepdims, **kwargs)(self)
    def dot(self, other, **kwargs): return OpNode(op="@")(self, other, **kwargs)
    def transpose(self, axes, **kwargs): return OpNode(op="T", axes=axes, **kwargs)(self)
    def T(self, axes, **kwargs): return OpNode(op="T", axes=axes, **kwargs)(self)
    def reshape(self, shape, **kwargs): return OpNode(op="reshape", shape=shape, **kwargs)(self)
    def pad(self, pad_width, **kwargs): return OpNode(op="pad", pad_width=pad_width, **kwargs)(self)

    @property
    def shape(self):
        try: return self.val.shape
        except Exception as e: print(e)

    def partial_derivative(self, with_respect_to):
        if not len(self.vars): return None
        if with_respect_to not in self.vars: return 0
        if with_respect_to.op == "const": 
            if type(with_respect_to.val) == np.ndarray: return np.zeros(with_respect_to.val.shape)
            return 0

        match(self.op):
            case "const":
                return 0
            case "neg":
                if type(self.val) == np.ndarray:
                    return -1 * np.ones(self.val.shape)
                return -1
            case "=="|"equals"|">="|"gte"|">"|"gt"|"<="|"lte"|"<"|"lt":
                if type(self.val) == np.ndarray:
                    partial = np.zeros(self.val.shape)
                    return partial
                return 0
            case "concat" | "concatenate":
                axis = self.kwargs.get("axis", -1)
                x_indices = [x_idx for x_idx, var in enumerate(self.vars) if var == with_respect_to]
                cumusum = [0] + list(np.cumsum([var.val.shape[axis] for var in self.vars]))
                partial = np.zeros(self.val.shape)
                ranges = [(cumusum[idx], cumusum[idx+1]) for idx in x_indices]
                for rng in ranges:
                    slices = [slice(None)] * self.val.ndim
                    slices[axis] = slice(*rng)
                    slices = tuple(slices)
                    partial[slices] = 1
                return partial
            case "mean" | "avg": # vars[2] should be reduced or vars[1] should be broadcasted
                if len(self.vars) == 1 and type(self.vars[0].val) == np.ndarray:
                    axis = self.kwargs.get("axis", None)
                    if axis is None: size = self.vars[0].val.size
                    else:
                        if not isinstance(axis, tuple): axis = (axis,)
                        size = np.prod([self.vars[0].val.shape[ax] for ax in axis])
                    return np.ones(self.vars[0].val.shape) / size
                count = 0
                for var in self.vars:
                    if var == with_respect_to:
                        count += 1
                if type(self.val) == np.ndarray:
                    partial = np.ones(with_respect_to.val.shape) * count / len(self.vars) if type(with_respect_to.val) == np.ndarray else count / len(self.vars)
                    return partial
                return count / len(self.vars)
            case "+" | "sum":
                if len(self.vars) == 1 and type(self.vars[0].val) == np.ndarray:
                    return np.ones(self.vars[0].val.shape)
                count = 0
                for var in self.vars:
                    if var == with_respect_to:
                        count += 1
                if type(self.val) == np.ndarray:
                    partial = np.ones(self.val.shape) * count
                    return partial
                return count
            case "-":
                if len(self.vars) != 2: raise Exception("minus op only accepts two inputs")
                scalar = 1
                if self.vars[0] == self.vars[1]: scalar = 0
                elif self.vars[0] == with_respect_to: scalar = 1
                else: scalar = -1
                if type(self.val) == np.ndarray:
                    partial = np.ones(self.val.shape) * scalar
                    return partial
                return scalar
            case "*" | "prod":
                if len(self.vars) == 1 and type(self.vars[0].val) == np.ndarray:
                    semi_broadcasted_shape = eval_reduce_semi_broadcast_shape(
                        og_shape=self.vars[0].val.shape,
                        axis=self.kwargs["axis"],
                    )
                    return self.val.reshape(semi_broadcasted_shape) / np.where(self.vars[0].val == 0, 1, self.vars[0].val)
                count = 0
                for var in self.vars:
                    if var == with_respect_to:
                        count += 1
                partial = count * self.val / np.where(with_respect_to.val == 0, 1, with_respect_to.val)
                return partial
            case "@" | "matmul":
                if type(self.vars[0].val) == np.ndarray and type(self.vars[1].val) == np.ndarray:
                    if self.vars[0] == with_respect_to:
                        ndim = self.vars[1].val.ndim
                        dims = list(range(ndim))
                        dims[-2], dims[-1] = dims[-1], dims[-2]
                        return self.vars[1].val.transpose(dims)
                    else:
                        ndim = self.vars[0].val.ndim
                        dims = list(range(ndim))
                        dims[-2], dims[-1] = dims[-1], dims[-2]
                        return self.vars[0].val.transpose(dims)
                else: raise Exception("matmul op only accepts two numpy array inputs")
            case "/":
                if len(self.vars) != 2: raise Exception("divide op only accepts two inputs")
                if self.vars[0] == self.vars[1]: ## and self.vars[0] == with_respect_to
                    if type(self.val) == np.ndarray:
                        partial = np.zeros(self.val.shape)
                        return partial
                    return 0
                if self.vars[0] == with_respect_to:
                    partial = 1 / self.vars[1].val
                    return partial
                partial = -1 * self.vars[0].val / (with_respect_to.val ** 2)
                return partial
            case "**":
                if len(self.vars) != 2: raise Exception("pow op only accepts two inputs")

                base, exponent = self.vars
                if base == exponent:
                    if base == with_respect_to:
                        return self.val * (1 + np.log(with_respect_to.val))
                    raise Exception("Unexpected case: both base and exponent are the same but not the target variable")

                if base == with_respect_to:
                    return exponent.val * (base.val ** (exponent.val - 1))

                if exponent == with_respect_to:
                    return self.val * np.log(base.val)

                raise Exception("with_respect_to not found in inputs")
            case "max" | "min":
                self_val = self.val
                if len(self.vars) == 1 and type(self.vars[0].val) == np.ndarray:
                # if type(with_respect_to.val) == np.ndarray:
                    semi_broadcasted_shape = eval_reduce_semi_broadcast_shape(
                        og_shape=self.vars[0].val.shape,
                        axis=self.kwargs["axis"] if "axis" in self.kwargs else None,
                    )
                    self_val = self_val.reshape(semi_broadcasted_shape)
                partial = with_respect_to.val == self_val
                return partial
            case "transpose" | "T":
                if type(with_respect_to.val) == np.ndarray:
                    return np.ones(with_respect_to.val.shape)
                return 1
            case "reshape":
                if type(with_respect_to.val) == np.ndarray:
                    return np.ones(with_respect_to.val.shape)
                return 1
            case "pad":
                if type(with_respect_to.val) == np.ndarray:
                    return np.ones(with_respect_to.val.shape)
                return 1
            case "im2col" | "col2im" | "im2col_pool" | "col2im_pool": return np.ones(self.val.shape)
            case "ln": return 1 / (with_respect_to.val + 1e-5) ## and self.vars[0] == with_respect_to
            case "log10": return 1 / (with_respect_to.val * np.log(10) + 1e-5)
            case "log2": return 1 / (with_respect_to.val * np.log(2) + 1e-5)
            case "cos": return -np.sin(with_respect_to.val) ## and self.vars[0] == with_respect_to
            case "sin": return np.cos(with_respect_to.val) ## and self.vars[0] == with_respect_to
            case "tan": return 1 / (np.cos(with_respect_to.val) ** 2 + 1e-5) ## and self.vars[0] == with_respect_to
            case "cosh": return np.sinh(with_respect_to.val) ## and self.vars[0] == with_respect_to
            case "sinh": return np.cosh(with_respect_to.val) ## and self.vars[0] == with_respect_to
            case "tanh": return 1 / (np.cosh(with_respect_to.val) ** 2 + 1e-5) ## and self.vars[0] == with_respect_to
            case "sqrt": return 1 / (2 * self.val + 1e-5) ## and self.vars[0] == with_respect_to
            case "exp": return self.val
            case "merge": return np.ones(with_respect_to.val.shape) if type(with_respect_to.val) == np.ndarray else 1
            case _:
                return self.vars[0].val

    def compute_adjoint(self):
        if not len(self.leafs):
            if self.adjoint is None:
                if type(self.val) == np.ndarray:
                    self.adjoint = np.ones(self.val.shape)
                    return
                self.adjoint = 1
            return
        if self.adjoint is None or self.op == "const": 
            self.adjoint = 0
        all_false_or_merge = True
        for lef in self.leafs:
            if (lef.op != "merge" and (type(lef.adjoint) == np.ndarray or lef.adjoint != "false")) or\
                (
                    lef.op == "merge" and\
                    (
                        (lef.kwargs["condition"] and lef.vars[0] == self) or\
                        (not lef.kwargs["condition"] and lef.vars[1] == self)
                    )
                ):
                all_false_or_merge = False
        if all_false_or_merge: 
            self.adjoint = "false"
            return
        for leaf in self.leafs:
            if leaf.adjoint is not None and (type(leaf.adjoint) == np.ndarray or leaf.adjoint != "false"):
                match (leaf.op):
                    case "slice" | "[]":
                        input_shape = self.val.shape
                        self_adjoint = np.zeros(input_shape)
                        indices = self.kwargs["indices"]
                        self_adjoint[indices] = leaf.adjoint
                        self.adjoint += self_adjoint
                    case "merge":
                        if  (
                                (leaf.kwargs["condition"] and leaf.vars[0] == self) or\
                                (not leaf.kwargs["condition"] and leaf.vars[1] == self)
                            ):
                            partial = leaf.partial_derivative(self)
                            self.adjoint += leaf.adjoint * partial
                    case "@" | "matmul":
                        if type(leaf.adjoint) != np.ndarray or not bool(leaf.adjoint.shape): 
                            leaf.adjoint = np.ones(leaf.val.shape)
                        partial_derivative = leaf.partial_derivative(self)
                        if self == leaf.vars[0]: self_adjoint = leaf.adjoint @ partial_derivative
                        else: self_adjoint = partial_derivative @ leaf.adjoint
                        if self.val.ndim != leaf.val.ndim or matmul_batch_axes_match(self.val.shape, leaf.val.shape):
                            reduce_axes = infer_reduction_axis(leaf.val.shape, self.val.shape)
                            self.adjoint += self_adjoint.sum(axis=reduce_axes)
                    case "sum"|"prod"|"+"|"-"|"/"|"*"|"max"|"min"|"mean"|"avg":
                        partial_derivative = leaf.partial_derivative(self)
                        if len(leaf.vars) == 1 :
                            if type(self.val) == np.ndarray: 
                                reshaped_leaf_adjoint = leaf.adjoint
                                keepdims_shape = eval_reduce_semi_broadcast_shape(
                                    og_shape=self.val.shape,
                                    axis=leaf.kwargs.get("axis", None),
                                )
                                if type(leaf.adjoint) == np.ndarray and leaf.adjoint.shape != keepdims_shape:
                                    reshaped_leaf_adjoint = leaf.adjoint.reshape(keepdims_shape)
                                self_adjoint = reshaped_leaf_adjoint * partial_derivative
                                self_adjoint = np.broadcast_to(self_adjoint, self.val.shape)
                            else:
                                self_adjoint = leaf.adjoint * partial_derivative
                            
                        if len(leaf.vars) == 2:
                            if type(leaf.adjoint) == np.ndarray and (type(self.val) != np.ndarray or self.val.shape != leaf.adjoint.shape):
                                axes = infer_reduction_axis(
                                    leaf.adjoint.shape,
                                    self.val.shape if type(self.val) == np.ndarray else len(leaf.adjoint.shape)*[1]
                                )
                                self_adjoint = leaf.adjoint * partial_derivative
                                self_adjoint = self_adjoint.sum(axis=axes, keepdims=True)
                            else:
                                self_adjoint = leaf.adjoint * partial_derivative

                        self.adjoint += self_adjoint
                    case "transpose" | "T": self.adjoint += leaf.adjoint.transpose(leaf.kwargs["axes"]) * leaf.partial_derivative(self)
                    case "reshape": self.adjoint += leaf.adjoint.reshape(self.val.shape) * leaf.partial_derivative(self)
                    case "pad": self.adjoint += reverse_pad(self.val, leaf.kwargs["pad_width"]) * leaf.partial_derivative(self)
                    case "concat" | "concatenate":
                        axis = leaf.kwargs.get("axis", -1)
                        x_indices = [x_idx for x_idx, var in enumerate(leaf.vars) if var == self]
                        cumusum = np.cumsum([var.val.shape[axis] for var in leaf.vars])[:-1]
                        split_adjoint = np.split(leaf.adjoint, cumusum, axis=axis)
                        sum_adjoint = np.zeros(self.val.shape)
                        for i in x_indices:
                            sum_adjoint += split_adjoint[i]
                        self.adjoint += sum_adjoint * leaf.partial_derivative(self)
                    case "im2col":
                        self_adjoint = leaf.adjoint * leaf.partial_derivative(self)
                        self_adjoint = col2im(
                            col_matrix=self_adjoint,
                            input_shape=self.val.shape,
                            kernal_shape=leaf.kwargs["kernal_shape"],
                            stride=leaf.kwargs["stride"],
                            padding=leaf.kwargs["padding"]
                        )
                        self.adjoint += self_adjoint
                    case "col2im":
                        self_adjoint = leaf.adjoint * leaf.partial_derivative(self)
                        self_adjoint = im2col(
                            input_tensor=self_adjoint,
                            kernal_shape=leaf.kwargs["kernal_shape"],
                            stride=leaf.kwargs["stride"],
                            padding=leaf.kwargs["padding"]
                        )
                        self.adjoint += self_adjoint
                    case "im2col_pool":
                        self_adjoint = leaf.adjoint * leaf.partial_derivative(self)
                        self_adjoint = col2im_pool(
                            col=self_adjoint,
                            input_shape=self.val.shape,
                            pool_shape=leaf.kwargs["pool_shape"],
                            stride=leaf.kwargs["stride"]
                        )
                        self.adjoint += self_adjoint
                    case "col2im_pool":
                        self_adjoint = leaf.adjoint * leaf.partial_derivative(self)
                        self_adjoint = im2col_pool(
                            input_tensor=self_adjoint,
                            pool_shape=leaf.kwargs["pool_shape"],
                            stride=leaf.kwargs["stride"]
                        )
                        self.adjoint += self_adjoint
                    case _:
                        self.adjoint += leaf.adjoint * leaf.partial_derivative(self)

        if type(self.val) == np.ndarray and type(self.adjoint) == np.ndarray and self.val.size == self.adjoint.size:
            self.adjoint = self.adjoint.reshape(self.val.shape)
        
        if type(self.val) != np.ndarray and type(self.adjoint) == np.ndarray and self.adjoint.size == 1:
            self.adjoint = self.adjoint.reshape((-1))[0]

    def compute(self):
        if not len(self.vars): return
        
        match(self.op):
            case "const": return
            case "neg":
                if len(self.vars) > 1: raise Exception("negation op only takes one input")
                self.val = -1 * self.vars[0].val
            case "==" | "equals":
                if len(self.vars) == 2:
                    if type(self.vars[0].val) == type(self.vars[1].val):
                        if type(self.vars[0].val) == np.ndarray and self.vars[0].val.shape != self.vars[1].val.shape:
                            raise Exception("inputs have mismatching shapes")
                        self.val = self.vars[0].val == self.vars[1].val
                    else: raise Exception("inputs have mismatching types")
                else: raise Exception("equals op only accepts two inputs")
            case ">=" | "gte":
                if len(self.vars) == 2:
                    if type(self.vars[0].val) == type(self.vars[1].val):
                        if type(self.vars[0].val) == np.ndarray and self.vars[0].val.shape != self.vars[1].val.shape:
                            raise Exception("inputs have mismatching shapes")
                        self.val = self.vars[0].val >= self.vars[1].val
                    else: raise Exception("inputs have mismatching types")
                else: raise Exception("gte op only accepts two inputs")
            case ">" | "gt":
                if len(self.vars) == 2:
                    if type(self.vars[0].val) == type(self.vars[1].val):
                        if type(self.vars[0].val) == np.ndarray and self.vars[0].val.shape != self.vars[1].val.shape:
                            raise Exception("inputs have mismatching shapes")
                        self.val = self.vars[0].val > self.vars[1].val
                    else: raise Exception("inputs have mismatching types")
                else: raise Exception("gte op only accepts two inputs")
            case "<=" | "lte":
                if len(self.vars) == 2:
                    if type(self.vars[0].val) == type(self.vars[1].val):
                        if type(self.vars[0].val) == np.ndarray and self.vars[0].val.shape != self.vars[1].val.shape:
                            raise Exception("inputs have mismatching shapes")
                        self.val = self.vars[0].val <= self.vars[1].val
                    else: raise Exception("inputs have mismatching types")
                else: raise Exception("gte op only accepts two inputs")
            case "<" | "lt":
                if len(self.vars) == 2:
                    if type(self.vars[0].val) == type(self.vars[1].val):
                        if type(self.vars[0].val) == np.ndarray and self.vars[0].val.shape != self.vars[1].val.shape:
                            raise Exception("inputs have mismatching shapes")
                        self.val = self.vars[0].val < self.vars[1].val
                    else: raise Exception("inputs have mismatching types")
                else: raise Exception("gte op only accepts two inputs")
            case "concat" | "concatenate":
                if len(self.vars) <= 1: raise Exception("concat requires at least 2 or more inputs")
                axis = self.kwargs.get("axis", -1)
                if type(axis) != int: raise Exception("concat only accepts one axis")
                self.val = np.concatenate((var.val for var in self.vars ), axis=axis)
            case "mean" | "avg":
                if len(self.vars) == 1:
                    if type(self.vars[0].val) == np.ndarray:
                        self.val = self.vars[0].val.mean(**self.kwargs)
                    else:
                        self.val = self.vars[0].val
                else:
                    summ = 0
                    for root in self.vars:
                        summ = summ + root.val
                    self.val = summ / len(self.vars)
            case "+" | "sum":
                if len(self.vars) == 1:
                    if type(self.vars[0].val) == np.ndarray:
                        self.val = self.vars[0].val.sum(**self.kwargs)
                    else:
                        self.val = self.vars[0].val
                else:
                    summ = 0
                    for root in self.vars:
                        summ = summ + root.val
                    self.val = summ
            case "-":
                if len(self.vars) == 2:
                    if self.vars[0] == self.vars[1]: self.val = 0
                    else: self.val = self.vars[0].val - self.vars[1].val
                else: raise Exception("minus op only accepts two inputs")
            case "*" | "prod":
                if len(self.vars) == 1:
                    if type(self.vars[0].val) == np.ndarray:
                        self.val = self.vars[0].val.prod(**self.kwargs)
                    else:
                        self.val = self.vars[0].val
                else:
                    prod = 1
                    for root in self.vars:
                        prod = prod * root.val
                    self.val = prod
            case "@" | "matmul":
                if len(self.vars) == 2:
                    if type(self.vars[0].val) == type(self.vars[1].val):
                        if type(self.vars[0].val) == np.ndarray: 
                            self.val = self.vars[0].val @ self.vars[1].val
                        else: self.val = self.vars[0].val * self.vars[1].val
                    else: raise Exception("mismatching input types in matmul")
                else: raise Exception("matmul op only accepts two inputs")
            case "/":
                if len(self.vars) == 2:
                    if self.vars[0] == self.vars[1]: self.val = 1
                    else: self.val = self.vars[0].val / self.vars[1].val
                else: raise Exception("divide op only accepts two inputs")
            case "**":
                if len(self.vars) == 2:
                    base = self.vars[0].val
                    power = self.vars[1].val
                    self.val = base ** power
                else: raise Exception("pow op only accepts two inputs")
            case "max":
                if len(self.vars) == 1:
                    if type(self.vars[0].val) == np.ndarray:
                        self.val = self.vars[0].val.max(**self.kwargs)
                    else:
                        self.val = self.vars[0].val
                else:
                    if any(map(lambda var: type(var.val) == np.ndarray, self.vars)):
                        if len(self.vars) > 2: raise Exception("You can only compare two ndarrays using max op !")
                        self.val = np.maximum(self.vars[0].val, self.vars[1].val)
                    else: self.val = max(list(map(lambda var: var.val, self.vars)))
            case "min":
                if len(self.vars) == 1:
                    if type(self.vars[0].val) == np.ndarray:
                        self.val = self.vars[0].val.min(**self.kwargs)
                    else:
                        self.val = self.vars[0].val
                else:
                    if any(map(lambda var: type(var.val) == np.ndarray, self.vars)):
                        if len(self.vars) > 2: raise Exception("You can only compare two ndarrays using min op !")
                        self.val = np.minimum(self.vars[0].val, self.vars[1].val)
                    else: self.val = min(list(map(lambda var: var.val, self.vars)))
            
            case "transpose" | "T":
                if len(self.vars) != 1: raise Exception("transpose operation only takes one tensor input")
                if type(self.vars[0].val) == np.ndarray:
                    if "axes" in self.kwargs:
                        self.val = self.vars[0].val.transpose(self.kwargs["axes"])
                    else:
                        self.val = self.vars[0].val.T
                else:
                    self.val = self.vars[0].val
            case "reshape":
                if len(self.vars) != 1: raise Exception("reshape operation only takes one tensor input")
                if type(self.vars[0].val) == np.ndarray:
                    if "shape" in self.kwargs:
                        self.val = self.vars[0].val.reshape(self.kwargs["shape"])
                    else:
                        raise Exception("reshape function requires a tuple as an arg")
                else:
                    self.val = self.vars[0].val
            case "pad":
                if len(self.vars) != 1: raise Exception("pad operation only takes one tensor input")
                if "pad_width" not in self.kwargs: raise Exception("pad operation requires argument 'pad_width'")
                pad_width = self.kwargs["pad_width"]
                inp = self.vars[0].val
                if type(inp) == np.ndarray:
                    self.val = np.pad(inp, pad_width)
                else:
                    broad = np.broadcast_to(inp, [1]*len(pad_width))
                    self.val = np.pad(broad, pad_width)
            case "im2col":
                if len(self.vars) != 1: raise Exception("im2col operation only takes one tensor input")
                if not ("kernal_shape" in self.kwargs and "stride" in self.kwargs and "padding" in self.kwargs):
                    raise Exception("im2col requires all of these arguments ['kernal_shape', 'stride', 'padding']")
                self.val = im2col(
                    input_tensor=self.vars[0].val, 
                    kernal_shape=self.kwargs["kernal_shape"],
                    stride=self.kwargs["stride"],
                    padding=self.kwargs["padding"]
                )
            case "col2im":
                if len(self.vars) != 1: raise Exception("col2im operation only takes one matrix input")
                if not ("kernal_shape" in self.kwargs and "stride" in self.kwargs and "padding" in self.kwargs):
                    raise Exception("col2im requires all of these arguments ['input_shape', 'kernal_shape', 'stride', 'padding']")
                self.val = col2im(
                    col_matrix=self.vars[0].val,
                    input_shape=self.kwargs["input_shape"],
                    kernal_shape=self.kwargs["kernal_shape"],
                    stride=self.kwargs["stride"],
                    padding=self.kwargs["padding"]
                )
            case "im2col_pool":
                if len(self.vars) != 1: raise Exception("im2col_pool operation only takes one tensor input")
                if not ("kernal_shape" in self.kwargs and "stride" in self.kwargs and "padding" in self.kwargs):
                    raise Exception("im2col_pool requires all of these arguments ['pool_shape', 'stride']")
                self.val = im2col_pool(
                    input_tensor=self.vars[0].val, 
                    pool_shape=self.kwargs["pool_shape"],
                    stride=self.kwargs["stride"]
                )
            case "col2im_pool":
                if len(self.vars) != 1: raise Exception("col2im_pool operation only takes one matrix input")
                if not ("kernal_shape" in self.kwargs and "stride" in self.kwargs and "padding" in self.kwargs):
                    raise Exception("col2im_pool requires all of these arguments ['input_shape', 'pool_shape', 'stride']")
                self.val = col2im_pool(
                    col=self.vars[0].val,
                    input_shape=self.kwargs["input_shape"],
                    pool_shape=self.kwargs["pool_shape"],
                    stride=self.kwargs["stride"]
                )
            case "ln": self.val = np.log(self.vars[0].val, **self.kwargs)
            case "log10": self.val = np.log10(self.vars[0].val, **self.kwargs)
            case "log2": self.val = np.log2(self.vars[0].val, **self.kwargs)
            case "cos": self.val = np.cos(self.vars[0].val, **self.kwargs)
            case "sin": self.val = np.sin(self.vars[0].val, **self.kwargs)
            case "tan": self.val = np.tan(self.vars[0].val, **self.kwargs)
            case "cosh": self.val = np.cosh(self.vars[0].val, **self.kwargs)
            case "sinh": self.val = np.sinh(self.vars[0].val, **self.kwargs)
            case "tanh": self.val = np.tanh(self.vars[0].val, **self.kwargs)
            case "sqrt": self.val = np.sqrt(self.vars[0].val, **self.kwargs)
            case "exp": self.val = np.exp(self.vars[0].val, **self.kwargs)
            case "merge":
                if len(self.vars) != 2: raise Exception("merge accepts two args, first for when the condition is true, second for false")
                if "condition" not in self.kwargs: raise Exception("no condition was provided to merge node !!")
                self.val = self.vars[not self.kwargs["condition"]].val
            case "slice" | "[]":
                if len(self.vars) != 1: raise Exception("slice op only excepts one input node")
                indices = self.kwargs["indices"]
                self.val = self.vars[0].val[indices]
            case _:
                raise Exception("Invalid Operation !!")
        return self.val

    def graph_forward(self, input_values=[], local_inputs_only=False):
        if not len(self.inputs): raise Exception("Cannont compute result of graph without setting the graph input nodes")
        if len(self.inputs) != len(input_values): raise Exception("input length mismatch")
        for input_node, input_value in zip(self.inputs, input_values):
            input_node.substitute(input_value)
        ordered_operations, _ = topological_DFS(self)
        ordered_operations = [node for node in ordered_operations if node.op != "const"]
        self.ordered_operations = ordered_operations
        if local_inputs_only:
            limited_operations = []
            num_input_found = 0
            for op in ordered_operations[::-1]:
                if op in self.inputs:
                    num_input_found += 1
                    if num_input_found < len(self.inputs): continue
                    else: break
                limited_operations.append(op)
            ordered_operations = reversed(limited_operations)
        for op in ordered_operations:
            if op.op != "var" and op.op != "const":
                op.compute()
                if op._cycle is not None:
                    op._cycle.val = op.val
            print("-->", op, op.val.shape)

        return self.val
    
    def graph_backward(self, y_adjoint=None):
        if y_adjoint is not None: self.adjoint = y_adjoint
        # ordered_operations, _ = branchless_DFS(self)
        ordered_operations, _ = topological_DFS(self)
        ordered_operations = [node for node in ordered_operations if node.op != "const"]
        self.ordered_operations = ordered_operations
        if any(map(lambda node: node.val is None, ordered_operations)):
            raise Exception("Forward pass didnt fill in all of the values of the nodes")
        
        trainable_params = []
        gradients = []
        # print(ordered_operations)
        for op in reversed(ordered_operations):
            # if op.op == "var" and not op.is_param: continue
            if op != self or y_adjoint is None:
                op.adjoint = None
                op.compute_adjoint()
                # print("backward>",op, ":", op.adjoint, op.is_param)
            if op.is_param and op.op == "var" and not op.frozen and not op.op == "const":
                trainable_params.append(op)
                gradients.append(op.adjoint)

        return trainable_params, gradients
    
    def graph_freeze(self, local_inputs_only=True):
        if local_inputs_only and not len(self.inputs): raise Exception("Cannont freeze local inputs, without defining the local inputs of the graph")
        ordered_operations, _ = topological_DFS(self)
        ordered_operations = [node for node in ordered_operations if node.op != "const"]
        if local_inputs_only:
            limited_operations = []
            num_input_found = 0
            for op in ordered_operations[::-1]:
                if op in self.inputs:
                    num_input_found += 1
                    if num_input_found < len(self.inputs): continue
                    else: break
                limited_operations.append(op)
            ordered_operations = reversed(limited_operations)

        for op in ordered_operations:
            if op.is_param and op.op == "var":
                op.frozen = True

        return self
    
    def graph_unfreeze(self, local_inputs_only=True):
        if local_inputs_only and not len(self.inputs): raise Exception("Cannont freeze local inputs, without defining the local inputs of the graph")
        ordered_operations, _ = topological_DFS(self)
        ordered_operations = [node for node in ordered_operations if node.op != "const"]
        if local_inputs_only:
            limited_operations = []
            num_input_found = 0
            for op in ordered_operations[::-1]:
                if op in self.inputs:
                    num_input_found += 1
                    if num_input_found < len(self.inputs): continue
                    else: break
                limited_operations.append(op)
            ordered_operations = reversed(limited_operations)

        for op in ordered_operations:
            if op.is_param and op.op == "var":
                op.frozen = False

        return self

    def reset_gradients(self):
        ordered_operations, _ = topological_DFS(self)
        for op in ordered_operations:
            op.adjoint = None

    def __hash__(self):
        return hash((self._id, self.prefix, self.name))

    def __eq__(self, other):
        return isinstance(other, OpNode) and self._id == other._id and self.op == other.op and self.name == other.name and self.prefix == other.prefix
    
    def __repr__(self) -> str:
        lim = len(self.op) + 6
        name = self.name[:lim]
        if lim < len(self.name):
            name += "..."
        is_param = ""
        if self.op == "var" and self.is_param:
            is_param = "{param}"
        return f'{self.prefix}["{name}"].{self.op}' + is_param