import uuid
import numpy as np

from collections.abc import Iterable

def eval_reduce_semi_broadcast_shape(og_shape, axis):
    if type(axis) not in [tuple, list]: axis=(axis,)
    shape_ls = list(og_shape) 
    for ax in axis: shape_ls[ax] = 1
    semi_broadcasted_shape = tuple(shape_ls)
    return semi_broadcasted_shape

class OpNode:
    NUMERIC_TYPES = [
        int, float, np.bool_, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16,
        np.float32, np.float64, np.complex64, np.complex128,
    ]
    CONSTANT_TYPES = NUMERIC_TYPES + [np.ndarray]

    def __init__(self, op, constant=None, **kwargs):
        if op == "const" and constant is None:
            raise Exception("Const node needs to be assigned a value !!")
        self.name = f"{op}_{str(uuid.uuid4())}"
        self.op = op
        self.vars = []
        self.leafs = set()
        self.val = constant
        self.adjoint = None
        self.kwargs = kwargs 
        self.inputs = []
        self.outputs = []
        if "axis" in self.kwargs and type(self.kwargs["axis"]) not in [list, tuple]:
            self.kwargs["axis"] = (self.kwargs["axis"])

    def set_graph_inputs(self, inputs: list):
        self.inputs = inputs
        return self
    
    def set_graph_outputs(self, outputs: list):
        self.outputs = outputs
        return self

    def substitute(self, val):
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
                op_arg.leafs.add(self)
        return self

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
    def __matmul__(node1, node2):
        op_node1 = node1
        if type(node1) in OpNode.CONSTANT_TYPES:
            op_node1 = OpNode("const", constant=node1)
        op_node2 = node2
        if type(node2) in OpNode.CONSTANT_TYPES:
            op_node2 = OpNode("const", constant=node2)
        res = OpNode("@")(op_node1, op_node2)
        return res
    
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
    def __pow__(node1, node2):
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

    def equals(self, other): return OpNode(op="==")(self, other)
    def sum(self, axis): return OpNode(op="+")(self, axis=axis)
    def prod(self, axis): return OpNode(op="*")(self, axis=axis)
    def transpose(self, axes): return OpNode(op="T")(self, axes=axes)
    def T(self, axes): return OpNode(op="T")(self, axes=axes)
    def max(self, axis): return OpNode(op="max")(self, axis=axis)
    def min(self, axis): return OpNode(op="min")(self, axis=axis)
    def reshape(self, shape): return OpNode(op="reshape")(self, shape=shape)

    def partial_derivative(self, with_respect_to):
        if not len(self.vars): return None
        if with_respect_to not in self.vars: return 0
        # if not len(self.leafs): return 1
        match(self.op):
            case "const":
                return 0
            case "neg":
                if type(self.val) == np.ndarray:
                    return -1 * np.ones(self.val.shape)
                return -1
            case "==" | "equals":
                if type(self.val) == np.ndarray:
                    return np.zeros(self.val.shape)
                return 0
            case "+" | "sum":
                if len(self.vars) == 1 and type(self.vars[0].val) == np.ndarray:
                    return np.ones(self.vars[0].val.shape)
                count = 0
                for var in self.vars:
                    if var == with_respect_to:
                        count += 1
                if type(self.val) == np.ndarray:
                    return np.ones(self.vars[0].val.shape) * count
                return count
            case "-":
                scalar = 1
                if len(self.vars) == 2:
                    if self.vars[0] == self.vars[1]: ## and self.vars[0] == with_respect_to
                        scalar = 0
                    elif self.vars[0] == with_respect_to:
                        scalar = 1
                    else:
                        scalar = -1
                    if type(self.val) == np.ndarray:
                        return np.ones(self.val.shape) * scalar
                    return scalar
                else: raise Exception("minus op only accepts two inputs")
            case "*" | "prod":
                if len(self.vars) == 1 and type(self.vars[0].val) == np.ndarray:
                    semi_broadcasted_shape = eval_reduce_semi_broadcast_shape(
                        og_shape=self.vars[0].val.shape,
                        axis=self.kwargs["axis"],
                    )
                    return self.val.reshape(semi_broadcasted_shape) / self.vars[0].val
                count = 0
                for var in self.vars:
                    if var == with_respect_to:
                        count += 1
                return count * self.val / with_respect_to.val
            case "@" | "matmul":
                if type(self.vars[0].val) == np.ndarray and type(self.vars[1].val) == np.ndarray:
                    if self.vars[0] == with_respect_to:
                        ndim = self.vars[1].val.ndim
                        dims = list(range(ndim))
                        dims[-2], dims[-1] = dims[-1], dims[-2]
                        return self.vars[1].val.transpose(dims)
                else:
                    count = 0
                    for var in self.vars:
                        if var == with_respect_to:
                            count += 1
                    return count * self.val / with_respect_to.val
            case "/":
                if len(self.vars) == 2:
                    if self.vars[0] == self.vars[1]: ## and self.vars[0] == with_respect_to
                        if type(self.val) == np.ndarray:
                            return np.zeros(self.val.shape)
                        return 0
                    if self.vars[0] == with_respect_to:
                        return 1 / self.vars[1].val
                    return -1 * self.vars[0].val / (with_respect_to.val ** 2)
                else: raise Exception("divide op only accepts two inputs")
            case "**":
                if len(self.vars) == 2:
                    if self.vars[0] == self.vars[1]: ## and self.vars[0] == with_respect_to
                        return self.val * (1 + np.log(with_respect_to.val))
                    if self.vars[0] == with_respect_to:
                        return self.vars[1].val * self.val / with_respect_to.val
                    return self.val * np.log(self.vars[0].val)
                else: raise Exception("pow op only accepts two inputs")
            case "max":
                self_val = self.val
                if type(with_respect_to.val) == np.ndarray:
                    semi_broadcasted_shape = eval_reduce_semi_broadcast_shape(
                        og_shape=self.vars[0].val.shape,
                        axis=self.kwargs["axis"],
                    )
                    self_val = self_val.reshape(semi_broadcasted_shape)
                return with_respect_to.val == self_val
            case "min": 
                self_val = self.val
                if type(with_respect_to.val) == np.ndarray:
                    semi_broadcasted_shape = eval_reduce_semi_broadcast_shape(
                        og_shape=self.vars[0].val.shape,
                        axis=self.kwargs["axis"],
                    )
                    self_val = self_val.reshape(semi_broadcasted_shape)
                return with_respect_to.val == self_val
            case "transpose" | "T":
                if type(with_respect_to.val) == np.ndarray:
                    return np.ones(with_respect_to.val.shape)
                return 1
            case "reshape":
                if type(with_respect_to.val) == np.ndarray:
                    return np.ones(with_respect_to.val.shape)
                return 1
            case "ln": return 1 / with_respect_to.val ## and self.vars[0] == with_respect_to
            case "log10": return 1 / (with_respect_to.val * np.log(10))
            case "log2": return 1 / (with_respect_to.val * np.log(2))
            case "cos": return -np.sin(with_respect_to.val) ## and self.vars[0] == with_respect_to
            case "sin": return np.cos(with_respect_to.val) ## and self.vars[0] == with_respect_to
            case "tan": return 1 / (np.cos(with_respect_to.val) ** 2) ## and self.vars[0] == with_respect_to
            case "cosh": return np.sinh(with_respect_to.val) ## and self.vars[0] == with_respect_to
            case "sinh": return np.cosh(with_respect_to.val) ## and self.vars[0] == with_respect_to
            case "tanh": return 1 / (np.cosh(with_respect_to.val) ** 2) ## and self.vars[0] == with_respect_to
            case "sqrt": return 1 / (2 * self.val) ## and self.vars[0] == with_respect_to
            case "exp": return self.val
            case _:
                if len(self.vars[0]): self.val = self.vars[0].val

    def compute_adjoint(self):
        if not len(self.leafs):
            if self.adjoint is None:
                self.adjoint = 1
            return
        if self.adjoint is None:
            self.adjoint = 0
        for leaf in self.leafs:
            if leaf.adjoint:
                match (leaf.op):
                    case "@" | "matmul":
                        if self == leaf.vars[0]: self.adjoint += leaf.partial_derivative(self) @ leaf.adjoint
                        else: self.adjoint += leaf.adjoint @ leaf.partial_derivative(self) 
                    case "sum"|"prod"|"+"|"*"|"max"|"min":
                        partial_derivative = leaf.partial_derivative(self)
                        broadcasted_adjoint = np.broadcast_to(leaf.adjoint.reshape(partial_derivative.shape), self.val.shape)
                        self.adjoint += broadcasted_adjoint * partial_derivative
                    case "transpose" | "T": self.adjoint += leaf.adjoint.transpose(leaf.kwargs["axes"]) * leaf.partial_derivative(self)
                    case "reshape": self.adjoint += leaf.adjoint.reshape(self.val.shape) * leaf.partial_derivative(self)
                    case _: self.adjoint += leaf.adjoint * leaf.partial_derivative(self)
    
    def compute(self):
        if not len(self.vars): return
        match(self.op):
            case "const": return
            case "neg":
                if len(self.vars) > 1: raise Exception("negation op only takes one input")
                self.val = -1 * self.val
            case "==" | "equals":
                if len(self.vars) == 2:
                    if type(self.vars[0].val) == type(self.vars[1].val):
                        if type(self.vars[0].val) == np.ndarray and self.vars[0].val.shape != self.vars[1].val.shape:
                            raise Exception("inputs have mismatching shapes")
                        self.val = self.vars[0].val.shap != self.vars[1].val
                    else: raise Exception("inputs have mismatching types")
                else: raise Exception("equals op only accepts two inputs")
            case "+" | "sum":
                if len(self.vars) == 1:
                    if type(self.vars[0].val) == np.ndarray:
                        self.val = self.vars[0].val.sum(**self.kwargs)
                    else:
                        self.val = self.vars[0].val
                else:
                    summ = 0
                    for root in self.vars:
                        summ += root.val
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
                        prod *= root.val
                    self.val = prod
            case "@" | "matmul":
                if len(self.vars) == 2:
                    if type(self.vars[0].val) == type(self.vars[1].val):
                        if type(self.vars[0].val) == np.ndarray: self.val = self.vars[0].val.dot(self.vars[1].val)
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
                if type(self.vars[0]) == np.ndarray:
                    if "axes" in self.kwargs:
                        self.val = self.vars[0].val.transpose(self.kwargs["axes"])
                    else:
                        self.val = self.vars[0].val.T
                else:
                    self.val = self.vars[0].val
            case "reshape":
                if len(self.vars) != 1: raise Exception("reshape operation only takes one tensor input")
                if type(self.vars[0]) == np.ndarray:
                    if "shape" in self.kwargs:
                        self.val = self.vars[0].val.reshape(self.kwargs["shape"])
                    else:
                        raise Exception("reshape function requires a tuple as an arg")
                else:
                    self.val = self.vars[0].val
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
            case _:
                raise Exception("Invalid Operation !!")
        return self.val

    def graph_forward(self, input_values):
        if not len(self.inputs): raise Exception("Cannont compute result of graph without setting the graph input nodes")
        if len(self.inputs) != len(input_values): raise Exception("input length mismatch")
        for input_node, input_value in zip(self.inputs, input_values):
            input_node.substitute(input_value)
        ordered_operations = topological_DFS(self)
        for op in ordered_operations:
            op.compute()
            print(op, ":", op.val)

        return self.val
    
    def graph_backward(self, y_adjoint=1):
        self.adjoint = y_adjoint
        ordered_operations = topological_DFS(self)
        if any(map(lambda node: node.val is None, ordered_operations)):
            raise Exception("Forward pass didnt fill in all of the values of the nodes")
        
        gradients = []
        for op in reversed(ordered_operations):
            if op != self: op.adjoint = None
            op.compute_adjoint()
            gradients.append((op, op.adjoint))
            print(f"[{op}] adjoint:", op.adjoint)
            print("---------------------")

        return gradients
        
    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        return isinstance(other, OpNode) and self.name == other.name and self.op == other.op
    
    def __repr__(self) -> str:
        lim = len(self.op) + 4
        name = self.name[:lim]
        return f"{name}({self.op})"