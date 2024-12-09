import uuid
import numpy as np

def im2col(input_tensor, filter_size, stride=1, padding=0):
    batch_size, num_channels, img_height, img_width = input_tensor.shape
    filter_height, filter_width = filter_size

    # Compute output dimensions
    out_height = (img_height + 2 * padding - filter_height) // stride + 1
    out_width = (img_width + 2 * padding - filter_width) // stride + 1

    # Add padding to the input
    if padding > 0:
        input_tensor = np.pad(
            input_tensor,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant',
            constant_values=0
        )

    # Create the im2col matrix
    col_matrix = np.zeros((batch_size, num_channels, filter_height, filter_width, out_height, out_width))

    # Fill the im2col matrix
    for h in range(filter_height):
        for w in range(filter_width):
            col_matrix[:, :, h, w, :, :] = input_tensor[
                :, 
                :, 
                h:h + stride * out_height:stride, 
                w:w + stride * out_width:stride
            ]
    # Reshape to create the column matrix
    col_matrix = col_matrix.transpose(0, 4, 5, 1, 2, 3)  # (batch_size, out_height, out_width, num_channels, filter_height, filter_width)
    col_matrix = col_matrix.reshape(batch_size, out_height * out_width, -1)  # (batch_size, num_patches, num_channels * filter_height * filter_width)
    return col_matrix


def col2im(col_matrix, input_shape, filter_size, stride=1, padding=0):
    batch_size, num_channels, img_height, img_width = input_shape
    filter_height, filter_width = filter_size

    # Compute output dimensions
    out_height = (img_height + 2 * padding - filter_height) // stride + 1
    out_width = (img_width + 2 * padding - filter_width) // stride + 1

    # Initialize output tensor with zeros
    padded_height = img_height + 2 * padding
    padded_width = img_width + 2 * padding
    output = np.zeros((batch_size, num_channels, padded_height, padded_width))

    # Reshape col_matrix to extract patches
    col_matrix = col_matrix.reshape(batch_size, out_height, out_width, num_channels, filter_height, filter_width)
    col_matrix = col_matrix.transpose(0, 3, 4, 5, 1, 2)  # (batch_size, num_channels, filter_height, filter_width, out_height, out_width)

    # Accumulate gradients for each patch
    for h in range(filter_height):
        for w in range(filter_width):
            output[:, :, h:h + stride * out_height:stride, w:w + stride * out_width:stride] += col_matrix[:, :, h, w, :, :]

    # Remove padding if applied
    if padding > 0:
        output = output[:, :, padding:-padding, padding:-padding]

    return output


def topological_DFS(node, visited=None, ordered_nodes=None, levels=None, level=None, key=lambda x: x.vars):
    if visited is None:
        visited = set()
    if ordered_nodes is None:
        ordered_nodes = []
    if levels is None:
        levels = []
    if level is None:
        level = 0

    if node not in visited:
        for root in key(node):
            topological_DFS(root, visited, ordered_nodes, levels, level + 1, key=key)
        ordered_nodes.append(node)
        levels.append(level)
        visited.add(node)
    return ordered_nodes, levels

def eval_reduce_semi_broadcast_shape(og_shape, axis):
    if type(axis) not in [tuple, list]: axis=(axis,)
    shape_ls = list(og_shape) 
    for ax in axis: shape_ls[ax] = 1
    semi_broadcasted_shape = tuple(shape_ls)
    return semi_broadcasted_shape


def infer_reduction_axis(adjoint_shape, x_shape):
    return tuple([i for i, (dz_dim, v_dim) in enumerate(zip(adjoint_shape, x_shape)) if v_dim == 1 and dz_dim > 1])


class OpNode:
    NUMERIC_TYPES = [
        int, float, np.bool_, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64, np.float16,
        np.float32, np.float64, np.complex64, np.complex128,
    ]
    CONSTANT_TYPES = NUMERIC_TYPES + [np.ndarray]

    def __init__(self, op, prefix="", constant=None, is_input=False, **kwargs):
        if op == "const" and constant is None:
            raise Exception("Const node needs to be assigned a value !!")
        self.name = f"[{prefix}]{op}_{str(uuid.uuid4())}"
        self.op = op
        self.vars = []
        self.leafs = set()
        self.val = constant
        self.adjoint = None
        self.kwargs = kwargs
        self.is_input = is_input
        self.inputs = []
        self.outputs = []
        self.frozen = False
        if "axis" in self.kwargs and type(self.kwargs["axis"]) not in [list, tuple]:
            self.kwargs["axis"] = (self.kwargs["axis"])

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

    def __neg__(self):
        return OpNode("neg")(self)

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
                        ndim = self.vars[0].val.ndim
                        dims = list(range(ndim))
                        dims[-2], dims[-1] = dims[-1], dims[-2]
                        return self.vars[0].val.transpose(dims)
                else:
                    count = 0
                    for var in self.vars:
                        if var == with_respect_to:
                            count += 1
                    return count * self.val / with_respect_to.va
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
                if len(self.vars) == 1 and type(self.vars[0].val) == np.ndarray:
                # if type(with_respect_to.val) == np.ndarray:
                    semi_broadcasted_shape = eval_reduce_semi_broadcast_shape(
                        og_shape=self.vars[0].val.shape,
                        axis=self.kwargs["axis"] if "axis" in self.kwargs else None,
                    )
                    self_val = self_val.reshape(semi_broadcasted_shape)
                return with_respect_to.val == self_val
            case "min": 
                self_val = self.val
                if len(self.vars) == 1 and type(self.vars[0].val) == np.ndarray:
                    semi_broadcasted_shape = eval_reduce_semi_broadcast_shape(
                        og_shape=self.vars[0].val.shape,
                        axis=self.kwargs["axis"] if "axis" in self.kwargs else None,
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
            case "im2col": return np.ones(self.val.shape)
            case "col2im": return np.ones(self.val.shape)
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
                return self.vars[0].val

    def compute_adjoint(self):
        if not len(self.leafs):
            if self.adjoint is None:
                self.adjoint = 1
            return
        if self.adjoint is None:
            self.adjoint = 0
        for leaf in self.leafs:
            if leaf.adjoint is not None:
                match (leaf.op):
                    case "@" | "matmul":
                        if self == leaf.vars[0]: self.adjoint += leaf.adjoint @ leaf.partial_derivative(self)
                        else: self.adjoint += leaf.partial_derivative(self) @ leaf.adjoint
                    case "sum"|"prod"|"+"|"*"|"max"|"min":
                        partial_derivative = leaf.partial_derivative(self)
                        broadcasted_adjoint = leaf.adjoint
                        if len(leaf.vars) == 1 :
                            if type(self.val) == np.ndarray: 
                                reshaped_leaf_adjoint = leaf.adjoint
                                if type(leaf.adjoint) == np.ndarray:
                                    reshaped_leaf_adjoint = leaf.adjoint.reshape(partial_derivative.shape)
                                broadcasted_adjoint = np.broadcast_to(reshaped_leaf_adjoint, self.val.shape)

                        self.adjoint += broadcasted_adjoint * partial_derivative
                        if len(leaf.vars) == 2:
                            if type(leaf.adjoint) == np.ndarray and type(self.val) != np.ndarray:
                                axes = infer_reduction_axis(
                                    leaf.adjoint.shape,
                                    self.val.shape if type(self.val) == np.ndarray else len(leaf.adjoint.shape)*[1]
                                )
                                self.adjoint = np.sum(self.adjoint, axis=axes)
                            elif (type(self.val) == np.ndarray and self.val.shape != leaf.adjoint.shape):
                                axes = infer_reduction_axis(
                                    leaf.adjoint.shape,
                                    self.val.shape if type(self.val) == np.ndarray else len(leaf.adjoint.shape)*[1]
                                )
                                self.adjoint = np.sum(self.adjoint, axis=axes)
                    case "transpose" | "T": self.adjoint += leaf.adjoint.transpose(leaf.kwargs["axes"]) * leaf.partial_derivative(self)
                    case "reshape": self.adjoint += leaf.adjoint.reshape(self.val.shape) * leaf.partial_derivative(self)
                    case "im2col":
                        transformed_adjoint = col2im(
                            col_matrix=leaf.adjoint,
                            input_shape=self.val.shape,
                            filter_size=leaf.kwargs["kernal_shape"],
                            stride=leaf.kwargs["stride"],
                            padding=leaf.kwargs["padding"]
                        )
                        self.adjoint += transformed_adjoint * leaf.partial_derivative(self)
                    case "col2im":
                        transformed_adjoint = im2col(
                            col_matrix=leaf.adjoint,
                            filter_size=leaf.kwargs["kernal_shape"],
                            stride=leaf.kwargs["stride"],
                            padding=leaf.kwargs["padding"]
                        )
                        self.adjoint += transformed_adjoint * leaf.partial_derivative(self)
                    case _: self.adjoint += leaf.adjoint * leaf.partial_derivative(self)
    
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
                        print(root, root.val.shape)
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
            case "im2col":
                if len(self.vars) != 1: raise Exception("im2col operation only takes one tensor input")
                if not ("kernal_shape" in self.kwargs and "stride" in self.kwargs and "padding" in self.kwargs):
                    raise Exception("im2col requires all of these arguments ['kernal_shape', 'stride', 'padding']")
                self.val = im2col(
                    input_tensor=self.vars[0].val, 
                    filter_size=self.kwargs["kernal_shape"],
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
                    filter_size=self.kwargs["kernal_shape"],
                    stride=self.kwargs["stride"],
                    padding=self.kwargs["padding"]
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
            case _:
                raise Exception("Invalid Operation !!")
        return self.val

    def graph_forward(self, input_values=[], local_inputs_only=False):
        if not len(self.inputs): raise Exception("Cannont compute result of graph without setting the graph input nodes")
        if len(self.inputs) != len(input_values): raise Exception("input length mismatch")
        for input_node, input_value in zip(self.inputs, input_values):
            input_node.substitute(input_value)
        ordered_operations, _ = topological_DFS(self)
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
            op.compute()
            # print(op, ":", op.val)

        return self.val
    
    def graph_backward(self, y_adjoint=None):
        if y_adjoint is not None: self.adjoint = y_adjoint
        ordered_operations, _ = topological_DFS(self)
        if any(map(lambda node: node.val is None, ordered_operations)):
            raise Exception("Forward pass didnt fill in all of the values of the nodes")
        
        trainable_params = []
        gradients = []
        for op in reversed(ordered_operations):
            if op.is_input: continue
            if op != self or y_adjoint is None:
                op.adjoint = None
                op.compute_adjoint()
            # print(op, "ADJOINT:", op.adjoint.shape if type(op.adjoint) == np.ndarray else op.adjoint)
            if not op.is_input and op.op == "var" and not op.frozen:
                trainable_params.append(op)
                gradients.append(op.adjoint)

        return trainable_params, gradients
    
    def graph_freeze(self, local_inputs_only=True):
        if local_inputs_only and not len(self.inputs): raise Exception("Cannont freeze local inputs, without defining the local inputs of the graph")
        ordered_operations, _ = topological_DFS(self)
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
            op.frozen = True

        return self

    def reset_gradients(self):
        ordered_operations, _ = topological_DFS(self)
        for op in ordered_operations:
            op.adjoint = None
            
    def backpropagate(self, callback):
        ordered_operations, _ = topological_DFS(self)
        if any(map(lambda node: node.val is None, ordered_operations)):
            raise Exception("Forward pass didnt fill in all of the values of the nodes")
        
        for op in reversed(ordered_operations):
            if op.is_input or op.op != "var" or op.frozen: continue
            callback(op)
            # print(f"[{op}] adjoint:", op.adjoint)
            # print("---------------------")
        
    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        return isinstance(other, OpNode) and self.name == other.name and self.op == other.op
    
    def __repr__(self) -> str:
        lim = len(self.op) + 14
        name = self.name[:lim]
        is_input = ""
        if self.op == "var" and self.is_input:
            is_input = "[input]"
        return f"{name}({self.op})" + is_input