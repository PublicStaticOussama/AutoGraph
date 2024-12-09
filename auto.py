from typing import Union, List
from abc import ABC, abstractmethod
import numpy as np
from autodiff_graph import OpNode

TYPES = Union[
    OpNode, int, float, np.bool_, np.int8,
    np.int16, np.int32, np.int64, np.uint8,
    np.uint16, np.uint32, np.uint64, np.float16,
    np.float32, np.float64, np.complex64, np.complex128,
    np.ndarray
]

def var(initial_val=None, is_input=False, prefix=""):
    x = OpNode(op="var", is_input=is_input, prefix=prefix)
    if initial_val is not None:
        x.substitute(initial_val)
    return x

def const(val):
    return OpNode(op="const", constant=val)

def sigmoid(X: OpNode = None, axis=-1):
    if X is None:
        X = var()
    sig = 1 / (1 + OpNode(op="exp", axis=axis)(-X))
    return sig.set_graph_inputs(inputs=[X])

# (b, c, w, h)
# (k, c, k_w, k_h)
# (b, k, (w - k_w + 2 * p) / s, (h - k_h + 2 * p) / s)

class Layer(ABC):
    def __init__(self, shape=None):
        self.shape: tuple = shape
        self.output_node: OpNode = None
        self.previous_layers = []

    def get_output_shape_(self): return self.shape
    
    def get_output_node_(self): return self.output_node

    @abstractmethod
    def __call__(self): pass

    @abstractmethod
    def forward(self): pass

    @abstractmethod
    def backward(self, y_grad=None): pass

    def __repr__(self):
        if_activation = ""
        if hasattr(self, 'activation'):
            if_activation = " | " + str(type(self.activation))
        return "Layer: " + str(type(self)) + " | " + str(self.get_output_shape_()) + if_activation

class ReLU:
    def __init__(self):
        pass

    def __call__(self, X_node: OpNode, prefix=""):
        self.y_node = OpNode(op="max", prefix=prefix)(0, X_node).set_graph_inputs(inputs=[X_node])
        return self.y_node

    def forward(self, X):
        return self.y_node.graph_forward(input_values=[X])
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass
    
class Sigmoid:
    def __init__(self):
        super().__init__()

    def __call__(self, X_node: OpNode, prefix=""):
        self.y_node = 1 / (1 + OpNode(op="exp", prefix=prefix)(-X_node))
        self.y_node.set_graph_inputs(inputs=[X_node])
        return self.y_node
    
    def forward(self, X):
        return self.y_node.graph_forward(input_values=[X])
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass

class Softmax:
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def __call__(self, X_node: OpNode, prefix=""):
        exp_node = OpNode(op="exp", prefix=prefix)(X_node - OpNode(op="max", axis=self.axis, keepdims=True, prefix=prefix)(X_node))
        self.y_node = exp_node / OpNode(op="sum", axis=self.axis, keepdims=True, prefix=prefix)(exp_node)
        self.y_node.set_graph_inputs(inputs=[X_node])
        return self.y_node
    
    def forward(self, X):
        return self.y_node.graph_forward(input_values=[X])
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass


def softmax(x):
    # Shift the input by the max value for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    # Normalize along the last axis
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

    
class Input(Layer):
    def __init__(self, shape):
        super().__init__(shape=shape)
        self.output_node = var(is_input=True)
    
    def __call__(self): 
        return self

    def forward(self, X): return X 

    def backward(self, y_grad=None): pass

class Dense(Layer):
    def __init__(self, units, activation = None, name=""):
        self.units = units
        self.activation = activation
        self.name = name

    def __call__(self, entry: Layer):
        self.previous_layers = [entry]
        input_shape = entry.get_output_shape_()
        self.shape = (input_shape[-1], self.units)
        W = np.random.randn(self.shape[0], self.shape[1]) * np.sqrt(2. / self.shape[0])
        # W = np.random.uniform(
        #     low=-np.sqrt(1 / self.shape[0]), high=np.sqrt(1 / self.shape[0]), size=(self.shape[0], self.shape[1])
        # )
        b = np.zeros((1, self.shape[1]))
        print(entry)
        print(entry.get_output_node_())
        self.X_node = entry.get_output_node_()
        self.W_node = var(initial_val=W, prefix=self.name)
        self.b_node = var(initial_val=b, prefix=self.name)
        self.z_node = OpNode(op="+", prefix=self.name)(OpNode(op="matmul", prefix=self.name)(self.X_node, self.W_node), self.b_node)
        self.output_node = self.activation(self.z_node, prefix=self.name)
        self.output_node.set_graph_inputs(inputs=[self.X_node])
        return self

    def forward(self, X):
        return self.output_node.graph_forward(input_values=[X])

    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass

class LossFunction(ABC):
    @abstractmethod
    def __call__(self): pass

    @abstractmethod
    def evaluate(self, y): pass


class CategoricalCrossEntropy(LossFunction):
    def __init__(self, epsilon=1e-15, axis=None):
        self.epsilon = epsilon
        self.axis = axis
        self.loss_node: OpNode = None
        
    def __call__(self, output: Layer):
        self.y_node = var(is_input=True)
        self.batch_size_node = var(is_input=True)
        self.preds_node = output.get_output_node_()
        self.preds_node_epsilon = OpNode(op="max")(self.preds_node, self.epsilon)
        self.preds_node_epsilon = OpNode(op="min")(self.preds_node_epsilon, 1 - self.epsilon)
        self.loss_node = -OpNode(op="+", axis=self.axis)(self.y_node * OpNode(op="ln")(self.preds_node_epsilon)) / self.batch_size_node
        return self

    def evaluate(self, y):
        self.loss_node.set_graph_inputs(inputs=[self.preds_node, self.y_node, self.batch_size_node])
        if self.preds_node.val is None: raise Exception("Model output is None, check if inference was executed correctly !!")
        return self.loss_node.graph_forward(input_values=[self.preds_node.val, y, y.shape[0]], local_inputs_only=True)

class Optimizer(ABC):
    @abstractmethod
    def update_node(self, node: OpNode): pass

    @abstractmethod
    def update_graph(self, nodes: List[OpNode], gradients: list): pass

def batch_numpy_array(dataset: np.ndarray, batch_size: int):
    padding_length = (batch_size - len(dataset) % batch_size) % batch_size
    indices = np.random.choice(dataset.shape[0], size=padding_length)
    padding = dataset[indices]
    padded_dataset = np.concatenate((dataset, padding), axis=0)
    return padded_dataset.reshape((-1, batch_size, *padded_dataset.shape[1:]))

class Model:
    def __init__(self, inputs: List[Layer], output: Layer):
        self.inputs = inputs
        self.output = output
        self.input_nodes = [input.get_output_node_() for input in self.inputs]
        self.output.get_output_node_().set_graph_inputs(inputs=self.input_nodes)
        self.layers, _ = topological_DFS(self.output, key=lambda x: x.previous_layers)
        self.loss = None
        self.optimizer = None

    def summary(self):
        print("="*90)
        print("-"*90)
        print("|| Layer\t\t\t|| Shape\t|| Activation\t\t||")
        for layer in self.layers:
            print("-"*90)
            print(layer)
        print("="*90)
        print("Loss Functions:", str(type(self.loss)))
        print("*"*90)
        print("Optimizer:", str(type(self.optimizer)))

    def set_loss_function(self, loss: LossFunction):
        self.loss = loss(self.output)
        return self

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
        return self
    
    def train_batch(self, x_train_inputs: list, y_train):
        _ = self.output.get_output_node_().graph_forward(input_values=x_train_inputs)
        loss = self.loss.evaluate(y_train)
        print("Batch loss:", loss)
        trainable_params, gradients = self.loss.loss_node.graph_backward()
        self.optimizer.update_graph(trainable_params, gradients)
        # for i, (weights_node, gradient) in enumerate(zip(trainable_params, gradients)):
        #     # if i == 0:
        #     #     print(
        #     #         weights_node, "=>",
        #     #         "trainable:", weights_node.val,
        #     #         "grad", gradient
        #     #     )
        #     self.optimizer.update_node(weights_node, gradient)

        return loss

    def fit(self, x_inputs: list, y, epochs, x_val_inputs: list=None, y_val=None, batch_size=32):
        batched_x_inputs = [batch_numpy_array(dataset=x_input, batch_size=batch_size) for x_input in x_inputs]
        batched_y = batch_numpy_array(dataset=y, batch_size=batch_size)
        print("X.shape:", batched_x_inputs[0].shape)
        print("y.shape:", batched_y.shape)
        for i in range(1, epochs + 1):
            print(f"Epochs: {i}/{epochs}")
            print("="*70)
            epoch_loss = 0
            for batch_tup in zip(*([*batched_x_inputs, batched_y])):
                y_batch = batch_tup[-1]
                input_batch = batch_tup[:-1]
                loss = self.train_batch(input_batch, y_batch)
                epoch_loss += loss
            print("-"*70)
            epoch_loss = epoch_loss / len(batched_y)
            print("Loss:", epoch_loss)
            print("-"*70)
            self.optimizer.lr *= 0.1


    def predict(self, x_inputs: list):
        logits = self.output.get_output_node_().graph_forward(input_values=[x for x in x_inputs])
        return logits


class SGD(Optimizer):
    def __init__(self, lr=1e-4, gradient_max_norm=1.0):
        self.lr = lr
        self.max_norm = gradient_max_norm

    def update_node(self, node: OpNode, gradient):
        self.max_norm = 1
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > self.max_norm:
            gradient = gradient * (self.max_norm / gradient_norm)
        node.val = node.val - self.lr * gradient

    def update_graph(self, nodes: List[OpNode], gradients: list):
        for node, gradient in zip(nodes, gradients):
            self.update_node(node, gradient)

class Adam(Optimizer):
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, gradient_max_norm=1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_norm = gradient_max_norm
        self.m = {}  # First moment vector (mean of gradients)
        self.v = {}  # Second moment vector (variance of gradients)
        self.t = 0   # Time step for bias correction

    def update_node(self, node: OpNode, gradient):
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > self.max_norm:
            gradient = gradient * (self.max_norm / gradient_norm)

        # print(np.linalg.norm(gradient))

        if node not in self.m:
            self.m[node] = np.zeros_like(gradient)
            self.v[node] = np.zeros_like(gradient)

        self.m[node] = self.beta1 * self.m[node] + (1 - self.beta1) * gradient
        self.v[node] = self.beta2 * self.v[node] + (1 - self.beta2) * (gradient ** 2)

        m_hat = self.m[node] / (1 - self.beta1 ** self.t)
        v_hat = self.v[node] / (1 - self.beta2 ** self.t)

        node.val -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def update_graph(self, nodes: List[OpNode], gradients: list):
        # gradient_norms = [np.linalg.norm(gradient) for gradient in gradients]
        # print(gradient_norms)
        self.t += 1
        for node, gradient in zip(nodes, gradients):
            self.update_node(node, gradient)

