import inspect
import numpy as np
from typing import Union, List, Tuple, Any
from abc import ABC, abstractmethod

from auto.graph import OpNode
from auto.define import (
    var,
    param,
    const
)

class Optimizer(ABC):
    @abstractmethod
    def update_node(self, node: OpNode): pass

    @abstractmethod
    def update_graph(self, nodes: List[OpNode], gradients: list): pass

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
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-7, gradient_max_norm=1.0):
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

str_to_opt = {
    "sgd": SGD,
    "adam": Adam
}

def optimizer_factory(class_name, **kwargs):
    instance = None
    if class_name.lower() not in str_to_opt:
        raise Exception(f"Invalid optimizer ({class_name})")
    cls = str_to_opt[class_name.lower()]
    instance = cls(**kwargs)
    return instance