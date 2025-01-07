import inspect
import numpy as np
from typing import Union, List, Tuple, Any, Dict
from abc import ABC, abstractmethod

from auto.graph import OpNode
from auto.define import (
    var,
    param,
    const
)

class ReLU:
    def __init__(self):
        pass

    def __call__(self, X_node: OpNode, prefix=""):
        self.y_node = OpNode(op="max", prefix="ReLU")(0, X_node).set_graph_inputs(inputs=[X_node])
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
        exp_node = OpNode(op="exp", prefix="softmax")(X_node - OpNode(op="max", axis=self.axis, keepdims=True, prefix="softmax")(X_node))
        self.y_node = exp_node / OpNode(op="sum", axis=self.axis, keepdims=True, prefix=prefix)(exp_node)
        self.y_node.set_graph_inputs(inputs=[X_node])
        return self.y_node
    
    def forward(self, X):
        return self.y_node.graph_forward(input_values=[X])
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass

def activation_factory(class_name):
    instance = None
    match(class_name.lower()):
        case "relu": instance = ReLU()
        case "sigmoid": instance = Sigmoid()
        case "softmax": instance = Softmax()
    return instance