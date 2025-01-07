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

from layers import Layer

class LossFunction(ABC):
    @abstractmethod
    def __call__(self): pass

    @abstractmethod
    def evaluate(self, y): pass

class CategoricalCrossEntropy(LossFunction):
    def __init__(self, epsilon=1e-7, axis=None):
        self.epsilon = epsilon
        self.axis = axis
        self.loss_node: OpNode = None
        
    def __call__(self, output: Layer):
        self.y_node = var()
        self.batch_size_node = var()
        self.preds_node = output.get_output_node_()
        self.preds_node_epsilon = OpNode(op="max", prefix="CrossEntropy")(self.preds_node, self.epsilon)
        self.preds_node_epsilon = OpNode(op="min", prefix="CrossEntropy")(self.preds_node_epsilon, 1 - self.epsilon)
        self.loss_node = -OpNode(op="+", axis=self.axis, prefix="CrossEntropy")(self.y_node * OpNode(op="ln", prefix="CrossEntropy")(self.preds_node_epsilon)) / self.batch_size_node
        return self

    def evaluate(self, y):
        self.loss_node.set_graph_inputs(inputs=[self.preds_node, self.y_node, self.batch_size_node])
        if self.preds_node.val is None: raise Exception("Model output is None, check if inference was executed correctly !!")
        return self.loss_node.graph_forward(input_values=[self.preds_node.val, y, y.shape[0]], local_inputs_only=True)

str_to_loss = {
    "categoricalcrossentropy": CategoricalCrossEntropy,
}

def loss_factory(class_name, **kwargs):
    instance = None
    if class_name.lower() not in str_to_loss:
        raise Exception(f"Invalid loss function ({class_name})")
    cls = str_to_loss[class_name.lower()]
    instance = cls(**kwargs)
    return instance