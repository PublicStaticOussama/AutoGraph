import os
import json
import numpy as np
from typing import Union, List, Tuple, Any, Dict
from abc import ABC, abstractmethod

from auto.graph import OpNode
from auto.define import (
    var,
    param,
    const
)
from auto.utils.tensor_utils import (
    batch_numpy_array
)
from auto.utils.tensor_storage import TensorStorage
from auto.utils.graph_utils import topological_DFS
from layers import Layer, layer_factory
from losses import LossFunction, loss_factory
from optimizers import Optimizer, optimizer_factory
from activations import activation_factory

class Model:
    def __init__(self, inputs: List[Layer], output: Layer):
        self.inputs = inputs
        self.output = output
        self.input_nodes = [input.get_output_node_() for input in self.inputs]
        self.output.get_output_node_().set_graph_inputs(inputs=self.input_nodes)
        self.layers: List[Layer]
        self.layers, _ = topological_DFS(self.output, key=lambda x: x.previous_layers)
        self.storage = TensorStorage()
        self.loss = None
        self.optimizer = None
        self.mode = "training"
        unique_names = {}
        for layer in self.layers:
            layer_class_name = layer.get_layer_class_name()
            layer_name = layer_class_name.lower()
            if layer_name not in unique_names:
                unique_names[layer_name] = 1
                layer_name += "_1"
            else:
                unique_names[layer_name] += 1
                num = unique_names[layer_name]
                layer_name += f"_{num}"
            layer_key = layer_class_name + "." + layer_name
            layer.model_layer_key = layer_key

    def training(self):
        self.mode = "training"
        for layer in self.layers:
            if "_training_mode" in layer.__dict__ or "training_mode" in layer.__dict__:
                layer.training_mode = True
        return self
    
    def inference(self):
        self.mode = "inference"
        for layer in self.layers:
            if "_training_mode" in layer.__dict__ or "training_mode" in layer.__dict__:
                layer.training_mode = False
        return self

    def total_parameter_count(self):
        total_count = 0
        for layer in self.layers:
            total_count += layer.parameter_count()
        return total_count

    def summary(self):
        print("="*90)
        print("-"*90)
        print("|| Layer     || Shape\t   || Activation\t\t||")
        for layer in self.layers:
            print("-"*90)
            print(layer)
        print("="*90)
        print("Loss Functions:", str(type(self.loss)))
        print("*"*90)
        print("Optimizer:", str(type(self.optimizer)))
        print("Total number of Parameters:", '{:,}'.format(self.total_parameter_count()))

    def set_layer_keys(self):
        unique_names = {}
        for layer in self.layers:
            layer_class_name = layer.get_layer_class_name()
            layer_name = layer_class_name.lower()
            if layer_name not in unique_names:
                unique_names[layer_name] = 1
                layer_name += "_1"
            else:
                unique_names[layer_name] += 1
                num = unique_names[layer_name]
                layer_name += f"_{num}"
            layer_key = layer_class_name + "." + layer_name
            layer.model_layer_key = layer_key
        return self


    def get_dag_architecture(self):
        self.set_layer_keys()
        model_dict = {
            "model": {},
            "optimizer": {"name": self.optimizer.__class__.__name__, "kwargs": {"lr": self.optimizer.lr}},
            "loss_func": {"name": self.loss.__class__.__name__, "kwargs": {}}
        }
        dict_dag = {}

        for layer in self.layers:
            dict_dag[layer.model_layer_key] = {
                "kwargs": layer.json_serialize_init_args(),
                "previous_layers": [l.model_layer_key for l in layer.previous_layers]
            }

        model_dict["model"] = dict_dag

        return model_dict
        
    def get_keyed_params(self):
        self.set_layer_keys()
        tensor_dict = {}
        for layer in self.layers:
            tensor_dict[layer.model_layer_key] = layer.get_params()
        return tensor_dict

    def save(self, path):
        architecture = self.get_dag_architecture()
        keyed_params = self.get_keyed_params()
        self.storage.save_tensors(keyed_params, os.path.join(path, 'param_tensors.h5'))
        with open(os.path.join(path, 'architecture.json'), "w") as f:
            json.dump(architecture, f, indent=4)
        print(f"Saved model in path: {path}")

    def set_loss_function(self, loss: LossFunction):
        self.loss = loss(self.output)
        return self

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
        return self
    
    def train_batch(self, x_train_inputs: list, y_train, verbose=2):
        _ = self.output.get_output_node_().graph_forward(input_values=x_train_inputs)
        loss = self.loss.evaluate(y_train)
        logits = self.output.get_output_node_().val
        mae = np.abs(logits - y_train).mean()
        acc = 1 - mae
        if verbose == 2:
            print("btch loss:", loss, "|", "batch acc:", acc)
        trainable_params, gradients = self.loss.loss_node.graph_backward()
        self.optimizer.update_graph(trainable_params, gradients)

        return loss, acc

    def fit(self, x_inputs: list, y, epochs, x_val_inputs: list=[], y_val=[], batch_size=32, verbose=2):
        if (len(x_inputs) or len(y)) and any([len(x_val_inp) != len(y) for x_val_inp in x_inputs]):
            xs = any([len(x_inp) for x_inp in x_inputs])
            raise Exception(f"Training set X and Y dont have the same lengths len(Y)={len(y)}, len(Xs)={str(xs)}")
        if (len(x_val_inputs) or len(y_val)) and any([len(x_val_inp) != len(y_val) for x_val_inp in x_val_inputs]):
            xs = any([len(x_val_inp) for x_val_inp in x_val_inputs])
            raise Exception(f"Validation set X and Y dont have the same lengths len(Y)={len(y_val)}, len(Xs)={str(xs)}")
        batched_x_inputs = [batch_numpy_array(dataset=x_input, batch_size=batch_size) for x_input in x_inputs]
        batched_y = batch_numpy_array(dataset=y, batch_size=batch_size)
        if verbose in [1,2]:
            print("X.shape:", batched_x_inputs[0].shape)
            print("y.shape:", batched_y.shape)
        for i in range(1, epochs + 1):
            self.training()
            if verbose in [1,2]:
                print(f"Epochs: {i}/{epochs}")
                print("="*70)
            epoch_loss = 0
            epoch_acc = 0
            for batch_tup in zip(*([*batched_x_inputs, batched_y])):
                y_batch = batch_tup[-1]
                input_batch = batch_tup[:-1]
                loss, acc = self.train_batch(input_batch, y_batch, verbose=verbose)
                epoch_loss += loss
                epoch_acc += acc
            if verbose in [1,2]: print("-"*70)
            epoch_loss = epoch_loss / len(batched_y)
            val_acc_str = ""
            if len(x_val_inputs) and len(y_val) and all([len(x_val_inp) == len(y_val) for x_val_inp in x_val_inputs]):
                self.inference()
                logits = self.predict(x_val_inputs)
                mae = np.abs(logits - y_val).mean()
                val_acc = 1 - mae
                val_acc_str = f"val accuracy: {val_acc}"
            if verbose in [1,2]:    
                print("Loss:", epoch_loss, "|", "Accuracy:", epoch_acc / len(batched_y), "|", val_acc_str)
                print("-"*70)
            self.optimizer.lr *= 0.1

    def predict(self, x_inputs: list):
        logits = self.output.get_output_node_().graph_forward(input_values=[x for x in x_inputs])
        return logits


def load_model(path):
    if not os.path.exists(path): raise Exception(f"Model path: [{path}] not found")
    if not os.path.exists(os.path.join(path, 'architecture.json')):
        raise Exception(f"Model architecture not found in: [{path}]")
    if not os.path.exists(os.path.join(path, 'param_tensors.h5')):
        raise Exception(f"Model params not found in: [{path}]")
    storage = TensorStorage()
    param_tensors, _ = storage.load_tensors(os.path.join(path, "param_tensors.h5"))
    with open(os.path.join(path, "architecture.json"), "r") as f:
        model_dag = json.load(f)
    loss_dict = model_dag["loss_func"]
    opt_dict = model_dag["optimizer"]
    loss_func: LossFunction = loss_factory(loss_dict["name"], **loss_dict["kwargs"])
    optimizer: Optimizer = optimizer_factory(opt_dict["name"], **opt_dict["kwargs"])
    layers_dict: Dict[str, Layer] = {}
    input_layer = None
    for layer_key, layer in model_dag["model"].items():
        layer_class, auto_name = layer_key.split(".", 1)
        kwargs = layer["kwargs"]
        if "name" not in kwargs or not kwargs:
            kwargs["name"] = auto_name
        layer_instance = layer_factory(layer_class, **kwargs)
        layers_dict[layer_key] = layer_instance
        prev_layer_keys = layer["previous_layers"]
        if len(prev_layer_keys) == 0 and input_layer is None:
            input_layer = layer_instance
        prev_layer_instances = []
        for l_key in prev_layer_keys:
            if l_key not in layers_dict: raise Exception(f"No previous layer [{l_key}] was found the architecture.json file")
            prev_layer_instances.append(layers_dict[l_key])
        layers_dict[layer_key] = layers_dict[layer_key](*prev_layer_instances)
        if layer_key in param_tensors:
            layers_dict[layer_key].set_params(*param_tensors[layer_key])
    if input_layer is None: raise Exception("No input layer was found in the architecture.json file")
    output_layer = list(layers_dict.values())[-1]
    model = Model(inputs=[input_layer], output=output_layer)
    if loss_func is not None: model.set_loss_function(loss_func)
    if optimizer is not None: model.set_optimizer(optimizer)
    del param_tensors

    return model

