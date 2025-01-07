
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
from auto.utils.json_utils import (
    serialize_iterables_in_dict,
    is_json_serializable
)
from auto.math import (
    softmax_graph,
    variance_graph
)

from activations import activation_factory

class Layer(ABC):

    layers_name_map = {}

    def __init__(self, shape=None, name=""):
        self.name = name
        self.model_layer_key = None
        self.shape: tuple = shape
        self.output_node: OpNode = None
        self.previous_layers = []
        self._trainable = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__.lower() not in Layer.layers_name_map:
            Layer.layers_name_map[cls.__name__.lower()] = cls

    @property
    def trainable(self):
        return self._trainable
    
    @trainable.setter
    def trainable(self, trainable):
        self._trainable = trainable
        if self._trainable:
            self.output_node.graph_unfreeze(local_inputs_only=True)
        else:
            self.output_node.graph_freeze(local_inputs_only=True)

    def get_output_shape_(self): return self.shape
    
    def get_output_node_(self): return self.output_node

    @abstractmethod
    def __call__(self): pass

    @abstractmethod
    def forward(self): pass

    @abstractmethod
    def backward(self, y_grad=None): pass

    def parameter_count(self):
        count = 0
        for attr_value in vars(self).values():
            if type(attr_value) == OpNode and attr_value.op == "var" and attr_value.is_param:
                count += attr_value.val.size if type(attr_value.val) == np.ndarray else 1
        return count

    def get_params(self):
        params = []
        for attr_value in vars(self).values():
            if type(attr_value) == OpNode and attr_value.op == "var" and attr_value.is_param:
                params.append(attr_value.val)
        return params
    
    def set_params(self, *params: Union[*OpNode.CONSTANT_TYPES]):
        if self.output_node is None: raise Exception("Layer has to be functionally called on previous layer before its params can be set manually [example: Layer(*args)(prev_layer)]")
        param_names = []
        prev_vals = []
        for attr_key, attr_value in vars(self).items():
            if type(attr_value) == OpNode and attr_value.op == "var" and attr_value.is_param:
                param_names.append(attr_key)
                prev_vals.append(attr_value)
        if len(param_names) != len(params): raise Exception(f"number of params provided doesnt fit the number required which is: {len(param_names)}, set any param to None if u wish to keep it as default")
        for i, (param_name, prev_val, param) in enumerate(zip(param_names, prev_vals, params)):
            if param is not None:
                if type(prev_val.val) != type(param): raise Exception(f"param '{param_name}' at position: [{i}] requries type == {type(prev_val.val)} but got {type(param)} instead")
                if type(param) == np.ndarray and (prev_val.val.shape != param.shape): raise Exception(f"param '{param_name}' at position: [{i}] requries shape == {prev_val.val.shape} but got {param.shape} instead")
                prev_val.substitute(param)
        return self

    def json_serialize_init_args(self):
        constructor_signature = inspect.signature(self.__class__.__init__)
        parameter_names = list(constructor_signature.parameters.keys())[1:]
        constructor_attributes = {attr: getattr(self, attr) for attr in parameter_names}
        constructor_attributes = serialize_iterables_in_dict(constructor_attributes)
        if "activation" in constructor_attributes:
            constructor_attributes["activation"] = constructor_attributes["activation"].__class__.__name__.lower()
        if not is_json_serializable(constructor_attributes): raise Exception(f"All constructor args in ({self.__class__}) need to be json serializable, for when the model architecture is saved as json")
        return constructor_attributes

    def get_layer_class_name(self):
        return self.__class__.__name__

    def __repr__(self):
        if_activation = ""
        if hasattr(self, 'activation'):
            if_activation = " | " + str(type(self.activation))
        return "Layer: " + self.get_layer_class_name() + " | " + str(self.get_output_shape_()) + if_activation

str_to_class_map = Layer.layers_name_map

def layer_factory(class_name, **kwargs):
    instance = None
    if class_name.lower() not in str_to_class_map:
        raise Exception(f"Invalid Layer ({class_name})")
    cls = str_to_class_map[class_name.lower()]
    if "activation" in kwargs:
        kwargs["activation"] = activation_factory(kwargs["activation"])
    instance = cls(**kwargs)
    return instance

class Input(Layer):
    def __init__(self, shape, name=""):
        super().__init__(shape=shape, name=name)
        self.output_node = var(prefix="INPUT")
    
    def __call__(self): 
        return self

    def forward(self, X): return X 

    def backward(self, y_grad=None): pass

class Add(Layer):
    def __init__(self, name="", axis=-1):
        super().__init__(name=name)
        self.axis = axis

    def __call__(self, *entries: Layer):
        self.previous_layers = list(entries)
        if len(entries) <= 1: raise Exception("Add layer requires at least two layers as inputs")
        some_shape = entries[0].get_output_shape_()
        if any(map(lambda l: l.get_output_shape_() != some_shape, entries)):
            raise Exception("Layer shape mismatch, all layers must have the same shape for Add layer")
        self.shape = some_shape
        self.X_nodes = [entry.get_output_node_() for entry in entries]
        self.output_node = OpNode(op="sum", axis=self.axis)(*self.X_nodes)
        self.output_node.set_graph_inputs(inputs=self.X_nodes)
        return self
    
    def forward(self, X):
        return self.output_node.graph_forward(input_values=[X])
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass

class Reshape(Layer):
    def __init__(self, shape, name=""):
        super().__init__(shape=shape, name=name)
 
    def __call__(self, entry: Layer): 
        self.previous_layers = [entry]
        self.X_nodes = entry.get_output_node_()
        self.output_node = self.X_nodes.reshape((-1, *self.shape))
        self.output_node.set_graph_inputs(inputs=self.X_nodes)
        return self

    def forward(self, X):
        return self.output_node.graph_forward(input_values=[X])
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass

class Flatten(Layer):
    def __init__(self, name=""):
        super().__init__(name=name)

    def __call__(self, entry: Layer):
        self.previous_layers = [entry]
        input_shape = entry.get_output_shape_()
        self.X_nodes = entry.get_output_node_()
        self.shape = (np.prod(input_shape),)
        self.output_node = self.X_nodes.reshape((-1, *self.shape))
        self.output_node.set_graph_inputs(inputs=self.X_nodes)
        return self
    
    def forward(self, X):
        return self.output_node.graph_forward(input_values=[X])
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass

class Concatenate(Layer):
    def __init__(self, name="", axis=-1):
        super().__init__(name=name)
        self.axis = axis

    def __call__(self, *entries: Layer):
        self.previous_layers = list(entries)
        if len(entries) <= 1: raise Exception("Concatenate layer requires at least two layers as inputs")
        some_ndim = len(entries[0].get_output_shape_())
        if any(map(lambda l: len(l.get_output_shape_()) != some_ndim, entries)):
            raise Exception("Layer ndim mismatch, all layers must have the same ndim for Concatenate layer")
        self.X_nodes = [entry.get_output_node_() for entry in entries]
        self.output_node = OpNode(op="concat", axis=self.axis)(*self.X_nodes)
        self.output_node.set_graph_inputs(inputs=self.X_nodes)
        return self
    
    def forward(self, X):
        return self.output_node.graph_forward(input_values=[X])
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass

class Dense(Layer):
    def __init__(self, units, activation = None, name=""):
        super().__init__(name=name)
        self.units = units
        self.activation = activation
        if type(activation) == str:
            self.activation = activation_factory(activation.lower())

    def __call__(self, entry: Layer):
        self.previous_layers = [entry]
        input_shape = entry.get_output_shape_()
        ndim = len(input_shape)
        self.shape = (*input_shape[:-1], self.units)
        w_shape = input_shape[-1], self.units
        print(w_shape)
        self.W = np.random.randn(input_shape[-1], self.units) * np.sqrt(2. / np.prod(self.shape))
        # W = np.random.uniform(
        #     low=-np.sqrt(1 / self.shape[0]), high=np.sqrt(1 / self.shape[0]), size=(self.shape[0], self.shape[1])
        # )
        self.b = np.zeros((1, self.shape[-1]))
        self.X_node = entry.get_output_node_()
        self.W_node = param(initial_val=self.W, prefix=self.name+".W")
        self.b_node = param(initial_val=self.b, prefix=self.name+".b")
        self.z_node = (self.X_node @ self.W_node) + self.b_node
        self.output_node = self.z_node
        if self.activation is not None:
            self.output_node = self.activation(self.z_node, prefix=self.name)
        self.output_node.set_graph_inputs(inputs=[self.X_node])
        return self
    
    def forward(self, X):
        return self.output_node.graph_forward(input_values=[X], local_inputs_only=True)

    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass

class Dropout(Layer):
    def __init__(self, drop_prob=0.5, name=""):
        """
        Initialize Dropout layer.
        
        Parameters:
        -----------
        drop_prob : float, optional
            Probability of dropping a neuron (default: 0.5)
        """
        super().__init__(name=name)
        self.drop_prob = drop_prob
        # Seed for reproducibility of dropout mask
        self.seed = np.random.randint(0, 1000)
        self._training_mode = True

    @property
    def training_mode(self):
        return self._training_mode
    
    @training_mode.setter
    def training_mode(self, training_mode):
        self._training_mode = training_mode
        self.output_node.kwargs["condition"] = training_mode
        return self

    def __call__(self, entry: Layer):
        """
        Create Dropout computation graph.
        
        Parameters:
        -----------
        entry : Layer
            Input layer

        Returns:
        --------
        Layer
            Dropout Layer
        """
        self.previous_layers = [entry]
        input_shape = entry.get_output_shape_()
        self.X_node = entry.get_output_node_()
        self.shape = input_shape
        def training_graph():
            np.random.seed(self.seed)
            mask = const(val=(np.random.rand(*self.shape) > self.drop_prob).astype(float))
            return self.X_node * mask / (1 - self.drop_prob)
        
        def inference_graph():
            return self.X_node
        
        self.output_node = OpNode(op="merge", condition=self._training_mode, prefix=self.name)(
            training_graph(), 
            inference_graph()
        )
        self.output_node.set_graph_inputs(inputs=[self.X_node])
        return self
    
    def forward(self, X):
        return self.output_node.graph_forward(input_values=[X], local_inputs_only=True)
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass

class BatchNorm(Layer):
    def __init__(self, epsilon=1e-3, momentum=0.99, name=""):
        """
        Initialize BatchNorm layer for OpNode-based computation.
        
        Parameters:
        -----------
        num_features : int
            Number of features/channels to normalize
        epsilon : float, optional
            Small constant to prevent division by zero (default: 1e-3)
        momentum : float, optional
            Momentum for running mean and variance calculation (default: 0.9)
        """
        super().__init__(name=name)
        self.epsilon = epsilon
        self.momentum = momentum
        self._training_mode = True

    @property
    def training_mode(self):
        return self._training_mode
    
    @training_mode.setter
    def training_mode(self, training_mode):
        self._training_mode = training_mode
        self.output_node.kwargs["condition"] = training_mode
        return self

    def __call__(self, entry: Layer):
        """
        Create BatchNorm computation graph.
        
        Parameters:
        -----------
        entry : Layer
            previous layer
        Returns:
        --------
        Layer
            self layer with normalization graph defined
        """
        self.previous_layers = [entry]
        input_shape = entry.get_output_shape_()
        self.X_node = entry.get_output_node_()
        self.shape = input_shape

        self.gamma = param(initial_val=np.ones(input_shape[-1]), prefix="Gamma")
        self.beta = param(initial_val=np.zeros(input_shape[-1]), prefix="BatchNorm")

        self.running_mean = var(initial_val=np.zeros(input_shape[-1]), prefix="BatchNorm")
        self.running_var = var(initial_val=np.ones(input_shape[-1]), prefix="BatchNorm")

        reduce_axes = tuple(range(len(input_shape)))
        # print("SHAPE-"*30)
        # print(self.shape)
        # print("REDUCE-AXES-"*30)
        # print(reduce_axes)
        
        # Training subgraph
        def training_graph():

            batch_mean = self.X_node.mean(axis=reduce_axes, keepdims=True, prefix="batch_X")
            batch_var = variance_graph(self.X_node, axis=reduce_axes, keepdims=True)

            self.new_running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.new_running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            self.new_running_mean.cycle = self.running_mean # self.running_mean = new_running_mean
            self.new_running_var.cycle = self.running_var # self.running_var = new_running_var

            normalized = (self.X_node - batch_mean) / ((batch_var + self.epsilon) ** 0.5)
            res = self.gamma * normalized + self.beta
            return res
        
        # Non-training (inference) subgraph
        def inference_graph():
            normalized = (self.X_node - self.new_running_mean) / ((self.new_running_var + self.epsilon) ** 0.5)
            res = self.gamma * normalized + self.beta
            return res
        
        # Use merge OpNode to select between training and inference graphs
        self.output_node = OpNode(op="merge", condition=self._training_mode)(
            training_graph(), 
            inference_graph()
        )
        self.output_node.set_graph_inputs(inputs=[self.X_node])
        return self

    def forward(self, X):
        return self.output_node.graph_forward(input_values=[X], local_inputs_only=True)
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass
    
class LayerNorm(Layer):
    def __init__(self, epsilon=1e-3, name=""):
        """
        Initialize LayerNorm.
        epsilon: Small constant to prevent division by zero
        """
        super().__init__(name=name)
        self.epsilon = epsilon
    
    def __call__(self, entry: Layer):
        """
        Create LayerNorm computation graph.
        
        Parameters:
        -----------
        entry : Layer
            previous layer
        Returns:
        --------
        Layer
            self layer with normalization graph defined
        """
        self.previous_layers = [entry]
        input_shape = entry.get_output_shape_()
        self.X_node = entry.get_output_node_()
        self.shape = input_shape

        self.gamma = param(initial_val=np.ones(input_shape))
        self.beta = param(initial_val=np.zeros(input_shape))

        reduce_axes = tuple(range(1, len(input_shape) + 1))
        mean = self.X_node.mean(axis=reduce_axes, keepdims=True)
        variance = variance_graph(self.X_node, axis=reduce_axes, keepdims=True)
        
        x_norm = (self.X_node - mean) / ((variance + self.epsilon) ** 0.5)
        
        self.output_node = self.gamma * x_norm + self.beta

        self.output_node.set_graph_inputs(inputs=[self.X_node])
        return self
    
    def forward(self, X):
        return self.output_node.graph_forward(input_values=[X], local_inputs_only=True)

    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass

class Conv2D(Layer):
    def __init__(self, num_kernals, kernal_shape=(3,3), stride=1, padding=0, activation = None, name=""):
        super().__init__(name=name)
        self.num_kernals = num_kernals
        self.kernal_shape = kernal_shape
        self.stride = stride
        self.padding = padding
        self.activation = activation
        if type(activation) == str:
            self.activation = activation_factory(activation.lower())

    def __call__(self, entry: Layer):
        self.previous_layers = [entry]
        kernal_height, kernal_width = self.kernal_shape
        img_height, img_width, num_channels = entry.get_output_shape_()
        output_height = (img_height + 2 * self.padding - kernal_height) // self.stride + 1
        output_width = (img_width + 2 * self.padding - kernal_width) // self.stride + 1
        self.shape = (output_height, output_width, self.num_kernals)

        self.X_node = entry.get_output_node_()
        self.col_mat_node = OpNode(op="im2col", kernal_shape=self.kernal_shape, stride=self.stride, padding=self.padding)(self.X_node)
        kernels_matrix = np.random.randn(num_channels * kernal_height * kernal_width, self.num_kernals) * np.sqrt(2 / (num_channels * kernal_height * kernal_width))
        b = np.zeros((1, self.num_kernals))
        self.ker_mat_node = param(initial_val=kernels_matrix)
        self.b_node = param(initial_val=b)
        self.Z_node = (self.col_mat_node @ self.ker_mat_node) + self.b_node
        self.Z_node = self.Z_node.reshape(shape=(-1, output_height, output_width, self.num_kernals))
        # self.Z_node = OpNode("reshape", shape=(-1, output_height, output_width, self.num_kernals))(self.Z_node)
        self.output_node = self.Z_node
        if self.activation is not None:
            self.output_node = self.activation(self.Z_node, prefix=self.name)
        self.output_node.set_graph_inputs(inputs=[self.X_node])
        return self

    def forward(self, X):
        return self.output_node.graph_forward(input_values=[X])
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass

# DOESNT WORK, still didnt figure out how to do transpose convolution
# class Conv2DTranspose(Layer):
#     def __init__(self, num_kernals, kernal_shape=(3,3), stride=1, padding=0, activation = None, name=""):
#         super().__init__(name=name)
#         self.num_kernals = num_kernals
#         self.kernal_shape = kernal_shape
#         self.stride = stride
#         self.padding = padding
#         self.activation = activation
#         if type(activation) == str:
#             self.activation = activation_factory(activation.lower())

#     def __call__(self, entry: Layer):
#         self.previous_layers = [entry]
#         kernal_height, kernal_width = self.kernal_shape
#         img_height, img_width, num_channels = entry.get_output_shape_()

#         # Calculate output dimensions
#         output_height = (img_height - 1) * self.stride - 2 * self.padding + kernal_height
#         output_width = (img_width - 1) * self.stride - 2 * self.padding + kernal_width

#         self.shape = (output_height, output_width, self.num_kernals)

#         self.X_node = entry.get_output_node_()
#         print(self.X_node)
#         kernels_matrix = np.random.randn(self.num_kernals, num_channels * kernal_height * kernal_width) * np.sqrt(2 / (num_channels * kernal_height * kernal_width))
#         b = np.zeros((1, self.num_kernals))

#         self.ker_mat_node = param(initial_val=kernels_matrix)
#         self.b_node = param(initial_val=b)
#         self.col_mat_node = OpNode(op="im2col", kernal_shape=self.kernal_shape, stride=self.stride, padding=self.padding, prefix=self.name)(self.X_node)
#         self.Z_node = (self.col_mat_node @ self.ker_mat_node.T(axes=(1,0))) + self.b_node
#         self.output_node = OpNode(
#             op="col2im",
#             input_shape=(output_height, output_width, self.num_kernals),
#             kernal_shape=self.kernal_shape, stride=self.stride,
#             padding=self.padding
#         )(self.Z_node)

#         if self.activation is not None:
#             self.output_node = self.activation(self.output_node, prefix=self.name)
#         self.output_node.set_graph_inputs(inputs=[self.X_node])

#         return self

#     def forward(self, X):
#         return self.output_node.graph_forward(input_values=[X])
    
#     def backward(self, y_grad=None):
#         # return self.output_node.graph_backward(y_adjoint=y_grad)
#         pass
 
class MultiHeadAttention(Layer):
    def __init__(self, num_heads, d_k, d_v, mask=None, name=""):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.mask = mask

    def split_heads(self, X_node: OpNode, d, seq_len):
        """Split the last dimension into (num_heads, d)
        d can be either d_k for query/key or d_v for value"""
        # batch_size, seq_length = x.shape[0], x.shape[1]
        X_node = X_node.reshape(shape=(-1, seq_len, self.num_heads, d))
        return X_node.transpose(axes=(0, 2, 1, 3))  # (batch_size, num_heads, seq_length, d)

    def scaled_dot_product_attention(self, Q_node, K_node, V_node):
        """Calculate scaled dot-product attention"""
        
        scores = (Q_node @ K_node.transpose((0, 1, 3, 2))) / np.sqrt(self.d_k)
        scores = scores + self.mask
        
        attention_weights = softmax_graph(scores, axis=-1)
        # attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        # attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True) + 1e-9

        output = attention_weights @ V_node
        return output, attention_weights

    def __call__(self, entry: Layer, context_entry: Layer = None):
        self.previous_layers = [entry, context_entry] if context_entry else [entry]
        self.input_shape = entry.get_output_shape_()
        self.context_shape = context_entry.get_output_shape_() if context_entry else self.input_shape
        self.seq_len, self.d_model = self.input_shape
        self.ctxt_seq_len, self.ctxt_d_model = self.context_shape

        assert self.d_model % self.num_heads == 0
        assert self.ctxt_d_model % self.num_heads == 0
        assert self.d_model == self.ctxt_d_model

        self.shape = (self.seq_len, self.d_model)

        self.X_node = entry.get_output_node_()
        self.context_node = context_entry.get_output_node_()
        
        self.W_q_node = param(np.random.randn(self.d_model, self.num_heads * self.d_k) / np.sqrt(self.d_model))
        self.W_k_node = param(np.random.randn(self.d_model, self.num_heads * self.d_k) / np.sqrt(self.d_model))
        self.W_v_node = param(np.random.randn(self.d_model, self.num_heads * self.d_v) / np.sqrt(self.d_model))
        self.W_o_node = param(np.random.randn(self.num_heads * self.d_v, self.d_model) / np.sqrt(self.num_heads * self.d_v))

        # Linear projections
        self.Q_node = self.X_node @ self.W_q  # (batch_size, seq_length, num_heads * d_k)
        self.K_node = self.context_node @ self.W_k # (batch_size, seq_length, num_heads * d_k)
        self.V_node = self.context_node @ self.W_v # (batch_size, seq_length, num_heads * d_v)
        
        # Split heads
        self.Q_node = self.split_heads(self.Q_node, self.d_k, self.seq_len)  # (batch_size, num_heads, seq_length, d_k)
        self.K_node = self.split_heads(self.K_node, self.d_k, self.ctxt_seq_len)  # (batch_size, num_heads, seq_length, d_k)
        self.V_node = self.split_heads(self.V_node, self.d_v, self.ctxt_seq_len)  # (batch_size, num_heads, seq_length, d_v)
        
        # Apply scaled dot-product attention
        self.scaled_attention, self.attention_weights = self.scaled_dot_product_attention(self.Q_node, self.K_node, self.V_node)
        
        # Concatenate heads
        self.scaled_attention = self.scaled_attention.transpose((0, 2, 1, 3))  # (batch_size, seq_length, num_heads, d_v)
        concat_attention = self.scaled_attention.reshape((-1, self.seq_len, self.num_heads * self.d_v))
        
        # Final linear projection
        self.output_node = concat_attention @ self.W_o  # (batch_size, seq_length, num_features)
        if context_entry:
            self.output_node.set_graph_inputs(inputs=[self.X_node, self.context_node])
        else:
            self.output_node.set_graph_inputs(inputs=[self.X_node])
        return self
    
    def forward(self, X, context=None):
        if len(self.output_node.inputs) == 2 and context is None: raise Exception("Cross attention requires additional context input")
        if len(self.output_node.inputs) == 1 and context is not None: raise Exception("Self attention doesnt accept context input")
        if context is not None:
            return self.output_node.graph_forward(input_values=[X, context])
        return self.output_node.graph_forward(input_values=[X])
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass
    
class AveragePooling2D(Layer):
    def __init__(self, pool_shape=(2,2), stride=2, name=""):
        super().__init__(name=name)
        self.pool_shape = pool_shape
        self.stride = stride

    def __call__(self, entry: Layer):
        self.previous_layers = [entry]
        pool_height, pool_width = self.pool_shape
        img_height, img_width, num_channels = entry.get_output_shape_()
        output_height = (img_height - pool_height) // self.stride + 1
        output_width = (img_width - pool_width) // self.stride + 1
        self.shape = (output_height, output_width, num_channels)

        self.X_node = entry.get_output_node_()
        self.col_mat_node = OpNode(op="im2col_pool", pool_shape=self.pool_shape, stride=self.stride)(self.X_node)
        self.mean_node = self.col_mat_node.mean(axis=-1, keepdims=True)
        self.output_node = self.mean_node.reshape(shape=(-1, *self.shape))
        self.output_node.set_graph_inputs(inputs=[self.X_node])
        return self

    def forward(self, X):
        return self.output_node.graph_forward(input_values=[X])
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass

class MaxPooling2D(Layer):
    def __init__(self, pool_shape=(2,2), stride=2, name=""):
        super().__init__(name=name)
        self.pool_shape = pool_shape
        self.stride = stride

    def __call__(self, entry: Layer):
        self.previous_layers = [entry]
        pool_height, pool_width = self.pool_shape
        img_height, img_width, num_channels = entry.get_output_shape_()
        output_height = (img_height - pool_height) // self.stride + 1
        output_width = (img_width - pool_width) // self.stride + 1
        self.shape = (output_height, output_width, num_channels)

        self.X_node = entry.get_output_node_()
        self.col_mat_node = OpNode(op="im2col_pool", pool_shape=self.pool_shape, stride=self.stride)(self.X_node)
        self.max_node = self.col_mat_node.max(axis=-1, keepdims=True)
        self.output_node = self.max_node.reshape(shape=(-1, *self.shape))
        self.output_node.set_graph_inputs(inputs=[self.X_node])
        return self

    def forward(self, X):
        return self.output_node.graph_forward(input_values=[X])
    
    def backward(self, y_grad=None):
        # return self.output_node.graph_backward(y_adjoint=y_grad)
        pass
