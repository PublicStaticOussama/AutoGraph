import numpy as np

def get_broadcast_shape(shape1, shape2):
    len_diff = abs(len(shape1) - len(shape2))
    if len(shape1) > len(shape2):
        shape2 = (1,) * len_diff + shape2
    else:
        shape1 = (1,) * len_diff + shape1

    broadcast_shape = []
    for dim1, dim2 in zip(shape1, shape2):
        if dim1 == dim2 or dim1 == 1 or dim2 == 1:
            broadcast_shape.append(max(dim1, dim2))
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
    
    return tuple(broadcast_shape)


def eval_reduce_semi_broadcast_shape(og_shape, axis):
    if axis is None: return tuple([1]*len(og_shape))
    if type(axis) not in [tuple, list]: axis=(axis,)
    shape_ls = list(og_shape) 
    for ax in axis: shape_ls[ax] = 1
    semi_broadcasted_shape = tuple(shape_ls)
    return semi_broadcasted_shape

def infer_matmul_broadcast_shape(unbroadcasted_shape, output_shape): # einsum is too hard
    matmul_dims = unbroadcasted_shape[-2:]
    batch_dims = output_shape[:-2]

    broadcast_shape = batch_dims + matmul_dims
    
    return broadcast_shape

def matmul_batch_axes_match(input_shape, output_shape): # einsum is too hard
    return input_shape[:-2] == output_shape[:-2]

def infer_reduction_axis(shape, reduced_shape):
    len_diff = len(shape) - len(reduced_shape)
    if len_diff > 0:
        reduced_shape = (1,) * len_diff + reduced_shape
    
    # Identify the axes where the reduction occurred
    reduction_axes = [i for i, (r_dim, red_dim) in enumerate(zip(shape, reduced_shape)) if red_dim == 1 and r_dim != 1]
    
    return tuple(reduction_axes)

def broadcast_adjoint(adjoint, wrt, this):
    if type(wrt.val) != np.ndarray or wrt.val.size < this.val.size:
        reduce_axes = infer_reduction_axis(
            shape=this.val.shape,
            reduced_shape=wrt.val.shape
        )
        adjoint = adjoint.sum(axis=reduce_axes)
    return adjoint


