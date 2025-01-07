from graph import OpNode

def softmax_graph(x_node: OpNode, axis=-1):
    exps = OpNode(op="exp")(x_node - x_node.max(axis=axis, keepdims=True))
    soft = exps / exps.sum(axis=axis, keepdims=True) + 1e-9
    return soft

def variance_graph(x_node: OpNode, axis=None, keepdims=False, prefix=""):
    """
    Operation graph for the variance of x_node.
    
    Parameters:
    - x_node: Input node.
    - axis: Axis or axes along which to compute the variance.
    - keepdims: If True, retains reduced dimensions with size 1.
    
    Returns:
    - variance op graph: Computed variance of x_node.
    """
    mean = x_node.mean(axis=axis, keepdims=True, prefix=prefix)
    # squared_diff = (x_node - mean) ** 2
    squared_diff = OpNode(op="**", prefix=prefix)(OpNode(op="-", prefix=prefix)(x_node, mean), 2)
    variance = squared_diff.mean(axis=axis, keepdims=keepdims, prefix=prefix)
    return variance

# need to add other math functions ...