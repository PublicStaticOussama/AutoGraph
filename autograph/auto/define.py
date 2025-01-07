from graph import OpNode

def var(initial_val=None, is_param=False, prefix=""):
    x = OpNode(op="var", is_param=is_param, prefix=prefix)
    if initial_val is not None:
        x.substitute(initial_val)
    return x

def param(initial_val=None, prefix=""):
    x = OpNode(op="var", is_param=True, prefix=prefix)
    if initial_val is not None:
        x.substitute(initial_val)
    return x

def const(val):
    return OpNode(op="const", constant=val)
