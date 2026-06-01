import numpy as np
from torch_candle.tensor import Tensor

class DispatchNode:
    def diff(self, var_name):
        raise NotImplementedError
    def expr_string(self):
        raise NotImplementedError

class VariableNode(DispatchNode):
    def __init__(self, name="x"):
        self.name = name
    def diff(self, var_name):
        if self.name == var_name:
            return ConstantNode(1.0)
        return ConstantNode(0.0)
    def expr_string(self):
        return self.name

class ConstantNode(DispatchNode):
    def __init__(self, value):
        self.value = value
    def diff(self, var_name):
        return ConstantNode(0.0)
    def expr_string(self):
        return str(self.value)

class AddNode(DispatchNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def diff(self, var_name):
        return AddNode(self.left.diff(var_name), self.right.diff(var_name))
    def expr_string(self):
        return f"({self.left.expr_string()} + {self.right.expr_string()})"

class SubNode(DispatchNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def diff(self, var_name):
        return SubNode(self.left.diff(var_name), self.right.diff(var_name))
    def expr_string(self):
        return f"({self.left.expr_string()} - {self.right.expr_string()})"

class MulNode(DispatchNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def diff(self, var_name):
        return AddNode(
            MulNode(self.left.diff(var_name), self.right),
            MulNode(self.left, self.right.diff(var_name))
        )
    def expr_string(self):
        return f"({self.left.expr_string()} * {self.right.expr_string()})"

class DivNode(DispatchNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def diff(self, var_name):
        num = SubNode(
            MulNode(self.left.diff(var_name), self.right),
            MulNode(self.left, self.right.diff(var_name))
        )
        denom = MulNode(self.right, self.right)
        return DivNode(num, denom)
    def expr_string(self):
        return f"({self.left.expr_string()} / {self.right.expr_string()})"

class PowNode(DispatchNode):
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent
    def diff(self, var_name):
        return MulNode(
            MulNode(ConstantNode(self.exponent), PowNode(self.base, self.exponent - 1)),
            self.base.diff(var_name)
        )
    def expr_string(self):
        return f"({self.base.expr_string()} ** {self.exponent})"

class SinNode(DispatchNode):
    def __init__(self, child):
        self.child = child
    def diff(self, var_name):
        return MulNode(CosNode(self.child), self.child.diff(var_name))
    def expr_string(self):
        return f"sin({self.child.expr_string()})"

class CosNode(DispatchNode):
    def __init__(self, child):
        self.child = child
    def diff(self, var_name):
        return MulNode(MulNode(ConstantNode(-1.0), SinNode(self.child)), self.child.diff(var_name))
    def expr_string(self):
        return f"cos({self.child.expr_string()})"

class ExpNode(DispatchNode):
    def __init__(self, child):
        self.child = child
    def diff(self, var_name):
        return MulNode(ExpNode(self.child), self.child.diff(var_name))
    def expr_string(self):
        return f"exp({self.child.expr_string()})"

class LogNode(DispatchNode):
    def __init__(self, child):
        self.child = child
    def diff(self, var_name):
        return DivNode(self.child.diff(var_name), self.child)
    def expr_string(self):
        return f"log({self.child.expr_string()})"

class SumNode(DispatchNode):
    def __init__(self, child):
        self.child = child
    def diff(self, var_name):
        return SumNode(self.child.diff(var_name))
    def expr_string(self):
        return f"({self.child.expr_string()})"

class TracingTensor(Tensor):
    def __init__(self, expr_node, shape=None):
        super().__init__(np.array([0.0]))
        self.expr_node = expr_node
        self._shape = shape if shape is not None else (1,)

    @property
    def shape(self):
        return self._shape

    def __add__(self, other):
        other_node = other.expr_node if isinstance(other, TracingTensor) else ConstantNode(other)
        return TracingTensor(AddNode(self.expr_node, other_node), self._shape)

    def __radd__(self, other):
        other_node = other.expr_node if isinstance(other, TracingTensor) else ConstantNode(other)
        return TracingTensor(AddNode(other_node, self.expr_node), self._shape)

    def __sub__(self, other):
        other_node = other.expr_node if isinstance(other, TracingTensor) else ConstantNode(other)
        return TracingTensor(SubNode(self.expr_node, other_node), self._shape)

    def __rsub__(self, other):
        other_node = other.expr_node if isinstance(other, TracingTensor) else ConstantNode(other)
        return TracingTensor(SubNode(other_node, self.expr_node), self._shape)

    def __mul__(self, other):
        other_node = other.expr_node if isinstance(other, TracingTensor) else ConstantNode(other)
        return TracingTensor(MulNode(self.expr_node, other_node), self._shape)

    def __rmul__(self, other):
        other_node = other.expr_node if isinstance(other, TracingTensor) else ConstantNode(other)
        return TracingTensor(MulNode(other_node, self.expr_node), self._shape)

    def __truediv__(self, other):
        other_node = other.expr_node if isinstance(other, TracingTensor) else ConstantNode(other)
        return TracingTensor(DivNode(self.expr_node, other_node), self._shape)

    def __rtruediv__(self, other):
        other_node = other.expr_node if isinstance(other, TracingTensor) else ConstantNode(other)
        return TracingTensor(DivNode(other_node, self.expr_node), self._shape)

    def __pow__(self, exponent):
        return TracingTensor(PowNode(self.expr_node, exponent), self._shape)

    def sum(self, *args, **kwargs):
        return TracingTensor(SumNode(self.expr_node), (1,))

    def sin(self):
        return TracingTensor(SinNode(self.expr_node), self._shape)

    def cos(self):
        return TracingTensor(CosNode(self.expr_node), self._shape)

    def exp(self):
        return TracingTensor(ExpNode(self.expr_node), self._shape)

    def log(self):
        return TracingTensor(LogNode(self.expr_node), self._shape)

    def diff(self):
        return TracingTensor(self.expr_node.diff("x"), self._shape)

    def expr_string(self):
        return self.expr_node.expr_string()

    def __torch_dispatch__(self, func_name, *args, **kwargs):
        def to_node(x):
            if isinstance(x, TracingTensor):
                return x.expr_node
            elif isinstance(x, Tensor):
                try:
                    return ConstantNode(float(x.numpy().item()))
                except Exception:
                    return ConstantNode(1.0)
            else:
                return ConstantNode(x)

        if func_name in ("add", "__add__", "__radd__"):
            left = to_node(args[0])
            right = to_node(args[1])
            return TracingTensor(AddNode(left, right), self._shape)
        elif func_name in ("sub", "__sub__", "__rsub__"):
            left = to_node(args[0])
            right = to_node(args[1])
            return TracingTensor(SubNode(left, right), self._shape)
        elif func_name in ("mul", "__mul__", "__rmul__"):
            left = to_node(args[0])
            right = to_node(args[1])
            return TracingTensor(MulNode(left, right), self._shape)
        elif func_name in ("div", "__div__", "__truediv__", "__rtruediv__"):
            left = to_node(args[0])
            right = to_node(args[1])
            return TracingTensor(DivNode(left, right), self._shape)
        elif func_name in ("pow", "__pow__"):
            base = to_node(args[0])
            exponent = args[1]
            return TracingTensor(PowNode(base, exponent), self._shape)
        elif func_name == "sum":
            child = to_node(args[0])
            return TracingTensor(SumNode(child), (1,))
        elif func_name == "sin":
            child = to_node(args[0])
            return TracingTensor(SinNode(child), self._shape)
        elif func_name == "cos":
            child = to_node(args[0])
            return TracingTensor(CosNode(child), self._shape)
        elif func_name == "exp":
            child = to_node(args[0])
            return TracingTensor(ExpNode(child), self._shape)
        elif func_name == "log":
            child = to_node(args[0])
            return TracingTensor(LogNode(child), self._shape)
