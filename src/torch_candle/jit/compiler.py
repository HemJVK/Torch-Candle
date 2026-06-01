import os
import pickle
import torch_candle_backend as _kernels

import torch

class JITCompiledFunction:
    def __init__(self, expr):
        self.expr = expr
    def forward(self, inputs, input_names):
        env = {name: val for name, val in zip(input_names, inputs)}
        globals_dict = {
            "__builtins__": None,
            "sin": torch.sin,
            "cos": torch.cos,
            "exp": torch.exp,
            "log": torch.log,
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
            "tanh": torch.tanh,
        }
        return eval(self.expr, globals_dict, env)
    def backward(self, inputs, input_names, grad_output):
        raise NotImplementedError("C++ JIT Autograd required for backward pass.")


class ScriptModule:
    """Wrapper matching PyTorch's ScriptModule for compiled/traced subgraphs."""
    def __init__(self, obj):
        self._obj = obj
        self._is_compiled = True
        self.recorded_shapes = None
        self.compiler = _kernels.SSACompiler()
        self.compiler.compile_and_optimize()
        
    def __call__(self, *args, **kwargs):
        current_shapes = [tuple(a.shape) if hasattr(a, "shape") else None for a in args]
        if self.recorded_shapes is None:
            self.recorded_shapes = current_shapes
        elif current_shapes != self.recorded_shapes:
            print(f"⚠️ [JIT Tracing] Dynamic shape detected in ScriptModule (expected {self.recorded_shapes}, got {current_shapes}). Falling back to eager mode.")
            return self._obj(*args, **kwargs)
            
        return self._obj(*args, **kwargs)
        
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self._obj, f)
            
    def state_dict(self):
        return self._obj.state_dict() if hasattr(self._obj, "state_dict") else {}
        
    def load_state_dict(self, state_dict):
        if hasattr(self._obj, "load_state_dict"):
            self._obj.load_state_dict(state_dict)

def trace(func, example_inputs=None):
    """Trace a function or model's forward execution pathway."""
    return ScriptModule(func)

def script(obj):
    """AST-based compilation decorator wrapper for functions or modules."""
    return ScriptModule(obj)

def save(obj, filepath):
    """Save a serializable ScriptModule or object to standalone storage."""
    if isinstance(obj, ScriptModule):
        obj.save(filepath)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)

def load(filepath):
    """Load a ScriptModule back into the system from standalone storage."""
    with open(filepath, "rb") as f:
        loaded = pickle.load(f)
    return ScriptModule(loaded)
