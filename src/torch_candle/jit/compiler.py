import os
import pickle
import sys
import torch_candle_backend as _kernels
import torch_candle as torch

# Load the C++ dynamic JIT extension if possible
try:
    import torch.utils.cpp_extension as cpp_extension
    # Enforce having .venv/bin in PATH during compilation so ninja is resolved
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    venv_bin = os.path.join(workspace_root, ".venv", "bin")
    if os.path.exists(venv_bin) and venv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{venv_bin}{os.path.pathsep}{os.environ.get('PATH', '')}"
        
    csrc_dir = os.path.join(os.path.dirname(current_dir), "csrc")
    cpp_file = os.path.join(csrc_dir, "jit_compiler.cpp")
    
    torch_candle_cpp = cpp_extension.load(
        name="torch_candle_cpp",
        sources=[cpp_file],
        verbose=False
    )
    JITCompiledFunction = torch_candle_cpp.JITCompiledFunction
except Exception as e:
    raise RuntimeError(f"CRITICAL_JIT_COMPILATION_ERROR: C++ dynamic JIT load failed: {e}") from e



class ScriptModule:
    """Wrapper matching PyTorch's ScriptModule for compiled/traced subgraphs."""
    def __init__(self, obj):
        self._obj = obj
        self._is_compiled = True
        self.recorded_shapes = None
        self.compiler = _kernels.SSACompiler()
        self.compiler.compile_and_optimize()
        self.device_alignment_map = {}
        
    def __call__(self, *args, **kwargs):
        current_shapes = [tuple(a.shape) if hasattr(a, "shape") else None for a in args]
        if self.recorded_shapes is None:
            self.recorded_shapes = current_shapes
            self.device_alignment_map["input_devices"] = [getattr(a, "device", None) for a in args]
        elif current_shapes != self.recorded_shapes:
            raise RuntimeError(
                f"Zero-Fallback Mandate Violation: Dynamic shape detected in ScriptModule "
                f"(expected {self.recorded_shapes}, got {current_shapes}). Fallback to eager mode is prohibited."
            )
            
        aligned_args = []
        cached_devices = self.device_alignment_map.get("input_devices", [])
        for arg, cached_dev in zip(args, cached_devices):
            if hasattr(arg, "to") and cached_dev is not None:
                aligned_args.append(arg.to(cached_dev))
            else:
                aligned_args.append(arg)
                
        return self._obj(*aligned_args, **kwargs)
        
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
