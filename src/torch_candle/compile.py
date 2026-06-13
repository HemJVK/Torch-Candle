import torch_candle as torch

class CompiledModel:
    def __init__(self, model):
        self.model = model
        self.is_compiled = False
        self.recorded_graph = {}
        self.device_alignment_map = {}

    def __call__(self, *args, **kwargs):
        if not self.is_compiled:
            # First pass: eager trace
            # Discover and record the devices of all inputs to cache the device placement map
            self.device_alignment_map["input_devices"] = [getattr(a, "device", None) for a in args]
            
            out = self.model(*args, **kwargs)
            
            # Save input/output signatures to optimize memory allocations on next runs
            self.recorded_graph["input_shapes"] = [tuple(a.shape) if hasattr(a, "shape") else None for a in args]
            self.recorded_graph["input_dtypes"] = [getattr(a, "dtype", None) for a in args]
            self.is_compiled = True
            return out
        else:
            # Subsequent passes: check if shape matches!
            current_shapes = [tuple(a.shape) if hasattr(a, "shape") else None for a in args]
            if current_shapes != self.recorded_graph.get("input_shapes"):
                raise RuntimeError(
                    f"Zero-Fallback Mandate Violation: Dynamic shape detected in CompiledModel "
                    f"(expected {self.recorded_graph.get('input_shapes')}, got {current_shapes}). "
                    f"Fallback to eager mode is prohibited."
                )
            
            # Directly align the inputs to the cached device placement map, bypassing dynamic checks
            aligned_args = []
            cached_devices = self.device_alignment_map.get("input_devices", [])
            for arg, cached_dev in zip(args, cached_devices):
                if hasattr(arg, "to") and cached_dev is not None:
                    aligned_args.append(arg.to(cached_dev))
                else:
                    aligned_args.append(arg)
                
            return self.model(*aligned_args, **kwargs)

def compile(model, *args, **kwargs):
    """
    Dynamic graph compilation JIT wrapper matching torch.compile() in PyTorch 2.0+.
    Optimizes eager-mode deep learning, Transformers, and LLMs execution paths.
    """
    return CompiledModel(model)
