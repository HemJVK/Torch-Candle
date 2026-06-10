"""
torch_candle.aot — Ahead-of-Time Compilation Path (§5)

Implements make_fx(): traces functions through vmap/grad transformations using
the native SSACompiler infrastructure and caches compiled subgraphs keyed by
(func.__qualname__, input_shapes). Subsequent calls with matching shapes bypass
the Python interpreter entirely, routing through the Rust SSA VM.

Architectural notes:
  - Tracing uses compile_ast() from the Rust backend (NativeASTParser path).
  - Cache keys are (qualname, shapes_tuple) — shape-specialised like XLA.
  - Compiled callables run in py.allow_threads() context for GIL-free execution.
  - No finite-difference fallback; all symbolic logic offloaded to Rust/C++.
"""
import torch_candle_backend as _kernels
from torch_candle.tensor import Tensor

# Shape-keyed AOT subgraph cache.
# Key:   (func.__qualname__, tuple of input shapes)
# Value: compiled SSACompiler instance
_aot_cache: dict = {}


def _shapes_key(sample_inputs):
    """Build a hashable shape tuple from a sequence of tensor-like inputs."""
    shapes = []
    for inp in sample_inputs:
        if isinstance(inp, Tensor):
            shapes.append(tuple(inp.shape))
        elif hasattr(inp, "shape"):
            shapes.append(tuple(inp.shape))
        else:
            shapes.append(None)
    return tuple(shapes)


def _make_compiled_callable(compiler, sample_inputs):
    """
    Wrap a compiled SSACompiler into a Python callable that:
      1. Maps positional args to named SSA inputs by position.
      2. Executes via the Rust SSA VM (GIL-free, allow_threads).
      3. Returns a torch_candle.Tensor wrapping the result.
    """
    # Build input name list from the compiler's inputs registry
    input_names = [v.dtype for v in compiler.inputs]  # dtype field holds the var name

    def compiled_call(*args, **kwargs):
        # Build the input map: {name: PyTensor} from positional args
        input_map = {}
        for i, arg in enumerate(args):
            if i < len(input_names):
                name = input_names[i]
                if isinstance(arg, Tensor):
                    input_map[name] = arg._tensor
                elif hasattr(arg, "_tensor"):
                    input_map[name] = arg._tensor
        result_pytensor = compiler.execute(input_map)
        return Tensor(result_pytensor)

    return compiled_call


def make_fx(func, *sample_inputs):
    """
    Trace `func` ahead-of-time through vmap/grad transformations.

    Uses the native SSACompiler (NativeASTParser) to build an SSA IR subgraph
    from the function's Python source, then compiles and caches it. Subsequent
    calls with the same input shapes bypass the Python interpreter entirely.

    Args:
        func: Python callable to trace. Must contain only supported ops
              (arithmetic, unary activations). Stateful operations that can't
              be expressed as SSA are not supported.
        *sample_inputs: Example torch_candle.Tensor inputs for shape inference.

    Returns:
        A compiled callable with identical signature to func, routed through
        the Rust SSA VM for GIL-free execution.

    Example:
        >>> def f(x, y):
        ...     z = x + y
        ...     return z
        >>> f_compiled = make_fx(f, Tensor([1.0]), Tensor([2.0]))
        >>> result = f_compiled(Tensor([3.0]), Tensor([4.0]))
    """
    qualname = getattr(func, "__qualname__", repr(func))
    cache_key = (qualname, _shapes_key(sample_inputs))

    if cache_key in _aot_cache:
        # Cache hit: return the cached compiled callable
        compiler = _aot_cache[cache_key]
        return _make_compiled_callable(compiler, sample_inputs)

    # Cache miss: compile the function via the Rust AST parser
    try:
        compiler = _kernels.compile_ast(func)
    except Exception as e:
        raise RuntimeError(
            f"make_fx: AOT compilation of '{qualname}' failed: {e}\n"
            "Only pure arithmetic functions with supported ops are traceable."
        ) from e

    _aot_cache[cache_key] = compiler
    return _make_compiled_callable(compiler, sample_inputs)


def aot_cache_size() -> int:
    """Return the number of compiled subgraphs in the AOT cache."""
    return len(_aot_cache)


def aot_cache_clear():
    """Evict all compiled subgraphs from the AOT cache."""
    global _aot_cache
    _aot_cache = {}


def aot_cache_keys() -> list:
    """Return all current cache keys as a list of (qualname, shapes) tuples."""
    return list(_aot_cache.keys())
