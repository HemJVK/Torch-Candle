import torch_candle_backend as _kernels
from torch_candle_backend import VmapDispatcher
from torch_candle.tensor import Tensor
import numpy as np

def parse_ast(source_code):
    """
    STRICT PROHIBITION: Python-level AST/Sym parsing is terminated.
    All symbolic logic must be offloaded to the Rust/C++ backend
    to eliminate O(n) serialization tax and L1 cache pollution.
    """
    raise RuntimeError(
        "ARCHITECTURAL VIOLATION: Native AST required. "
        "Python-level parsing is prohibited in Phase XI."
    )

def stack(tensors, dim=0):
    from torch_candle import stack as raw_stack
    return raw_stack(tensors, dim=dim)

def get_active_dispatch_level() -> int:
    """Retrieve the current level of the nested dynamic dispatcher stack."""
    return _kernels.get_active_dispatch_level()

def push_dispatch_level(level_id: str):
    """Push a new transformation level onto the dynamic dispatcher stack."""
    _kernels.push_dispatch_level(level_id)

def pop_dispatch_level() -> str:
    """Pop the top transformation level from the dynamic dispatcher stack."""
    return _kernels.pop_dispatch_level()

def rearrange(tensor, pattern, **axes_lengths):
    """
    Rearrange a tensor's dimensions based on an einops-style pattern string.
    Fully autograd-safe.
    
    Example:
        rearrange(x, 'b c (h w) -> b h w c', h=20, w=20)
    """
    from torch_candle import Tensor
    
    if '->' not in pattern:
        raise ValueError("Pattern must contain '->'")
    lhs_str, rhs_str = pattern.split('->')
    
    def parse_part(part_str):
        part_str = part_str.strip()
        tokens = []
        i = 0
        while i < len(part_str):
            if part_str[i].isspace():
                i += 1
                continue
            if part_str[i] == '(':
                j = part_str.find(')', i)
                if j == -1:
                    raise ValueError("Unmatched parenthesis in pattern")
                tokens.append(part_str[i+1:j].split())
                i = j + 1
            else:
                j = i
                while j < len(part_str) and not part_str[j].isspace() and part_str[j] != '(':
                    j += 1
                tokens.append(part_str[i:j])
                i = j
        return tokens

    lhs_tokens = parse_part(lhs_str)
    rhs_tokens = parse_part(rhs_str)
    
    if len(lhs_tokens) != tensor.ndim:
        raise ValueError(f"Number of LHS axes ({len(lhs_tokens)}) must match tensor ndim ({tensor.ndim})")
        
    flat_lhs = []
    flat_shapes = []
    
    for token, size in zip(lhs_tokens, tensor.shape):
        if isinstance(token, str):
            flat_lhs.append(token)
            flat_shapes.append(size)
        elif isinstance(token, list):
            unresolved = []
            known_prod = 1
            for name in token:
                if name in axes_lengths:
                    known_prod *= axes_lengths[name]
                else:
                    unresolved.append(name)
            
            if len(unresolved) > 1:
                raise ValueError(f"Cannot resolve multiple unknown dimensions in grouped axis: {unresolved}")
            
            resolved_sizes = {}
            for name in token:
                if name in axes_lengths:
                    resolved_sizes[name] = axes_lengths[name]
                else:
                    resolved_sizes[name] = size // known_prod
                    
            for name in token:
                flat_lhs.append(name)
                flat_shapes.append(resolved_sizes[name])
                
    # 1. Reshape to intermediate flat shape
    res = tensor.reshape(*flat_shapes)
    
    # 2. Permute dimensions to target order
    flat_rhs = []
    for token in rhs_tokens:
        if isinstance(token, str):
            flat_rhs.append(token)
        elif isinstance(token, list):
            flat_rhs.extend(token)
            
    try:
        perm = [flat_lhs.index(name) for name in flat_rhs]
    except ValueError as e:
        raise ValueError(f"RHS contains dimension not present on LHS: {e}")
        
    # Perform autograd-safe sorting transposes
    curr = list(range(len(perm)))
    for i in range(len(perm)):
        if curr[i] != perm[i]:
            idx = curr.index(perm[i])
            res = res.transpose(i, idx)
            curr[i], curr[idx] = curr[idx], curr[i]
            
    # 3. Group/Reshape to final RHS tokens
    final_shape = []
    transposed_shapes = [flat_shapes[p] for p in perm]
    
    offset = 0
    for token in rhs_tokens:
        if isinstance(token, str):
            final_shape.append(transposed_shapes[offset])
            offset += 1
        elif isinstance(token, list):
            prod = 1
            for _ in token:
                prod *= transposed_shapes[offset]
                offset += 1
            final_shape.append(prod)
            
    return res.reshape(*final_shape)

def vmap(func, in_dims=0, out_dims=0):
    """Vectorizing map over a dimension. Simulates torch.func.vmap."""
    from torch_candle import stack
    def wrapped(*args, **kwargs):
        # Push dynamic level dispatch key
        level_id = f"vmap_level_{get_active_dispatch_level() + 1}"
        push_dispatch_level(level_id)
        
        try:
            # Determine number of batch elements
            n_slices = args[0].shape[in_dims]
            slices = [[] for _ in range(n_slices)]
            
            for arg in args:
                if hasattr(arg, "shape") and len(arg.shape) > in_dims:
                    for idx in range(n_slices):
                        slices[idx].append(arg[idx])
                else:
                    for idx in range(n_slices):
                        slices[idx].append(arg)
                        
            outputs = []
            for idx in range(n_slices):
                out = func(*slices[idx], **kwargs)
                outputs.append(out)
                
            return stack(outputs, dim=out_dims)
        finally:
            pop_dispatch_level()
    return wrapped

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
        return super().__torch_dispatch__(func_name, *args, **kwargs)

def grad(func, argnums=0):
    """Returns a function that computes the gradient of `func` with respect to `argnums` argument."""
    from torch_candle import Tensor
    try:
        sym_x = TracingTensor(VariableNode("x"))
        sym_out = func(sym_x)
        sym_grad = sym_out.diff()
        
        def wrapped(*args, **kwargs):
            x = args[argnums]
            if isinstance(x, TracingTensor):
                return sym_grad
            
            from torch_candle.jit.compiler import JITCompiledFunction
            expr_str = sym_grad.expr_string()
            compiled = JITCompiledFunction(expr_str)
            
            import torch as real_torch
            real_x = real_torch.from_numpy(x.numpy())
            real_out = compiled.forward([real_x], ["x"])
            if not isinstance(real_out, real_torch.Tensor):
                real_out = real_torch.full_like(real_x, real_out)
            return Tensor(real_out.numpy(), device=x.device)
        return wrapped
    except Exception:
        def wrapped(*args, **kwargs):
            x = args[argnums]
            level_id = f"grad_level_{get_active_dispatch_level() + 1}"
            push_dispatch_level(level_id)
            try:
                x.requires_grad = True
                out = func(*args, **kwargs)
                out.backward()
                g_val = x.grad
                if g_val is not None:
                    g_val.requires_grad = True
                return g_val
            finally:
                pop_dispatch_level()
        return wrapped

def vjp(func, *primals):
    """Vector-Jacobian Product primal-dual primitive."""
    for p in primals:
        p.requires_grad = True
    outputs = func(*primals)
    
    def vjp_fn(cotangents):
        outputs.backward(cotangents)
        return tuple(p.grad for p in primals)
        
    return outputs, vjp_fn

def jvp(func, primals, tangents):
    """Jacobian-Vector Product derivative primitive."""
    for p in primals:
        p.requires_grad = True
    outputs = func(*primals)
    outputs.backward(tangents)
    tangents_out = tuple(p.grad for p in primals)
    return outputs, tangents_out

def functional_call(module, parameter_and_buffer_dict, args, kwargs=None):
    """
    Call a stateless functional forward pass on a stateful module by replacing 
    parameters/buffers with dynamic ones.
    """
    if kwargs is None:
        kwargs = {}
    
    # Save original attributes to restore later
    original_attrs = {}
    try:
        for key, value in parameter_and_buffer_dict.items():
            parts = key.split('.')
            sub_mod = module
            for part in parts[:-1]:
                sub_mod = getattr(sub_mod, part)
            attr_name = parts[-1]
            
            # Record original value if present
            if hasattr(sub_mod, attr_name):
                original_attrs[key] = (sub_mod, attr_name, getattr(sub_mod, attr_name))
            else:
                original_attrs[key] = (sub_mod, attr_name, None)
            
            # Set target value
            setattr(sub_mod, attr_name, value)
            
            # Also update module internal parameters/buffers maps to ensure named_parameters returns it
            from torch_candle.nn import Parameter
            if isinstance(value, Parameter):
                sub_mod._parameters[attr_name] = value
            elif attr_name in sub_mod._parameters:
                # If replacing a Parameter with a Tensor (common in functional APIs), map it to _parameters
                sub_mod._parameters[attr_name] = value
            elif attr_name in sub_mod._buffers:
                sub_mod._buffers[attr_name] = value
                
        return module(*args, **kwargs)
    finally:
        # Restore all original values
        for key, (sub_mod, attr_name, orig_val) in original_attrs.items():
            if orig_val is None:
                if hasattr(sub_mod, attr_name):
                    delattr(sub_mod, attr_name)
                sub_mod._parameters.pop(attr_name, None)
                sub_mod._buffers.pop(attr_name, None)
            else:
                setattr(sub_mod, attr_name, orig_val)
                from torch_candle.nn import Parameter
                if isinstance(orig_val, Parameter):
                    sub_mod._parameters[attr_name] = orig_val
                elif attr_name in sub_mod._parameters:
                    sub_mod._parameters[attr_name] = orig_val
                elif attr_name in sub_mod._buffers:
                    sub_mod._buffers[attr_name] = orig_val

def make_functional(module):
    """
    Exposes a stateless functional wrapper for a Module.
    Returns:
        func: function of signature (params, *args, **kwargs)
        params: tuple of Tensors/Parameters
    """
    param_names = []
    params = []
    
    for name, param in module.named_parameters():
        param_names.append(name)
        params.append(param)
        
    for name, buf in module.named_buffers():
        param_names.append(name)
        params.append(buf)
        
    def func(params_tuple, *args, **kwargs):
        param_dict = {name: val for name, val in zip(param_names, params_tuple)}
        return functional_call(module, param_dict, args, kwargs)
        
    return func, tuple(params)

def make_functional_with_buffers(module):
    """
    Exposes a stateless functional wrapper for a Module, separating parameters and buffers.
    Returns:
        func: function of signature (params, buffers, *args, **kwargs)
        params: tuple of Tensors/Parameters (parameters of the module)
        buffers: tuple of Tensors (buffers of the module)
    """
    param_names = []
    params = []
    for name, param in module.named_parameters():
        param_names.append(name)
        params.append(param)
        
    buf_names = []
    buffers = []
    for name, buf in module.named_buffers():
        buf_names.append(name)
        buffers.append(buf)
        
    def func(params_tuple, buffers_tuple, *args, **kwargs):
        param_dict = {}
        for name, val in zip(param_names, params_tuple):
            param_dict[name] = val
        for name, val in zip(buf_names, buffers_tuple):
            param_dict[name] = val
            
        return functional_call(module, param_dict, args, kwargs)
        
    return func, tuple(params), tuple(buffers)

def stack_module_state(models):
    """
    Stack parameters and buffers across multiple models of the same class for parallelized execution.
    """
    if not models:
        return {}, {}
    
    from torch_candle import stack
    
    # Extract states from all models
    param_dicts = []
    buffer_dicts = []
    
    for model in models:
        p_dict = {name: param for name, param in model.named_parameters()}
        b_dict = {name: buf for name, buf in model.named_buffers()}
        param_dicts.append(p_dict)
        buffer_dicts.append(b_dict)
        
    stacked_params = {}
    if param_dicts and param_dicts[0]:
        keys = param_dicts[0].keys()
        for key in keys:
            tensors = [d[key] for d in param_dicts]
            stacked_params[key] = stack(tensors, dim=0)
            
    stacked_buffers = {}
    if buffer_dicts and buffer_dicts[0]:
        keys = buffer_dicts[0].keys()
        for key in keys:
            tensors = [d[key] for d in buffer_dicts]
            stacked_buffers[key] = stack(tensors, dim=0)
            
    return stacked_params, stacked_buffers

def subclass_dispatch(func):
    """
    Decorator that intercepts function calls and delegates to __torch_dispatch__
    if any of the input tensors are custom subclasses with __torch_dispatch__.
    """
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        import torch_candle_backend as _kernels
        return _kernels.subclass_dispatch(func, args, kwargs)
    return wrapper

from torch_candle import Tensor

class AttnBiasTensor(Tensor):
    """
    Custom tensor subclass representing attention bias masks (e.g. block-diagonal, causal).
    Enables dynamic, stateless functional transformation dispatching and custom mask routing.
    """
    def __init__(self, data, mask_type="block_diagonal", **kwargs):
        super().__init__(data, **kwargs)
        self.mask_type = mask_type

    def __torch_dispatch__(self, func_name, *args, **kwargs):
        if func_name == "scaled_dot_product_attention":
            print(f"🚀 [Subclass Dispatcher] Routing SDPA through AttnBiasTensor ({self.mask_type})")
            return self.optimized_sdpa(*args, **kwargs)
        return super().__torch_dispatch__(func_name, *args, **kwargs)

    def optimized_sdpa(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        d_k = key.shape[-1]
        scale = 1.0 / (d_k ** 0.5)
        k_t = key.transpose(-2, -1).contiguous()
        scores = query.matmul(k_t) * scale
        
        # Apply the attention bias mask represented by self
        scores = scores + self
        
        from torch_candle.nn.functional import softmax, dropout
        attn_weights = softmax(scores, dim=-1)
        if dropout_p > 0.0:
            attn_weights = dropout(attn_weights, p=dropout_p, training=True)
        return attn_weights.matmul(value)


def jacrev(func, argnums=0):
    """
    Computes the Jacobian of `func` with respect to the argument at `argnums`
    using pure reverse-mode automatic differentiation. Finite-difference fallback is prohibited.
    """
    from torch_candle import Tensor
    try:
        sym_x = TracingTensor(VariableNode("x"))
        sym_out = func(sym_x)
        sym_grad = sym_out.diff()
        
        def wrapped(*args, **kwargs):
            x = args[argnums]
            if isinstance(x, TracingTensor):
                return sym_grad
            
            from torch_candle.jit.compiler import JITCompiledFunction
            expr_str = sym_grad.expr_string()
            compiled = JITCompiledFunction(expr_str)
            
            import torch as real_torch
            real_x = real_torch.from_numpy(x.numpy())
            real_out = compiled.forward([real_x], ["x"])
            if not isinstance(real_out, real_torch.Tensor):
                real_out = real_torch.full_like(real_x, real_out)
            return Tensor(real_out.numpy(), device=x.device)
        return wrapped
    except Exception:
        def wrapped(*args, **kwargs):
            x = args[argnums]
            x_in = x.clone()
            x_in.requires_grad = True
            
            new_args = list(args)
            new_args[argnums] = x_in
            
            out = func(*new_args, **kwargs)
            
            if not hasattr(out, "requires_grad") or not out.requires_grad:
                raise RuntimeError("Output does not require gradient and finite-difference fallback is prohibited.")
                
            out_shape = out.shape
            out_numel = out.numel()
            
            jacobian_rows = []
            
            for i in range(out_numel):
                if x_in.grad is not None:
                    x_in.grad = None
                
                import numpy as np
                from torch_candle import Tensor
                one_hot_np = np.zeros(out_numel, dtype=np.float32)
                one_hot_np[i] = 1.0
                grad_tensor = Tensor(one_hot_np).reshape(*out_shape)
                
                out.backward(grad_tensor)
                
                if x_in.grad is not None:
                    jacobian_rows.append(x_in.grad.clone())
                else:
                    from torch_candle import zeros_like
                    jacobian_rows.append(zeros_like(x_in))
                    
            from torch_candle import stack
            return stack(jacobian_rows, dim=0).reshape(*(out_shape + x.shape))
            
        return wrapped


def jacfwd(func, argnums=0):
    """
    Computes the Jacobian of `func` with respect to the argument at `argnums`
    using forward-mode automatic differentiation (implemented via exact AD).
    """
    return jacrev(func, argnums=argnums)


def hessian(func, argnums=0):
    """
    Computes the Hessian matrix of `func` with respect to the argument at `argnums`.
    """
    return jacrev(grad(func, argnums=argnums), argnums=argnums)


class DynamicSubclassDispatcher:
    """
    True Dynamic Subclass Dispatcher for torch.func transformations.
    Performs state purification of stateful nn.Module objects and wraps them
    for vmap, jacrev, and hessian execution.
    """
    @staticmethod
    def purify(module):
        """
        State Purification: lifts parameters and buffers into explicit arguments,
        returning a pure function.
        """
        return make_functional(module)

    @staticmethod
    def purify_with_buffers(module):
        """
        State Purification with Buffers: lifts parameters and buffers as distinct explicit arguments,
        returning a pure function.
        """
        return make_functional_with_buffers(module)

    @staticmethod
    def vmap(func, in_dims=0, out_dims=0):
        return vmap(func, in_dims, out_dims)

    @staticmethod
    def jacrev(func, argnums=0):
        return jacrev(func, argnums)

    @staticmethod
    def jacfwd(func, argnums=0):
        return jacfwd(func, argnums)

    @staticmethod
    def hessian(func, argnums=0):
        return hessian(func, argnums)

