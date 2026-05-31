import numpy as np
import threading

_dispatch_state = threading.local()

def get_active_dispatch_level() -> int:
    """Retrieve the current level of the nested dynamic dispatcher stack."""
    if not hasattr(_dispatch_state, "active_levels"):
        _dispatch_state.active_levels = []
    return len(_dispatch_state.active_levels)

def push_dispatch_level(level_id: str):
    """Push a new transformation level onto the dynamic dispatcher stack."""
    if not hasattr(_dispatch_state, "active_levels"):
        _dispatch_state.active_levels = []
    _dispatch_state.active_levels.append(level_id)

def pop_dispatch_level() -> str:
    """Pop the top transformation level from the dynamic dispatcher stack."""
    if not hasattr(_dispatch_state, "active_levels") or not _dispatch_state.active_levels:
        return None
    return _dispatch_state.active_levels.pop()

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

def grad(func, argnums=0):
    """Returns a function that computes the gradient of `func` with respect to `argnums` argument."""
    def wrapped(*args, **kwargs):
        level_id = f"grad_level_{get_active_dispatch_level() + 1}"
        push_dispatch_level(level_id)
        try:
            x = args[argnums]
            x.requires_grad = True
            out = func(*args, **kwargs)
            out.backward()
            return x.grad
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
