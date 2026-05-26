import re

def generate_backend_signatures(schema_str):
    """
    Parse PyTorch-style operator schemas and output target Rust bindgen signatures.
    Example schema: 'add(Tensor self, Tensor other) -> Tensor'
    """
    match = re.match(r"(\w+)\((.*)\)\s*->\s*(\w+)", schema_str.strip())
    if not match:
        raise ValueError("Invalid schema format. Expected 'name(arg1Type arg1, ...) -> returnType'")
        
    name, args_str, ret_type = match.groups()
    args = [arg.strip().split() for arg in args_str.split(",") if arg.strip()]
    
    rust_args = []
    for arg in args:
        if len(arg) != 2:
            continue
        arg_type, arg_name = arg
        if arg_type == "Tensor":
            rust_args.append(f"{arg_name}: &PyTensor")
        else:
            rust_args.append(f"{arg_name}: {arg_type.lower()}")
            
    rust_signature = f"#[pyo3(name = \"{name}\")]\nfn {name}(&self, {', '.join(rust_args)}) -> PyResult<PyTensor>;"
    return rust_signature
