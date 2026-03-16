import sys
import inspect
from typing import Set, Tuple, Dict, List

# Try importing torch
try:
    import torch
except ImportError:
    print("Error: PyTorch is not installed. Please install it to run the coverage check.")
    sys.exit(1)

# Try importing torch_candle
try:
    import torch_candle as tc
except ImportError:
    print("Error: torch_candle is not installed or not in PYTHONPATH.")
    sys.exit(1)

def get_public_api(module) -> Set[str]:
    api: Set[str] = set()
    for name in dir(module):
        if not name.startswith("_"):
            api.add(name)
    return api

def compare_modules(torch_mod, tc_mod, mod_name: str) -> Dict[str, List[str]]:
    torch_api = get_public_api(torch_mod)
    tc_api = get_public_api(tc_mod)
    
    implemented = torch_api.intersection(tc_api)
    missing = torch_api.difference(tc_api)
    extra = tc_api.difference(torch_api)
    
    return {
        "implemented": sorted(list(implemented)),
        "missing": sorted(list(missing)),
        "extra": sorted(list(extra)),
        "coverage": len(implemented) / len(torch_api) if len(torch_api) > 0 else 1.0
    }

def main():
    modules_to_check = [
        ("torch", torch, tc),
        ("torch.nn", torch.nn, getattr(tc, "nn", None)),
        ("torch.nn.functional", torch.nn.functional, getattr(getattr(tc, "nn", None), "functional", None)),
        ("torch.optim", torch.optim, getattr(tc, "optim", None)),
        ("torch.autograd", torch.autograd, getattr(tc, "autograd", None)),
        ("torch.cuda", torch.cuda, getattr(tc, "cuda", None)),
    ]
    
    print("# PyTorch API Coverage Report")
    print("This report outlines the current API coverage of `torch_candle` compared to `torch`.\\n")
    
    for name, t_mod, tc_mod in modules_to_check:
        if tc_mod is None:
            print(f"## {name}")
            print(f"**Error:** {name} is not implemented in torch_candle.\\n")
            print("---\\n")
            continue
        try:
            res = compare_modules(t_mod, tc_mod, name)
            print(f"## {name}")
            print(f"- **Coverage:** {res['coverage']*100:.2f}% ({len(res['implemented'])} / {len(res['implemented']) + len(res['missing'])})")
            print(f"- **Implemented APIs:** {len(res['implemented'])}")
            print(f"- **Missing APIs:** {len(res['missing'])}")
            print("")
            
            # Print a few missing ones as examples
            examples = res['missing'][:15]
            if examples:
                print(f"**Top Missing APIs (Sample):**")
                print(", ".join(examples) + ("..." if len(res['missing']) > 15 else ""))
            print("\n---\n")
            
        except AttributeError as e:
            print(f"## {name}")
            print(f"**Error:** {name} is not fully accessible in torch_candle. ({e})\n")
            print("---\n")

if __name__ == "__main__":
    main()
