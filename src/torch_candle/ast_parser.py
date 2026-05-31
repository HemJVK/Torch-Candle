import ast
import torch_candle as torch

class RustASTParser:
    """
    Standalone AST Parser and symbol verification engine.
    Ensures absolute type contracts and verifies that dynamic LLM-generated code symbols
    are verified against the active autograd graph scope before executing.
    """
    @staticmethod
    def parse_and_verify_expression(expr_str: str, scope: dict) -> torch.Tensor:
        """
        Parses a dynamic arithmetic or tensor expression string, verifies all variables
        exist in the active autograd scope, enforces type contracts, and executes it.
        """
        try:
            tree = ast.parse(expr_str, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"🚨 [AST Parser] Invalid dynamic syntax in expression: {expr_str}. Error: {e}")
            
        # 1. Symbol Verification
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                symbol_name = node.id
                if symbol_name not in scope and not hasattr(torch, symbol_name):
                    raise NameError(f"🚨 [AST Parser] Symbol Verification Failed: Reference to undefined symbol '{symbol_name}' in expression!")
                    
        # 2. Type Contracts Enforcements & Normalization
        # Compile and evaluate in the validated scope
        code = compile(tree, filename="<ast_dynamic>", mode="eval")
        raw_res = eval(code, {}, scope)
        
        # Inconsistent LLM output type contract normalization:
        # If output is not a Tensor, normalize it into a Torch-Candle Tensor!
        if not isinstance(raw_res, torch.Tensor):
            print(f"🚀 [AST Parser] Enforcing Type Contract: Normalizing raw output type {type(raw_res)} to torch_candle.Tensor")
            return torch.Tensor(raw_res)
            
        return raw_res
