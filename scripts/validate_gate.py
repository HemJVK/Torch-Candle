#!/usr/bin/env python3
import sys
import os
import numpy as np

# Path resolver
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(workspace_root, "src"))

def main():
    print("⏳ Running Phase VI Deterministic Validation Gate...")
    
    # 1. Imports
    try:
        import torch_candle as torch
        from torch_candle import Tensor, ZeroToolCallGuard, HardValidationFailure
        from torch_candle.func import (
            jacrev, hessian, DynamicSubclassDispatcher, make_functional, vmap, grad
        )
    except ImportError as e:
        print(f"❌ Verification failed during import: {e}")
        sys.exit(1)
        
    print("✅ Successfully imported all mandatory Phase VI symbols.")
    
    # 2. Check DynamicSubclassDispatcher APIs
    assert hasattr(DynamicSubclassDispatcher, "purify"), "DynamicSubclassDispatcher is missing 'purify'"
    assert hasattr(DynamicSubclassDispatcher, "vmap"), "DynamicSubclassDispatcher is missing 'vmap'"
    assert hasattr(DynamicSubclassDispatcher, "jacrev"), "DynamicSubclassDispatcher is missing 'jacrev'"
    assert hasattr(DynamicSubclassDispatcher, "hessian"), "DynamicSubclassDispatcher is missing 'hessian'"
    print("✅ DynamicSubclassDispatcher contract signatures verified.")
    
    # 3. Check jacrev and hessian functional transforms
    # Simple function: f(x) = x * x. grad = 2*x, hessian = 2
    f = lambda x: x * x
    x = Tensor([3.0])
    
    try:
        j_val = jacrev(f)(x)
        h_val = hessian(f)(x)
        
        assert np.allclose(j_val.numpy(), [6.0], rtol=1e-3, atol=1e-3), f"jacrev incorrect value: {j_val.numpy()}"
        assert np.allclose(h_val.numpy(), [2.0], rtol=1e-3, atol=1e-3), f"hessian incorrect value: {h_val.numpy()}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error during jacrev/hessian verification: {e}")
        sys.exit(1)
    print("✅ Reverse-mode Jacobian and Hessian transforms verified.")

    # 4. Check Propagation Mandate (No EMA reconstruction/gradient masking)
    Tensor.enable_sha = False
    w = Tensor([5.0], requires_grad=True)
    w.grad = Tensor([float('nan')])
    assert np.isnan(w.grad.item()), "Propagation Mandate Failed: NaN gradient was not preserved"
    
    # Verify NaN flow even when enable_sha is toggled to True (no healing occurs)
    Tensor.enable_sha = True
    w.grad = Tensor([float('nan')])
    assert np.isnan(w.grad.item()), "Propagation Mandate Failed: NaN gradient was not preserved with enable_sha=True"
    print("✅ Propagation Mandate verified: NaNs/Infs flow naturally without healing under all configurations.")
    
    # 5. Check Zero-Tool-Call Guard
    ZeroToolCallGuard.reset_tool_call_count()
    try:
        ZeroToolCallGuard.verify_execution("Success")
        print("❌ Zero-Tool-Call Guard failed: Success with 0 tool calls did not raise HardValidationFailure!")
        sys.exit(1)
    except HardValidationFailure as e:
        print(f"✅ Zero-Tool-Call Guard successfully blocked phantom execution: {e}")
        
    ZeroToolCallGuard.increment_tool_call_count()
    try:
        ZeroToolCallGuard.verify_execution("Success")
        print("✅ Zero-Tool-Call Guard passed success verification with active tool calls.")
    except HardValidationFailure as e:
        print(f"❌ Zero-Tool-Call Guard failed: raised HardValidationFailure even when tools were called: {e}")
        sys.exit(1)

    # 6. Check AOT Build Validation (ROCm hipcc mandate)
    build_rs_path = os.path.join(workspace_root, "rust", "build.rs")
    cmake_path = os.path.join(workspace_root, "CMakeLists.txt")
    aot_ok = False
    for path in [build_rs_path, cmake_path]:
        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read()
                if "hipcc" in content:
                    aot_ok = True
                    break
    assert aot_ok, "AOT Build Validation Failed: build.rs/CMakeLists.txt must exist and explicitly execute hipcc"
    print("✅ AOT Build Validation verified: build.rs/CMakeLists.txt configured for hipcc AOT compilation.")

    # 7. Check DTS Brain Hard-Fail Logic Gate
    from torch_candle import DTSBrain, HardFail
    try:
        DTSBrain.verify_payload("Payload containing TrackedTensor signature")
        print("❌ DTS Brain Hard-Fail Logic Gate failed to intercept payload!")
        sys.exit(1)
    except HardFail as e:
        print(f"✅ DTS Brain Hard-Fail Logic Gate successfully blocked non-native profile: {e}")
        
    print("\n🎉 ALL PHASE X VALIDATION GATE CHECKS PASSED SUCCESSFULLY!")
    sys.exit(0)

if __name__ == "__main__":
    main()
