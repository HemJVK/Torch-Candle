#!/usr/bin/env python3
import os
import sys
import re
import subprocess
import shutil

def modify_pyproject(name, features):
    with open("pyproject.toml", "r") as f:
        content = f.read()
    
    # Replace name = "..." in the [project] section
    content = re.sub(r'(name\s*=\s*)"[^"]+"', f'\\1"{name}"', content, count=1)
    
    # Format the features list
    features_str = ", ".join(f'"{f}"' for f in features)
    features_line = f'features = [{features_str}]'
    
    # Replace features = [...] in the [tool.maturin] section
    content = re.sub(r'(features\s*=\s*\[[^\]]+\])', features_line, content, count=1)
    
    with open("pyproject.toml", "w") as f:
        f.write(content)

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ["--cpu", "--cuda"]:
        print("Usage: python build_wheels.py [--cpu | --cuda]")
        sys.exit(1)
        
    mode = sys.argv[1]
    original_name = "torch-candle"
    original_features = ["pyo3/extension-module"]
    
    # Check for rust toolchain
    if not shutil.which("cargo"):
        print("❌ Error: Rust toolchain (cargo) is required to build wheels.")
        sys.exit(1)
        
    # Check for maturin
    if not shutil.which("maturin"):
        print("❌ Error: maturin is required to build wheels. Install it via pip/uv first.")
        sys.exit(1)

    try:
        if mode == "--cuda":
            print("=" * 80)
            print("🔨 Building CUDA Package: torch-candle-cuda")
            print("=" * 80)
            
            # Auto-resolve CUDA capability and path for local compilation in the build environment
            env = os.environ.copy()
            
            # Find nvcc
            nvcc_path = shutil.which("nvcc")
            if not nvcc_path and os.path.exists("/usr/local/cuda/bin/nvcc"):
                nvcc_path = "/usr/local/cuda/bin/nvcc"
                
            if nvcc_path:
                if not env.get("CUDA_HOME"):
                    env["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc_path))
                if not env.get("CUDA_PATH"):
                    env["CUDA_PATH"] = env["CUDA_HOME"]
                    
            # Set default compute capability if not set
            if not env.get("CUDA_COMPUTE_CAP"):
                # Try getting compute cap via nvidia-smi
                nvidia_smi = shutil.which("nvidia-smi")
                if nvidia_smi:
                    try:
                        out = subprocess.check_output([nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"], text=True)
                        cap = out.strip().replace(".", "")
                        if cap.isdigit():
                            env["CUDA_COMPUTE_CAP"] = cap
                    except Exception:
                        pass
                if not env.get("CUDA_COMPUTE_CAP"):
                    env["CUDA_COMPUTE_CAP"] = "75"  # Standard fallback
            
            print(f"   CUDA_HOME:        {env.get('CUDA_HOME')}")
            print(f"   CUDA_COMPUTE_CAP: {env.get('CUDA_COMPUTE_CAP')}")
            
            # Temporarily configure name and features for the CUDA wheel/sdist
            modify_pyproject("torch-candle-cuda", ["pyo3/extension-module", "cuda"])
            
            # Build wheel
            print("\n📦 Generating wheels...")
            subprocess.check_call(["maturin", "build", "--release", "--manifest-path", "rust/Cargo.toml"], env=env)
            
            # Build sdist (source distribution)
            print("\n📦 Generating source distribution (sdist)...")
            subprocess.check_call(["maturin", "sdist", "--manifest-path", "rust/Cargo.toml"], env=env)
        else:
            print("=" * 80)
            print("🔨 Building CPU/Metal Package: torch-candle")
            print("=" * 80)
            
            features = ["pyo3/extension-module"]
            if sys.platform == "darwin":
                print("   Enabling macOS Metal and Accelerate backends...")
                features.extend(["metal", "accelerate"])
                
            # Temporarily configure name and features for standard wheel/sdist
            modify_pyproject("torch-candle", features)
            
            # Build wheel
            print("\n📦 Generating wheels...")
            subprocess.check_call(["maturin", "build", "--release", "--manifest-path", "rust/Cargo.toml"])
            
            # Build sdist (source distribution)
            print("\n📦 Generating source distribution (sdist)...")
            subprocess.check_call(["maturin", "sdist", "--manifest-path", "rust/Cargo.toml"])
            
        print("\n✨ Build successful! Output stored in: target/wheels/")
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)
    finally:
        # Always restore default package configuration
        modify_pyproject(original_name, original_features)

if __name__ == "__main__":
    main()
