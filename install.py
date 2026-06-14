#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess

def detect_cuda_compute_cap():
    # Check for nvidia-smi to query compute capability
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
        
    try:
        out = subprocess.check_output([nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"], text=True)
        # Parse e.g. "7.5\n" or "8.9\n" -> "75" or "89"
        cap = out.strip().replace(".", "")
        if cap.isdigit():
            return int(cap)
    except Exception:
        pass
    return None

def main():
    print("=" * 80)
    print("🕯️  Torch-Candle Auto-Hardware Installer")
    print("=" * 80)
    
    # 1. Check for rust toolchain
    if not shutil.which("rustc"):
        print("❌ Error: Rust toolchain not found. Please install Rust first:")
        print("   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
        sys.exit(1)

    # 2. Detect package manager (uv vs pip)
    use_uv = shutil.which("uv") is not None
    pkg_mgr = "uv pip" if use_uv else "pip"
    print(f"🔹 Package Manager: {pkg_mgr.upper()}")

    # 3. Determine build flags and environment variables
    env = os.environ.copy()
    features = ["pyo3/extension-module"]
    
    force_cuda = False
    if "--cuda" in sys.argv:
        force_cuda = True
        sys.argv.remove("--cuda")

    # 4. Detect hardware
    if sys.platform == "darwin":
        print("🍏 macOS detected. Enabling Apple Metal GPU and Accelerate CPU acceleration.")
        features.extend(["metal", "accelerate"])
    else:
        # Check for CUDA
        compute_cap = detect_cuda_compute_cap()
        nvcc_path = shutil.which("nvcc")
        
        # Check standard CUDA paths if nvcc is not in PATH
        cuda_home = "/usr/local/cuda"
        if not nvcc_path and os.path.exists(os.path.join(cuda_home, "bin/nvcc")):
            nvcc_path = os.path.join(cuda_home, "bin/nvcc")
            
        if compute_cap or nvcc_path or force_cuda:
            print("⚡ NVIDIA GPU detected. Enabling CUDA hardware acceleration.")
            features.append("cuda")
            
            # Resolve CUDA Paths
            if not env.get("CUDA_HOME"):
                if nvcc_path:
                    # e.g., /usr/local/cuda/bin/nvcc -> /usr/local/cuda
                    env["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc_path))
                else:
                    env["CUDA_HOME"] = cuda_home
            if not env.get("CUDA_PATH"):
                env["CUDA_PATH"] = env["CUDA_HOME"]
                
            # Resolve Compute Capability
            if not env.get("CUDA_COMPUTE_CAP"):
                if compute_cap:
                    env["CUDA_COMPUTE_CAP"] = str(compute_cap)
                else:
                    env["CUDA_COMPUTE_CAP"] = "80"  # Default to Ampere/A100 capability
            
            print(f"   CUDA_HOME:        {env['CUDA_HOME']}")
            print(f"   CUDA_COMPUTE_CAP: {env['CUDA_COMPUTE_CAP']}")
        else:
            print("💻 CPU-only environment detected. Building optimized CPU version.")

    # Configure Maturin feature flags
    env["MATURIN_PEP517_ARGS"] = f"--features {','.join(features)}"
    
    # 5. Construct command
    cmd = []
    if use_uv:
        cmd.extend(["uv", "pip", "install", "--force-reinstall", "--no-cache"])
    else:
        cmd.extend(["pip", "install", "--force-reinstall", "--no-cache-dir"])
        
    # Editable vs standard install flag
    if "--editable" in sys.argv or "-e" in sys.argv:
        cmd.append("-e")
        print("   Install Type:     Editable (-e)")
    else:
        print("   Install Type:     Standard")
        
    cmd.append(".")
    
    print(f"\n🚀 Running: {' '.join(cmd)}")
    print("=" * 80 + "\n")
    
    try:
        subprocess.check_call(cmd, env=env)
        print("\n✨ Torch-Candle installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
