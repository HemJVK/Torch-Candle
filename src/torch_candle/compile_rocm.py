import subprocess
import shutil
import os

class ROCmAOTCompiler:
    """
    Ahead-of-Time (AOT) compiler driver invoking native hipcc to compile
    custom HIP/ROCm kernels into highly-optimized binary extensions.
    """
    @staticmethod
    def is_hipcc_available() -> bool:
        return shutil.which("hipcc") is not None

    @staticmethod
    def compile_kernels_aot(source_file: str, output_so: str) -> bool:
        if not ROCmAOTCompiler.is_hipcc_available():
            print("⚠️ [ROCmAOTCompiler] hipcc compiler not found in PATH. Skipping AOT compilation.")
            return False
            
        print(f"🚀 [ROCmAOTCompiler] Compiling {source_file} Ahead-of-Time using hipcc...")
        cmd = [
            "hipcc",
            "-shared",
            "-fPIC",
            "-O3",
            source_file,
            "-o",
            output_so
        ]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"🚀 [ROCmAOTCompiler] AOT compilation successful: {output_so}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ [ROCmAOTCompiler] hipcc compilation failed:\n{e.stderr}")
            return False
