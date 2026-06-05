import subprocess
import shutil
import logging

logger = logging.getLogger(__name__)

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
            logger.warning("[ROCmAOTCompiler] hipcc compiler not found in PATH. Skipping AOT compilation.")
            return False

        logger.info("[ROCmAOTCompiler] Compiling %s ahead-of-time using hipcc...", source_file)
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
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("[ROCmAOTCompiler] AOT compilation successful: %s", output_so)
            return True
        except subprocess.CalledProcessError as e:
            logger.error("[ROCmAOTCompiler] hipcc compilation failed:\n%s", e.stderr)
            return False
