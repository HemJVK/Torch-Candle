import ctypes
import os
import subprocess
import platform

class CtypesMmapOptimizer:
    """
    Manages physical and virtual memory areas (VMAs), configures high-frequency
    mmap thresholds, and enforces programmatic kernel map limit adjustments.
    """
    @staticmethod
    def adjust_kernel_mmap_limit() -> bool:
        """
        Attempts to adjust the kernel's max_map_count threshold to prevent multi-process
        memory allocation overflows (OOM cannot allocate memory 12 errors).
        """
        # Enforce strict glibc memory threshold: MALLOC_MMAP_THRESHOLD_ = 65536
        os.environ["MALLOC_MMAP_THRESHOLD_"] = "65536"
        try:
            if platform.system() == "Linux":
                try:
                    libc = ctypes.CDLL("libc.so.6")
                except Exception:
                    libc = ctypes.CDLL(None)
                # M_MMAP_THRESHOLD is 3
                res = libc.mallopt(3, 65536)
                print(f"🚀 [Memory Optimization] Enforced M_MMAP_THRESHOLD = 65536 via mallopt (status: {res})")
        except Exception as e:
            print(f"⚠️ [Memory Optimization] Failed to set mallopt: {e}")

        if platform.system() != "Linux":
            return False
            
        print("🚀 [Memory Optimization] Checking vm.max_map_count configuration...")
        try:
            # Check current limit
            current_limit = int(subprocess.check_output(["sysctl", "-n", "vm.max_map_count"]).strip())
            if current_limit >= 1048576:
                print(f"🚀 [Memory Optimization] Current map limit ({current_limit}) is already optimized.")
                return True
                
            print(f"🚀 [Memory Optimization] Map limit is {current_limit}. Attempting to optimize to 1048576...")
            # We run it with subprocess; if we are not root, it will print a guidance tip.
            res = subprocess.run(
                ["sudo", "sysctl", "-w", "vm.max_map_count=1048576"],
                capture_output=True,
                text=True
            )
            if res.returncode == 0:
                print("🚀 [Memory Optimization] Successfully optimized vm.max_map_count to 1048576!")
                return True
            else:
                print("⚠️ [Memory Optimization] Permission denied. Please execute manually: 'sudo sysctl -w vm.max_map_count=1048576'")
                return False
        except Exception as e:
            print(f"⚠️ [Memory Optimization] Unable to automatically configure map limits: {e}")
            return False

    @staticmethod
    def create_ctypes_mmap(size: int):
        """
        Allocates and returns an optimized ctypes mmap segment for high-frequency shared memory.
        """
        libc = ctypes.CDLL(None)
        
        # Configure standard mmap constants
        PROT_READ = 0x1
        PROT_WRITE = 0x2
        MAP_SHARED = 0x01
        MAP_ANONYMOUS = 0x20
        
        # Define mmap signature
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.mmap.restype = ctypes.c_void_p
        
        ptr = libc.mmap(None, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0)
        if ptr == -1:
            raise OSError("🚨 [Memory Optimization] ctypes mmap allocation failed: Cannot allocate memory")
            
        return ptr

# Automatically attempt kernel map adjustments on import
CtypesMmapOptimizer.adjust_kernel_mmap_limit()
