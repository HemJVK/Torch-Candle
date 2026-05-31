use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels_rocm.hip");
    
    // Check if hipcc is available in the environment
    let has_hipcc = Command::new("hipcc").arg("--version").status().is_ok();
    let use_rocm = std::env::var("USE_ROCM").map(|v| v == "1" || v == "true" || v == "True").unwrap_or(false);
    
    if use_rocm && !has_hipcc {
        panic!("❌ [hipcc Build] hipcc compiler is MANDATORY when USE_ROCM=1 is configured, but was not found in PATH!");
    }
    
    if has_hipcc {
        println!("cargo:warning=🚀 [hipcc Build] Found hipcc! Compiling AMD ROCm HIP kernels Ahead-of-Time (AOT)...");
        let out_dir = std::env::var("OUT_DIR").unwrap();
        
        let status = Command::new("hipcc")
            .args(&[
                "-c",
                "src/kernels_rocm.hip",
                "-fPIC",
                "-o",
                &format!("{}/kernels_rocm.o", out_dir),
            ])
            .status();
            
        let compile_success = match status {
            Ok(s) => s.success(),
            _ => false,
        };
        
        if !compile_success {
            if use_rocm {
                panic!("❌ [hipcc Build] hipcc compilation failed, which is fatal when USE_ROCM=1!");
            }
            println!("cargo:warning=⚠️ [hipcc Build] hipcc compilation failed. Falling back to default emulation.");
            return;
        }
        
        let lib_status = Command::new("ar")
            .args(&[
                "crus",
                &format!("{}/libkernels_rocm.a", out_dir),
                &format!("{}/kernels_rocm.o", out_dir),
            ])
            .status();
            
        let ar_success = match lib_status {
            Ok(ls) => ls.success(),
            _ => false,
        };
        
        if ar_success {
            println!("cargo:rustc-link-search=native={}", out_dir);
            println!("cargo:rustc-link-lib=static=kernels_rocm");
            println!("cargo:rustc-link-lib=dylib=amdhip64");
            println!("cargo:warning=🚀 [hipcc Build] ROCm/HIP kernels successfully compiled AOT and linked!");
        } else {
            if use_rocm {
                panic!("❌ [hipcc Build] ar static library creation failed, which is fatal when USE_ROCM=1!");
            }
            println!("cargo:warning=⚠️ [hipcc Build] ar command failed. Falling back to default emulation.");
        }
    } else {
        println!("cargo:warning=⚠️ [hipcc Build] hipcc compiler not found in PATH. Skipping AOT AMD compilation.");
    }
}
