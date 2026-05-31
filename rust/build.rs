use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels_rocm.hip");
    
    // Check if hipcc is available in the environment
    let has_hipcc = Command::new("hipcc").arg("--version").status().is_ok();
    
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
            
        if let Ok(s) = status {
            if s.success() {
                let lib_status = Command::new("ar")
                    .args(&[
                        "crus",
                        &format!("{}/libkernels_rocm.a", out_dir),
                        &format!("{}/kernels_rocm.o", out_dir),
                    ])
                    .status();
                if let Ok(ls) = lib_status {
                    if ls.success() {
                        println!("cargo:rustc-link-search=native={}", out_dir);
                        println!("cargo:rustc-link-lib=static=kernels_rocm");
                        println!("cargo:rustc-link-lib=dylib=amdhip64");
                        println!("cargo:warning=🚀 [hipcc Build] ROCm/HIP kernels successfully compiled AOT and linked!");
                        return;
                    }
                }
            }
        }
        println!("cargo:warning=⚠️ [hipcc Build] hipcc compilation failed. Falling back to default emulation.");
    } else {
        println!("cargo:warning=⚠️ [hipcc Build] hipcc compiler not found in PATH. Skipping AOT AMD compilation.");
    }
}
