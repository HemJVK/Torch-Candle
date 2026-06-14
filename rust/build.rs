use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels_rocm.hip");
    
    // Check if hipcc is available in the environment
    let version_output = Command::new("hipcc").arg("--version").output();
    let has_hipcc = version_output.is_ok();
    let is_nvcc = version_output
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("nvcc"))
        .unwrap_or(false);
    let use_rocm = std::env::var("USE_ROCM").map(|v| v == "1" || v == "true" || v == "True").unwrap_or(false);
    
    if use_rocm && !has_hipcc {
        panic!("❌ [hipcc Build] hipcc compiler is MANDATORY when USE_ROCM=1 is configured, but was not found in PATH!");
    }
    
    if has_hipcc {
        println!("cargo:warning=🚀 [hipcc Build] Found hipcc! Compiling AMD ROCm HIP kernels Ahead-of-Time (AOT)...");
        let out_dir = std::env::var("OUT_DIR").unwrap();
        
        let mut compile_args = vec![
            "-c".to_string(),
            "src/kernels_rocm.hip".to_string(),
            "-fPIC".to_string(),
        ];
        if is_nvcc {
            compile_args.push("-x".to_string());
            compile_args.push("cu".to_string());
        } else {
            compile_args.extend(
                vec![
                    "--offload-arch=gfx906",
                    "--offload-arch=gfx908",
                    "--offload-arch=gfx90a",
                    "--offload-arch=gfx90c",
                    "--offload-arch=gfx1030",
                    "--offload-arch=gfx1100",
                ]
                .into_iter()
                .map(|s| s.to_string()),
            );
        }
        compile_args.push("-o".to_string());
        compile_args.push(format!("{}/kernels_rocm.o", out_dir));

        let output = Command::new("hipcc")
            .args(&compile_args)
            .output();
            
        let compile_success = match &output {
            Ok(o) => o.status.success(),
            _ => false,
        };
        
        if !compile_success {
            if let Ok(o) = output {
                let stdout = String::from_utf8_lossy(&o.stdout);
                let stderr = String::from_utf8_lossy(&o.stderr);
                for line in stdout.lines() {
                    println!("cargo:warning=[hipcc stdout] {}", line);
                }
                for line in stderr.lines() {
                    println!("cargo:warning=[hipcc stderr] {}", line);
                }
            }
            if use_rocm {
                panic!("❌ [hipcc Build] hipcc compilation failed, which is fatal when USE_ROCM=1!");
            }
            println!("cargo:warning=❌ [hipcc Build] hipcc compilation failed. No fallback permitted.");
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
            println!("cargo:warning=❌ [hipcc Build] ar command failed. No fallback permitted.");
        }
    } else {
        println!("cargo:warning=⚠️ [hipcc Build] hipcc compiler not found in PATH. Skipping AOT AMD compilation.");
    }

    // Link MKL and OpenMP only if the 'mkl' feature is enabled.
    if std::env::var("CARGO_FEATURE_MKL").is_ok() {
        let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        
        let mut mkl_search_paths = Vec::new();

        // 1. Prioritize paths from the environment variable LIB
        if let Ok(lib_env) = std::env::var("LIB") {
            for path_str in lib_env.split(|c| c == ':' || c == ';') {
                if !path_str.is_empty() {
                    mkl_search_paths.push(std::path::PathBuf::from(path_str));
                }
            }
        }

        // 2. Prioritize paths derived from CMAKE_INCLUDE_PATH (map /include -> /lib)
        if let Ok(cmake_include) = std::env::var("CMAKE_INCLUDE_PATH") {
            for path_str in cmake_include.split(|c| c == ':' || c == ';') {
                if !path_str.is_empty() {
                    let include_path = std::path::PathBuf::from(path_str);
                    mkl_search_paths.push(include_path.clone());
                    if let Some(parent) = include_path.parent() {
                        mkl_search_paths.push(parent.join("lib"));
                    }
                    if path_str.contains("include") {
                        let lib_str = path_str.replace("include", "lib");
                        mkl_search_paths.push(std::path::PathBuf::from(lib_str));
                    }
                }
            }
        }

        // 3. Fallback/default search paths
        mkl_search_paths.extend(vec![
            std::path::PathBuf::from("/home/hem/personal/Library/Torch-Candle/.venv/lib"),
            std::path::PathBuf::from("/usr/lib/x86_64-linux-gnu"),
            std::path::PathBuf::from("/usr/local/lib"),
            std::path::PathBuf::from("/usr/lib"),
        ]);

        if let Some(parent) = manifest_dir.parent() {
            mkl_search_paths.push(parent.join(".venv").join("lib"));
            if let Some(grandparent) = parent.parent() {
                mkl_search_paths.push(grandparent.join(".venv").join("lib"));
            }
        }

        if let Ok(pwd) = std::env::var("PWD") {
            let pwd_path = std::path::PathBuf::from(pwd);
            mkl_search_paths.push(pwd_path.join(".venv").join("lib"));
            mkl_search_paths.push(pwd_path.join("..").join(".venv").join("lib"));
            mkl_search_paths.push(pwd_path.join("..").join("Torch-Candle").join(".venv").join("lib"));
            mkl_search_paths.push(pwd_path.join("..").join("..").join("Torch-Candle").join(".venv").join("lib"));
            mkl_search_paths.push(pwd_path.join("..").join("..").join(".venv").join("lib"));
        }

        if let Ok(conda) = std::env::var("CONDA_PREFIX") {
            mkl_search_paths.push(std::path::PathBuf::from(conda).join("lib"));
        }

        // Output search path directions for every path that exists to support multi-directory layouts (MKL vs OpenMP)
        for path in &mkl_search_paths {
            if path.exists() {
                println!("cargo:rustc-link-search=native={}", path.display());
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
            }
        }

        // Find the first path that actually contains libmkl_rt.so
        let mut found_path = None;
        for path in &mkl_search_paths {
            if path.join("libmkl_rt.so").exists() {
                found_path = Some(path.clone());
                break;
            }
        }

        let final_mkl_path = found_path.unwrap_or_else(|| {
            manifest_dir.parent().unwrap().join(".venv").join("lib")
        });

        println!("cargo:rustc-link-search=native={}", final_mkl_path.display());
        println!("cargo:rustc-link-lib=dylib=mkl_rt");
        println!("cargo:rustc-link-lib=dylib=iomp5");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", final_mkl_path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../../../../");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../../../../.venv/lib");
    }
}
