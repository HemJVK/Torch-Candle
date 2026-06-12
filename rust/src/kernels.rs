use crate::simd::*;
use ndarray::ArrayViewMutD;
use rayon::prelude::*;

// ─── Local AVX2 helpers (x86_64 only) ────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ─── Element-wise ops ────────────────────────────────────────────────────────

pub fn fast_relu(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    unsafe { simd_relu_slice(data); }
}

pub fn fast_exp(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    unsafe { simd_exp_slice(data); }
}

pub fn fast_log(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    unsafe { simd_log_slice(data); }
}

pub fn fast_sqrt(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    unsafe { simd_sqrt_slice(data); }
}

pub fn fast_sigmoid(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    unsafe { simd_sigmoid_slice(data); }
}

pub fn fast_tanh(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    unsafe { simd_tanh_slice(data); }
}

pub fn fast_silu(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    unsafe { simd_silu_slice(data); }
}

pub fn fast_gelu(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    unsafe { simd_gelu_slice(data); }
}

// ─── Reductions ──────────────────────────────────────────────────────────────

pub fn fast_sum_all(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        const PAR_THRESH: usize = 4 * 1024 * 1024;
        if data.len() <= PAR_THRESH {
            unsafe {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();
                let mut i = 0;
                while i + 32 <= data.len() {
                    acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data.as_ptr().add(i)));
                    acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(data.as_ptr().add(i + 8)));
                    acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(data.as_ptr().add(i + 16)));
                    acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(data.as_ptr().add(i + 24)));
                    i += 32;
                }
                while i + 8 <= data.len() {
                    acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data.as_ptr().add(i)));
                    i += 8;
                }
                let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                let mut buf = [0.0f32; 8];
                _mm256_storeu_ps(buf.as_mut_ptr(), acc);
                buf.iter().sum::<f32>() + data[i..].iter().sum::<f32>()
            }
        } else {
            let chunk_sums: Vec<f32> = data.par_chunks(65536).map(|chunk| unsafe {
                let mut acc = _mm256_setzero_ps();
                let mut i = 0;
                while i + 8 <= chunk.len() {
                    acc = _mm256_add_ps(acc, _mm256_loadu_ps(chunk.as_ptr().add(i)));
                    i += 8;
                }
                let mut buf = [0.0f32; 8];
                _mm256_storeu_ps(buf.as_mut_ptr(), acc);
                buf.iter().sum::<f32>() + chunk[i..].iter().sum::<f32>()
            }).collect();
            chunk_sums.iter().sum()
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 16 <= data.len() {
                acc0 = vaddq_f32(acc0, vld1q_f32(data.as_ptr().add(i)));
                acc1 = vaddq_f32(acc1, vld1q_f32(data.as_ptr().add(i + 4)));
                acc2 = vaddq_f32(acc2, vld1q_f32(data.as_ptr().add(i + 8)));
                acc3 = vaddq_f32(acc3, vld1q_f32(data.as_ptr().add(i + 12)));
                i += 16;
            }
            let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            let mut buf = [0.0f32; 4];
            vst1q_f32(buf.as_mut_ptr(), acc);
            buf.iter().sum::<f32>() + data[i..].iter().sum::<f32>()
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { data.iter().sum() }
}

pub fn fast_max_all(data: &[f32]) -> f32 {
    if data.is_empty() { return f32::NEG_INFINITY; }
    #[cfg(target_arch = "x86_64")]
    {
        let chunk_maxes: Vec<f32> = data.par_chunks(2048).map(|chunk| unsafe {
            let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut i = 0;
            while i + 8 <= chunk.len() {
                acc = _mm256_max_ps(acc, _mm256_loadu_ps(chunk.as_ptr().add(i)));
                i += 8;
            }
            let mut buf = [f32::NEG_INFINITY; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), acc);
            let mut m = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            for &v in &chunk[i..] { if v > m { m = v; } }
            m
        }).collect();
        chunk_maxes.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
            let mut i = 0;
            while i + 4 <= data.len() {
                acc = vmaxq_f32(acc, vld1q_f32(data.as_ptr().add(i)));
                i += 4;
            }
            let mut buf = [f32::NEG_INFINITY; 4];
            vst1q_f32(buf.as_mut_ptr(), acc);
            let mut m = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            for &v in &data[i..] { if v > m { m = v; } }
            m
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) }
}

pub fn fast_min_all(data: &[f32]) -> f32 {
    if data.is_empty() { return f32::INFINITY; }
    #[cfg(target_arch = "x86_64")]
    {
        let chunk_mins: Vec<f32> = data.par_chunks(2048).map(|chunk| unsafe {
            let mut acc = _mm256_set1_ps(f32::INFINITY);
            let mut i = 0;
            while i + 8 <= chunk.len() {
                acc = _mm256_min_ps(acc, _mm256_loadu_ps(chunk.as_ptr().add(i)));
                i += 8;
            }
            let mut buf = [f32::INFINITY; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), acc);
            let mut m = buf.iter().cloned().fold(f32::INFINITY, f32::min);
            for &v in &chunk[i..] { if v < m { m = v; } }
            m
        }).collect();
        chunk_mins.iter().cloned().fold(f32::INFINITY, f32::min)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let mut acc = vdupq_n_f32(f32::INFINITY);
            let mut i = 0;
            while i + 4 <= data.len() {
                acc = vminq_f32(acc, vld1q_f32(data.as_ptr().add(i)));
                i += 4;
            }
            let mut buf = [f32::INFINITY; 4];
            vst1q_f32(buf.as_mut_ptr(), acc);
            let mut m = buf.iter().cloned().fold(f32::INFINITY, f32::min);
            for &v in &data[i..] { if v < m { m = v; } }
            m
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { data.iter().cloned().fold(f32::INFINITY, f32::min) }
}

pub fn fast_mean_all(data: &[f32]) -> f32 {
    if data.is_empty() { return 0.0; }
    fast_sum_all(data) / data.len() as f32
}

pub fn fast_norm_l2(data: &[f32]) -> f32 {
    const PAR_THRESH: usize = 512 * 1024;
    if data.len() <= PAR_THRESH {
        let mut sum = 0.0f32;
        for &x in data {
            sum += x * x;
        }
        sum.sqrt()
    } else {
        let sq_sum: f32 = data.par_chunks(65536).map(|chunk| {
            chunk.iter().map(|&x| x * x).sum::<f32>()
        }).sum();
        sq_sum.sqrt()
    }
}

pub fn fast_std_all(data: &[f32], ddof: usize) -> f32 {
    let n = data.len();
    if n <= ddof { return 0.0; }
    let mean = fast_mean_all(data);
    const PAR_THRESH: usize = 512 * 1024;
    if n <= PAR_THRESH {
        let mut sq_sum = 0.0f32;
        for &x in data {
            let diff = x - mean;
            sq_sum += diff * diff;
        }
        (sq_sum / (n - ddof) as f32).sqrt()
    } else {
        let sq_sum: f32 = data.par_chunks(65536)
            .map(|chunk| chunk.iter().map(|&x| {
                let d = x - mean;
                d * d
            }).sum::<f32>())
            .sum();
        (sq_sum / (n - ddof) as f32).sqrt()
    }
}

/// Fused cross-entropy: log_softmax then pick target index.
pub fn fast_cross_entropy(logits: &[f32], targets: &[i64], n: usize, c: usize) -> f32 {
    assert_eq!(logits.len(), n * c);
    assert_eq!(targets.len(), n);
    let losses: Vec<f32> = (0..n).into_par_iter().map(|i| {
        let row = &logits[i * c..(i + 1) * c];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
        let log_sum_exp = sum_exp.ln() + max_val;
        let tgt = (targets[i].max(0) as usize).min(c - 1);
        log_sum_exp - row[tgt]
    }).collect();
    losses.iter().sum::<f32>() / n as f32
}

// ─── Softmax ─────────────────────────────────────────────────────────────────

pub fn fast_softmax(mut x: ArrayViewMutD<'_, f32>, dim: isize) {
    let shape = x.shape().to_vec();
    let ndim  = shape.len();
    let axis  = if dim < 0 { (ndim as isize + dim) as usize } else { dim as usize };
    if axis == ndim - 1 {
        let chunk_size = shape[axis];
        let data = x.as_slice_mut().expect("contiguous");
        data.par_chunks_mut(chunk_size).for_each(|row| {
            // find max
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            // exp(x - max)
            let mut total = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                total += *v;
            }
            // normalise
            let inv = 1.0 / total;
            for v in row.iter_mut() { *v *= inv; }
        });
    }
}

// ─── Layer norm ──────────────────────────────────────────────────────────────

pub fn fast_layer_norm(
    mut x: ArrayViewMutD<'_, f32>,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) {
    let last_dim = *x.shape().last().unwrap();
    let data = x.as_slice_mut().expect("contiguous");
    data.par_chunks_mut(last_dim).for_each(|row| {
        let n = row.len() as f32;
        let mean: f32 = row.iter().sum::<f32>() / n;
        let var: f32  = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std   = 1.0 / (var + eps).sqrt();
        for (j, v) in row.iter_mut().enumerate() {
            let norm = (*v - mean) * inv_std;
            *v = match (weight, bias) {
                (Some(w), Some(b)) => norm * w[j] + b[j],
                _ => norm,
            };
        }
    });
}

// ─── Adam / AdamW ────────────────────────────────────────────────────────────

pub fn fast_adam_step(
    mut param: ArrayViewMutD<'_, f32>,
    grad: &[f32],
    mut m: ArrayViewMutD<'_, f32>,
    mut v: ArrayViewMutD<'_, f32>,
    beta1: f32, beta2: f32, lr: f32, eps: f32, step: i32,
) {
    let p   = param.as_slice_mut().expect("contiguous");
    let m_d = m.as_slice_mut().expect("contiguous");
    let v_d = v.as_slice_mut().expect("contiguous");
    let bc1 = 1.0 - beta1.powi(step);
    let bc2 = 1.0 - beta2.powi(step);
    let slr = lr * bc2.sqrt() / bc1;
    p.par_chunks_mut(1024)
     .zip(m_d.par_chunks_mut(1024))
     .zip(v_d.par_chunks_mut(1024))
     .enumerate()
     .for_each(|(ci, ((pc, mc), vc))| {
         let off = ci * 1024;
         for j in 0..pc.len() {
             let g = grad[off + j];
             mc[j] = beta1 * mc[j] + (1.0 - beta1) * g;
             vc[j] = beta2 * vc[j] + (1.0 - beta2) * g * g;
             pc[j] -= slr * mc[j] / (vc[j].sqrt() + eps);
         }
     });
}

#[allow(dead_code)]
pub fn fast_adamw_step(
    mut param: ArrayViewMutD<'_, f32>,
    grad: &[f32],
    mut m: ArrayViewMutD<'_, f32>,
    mut v: ArrayViewMutD<'_, f32>,
    beta1: f32, beta2: f32, lr: f32, wd: f32, eps: f32, step: i32,
) {
    let p   = param.as_slice_mut().expect("contiguous");
    let m_d = m.as_slice_mut().expect("contiguous");
    let v_d = v.as_slice_mut().expect("contiguous");
    let bc1    = 1.0 - beta1.powi(step);
    let bc2    = 1.0 - beta2.powi(step);
    let step_lr = lr * bc2.sqrt() / bc1;
    let wd_f   = 1.0 - lr * wd;
    p.par_chunks_mut(1024)
     .zip(m_d.par_chunks_mut(1024))
     .zip(v_d.par_chunks_mut(1024))
     .enumerate()
     .for_each(|(ci, ((pc, mc), vc))| {
         let off = ci * 1024;
         for j in 0..pc.len() {
             let g = grad[off + j];
             mc[j] = beta1 * mc[j] + (1.0 - beta1) * g;
             vc[j] = beta2 * vc[j] + (1.0 - beta2) * g * g;
             pc[j] = pc[j] * wd_f - step_lr * mc[j] / (vc[j].sqrt() + eps);
         }
     });
}
