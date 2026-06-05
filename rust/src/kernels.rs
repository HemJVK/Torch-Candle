use crate::simd::*;
use ndarray::ArrayViewMutD;
use rayon::prelude::*;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn v_silu_ps(v: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let neg_v = _mm256_sub_ps(zero, v);
    let e = v_exp_ps(neg_v);
    let sig = v_recip_ps(_mm256_add_ps(one, e));
    _mm256_mul_ps(v, sig)
}

// ─── Element-wise ops ────────────────────────────────────────────────────────

pub fn fast_relu(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    data.par_chunks_mut(2048).for_each(|chunk| unsafe {
        let mut i = 0;
        let zero = _mm256_setzero_ps();
        while i + 8 <= chunk.len() {
            let v = _mm256_loadu_ps(chunk.as_ptr().add(i));
            _mm256_storeu_ps(chunk.as_mut_ptr().add(i), _mm256_max_ps(v, zero));
            i += 8;
        }
        for j in i..chunk.len() { if chunk[j] < 0.0 { chunk[j] = 0.0; } }
    });
}

pub fn fast_exp(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    // LLVM auto-vectorizes f32::exp() with target-cpu=native — parallel across cores
    data.par_iter_mut().for_each(|v| { *v = v.exp(); });
}

pub fn fast_log(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    data.par_chunks_mut(2048).for_each(|chunk| unsafe {
        let mut i = 0;
        while i + 8 <= chunk.len() {
            let v = _mm256_loadu_ps(chunk.as_ptr().add(i));
            _mm256_storeu_ps(chunk.as_mut_ptr().add(i), v_log_ps(v));
            i += 8;
        }
        for j in i..chunk.len() { chunk[j] = chunk[j].ln(); }
    });
}

pub fn fast_sqrt(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    data.par_chunks_mut(2048).for_each(|chunk| unsafe {
        let mut i = 0;
        while i + 8 <= chunk.len() {
            let v = _mm256_loadu_ps(chunk.as_ptr().add(i));
            _mm256_storeu_ps(chunk.as_mut_ptr().add(i), v_sqrt_ps(v));
            i += 8;
        }
        for j in i..chunk.len() { chunk[j] = chunk[j].sqrt(); }
    });
}

pub fn fast_sigmoid(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    // Use Rayon parallel iterator with scalar exp — Rust's exp() uses libm which
    // is typically auto-vectorized by LLVM, and Rayon gives multi-core speedup.
    data.par_iter_mut().for_each(|v| {
        // Numerically stable sigmoid
        if *v >= 0.0 {
            let e = (-*v).exp();
            *v = 1.0 / (1.0 + e);
        } else {
            let e = v.exp();
            *v = e / (1.0 + e);
        }
    });
}

pub fn fast_tanh(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    data.par_iter_mut().for_each(|v| { *v = v.tanh(); });
}

pub fn fast_silu(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    data.par_iter_mut().for_each(|v| {
        let e = (-*v).exp();
        *v = *v / (1.0 + e);
    });
}

pub fn fast_gelu(mut x: ArrayViewMutD<'_, f32>) {
    let data = x.as_slice_mut().expect("contiguous");
    // tanh-approximation GELU
    data.par_iter_mut().for_each(|v| {
        let xv = *v;
        let inner = 0.7978845608 * (xv + 0.044715 * xv * xv * xv);
        *v = 0.5 * xv * (1.0 + inner.tanh());
    });
}

// ─── Reductions ──────────────────────────────────────────────────────────────

pub fn fast_sum_all(data: &[f32]) -> f32 {
    const PARALLEL_THRESHOLD: usize = 4 * 1024 * 1024; // 16MB — use parallel above this
    if data.len() <= PARALLEL_THRESHOLD {
        // Single-threaded 4-way unrolled AVX2 reduction
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

pub fn fast_max_all(data: &[f32]) -> f32 {
    if data.is_empty() { return f32::NEG_INFINITY; }
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

pub fn fast_min_all(data: &[f32]) -> f32 {
    if data.is_empty() { return f32::INFINITY; }
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

pub fn fast_mean_all(data: &[f32]) -> f32 {
    if data.is_empty() { return 0.0; }
    fast_sum_all(data) / data.len() as f32
}

pub fn fast_norm_l2(data: &[f32]) -> f32 {
    let chunk_sqs: Vec<f32> = data.par_chunks(2048).map(|chunk| unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= chunk.len() {
            let v = _mm256_loadu_ps(chunk.as_ptr().add(i));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
            i += 8;
        }
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        buf.iter().sum::<f32>() + chunk[i..].iter().map(|&x| x * x).sum::<f32>()
    }).collect();
    chunk_sqs.iter().sum::<f32>().sqrt()
}

pub fn fast_std_all(data: &[f32], ddof: usize) -> f32 {
    let n = data.len();
    if n <= ddof { return 0.0; }
    let mean = fast_mean_all(data);
    let sq_sum: f32 = data.par_chunks(2048).map(|chunk| unsafe {
        let mean_v = _mm256_set1_ps(mean);
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= chunk.len() {
            let v = _mm256_loadu_ps(chunk.as_ptr().add(i));
            let d = _mm256_sub_ps(v, mean_v);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(d, d));
            i += 8;
        }
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        buf.iter().sum::<f32>() + chunk[i..].iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>()
    }).sum();
    (sq_sum / (n - ddof) as f32).sqrt()
}

/// Fused cross-entropy: for each row, compute log_softmax then pick target index.
/// logits: (N, C) row-major, targets: (N,) integer indices
pub fn fast_cross_entropy(logits: &[f32], targets: &[i64], n: usize, c: usize) -> f32 {
    assert_eq!(logits.len(), n * c);
    assert_eq!(targets.len(), n);

    let losses: Vec<f32> = (0..n).into_par_iter().map(|i| {
        let row = &logits[i * c..(i + 1) * c];
        // max for numerical stability
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // compute log(sum(exp(x - max)))
        let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
        let log_sum_exp = sum_exp.ln() + max_val;
        // NLL at target
        let tgt = targets[i].max(0) as usize;
        let tgt = tgt.min(c - 1);
        log_sum_exp - row[tgt]
    }).collect();
    losses.iter().sum::<f32>() / n as f32
}

// ─── Softmax / LayerNorm ─────────────────────────────────────────────────────

pub fn fast_softmax(mut x: ArrayViewMutD<'_, f32>, dim: isize) {
    let shape = x.shape().to_vec();
    let ndim = shape.len();
    let axis = if dim < 0 { (ndim as isize + dim) as usize } else { dim as usize };
    if axis == ndim - 1 {
        let chunk_size = shape[axis];
        let data = x.as_slice_mut().expect("contiguous");
        data.par_chunks_mut(chunk_size).for_each(|row| unsafe {
            let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut i = 0;
            while i + 8 <= row.len() {
                max_vec = _mm256_max_ps(max_vec, _mm256_loadu_ps(row.as_ptr().add(i)));
                i += 8;
            }
            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), max_vec);
            let mut max_val = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            for j in i..row.len() { if row[j] > max_val { max_val = row[j]; } }
            let max_vec = _mm256_set1_ps(max_val);
            let mut sum_vec = _mm256_setzero_ps();
            i = 0;
            while i + 8 <= row.len() {
                let e = v_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(row.as_ptr().add(i)), max_vec));
                _mm256_storeu_ps(row.as_mut_ptr().add(i), e);
                sum_vec = _mm256_add_ps(sum_vec, e);
                i += 8;
            }
            let mut total = 0.0f32;
            _mm256_storeu_ps(buf.as_mut_ptr(), sum_vec);
            for v in buf { total += v; }
            for j in i..row.len() { let e = (row[j] - max_val).exp(); row[j] = e; total += e; }
            let inv = _mm256_set1_ps(1.0 / total);
            i = 0;
            while i + 8 <= row.len() {
                let v = _mm256_loadu_ps(row.as_ptr().add(i));
                _mm256_storeu_ps(row.as_mut_ptr().add(i), _mm256_mul_ps(v, inv));
                i += 8;
            }
            let inv_s = 1.0 / total;
            for j in i..row.len() { row[j] *= inv_s; }
        });
    }
}

pub fn fast_layer_norm(mut x: ArrayViewMutD<'_, f32>, weight: Option<&[f32]>, bias: Option<&[f32]>, eps: f32) {
    let last_dim = *x.shape().last().unwrap();
    let data = x.as_slice_mut().expect("contiguous");
    data.par_chunks_mut(last_dim).for_each(|row| unsafe {
        let mut sum_v = _mm256_setzero_ps();
        let mut sq_v = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= row.len() {
            let v = _mm256_loadu_ps(row.as_ptr().add(i));
            sum_v = _mm256_add_ps(sum_v, v);
            sq_v = _mm256_add_ps(sq_v, _mm256_mul_ps(v, v));
            i += 8;
        }
        let mut buf = [0.0f32; 8];
        let mut sum = 0.0f32; let mut sq_sum = 0.0f32;
        _mm256_storeu_ps(buf.as_mut_ptr(), sum_v); for v in buf { sum += v; }
        _mm256_storeu_ps(buf.as_mut_ptr(), sq_v);  for v in buf { sq_sum += v; }
        for j in i..row.len() { sum += row[j]; sq_sum += row[j] * row[j]; }
        let n = last_dim as f32;
        let mean = sum / n;
        let var = (sq_sum / n) - (mean * mean);
        let inv_std = 1.0 / (var + eps).sqrt();
        let mean_v = _mm256_set1_ps(mean);
        let inv_v = _mm256_set1_ps(inv_std);
        i = 0;
        while i + 8 <= row.len() {
            let v = _mm256_loadu_ps(row.as_ptr().add(i));
            let mut r = _mm256_mul_ps(_mm256_sub_ps(v, mean_v), inv_v);
            if let (Some(w), Some(b)) = (weight, bias) {
                r = _mm256_add_ps(_mm256_mul_ps(r, _mm256_loadu_ps(w.as_ptr().add(i))), _mm256_loadu_ps(b.as_ptr().add(i)));
            }
            _mm256_storeu_ps(row.as_mut_ptr().add(i), r);
            i += 8;
        }
        for j in i..row.len() {
            let mut r = (row[j] - mean) * inv_std;
            if let (Some(w), Some(b)) = (weight, bias) { r = r * w[j] + b[j]; }
            row[j] = r;
        }
    });
}

// ─── Adam / AdamW ────────────────────────────────────────────────────────────

pub fn fast_adam_step(mut param: ArrayViewMutD<'_, f32>, grad: &[f32], mut m: ArrayViewMutD<'_, f32>, mut v: ArrayViewMutD<'_, f32>, beta1: f32, beta2: f32, lr: f32, eps: f32, step: i32) {
    let p = param.as_slice_mut().expect("contiguous");
    let m_d = m.as_slice_mut().expect("contiguous");
    let v_d = v.as_slice_mut().expect("contiguous");
    let step_lr = unsafe { _mm256_set1_ps(lr * (1.0 - beta2.powi(step)).sqrt() / (1.0 - beta1.powi(step))) };
    let (b1, b2, ib1, ib2, eps_v) = unsafe { (
        _mm256_set1_ps(beta1), _mm256_set1_ps(beta2),
        _mm256_set1_ps(1.0 - beta1), _mm256_set1_ps(1.0 - beta2),
        _mm256_set1_ps(eps),
    )};
    p.par_chunks_mut(1024).zip(m_d.par_chunks_mut(1024)).zip(v_d.par_chunks_mut(1024)).enumerate()
        .for_each(|(ci, ((pc, mc), vc))| {
            let off = ci * 1024;
            unsafe {
                let mut i = 0;
                while i + 8 <= pc.len() {
                    let gv = _mm256_loadu_ps(grad.as_ptr().add(off + i));
                    let mv = _mm256_loadu_ps(mc.as_ptr().add(i));
                    let vv = _mm256_loadu_ps(vc.as_ptr().add(i));
                    let mn = _mm256_add_ps(_mm256_mul_ps(b1, mv), _mm256_mul_ps(ib1, gv));
                    let vn = _mm256_add_ps(_mm256_mul_ps(b2, vv), _mm256_mul_ps(ib2, _mm256_mul_ps(gv, gv)));
                    _mm256_storeu_ps(mc.as_mut_ptr().add(i), mn);
                    _mm256_storeu_ps(vc.as_mut_ptr().add(i), vn);
                    let pv = _mm256_loadu_ps(pc.as_ptr().add(i));
                    let denom = _mm256_add_ps(_mm256_sqrt_ps(vn), eps_v);
                    let delta = _mm256_mul_ps(step_lr, _mm256_mul_ps(mn, v_recip_ps(denom)));
                    _mm256_storeu_ps(pc.as_mut_ptr().add(i), _mm256_sub_ps(pv, delta));
                    i += 8;
                }
                let bc1 = 1.0 - beta1.powi(step);
                let bc2 = 1.0 - beta2.powi(step);
                let slr = lr * bc2.sqrt() / bc1;
                for j in i..pc.len() {
                    let g = grad[off + j];
                    mc[j] = beta1 * mc[j] + (1.0 - beta1) * g;
                    vc[j] = beta2 * vc[j] + (1.0 - beta2) * g * g;
                    pc[j] -= slr * mc[j] / (vc[j].sqrt() + eps);
                }
            }
        });
}

#[allow(dead_code)]
pub fn fast_adamw_step(mut param: ArrayViewMutD<'_, f32>, grad: &[f32], mut m: ArrayViewMutD<'_, f32>, mut v: ArrayViewMutD<'_, f32>, beta1: f32, beta2: f32, lr: f32, wd: f32, eps: f32, step: i32) {
    let p = param.as_slice_mut().expect("contiguous");
    let m_d = m.as_slice_mut().expect("contiguous");
    let v_d = v.as_slice_mut().expect("contiguous");
    let (b1, b2, ib1, ib2, eps_v, slr, wdf) = unsafe { (
        _mm256_set1_ps(beta1), _mm256_set1_ps(beta2),
        _mm256_set1_ps(1.0 - beta1), _mm256_set1_ps(1.0 - beta2),
        _mm256_set1_ps(eps),
        _mm256_set1_ps(lr * (1.0 - beta2.powi(step)).sqrt() / (1.0 - beta1.powi(step))),
        _mm256_set1_ps(1.0 - lr * wd),
    )};
    p.par_chunks_mut(1024).zip(m_d.par_chunks_mut(1024)).zip(v_d.par_chunks_mut(1024)).enumerate()
        .for_each(|(ci, ((pc, mc), vc))| {
            let off = ci * 1024;
            unsafe {
                let mut i = 0;
                while i + 8 <= pc.len() {
                    let gv = _mm256_loadu_ps(grad.as_ptr().add(off + i));
                    let mv = _mm256_loadu_ps(mc.as_ptr().add(i));
                    let vv = _mm256_loadu_ps(vc.as_ptr().add(i));
                    let mn = _mm256_add_ps(_mm256_mul_ps(b1, mv), _mm256_mul_ps(ib1, gv));
                    let vn = _mm256_add_ps(_mm256_mul_ps(b2, vv), _mm256_mul_ps(ib2, _mm256_mul_ps(gv, gv)));
                    _mm256_storeu_ps(mc.as_mut_ptr().add(i), mn);
                    _mm256_storeu_ps(vc.as_mut_ptr().add(i), vn);
                    let pd = _mm256_mul_ps(_mm256_loadu_ps(pc.as_ptr().add(i)), wdf);
                    let denom = _mm256_add_ps(_mm256_sqrt_ps(vn), eps_v);
                    let delta = _mm256_mul_ps(slr, _mm256_mul_ps(mn, v_recip_ps(denom)));
                    _mm256_storeu_ps(pc.as_mut_ptr().add(i), _mm256_sub_ps(pd, delta));
                    i += 8;
                }
                let bc1 = 1.0 - beta1.powi(step);
                let bc2 = 1.0 - beta2.powi(step);
                let step_lr = lr * bc2.sqrt() / bc1;
                let wd_f = 1.0 - lr * wd;
                for j in i..pc.len() {
                    let g = grad[off + j];
                    mc[j] = beta1 * mc[j] + (1.0 - beta1) * g;
                    vc[j] = beta2 * vc[j] + (1.0 - beta2) * g * g;
                    pc[j] = pc[j] * wd_f - step_lr * mc[j] / (vc[j].sqrt() + eps);
                }
            }
        });
}
