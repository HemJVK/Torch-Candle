use std::arch::x86_64::*;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Parallel threshold: below this many f32 elements, avoid Rayon entirely.
/// 256K floats = 1MB. Rayon dispatch overhead (~10-50μs) dominates for tensors
/// this small, where single-core AVX2 finishes in <100μs.
pub const SINGLE_THREAD_THRESHOLD: usize = 256 * 1024;

/// Optimal chunk size for Rayon parallel iteration.
/// 64K floats = 256KB — fits in L2 cache on Zen 3 (512KB per core).
/// This ensures each core's work fits in L2, avoiding L3 thrashing.
pub const OPTIMAL_CHUNK_SIZE: usize = 64 * 1024;

// ─── Core Primitives ─────────────────────────────────────────────────────────

/// Vectorized reciprocal (1/x) using AVX2 rcp_ps + one Newton-Raphson step for precision.
#[inline(always)]
pub unsafe fn v_recip_ps(x: __m256) -> __m256 {
    let r = _mm256_rcp_ps(x);
    // Newton-Raphson: r_new = r * (2 - x * r) = 2*r - x*r*r
    // Using FMA: r_new = fma(-x, r*r, 2*r)  — saves one mul
    let two = _mm256_set1_ps(2.0);
    let two_r = _mm256_mul_ps(two, r);
    let r_sq = _mm256_mul_ps(r, r);
    _mm256_fnmadd_ps(x, r_sq, two_r) // 2*r - x*r*r
}

/// Vectorized fast exp(x) approximation using AVX2+FMA.
/// Uses range reduction (x = n*ln2 + r) and 6th-order polynomial.
/// Accuracy: ~1 ULP for |x| < 88 (full float range).
#[inline(always)]
pub unsafe fn v_exp_ps(x: __m256) -> __m256 {
    let ln2_inv = _mm256_set1_ps(1.4426950408889634);
    let ln2_hi = _mm256_set1_ps(0.693145751953125);
    let ln2_lo = _mm256_set1_ps(1.4286068203094172e-06);

    // n = round(x * log2(e))
    let n = _mm256_round_ps(
        _mm256_mul_ps(x, ln2_inv),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    // r = x - n * ln2 (high + low parts for precision)
    let r = _mm256_fmadd_ps(_mm256_set1_ps(-1.0), _mm256_mul_ps(n, ln2_hi),
                            _mm256_fmadd_ps(_mm256_set1_ps(-1.0), _mm256_mul_ps(n, ln2_lo), x));

    // Polynomial approximation for exp(r) on [-0.5*ln2, 0.5*ln2]
    // Horner's method with FMA
    let c6 = _mm256_set1_ps(1.9875691500e-04);
    let c5 = _mm256_set1_ps(1.3981999507e-03);
    let c4 = _mm256_set1_ps(8.3334519073e-03);
    let c3 = _mm256_set1_ps(4.1665858030e-02);
    let c2 = _mm256_set1_ps(1.6666625738e-01);
    let c1 = _mm256_set1_ps(4.9999990463e-01);
    let one = _mm256_set1_ps(1.0);

    let mut p = c6;
    p = _mm256_fmadd_ps(p, r, c5);
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, one);
    p = _mm256_fmadd_ps(p, r, one);

    // result = p * 2^n
    let n_int = _mm256_cvtps_epi32(n);
    let twon = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(n_int, _mm256_set1_epi32(127)),
        23,
    ));

    // Clamp to valid float range to handle extreme inputs
    let max_x = _mm256_set1_ps(88.3762626647949);
    let min_x = _mm256_set1_ps(-87.33654475);
    let clamped_result = _mm256_mul_ps(p, twon);
    let result = _mm256_blendv_ps(clamped_result, _mm256_setzero_ps(),
                                   _mm256_cmp_ps(x, min_x, _CMP_LT_OQ));
    _mm256_blendv_ps(result, _mm256_set1_ps(f32::INFINITY),
                     _mm256_cmp_ps(x, max_x, _CMP_GT_OQ))
}

/// Vectorized fast natural log(x) using AVX2+FMA.
/// Uses range reduction: x = 2^e * m where m ∈ [1, 2)
/// log(x) = e*ln(2) + log(m) with minimax polynomial on [1, 2)
#[inline(always)]
pub unsafe fn v_log_ps(x: __m256) -> __m256 {
    let ln2 = _mm256_set1_ps(0.693147180559945f32);
    let one = _mm256_set1_ps(1.0f32);

    // Extract exponent: e = floor(log2(x))
    let xi = _mm256_castps_si256(x);
    let exp_i = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
    let e = _mm256_cvtepi32_ps(exp_i);

    // Mantissa: set exponent to 0 → value in [1, 2)
    let mantissa_bits = _mm256_or_si256(
        _mm256_and_si256(xi, _mm256_set1_epi32(0x007fffff)),
        _mm256_set1_epi32(0x3f800000),
    );
    let m = _mm256_castsi256_ps(mantissa_bits);

    // Polynomial approximation of log(m) on [1, 2): t = m - 1 ∈ [0, 1)
    let t = _mm256_sub_ps(m, one);
    let p5 = _mm256_set1_ps(0.15313837f32);
    let p4 = _mm256_set1_ps(-0.20262926f32);
    let p3 = _mm256_set1_ps(0.24999100f32);
    let p2 = _mm256_set1_ps(-0.33333942f32);
    let p1 = _mm256_set1_ps(0.50000000f32);
    let p0 = _mm256_set1_ps(1.00000000f32);

    // Horner's with FMA
    let mut poly = p5;
    poly = _mm256_fmadd_ps(poly, t, p4);
    poly = _mm256_fmadd_ps(poly, t, p3);
    poly = _mm256_fmadd_ps(poly, t, p2);
    poly = _mm256_fmadd_ps(poly, t, p1);
    poly = _mm256_fmadd_ps(poly, t, p0);
    let log_m = _mm256_mul_ps(poly, t);

    // log(x) = e * ln(2) + log(m)
    _mm256_fmadd_ps(e, ln2, log_m)
}

/// Vectorized sqrt via AVX2 _mm256_sqrt_ps (hardware instruction, exact).
#[inline(always)]
pub unsafe fn v_sqrt_ps(x: __m256) -> __m256 {
    _mm256_sqrt_ps(x)
}

// ─── Activation Functions ────────────────────────────────────────────────────

/// Vectorized sigmoid: σ(x) = 1 / (1 + exp(-x))
/// Uses v_exp_ps for the exponential, then fast reciprocal.
#[inline(always)]
pub unsafe fn v_sigmoid_ps(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    let e = v_exp_ps(neg_x);
    let denom = _mm256_add_ps(one, e);
    // Use division instead of reciprocal for accuracy in sigmoid
    _mm256_div_ps(one, denom)
}

/// Vectorized tanh: tanh(x) = 2*sigmoid(2x) - 1
/// More efficient than (exp(x)-exp(-x))/(exp(x)+exp(-x)) — only one exp call.
#[inline(always)]
pub unsafe fn v_tanh_ps(x: __m256) -> __m256 {
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    let two_x = _mm256_mul_ps(two, x);
    let sig = v_sigmoid_ps(two_x);
    _mm256_fmsub_ps(two, sig, one) // 2*sig - 1
}

/// Vectorized SiLU (Swish): silu(x) = x * sigmoid(x)
#[inline(always)]
pub unsafe fn v_silu_ps(x: __m256) -> __m256 {
    let sig = v_sigmoid_ps(x);
    _mm256_mul_ps(x, sig)
}

/// Vectorized GELU (tanh approximation):
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// All FMA — 8 elements per cycle.
#[inline(always)]
pub unsafe fn v_gelu_ps(x: __m256) -> __m256 {
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let sqrt_2_over_pi = _mm256_set1_ps(0.7978845608028654); // sqrt(2/π)
    let c = _mm256_set1_ps(0.044715);

    // inner = sqrt(2/π) * (x + 0.044715 * x³)
    let x_sq = _mm256_mul_ps(x, x);
    let x_cube = _mm256_mul_ps(x_sq, x);
    let inner_sum = _mm256_fmadd_ps(c, x_cube, x); // x + 0.044715 * x³
    let inner = _mm256_mul_ps(sqrt_2_over_pi, inner_sum);

    let tanh_val = v_tanh_ps(inner);
    let one_plus_tanh = _mm256_add_ps(one, tanh_val);
    _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus_tanh))
}

// ─── Bulk Processing Functions ───────────────────────────────────────────────
// These apply SIMD kernels over contiguous f32 slices with scalar tail handling.

/// Apply AVX2 sigmoid over a contiguous f32 slice.
#[inline]
pub unsafe fn simd_sigmoid_slice(data: &mut [f32]) {
    let mut i = 0;
    while i + 32 <= data.len() {
        let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(data.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(data.as_ptr().add(i + 24));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), v_sigmoid_ps(v0));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 8), v_sigmoid_ps(v1));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 16), v_sigmoid_ps(v2));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 24), v_sigmoid_ps(v3));
        i += 32;
    }
    while i + 8 <= data.len() {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), v_sigmoid_ps(v));
        i += 8;
    }
    // Scalar tail
    for j in i..data.len() {
        let x = data[j];
        data[j] = 1.0 / (1.0 + (-x).exp());
    }
}

/// Apply AVX2 tanh over a contiguous f32 slice.
#[inline]
pub unsafe fn simd_tanh_slice(data: &mut [f32]) {
    let mut i = 0;
    while i + 32 <= data.len() {
        let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(data.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(data.as_ptr().add(i + 24));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), v_tanh_ps(v0));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 8), v_tanh_ps(v1));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 16), v_tanh_ps(v2));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 24), v_tanh_ps(v3));
        i += 32;
    }
    while i + 8 <= data.len() {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), v_tanh_ps(v));
        i += 8;
    }
    for j in i..data.len() {
        data[j] = data[j].tanh();
    }
}

/// Apply AVX2 exp over a contiguous f32 slice.
#[inline]
pub unsafe fn simd_exp_slice(data: &mut [f32]) {
    let mut i = 0;
    while i + 32 <= data.len() {
        let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(data.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(data.as_ptr().add(i + 24));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), v_exp_ps(v0));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 8), v_exp_ps(v1));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 16), v_exp_ps(v2));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 24), v_exp_ps(v3));
        i += 32;
    }
    while i + 8 <= data.len() {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), v_exp_ps(v));
        i += 8;
    }
    for j in i..data.len() {
        data[j] = data[j].exp();
    }
}

/// Apply AVX2 log over a contiguous f32 slice.
#[inline]
pub unsafe fn simd_log_slice(data: &mut [f32]) {
    let mut i = 0;
    while i + 32 <= data.len() {
        let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(data.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(data.as_ptr().add(i + 24));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), v_log_ps(v0));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 8), v_log_ps(v1));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 16), v_log_ps(v2));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 24), v_log_ps(v3));
        i += 32;
    }
    while i + 8 <= data.len() {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), v_log_ps(v));
        i += 8;
    }
    for j in i..data.len() {
        data[j] = data[j].ln();
    }
}

/// Apply AVX2 sqrt over a contiguous f32 slice.
#[inline]
pub unsafe fn simd_sqrt_slice(data: &mut [f32]) {
    let mut i = 0;
    while i + 32 <= data.len() {
        let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(data.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(data.as_ptr().add(i + 24));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), _mm256_sqrt_ps(v0));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 8), _mm256_sqrt_ps(v1));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 16), _mm256_sqrt_ps(v2));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 24), _mm256_sqrt_ps(v3));
        i += 32;
    }
    while i + 8 <= data.len() {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), _mm256_sqrt_ps(v));
        i += 8;
    }
    for j in i..data.len() {
        data[j] = data[j].sqrt();
    }
}

/// Apply AVX2 SiLU over a contiguous f32 slice.
#[inline]
pub unsafe fn simd_silu_slice(data: &mut [f32]) {
    let mut i = 0;
    while i + 32 <= data.len() {
        let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(data.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(data.as_ptr().add(i + 24));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), v_silu_ps(v0));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 8), v_silu_ps(v1));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 16), v_silu_ps(v2));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 24), v_silu_ps(v3));
        i += 32;
    }
    while i + 8 <= data.len() {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), v_silu_ps(v));
        i += 8;
    }
    for j in i..data.len() {
        let x = data[j];
        data[j] = x / (1.0 + (-x).exp());
    }
}

/// Apply AVX2 GELU over a contiguous f32 slice.
#[inline]
pub unsafe fn simd_gelu_slice(data: &mut [f32]) {
    let mut i = 0;
    while i + 32 <= data.len() {
        let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(data.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(data.as_ptr().add(i + 24));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), v_gelu_ps(v0));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 8), v_gelu_ps(v1));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 16), v_gelu_ps(v2));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 24), v_gelu_ps(v3));
        i += 32;
    }
    while i + 8 <= data.len() {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), v_gelu_ps(v));
        i += 8;
    }
    for j in i..data.len() {
        let xv = data[j];
        let inner = 0.7978845608 * (xv + 0.044715 * xv * xv * xv);
        data[j] = 0.5 * xv * (1.0 + inner.tanh());
    }
}

/// Apply AVX2 ReLU over a contiguous f32 slice.
/// 4x unrolled for maximum throughput.
#[inline]
pub unsafe fn simd_relu_slice(data: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 32 <= data.len() {
        let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(data.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(data.as_ptr().add(i + 24));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), _mm256_max_ps(v0, zero));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 8), _mm256_max_ps(v1, zero));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 16), _mm256_max_ps(v2, zero));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 24), _mm256_max_ps(v3, zero));
        i += 32;
    }
    while i + 8 <= data.len() {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), _mm256_max_ps(v, zero));
        i += 8;
    }
    for j in i..data.len() {
        if data[j] < 0.0 { data[j] = 0.0; }
    }
}
