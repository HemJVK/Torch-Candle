use std::arch::x86_64::*;

/// Vectorized reciprocal (1/x) using AVX2 rcp_ps + one Newton-Raphson step for precision.
#[inline(always)]
pub unsafe fn v_recip_ps(x: __m256) -> __m256 {
    let r = _mm256_rcp_ps(x);
    // Newton-Raphson: r_new = r * (2 - x * r)
    let two = _mm256_set1_ps(2.0);
    _mm256_mul_ps(r, _mm256_sub_ps(two, _mm256_mul_ps(x, r)))
}

/// Vectorized fast exp(x) approximation using AVX2.
/// Uses a 6th-order polynomial approximation: exp(x) ≈ (1 + x/n)^n approach or minimax polynomial.
/// This implementation uses a fast minimax polynomial for exp(x) on [0, ln2] and range reduction.
#[inline(always)]
pub unsafe fn v_exp_ps(x: __m256) -> __m256 {
    let ln2_inv = _mm256_set1_ps(1.4426950408889634);
    let ln2_hi = _mm256_set1_ps(-0.693145751953125);
    let ln2_lo = _mm256_set1_ps(-1.4286068203094172e-06);

    // n = floor(x * log2(e) + 0.5)
    let n = _mm256_round_ps(_mm256_mul_ps(x, ln2_inv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    
    // r = x - n * ln2
    let mut r = _mm256_add_ps(x, _mm256_mul_ps(n, ln2_hi));
    r = _mm256_add_ps(r, _mm256_mul_ps(n, ln2_lo));

    // Polynomial approximation for exp(r) on [-0.5, 0.5]
    // p = (((((c6*r + c5)*r + c4)*r + c3)*r + c2)*r + c1)*r + c0
    let c6 = _mm256_set1_ps(1.3888949082884814e-03);
    let c5 = _mm256_set1_ps(8.3333649983008703e-03);
    let c4 = _mm256_set1_ps(4.1666463212871032e-02);
    let c3 = _mm256_set1_ps(1.6666673620583713e-01);
    let c2 = _mm256_set1_ps(5.0000000000000000e-01);
    let c1 = _mm256_set1_ps(1.0000000000000000e+00);
    let c0 = _mm256_set1_ps(1.0000000000000000e+00);

    let mut p = _mm256_mul_ps(c6, r);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c5);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c4);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c3);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c2);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c1);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c0);

    // result = p * 2^n
    // Use scale_ps or manual bit manipulation
    let n_int = _mm256_cvtps_epi32(n);
    let twon = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n_int, _mm256_set1_epi32(127)), 23));
    
    _mm256_mul_ps(p, twon)
}

/// Vectorized fast natural log(x) using AVX2.
/// Uses range reduction: x = 2^e * m where m ∈ [1, 2)
/// log(x) = e*log(2) + log(m)
/// log(m) approximated by minimax polynomial on [1, 2)
#[inline(always)]
pub unsafe fn v_log_ps(x: __m256) -> __m256 {
    let log2e_inv = _mm256_set1_ps(0.693147180559945f32); // ln(2)
    let one = _mm256_set1_ps(1.0f32);

    // Extract exponent bits: e = floor(log2(x))
    let xi = _mm256_castps_si256(x);
    // Exponent: (bits >> 23) - 127
    let exp_i = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
    let e = _mm256_cvtepi32_ps(exp_i);

    // Mantissa: set exponent to 0 → value in [1, 2)
    let mantissa_bits = _mm256_or_si256(
        _mm256_and_si256(xi, _mm256_set1_epi32(0x007fffff)),
        _mm256_set1_epi32(0x3f800000),
    );
    let m = _mm256_castsi256_ps(mantissa_bits);

    // Polynomial approximation of log(m) on [1, 2): m' = m - 1 ∈ [0, 1)
    // Use: log(1+t) ≈ t*(p0 + t*(p1 + t*(p2 + t*(p3 + t*p4))))
    let t = _mm256_sub_ps(m, one);
    let p4 = _mm256_set1_ps(-0.20262926f32);
    let p3 = _mm256_set1_ps( 0.24999100f32);
    let p2 = _mm256_set1_ps(-0.33333942f32);
    let p1 = _mm256_set1_ps( 0.50000000f32);
    let p0 = _mm256_set1_ps( 1.00000000f32);

    let mut poly = _mm256_mul_ps(p4, t);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, t), p3);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, t), p2);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, t), p1);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, t), p0);
    let log_m = _mm256_mul_ps(poly, t);

    // log(x) = e * ln(2) + log(m)
    _mm256_add_ps(_mm256_mul_ps(e, log2e_inv), log_m)
}

/// Vectorized sqrt via AVX2 _mm256_sqrt_ps (hardware instruction, exact).
#[inline(always)]
pub unsafe fn v_sqrt_ps(x: __m256) -> __m256 {
    _mm256_sqrt_ps(x)
}
