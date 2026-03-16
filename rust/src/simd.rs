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
