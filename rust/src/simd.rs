// ─── Architecture-gated SIMD ─────────────────────────────────────────────────
//
// x86_64  → AVX2 + FMA
// aarch64 → ARM NEON (Apple M-series, Graviton, etc.)
// other   → Scalar (Rayon + LLVM auto-vectorisation)

pub const SINGLE_THREAD_THRESHOLD: usize = 256 * 1024;
pub const OPTIMAL_CHUNK_SIZE: usize = 64 * 1024;

// ═════════════════════════════════════════════════════════════════════════════
// x86_64 — AVX2 + FMA
// ═════════════════════════════════════════════════════════════════════════════
#[cfg(target_arch = "x86_64")]
pub use x86_impl::*;

#[cfg(target_arch = "x86_64")]
mod x86_impl {
    use std::arch::x86_64::*;

    #[inline(always)]
    pub unsafe fn v_recip_ps(x: __m256) -> __m256 {
        let r = _mm256_rcp_ps(x);
        let two = _mm256_set1_ps(2.0);
        let two_r = _mm256_mul_ps(two, r);
        let r_sq = _mm256_mul_ps(r, r);
        _mm256_fnmadd_ps(x, r_sq, two_r)
    }

    #[inline(always)]
    pub unsafe fn v_exp_ps(x: __m256) -> __m256 {
        let ln2_inv = _mm256_set1_ps(1.4426950408889634);
        let ln2_hi  = _mm256_set1_ps(0.693145751953125);
        let ln2_lo  = _mm256_set1_ps(1.4286068203094172e-06);
        let n = _mm256_round_ps(_mm256_mul_ps(x, ln2_inv),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let r = _mm256_fmadd_ps(_mm256_set1_ps(-1.0), _mm256_mul_ps(n, ln2_hi),
                _mm256_fmadd_ps(_mm256_set1_ps(-1.0), _mm256_mul_ps(n, ln2_lo), x));
        let c6  = _mm256_set1_ps(1.9875691500e-04);
        let c5  = _mm256_set1_ps(1.3981999507e-03);
        let c4  = _mm256_set1_ps(8.3334519073e-03);
        let c3  = _mm256_set1_ps(4.1665858030e-02);
        let c2  = _mm256_set1_ps(1.6666625738e-01);
        let c1  = _mm256_set1_ps(4.9999990463e-01);
        let one = _mm256_set1_ps(1.0);
        let mut p = c6;
        p = _mm256_fmadd_ps(p, r, c5);
        p = _mm256_fmadd_ps(p, r, c4);
        p = _mm256_fmadd_ps(p, r, c3);
        p = _mm256_fmadd_ps(p, r, c2);
        p = _mm256_fmadd_ps(p, r, c1);
        p = _mm256_fmadd_ps(p, r, one);
        p = _mm256_fmadd_ps(p, r, one);
        let n_int = _mm256_cvtps_epi32(n);
        let twon  = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_add_epi32(n_int, _mm256_set1_epi32(127)), 23));
        let max_x = _mm256_set1_ps(88.3762626647949);
        let min_x = _mm256_set1_ps(-87.33654475);
        let res   = _mm256_mul_ps(p, twon);
        let res   = _mm256_blendv_ps(res, _mm256_setzero_ps(),
                                     _mm256_cmp_ps(x, min_x, _CMP_LT_OQ));
        _mm256_blendv_ps(res, _mm256_set1_ps(f32::INFINITY),
                         _mm256_cmp_ps(x, max_x, _CMP_GT_OQ))
    }

    #[inline(always)]
    pub unsafe fn v_log_ps(x: __m256) -> __m256 {
        let ln2 = _mm256_set1_ps(0.693147180559945f32);
        let one = _mm256_set1_ps(1.0f32);
        let xi      = _mm256_castps_si256(x);
        let exp_i   = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
        let e       = _mm256_cvtepi32_ps(exp_i);
        let mantissa_bits = _mm256_or_si256(
            _mm256_and_si256(xi, _mm256_set1_epi32(0x007fffff)),
            _mm256_set1_epi32(0x3f800000));
        let m  = _mm256_castsi256_ps(mantissa_bits);
        let t  = _mm256_sub_ps(m, one);
        let p5 = _mm256_set1_ps(0.15313837f32);
        let p4 = _mm256_set1_ps(-0.20262926f32);
        let p3 = _mm256_set1_ps(0.24999100f32);
        let p2 = _mm256_set1_ps(-0.33333942f32);
        let p1 = _mm256_set1_ps(0.50000000f32);
        let p0 = _mm256_set1_ps(1.00000000f32);
        let mut poly = p5;
        poly = _mm256_fmadd_ps(poly, t, p4);
        poly = _mm256_fmadd_ps(poly, t, p3);
        poly = _mm256_fmadd_ps(poly, t, p2);
        poly = _mm256_fmadd_ps(poly, t, p1);
        poly = _mm256_fmadd_ps(poly, t, p0);
        let log_m = _mm256_mul_ps(poly, t);
        _mm256_fmadd_ps(e, ln2, log_m)
    }

    #[inline(always)]
    pub unsafe fn v_sqrt_ps(x: __m256) -> __m256 { _mm256_sqrt_ps(x) }

    #[inline(always)]
    pub unsafe fn v_sigmoid_ps(x: __m256) -> __m256 {
        let one   = _mm256_set1_ps(1.0);
        let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        let e     = v_exp_ps(neg_x);
        _mm256_div_ps(one, _mm256_add_ps(one, e))
    }

    #[inline(always)]
    pub unsafe fn v_tanh_ps(x: __m256) -> __m256 {
        let two = _mm256_set1_ps(2.0);
        let one = _mm256_set1_ps(1.0);
        let sig = v_sigmoid_ps(_mm256_mul_ps(two, x));
        _mm256_fmsub_ps(two, sig, one)
    }

    #[inline(always)]
    pub unsafe fn v_silu_ps(x: __m256) -> __m256 {
        _mm256_mul_ps(x, v_sigmoid_ps(x))
    }

    #[inline(always)]
    pub unsafe fn v_gelu_ps(x: __m256) -> __m256 {
        let half          = _mm256_set1_ps(0.5);
        let one           = _mm256_set1_ps(1.0);
        let sqrt_2_over_pi = _mm256_set1_ps(0.7978845608028654);
        let c             = _mm256_set1_ps(0.044715);
        let x_sq          = _mm256_mul_ps(x, x);
        let x_cube        = _mm256_mul_ps(x_sq, x);
        let inner_sum     = _mm256_fmadd_ps(c, x_cube, x);
        let inner         = _mm256_mul_ps(sqrt_2_over_pi, inner_sum);
        let tanh_val      = v_tanh_ps(inner);
        _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, tanh_val)))
    }

    // ── Bulk slice helpers ───────────────────────────────────────────────────

    macro_rules! slice_op8 {
        ($name:ident, $vop:ident, $scalar:expr) => {
            #[inline]
            pub unsafe fn $name(data: &mut [f32]) {
                let mut i = 0;
                while i + 32 <= data.len() {
                    let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
                    let v1 = _mm256_loadu_ps(data.as_ptr().add(i +  8));
                    let v2 = _mm256_loadu_ps(data.as_ptr().add(i + 16));
                    let v3 = _mm256_loadu_ps(data.as_ptr().add(i + 24));
                    _mm256_storeu_ps(data.as_mut_ptr().add(i),      $vop(v0));
                    _mm256_storeu_ps(data.as_mut_ptr().add(i +  8), $vop(v1));
                    _mm256_storeu_ps(data.as_mut_ptr().add(i + 16), $vop(v2));
                    _mm256_storeu_ps(data.as_mut_ptr().add(i + 24), $vop(v3));
                    i += 32;
                }
                while i + 8 <= data.len() {
                    let v = _mm256_loadu_ps(data.as_ptr().add(i));
                    _mm256_storeu_ps(data.as_mut_ptr().add(i), $vop(v));
                    i += 8;
                }
                #[allow(clippy::redundant_closure_call)]
                for j in i..data.len() { data[j] = $scalar(data[j]); }
            }
        };
    }

    slice_op8!(simd_sigmoid_slice, v_sigmoid_ps, |x: f32| 1.0 / (1.0 + (-x).exp()));
    slice_op8!(simd_tanh_slice,    v_tanh_ps,    |x: f32| x.tanh());
    slice_op8!(simd_exp_slice,     v_exp_ps,     |x: f32| x.exp());
    slice_op8!(simd_log_slice,     v_log_ps,     |x: f32| x.ln());
    slice_op8!(simd_sqrt_slice,    v_sqrt_ps,    |x: f32| x.sqrt());
    slice_op8!(simd_silu_slice,    v_silu_ps,    |x: f32| x / (1.0 + (-x).exp()));
    slice_op8!(simd_gelu_slice,    v_gelu_ps,    |x: f32| {
        let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    });

    #[inline]
    pub unsafe fn simd_relu_slice(data: &mut [f32]) {
        let zero = _mm256_setzero_ps();
        let mut i = 0;
        while i + 32 <= data.len() {
            let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(data.as_ptr().add(i +  8));
            let v2 = _mm256_loadu_ps(data.as_ptr().add(i + 16));
            let v3 = _mm256_loadu_ps(data.as_ptr().add(i + 24));
            _mm256_storeu_ps(data.as_mut_ptr().add(i),      _mm256_max_ps(v0, zero));
            _mm256_storeu_ps(data.as_mut_ptr().add(i +  8), _mm256_max_ps(v1, zero));
            _mm256_storeu_ps(data.as_mut_ptr().add(i + 16), _mm256_max_ps(v2, zero));
            _mm256_storeu_ps(data.as_mut_ptr().add(i + 24), _mm256_max_ps(v3, zero));
            i += 32;
        }
        while i + 8 <= data.len() {
            let v = _mm256_loadu_ps(data.as_ptr().add(i));
            _mm256_storeu_ps(data.as_mut_ptr().add(i), _mm256_max_ps(v, zero));
            i += 8;
        }
        for j in i..data.len() { if data[j] < 0.0 { data[j] = 0.0; } }
    }

    #[inline]
    pub unsafe fn simd_add_slice(lhs: &mut [f32], rhs: &[f32]) {
        use rayon::prelude::*;
        lhs.par_chunks_mut(2048)
           .zip(rhs.par_chunks(2048))
           .for_each(|(l_chunk, r_chunk)| unsafe {
               let mut i = 0;
               while i + 32 <= l_chunk.len() {
                   let vl0 = _mm256_loadu_ps(l_chunk.as_ptr().add(i));
                   let vr0 = _mm256_loadu_ps(r_chunk.as_ptr().add(i));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i), _mm256_add_ps(vl0, vr0));

                   let vl1 = _mm256_loadu_ps(l_chunk.as_ptr().add(i + 8));
                   let vr1 = _mm256_loadu_ps(r_chunk.as_ptr().add(i + 8));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i + 8), _mm256_add_ps(vl1, vr1));

                   let vl2 = _mm256_loadu_ps(l_chunk.as_ptr().add(i + 16));
                   let vr2 = _mm256_loadu_ps(r_chunk.as_ptr().add(i + 16));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i + 16), _mm256_add_ps(vl2, vr2));

                   let vl3 = _mm256_loadu_ps(l_chunk.as_ptr().add(i + 24));
                   let vr3 = _mm256_loadu_ps(r_chunk.as_ptr().add(i + 24));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i + 24), _mm256_add_ps(vl3, vr3));

                   i += 32;
               }
               while i + 8 <= l_chunk.len() {
                   let vl = _mm256_loadu_ps(l_chunk.as_ptr().add(i));
                   let vr = _mm256_loadu_ps(r_chunk.as_ptr().add(i));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i), _mm256_add_ps(vl, vr));
                   i += 8;
               }
               for j in i..l_chunk.len() {
                   l_chunk[j] += r_chunk[j];
               }
           });
    }

    #[inline]
    pub unsafe fn simd_mul_slice(lhs: &mut [f32], rhs: &[f32]) {
        use rayon::prelude::*;
        lhs.par_chunks_mut(2048)
           .zip(rhs.par_chunks(2048))
           .for_each(|(l_chunk, r_chunk)| unsafe {
               let mut i = 0;
               while i + 32 <= l_chunk.len() {
                   let vl0 = _mm256_loadu_ps(l_chunk.as_ptr().add(i));
                   let vr0 = _mm256_loadu_ps(r_chunk.as_ptr().add(i));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i), _mm256_mul_ps(vl0, vr0));

                   let vl1 = _mm256_loadu_ps(l_chunk.as_ptr().add(i + 8));
                   let vr1 = _mm256_loadu_ps(r_chunk.as_ptr().add(i + 8));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i + 8), _mm256_mul_ps(vl1, vr1));

                   let vl2 = _mm256_loadu_ps(l_chunk.as_ptr().add(i + 16));
                   let vr2 = _mm256_loadu_ps(r_chunk.as_ptr().add(i + 16));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i + 16), _mm256_mul_ps(vl2, vr2));

                   let vl3 = _mm256_loadu_ps(l_chunk.as_ptr().add(i + 24));
                   let vr3 = _mm256_loadu_ps(r_chunk.as_ptr().add(i + 24));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i + 24), _mm256_mul_ps(vl3, vr3));

                   i += 32;
               }
               while i + 8 <= l_chunk.len() {
                   let vl = _mm256_loadu_ps(l_chunk.as_ptr().add(i));
                   let vr = _mm256_loadu_ps(r_chunk.as_ptr().add(i));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i), _mm256_mul_ps(vl, vr));
                   i += 8;
               }
               for j in i..l_chunk.len() {
                   l_chunk[j] *= r_chunk[j];
               }
           });
    }

    #[inline]
    pub unsafe fn simd_sub_slice(lhs: &mut [f32], rhs: &[f32]) {
        use rayon::prelude::*;
        lhs.par_chunks_mut(2048)
           .zip(rhs.par_chunks(2048))
           .for_each(|(l_chunk, r_chunk)| unsafe {
               let mut i = 0;
               while i + 32 <= l_chunk.len() {
                   let vl0 = _mm256_loadu_ps(l_chunk.as_ptr().add(i));
                   let vr0 = _mm256_loadu_ps(r_chunk.as_ptr().add(i));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i), _mm256_sub_ps(vl0, vr0));

                   let vl1 = _mm256_loadu_ps(l_chunk.as_ptr().add(i + 8));
                   let vr1 = _mm256_loadu_ps(r_chunk.as_ptr().add(i + 8));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i + 8), _mm256_sub_ps(vl1, vr1));

                   let vl2 = _mm256_loadu_ps(l_chunk.as_ptr().add(i + 16));
                   let vr2 = _mm256_loadu_ps(r_chunk.as_ptr().add(i + 16));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i + 16), _mm256_sub_ps(vl2, vr2));

                   let vl3 = _mm256_loadu_ps(l_chunk.as_ptr().add(i + 24));
                   let vr3 = _mm256_loadu_ps(r_chunk.as_ptr().add(i + 24));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i + 24), _mm256_sub_ps(vl3, vr3));

                   i += 32;
               }
               while i + 8 <= l_chunk.len() {
                   let vl = _mm256_loadu_ps(l_chunk.as_ptr().add(i));
                   let vr = _mm256_loadu_ps(r_chunk.as_ptr().add(i));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i), _mm256_sub_ps(vl, vr));
                   i += 8;
               }
               for j in i..l_chunk.len() {
                   l_chunk[j] -= r_chunk[j];
               }
           });
    }

    #[inline]
    pub unsafe fn simd_div_slice(lhs: &mut [f32], rhs: &[f32]) {
        use rayon::prelude::*;
        lhs.par_chunks_mut(2048)
           .zip(rhs.par_chunks(2048))
           .for_each(|(l_chunk, r_chunk)| unsafe {
               let mut i = 0;
               while i + 32 <= l_chunk.len() {
                   let vl0 = _mm256_loadu_ps(l_chunk.as_ptr().add(i));
                   let vr0 = _mm256_loadu_ps(r_chunk.as_ptr().add(i));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i), _mm256_div_ps(vl0, vr0));

                   let vl1 = _mm256_loadu_ps(l_chunk.as_ptr().add(i + 8));
                   let vr1 = _mm256_loadu_ps(r_chunk.as_ptr().add(i + 8));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i + 8), _mm256_div_ps(vl1, vr1));

                   let vl2 = _mm256_loadu_ps(l_chunk.as_ptr().add(i + 16));
                   let vr2 = _mm256_loadu_ps(r_chunk.as_ptr().add(i + 16));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i + 16), _mm256_div_ps(vl2, vr2));

                   let vl3 = _mm256_loadu_ps(l_chunk.as_ptr().add(i + 24));
                   let vr3 = _mm256_loadu_ps(r_chunk.as_ptr().add(i + 24));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i + 24), _mm256_div_ps(vl3, vr3));

                   i += 32;
               }
               while i + 8 <= l_chunk.len() {
                   let vl = _mm256_loadu_ps(l_chunk.as_ptr().add(i));
                   let vr = _mm256_loadu_ps(r_chunk.as_ptr().add(i));
                   _mm256_storeu_ps(l_chunk.as_mut_ptr().add(i), _mm256_div_ps(vl, vr));
                   i += 8;
               }
               for j in i..l_chunk.len() {
                   l_chunk[j] /= r_chunk[j];
               }
           });
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// aarch64 — ARM NEON (Apple M1/M2/M3, AWS Graviton, Android)
// ═════════════════════════════════════════════════════════════════════════════
#[cfg(target_arch = "aarch64")]
pub use neon_impl::*;

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use std::arch::aarch64::*;

    // ── Scalar helpers used in NEON implementations ──────────────────────────

    #[inline(always)]
    fn scalar_exp(x: f32) -> f32 { x.exp() }
    #[inline(always)]
    fn scalar_ln(x: f32) -> f32 { x.ln() }

    // ── Core vectorised primitives (4-wide float32x4_t) ─────────────────────

    /// Apply a scalar function lane-by-lane to a float32x4_t.
    macro_rules! map4 {
        ($v:expr, $f:expr) => {{
            let mut buf = [0f32; 4];
            vst1q_f32(buf.as_mut_ptr(), $v);
            buf[0] = $f(buf[0]);
            buf[1] = $f(buf[1]);
            buf[2] = $f(buf[2]);
            buf[3] = $f(buf[3]);
            vld1q_f32(buf.as_ptr())
        }};
    }

    #[inline(always)]
    pub unsafe fn v_exp_ps(x: float32x4_t) -> float32x4_t { map4!(x, scalar_exp) }

    #[inline(always)]
    pub unsafe fn v_log_ps(x: float32x4_t) -> float32x4_t { map4!(x, scalar_ln) }

    #[inline(always)]
    pub unsafe fn v_sqrt_ps(x: float32x4_t) -> float32x4_t { vsqrtq_f32(x) }

    #[inline(always)]
    pub unsafe fn v_recip_ps(x: float32x4_t) -> float32x4_t {
        // Newton-Raphson refinement: r1 = r0 * (2 - x*r0)
        let r0 = vrecpeq_f32(x);
        vmulq_f32(r0, vrecpsq_f32(x, r0))
    }

    #[inline(always)]
    pub unsafe fn v_sigmoid_ps(x: float32x4_t) -> float32x4_t {
        let one   = vdupq_n_f32(1.0f32);
        let neg_x = vnegq_f32(x);
        let e     = v_exp_ps(neg_x);
        let denom = vaddq_f32(one, e);
        v_recip_ps(denom)
    }

    #[inline(always)]
    pub unsafe fn v_tanh_ps(x: float32x4_t) -> float32x4_t {
        let two = vdupq_n_f32(2.0f32);
        let one = vdupq_n_f32(1.0f32);
        let sig = v_sigmoid_ps(vmulq_f32(two, x));
        // 2*sig - 1
        vsubq_f32(vmulq_f32(two, sig), one)
    }

    #[inline(always)]
    pub unsafe fn v_silu_ps(x: float32x4_t) -> float32x4_t {
        vmulq_f32(x, v_sigmoid_ps(x))
    }

    #[inline(always)]
    pub unsafe fn v_gelu_ps(x: float32x4_t) -> float32x4_t {
        let half           = vdupq_n_f32(0.5f32);
        let one            = vdupq_n_f32(1.0f32);
        let sqrt_2_over_pi = vdupq_n_f32(0.7978845608028654f32);
        let c              = vdupq_n_f32(0.044715f32);
        let x_sq           = vmulq_f32(x, x);
        let x_cube         = vmulq_f32(x_sq, x);
        let inner_sum      = vaddq_f32(x, vmulq_f32(c, x_cube));
        let inner          = vmulq_f32(sqrt_2_over_pi, inner_sum);
        let tanh_val       = v_tanh_ps(inner);
        vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, tanh_val)))
    }

    // ── Bulk slice helpers (4-wide, 4× unrolled → 16 per iteration) ─────────

    macro_rules! slice_op4 {
        ($name:ident, $vop:ident, $scalar:expr) => {
            #[inline]
            pub unsafe fn $name(data: &mut [f32]) {
                let mut i = 0;
                while i + 16 <= data.len() {
                    let v0 = vld1q_f32(data.as_ptr().add(i));
                    let v1 = vld1q_f32(data.as_ptr().add(i +  4));
                    let v2 = vld1q_f32(data.as_ptr().add(i +  8));
                    let v3 = vld1q_f32(data.as_ptr().add(i + 12));
                    vst1q_f32(data.as_mut_ptr().add(i),      $vop(v0));
                    vst1q_f32(data.as_mut_ptr().add(i +  4), $vop(v1));
                    vst1q_f32(data.as_mut_ptr().add(i +  8), $vop(v2));
                    vst1q_f32(data.as_mut_ptr().add(i + 12), $vop(v3));
                    i += 16;
                }
                while i + 4 <= data.len() {
                    let v = vld1q_f32(data.as_ptr().add(i));
                    vst1q_f32(data.as_mut_ptr().add(i), $vop(v));
                    i += 4;
                }
                #[allow(clippy::redundant_closure_call)]
                for j in i..data.len() { data[j] = $scalar(data[j]); }
            }
        };
    }

    slice_op4!(simd_sigmoid_slice, v_sigmoid_ps, |x: f32| 1.0 / (1.0 + (-x).exp()));
    slice_op4!(simd_tanh_slice,    v_tanh_ps,    |x: f32| x.tanh());
    slice_op4!(simd_exp_slice,     v_exp_ps,     |x: f32| x.exp());
    slice_op4!(simd_log_slice,     v_log_ps,     |x: f32| x.ln());
    slice_op4!(simd_sqrt_slice,    v_sqrt_ps,    |x: f32| x.sqrt());
    slice_op4!(simd_silu_slice,    v_silu_ps,    |x: f32| x / (1.0 + (-x).exp()));
    slice_op4!(simd_gelu_slice,    v_gelu_ps,    |x: f32| {
        let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    });

    #[inline]
    pub unsafe fn simd_relu_slice(data: &mut [f32]) {
        let zero = vdupq_n_f32(0.0f32);
        let mut i = 0;
        while i + 16 <= data.len() {
            vst1q_f32(data.as_mut_ptr().add(i),      vmaxq_f32(vld1q_f32(data.as_ptr().add(i)),      zero));
            vst1q_f32(data.as_mut_ptr().add(i +  4), vmaxq_f32(vld1q_f32(data.as_ptr().add(i +  4)), zero));
            vst1q_f32(data.as_mut_ptr().add(i +  8), vmaxq_f32(vld1q_f32(data.as_ptr().add(i +  8)), zero));
            vst1q_f32(data.as_mut_ptr().add(i + 12), vmaxq_f32(vld1q_f32(data.as_ptr().add(i + 12)), zero));
            i += 16;
        }
        while i + 4 <= data.len() {
            vst1q_f32(data.as_mut_ptr().add(i), vmaxq_f32(vld1q_f32(data.as_ptr().add(i)), zero));
            i += 4;
        }
        for j in i..data.len() { if data[j] < 0.0 { data[j] = 0.0; } }
    }

    #[inline]
    pub unsafe fn simd_add_slice(lhs: &mut [f32], rhs: &[f32]) {
        use rayon::prelude::*;
        lhs.par_chunks_mut(2048)
           .zip(rhs.par_chunks(2048))
           .for_each(|(l_chunk, r_chunk)| unsafe {
               let mut i = 0;
               while i + 16 <= l_chunk.len() {
                   vst1q_f32(l_chunk.as_mut_ptr().add(i), vaddq_f32(vld1q_f32(l_chunk.as_ptr().add(i)), vld1q_f32(r_chunk.as_ptr().add(i))));
                   vst1q_f32(l_chunk.as_mut_ptr().add(i + 4), vaddq_f32(vld1q_f32(l_chunk.as_ptr().add(i + 4)), vld1q_f32(r_chunk.as_ptr().add(i + 4))));
                   vst1q_f32(l_chunk.as_mut_ptr().add(i + 8), vaddq_f32(vld1q_f32(l_chunk.as_ptr().add(i + 8)), vld1q_f32(r_chunk.as_ptr().add(i + 8))));
                   vst1q_f32(l_chunk.as_mut_ptr().add(i + 12), vaddq_f32(vld1q_f32(l_chunk.as_ptr().add(i + 12)), vld1q_f32(r_chunk.as_ptr().add(i + 12))));
                   i += 16;
               }
               while i + 4 <= l_chunk.len() {
                   vst1q_f32(l_chunk.as_mut_ptr().add(i), vaddq_f32(vld1q_f32(l_chunk.as_ptr().add(i)), vld1q_f32(r_chunk.as_ptr().add(i))));
                   i += 4;
               }
               for j in i..l_chunk.len() { l_chunk[j] += r_chunk[j]; }
           });
    }

    #[inline]
    pub unsafe fn simd_mul_slice(lhs: &mut [f32], rhs: &[f32]) {
        use rayon::prelude::*;
        lhs.par_chunks_mut(2048)
           .zip(rhs.par_chunks(2048))
           .for_each(|(l_chunk, r_chunk)| unsafe {
               let mut i = 0;
               while i + 16 <= l_chunk.len() {
                   vst1q_f32(l_chunk.as_mut_ptr().add(i), vmulq_f32(vld1q_f32(l_chunk.as_ptr().add(i)), vld1q_f32(r_chunk.as_ptr().add(i))));
                   vst1q_f32(l_chunk.as_mut_ptr().add(i + 4), vmulq_f32(vld1q_f32(l_chunk.as_ptr().add(i + 4)), vld1q_f32(r_chunk.as_ptr().add(i + 4))));
                   vst1q_f32(l_chunk.as_mut_ptr().add(i + 8), vmulq_f32(vld1q_f32(l_chunk.as_ptr().add(i + 8)), vld1q_f32(r_chunk.as_ptr().add(i + 8))));
                   vst1q_f32(l_chunk.as_mut_ptr().add(i + 12), vmulq_f32(vld1q_f32(l_chunk.as_ptr().add(i + 12)), vld1q_f32(r_chunk.as_ptr().add(i + 12))));
                   i += 16;
               }
               while i + 4 <= l_chunk.len() {
                   vst1q_f32(l_chunk.as_mut_ptr().add(i), vmulq_f32(vld1q_f32(l_chunk.as_ptr().add(i)), vld1q_f32(r_chunk.as_ptr().add(i))));
                   i += 4;
               }
               for j in i..l_chunk.len() { l_chunk[j] *= r_chunk[j]; }
           });
    }

    #[inline]
    pub unsafe fn simd_sub_slice(lhs: &mut [f32], rhs: &[f32]) {
        use rayon::prelude::*;
        lhs.par_chunks_mut(2048)
           .zip(rhs.par_chunks(2048))
           .for_each(|(l_chunk, r_chunk)| unsafe {
               let mut i = 0;
               while i + 16 <= l_chunk.len() {
                   vst1q_f32(l_chunk.as_mut_ptr().add(i), vsubq_f32(vld1q_f32(l_chunk.as_ptr().add(i)), vld1q_f32(r_chunk.as_ptr().add(i))));
                   vst1q_f32(l_chunk.as_mut_ptr().add(i + 4), vsubq_f32(vld1q_f32(l_chunk.as_ptr().add(i + 4)), vld1q_f32(r_chunk.as_ptr().add(i + 4))));
                   vst1q_f32(l_chunk.as_mut_ptr().add(i + 8), vsubq_f32(vld1q_f32(l_chunk.as_ptr().add(i + 8)), vld1q_f32(r_chunk.as_ptr().add(i + 8))));
                   vst1q_f32(l_chunk.as_mut_ptr().add(i + 12), vsubq_f32(vld1q_f32(l_chunk.as_ptr().add(i + 12)), vld1q_f32(r_chunk.as_ptr().add(i + 12))));
                   i += 16;
               }
               while i + 4 <= l_chunk.len() {
                   vst1q_f32(l_chunk.as_mut_ptr().add(i), vsubq_f32(vld1q_f32(l_chunk.as_ptr().add(i)), vld1q_f32(r_chunk.as_ptr().add(i))));
                   i += 4;
                }
               for j in i..l_chunk.len() { l_chunk[j] -= r_chunk[j]; }
           });
    }

    #[inline]
    pub unsafe fn simd_div_slice(lhs: &mut [f32], rhs: &[f32]) {
        use rayon::prelude::*;
        lhs.par_chunks_mut(2048)
           .zip(rhs.par_chunks(2048))
           .for_each(|(l_chunk, r_chunk)| unsafe {
               let mut i = 0;
               while i + 16 <= l_chunk.len() {
                   vst1q_f32(l_chunk.as_mut_ptr().add(i), vdivq_f32(vld1q_f32(l_chunk.as_ptr().add(i)), vld1q_f32(r_chunk.as_ptr().add(i))));
                   vst1q_f32(l_chunk.as_mut_ptr().add(i + 4), vdivq_f32(vld1q_f32(l_chunk.as_ptr().add(i + 4)), vld1q_f32(r_chunk.as_ptr().add(i + 4))));
                   vst1q_f32(l_chunk.as_mut_ptr().add(i + 8), vdivq_f32(vld1q_f32(l_chunk.as_ptr().add(i + 8)), vld1q_f32(r_chunk.as_ptr().add(i + 8))));
                   vst1q_f32(l_chunk.as_mut_ptr().add(i + 12), vdivq_f32(vld1q_f32(l_chunk.as_ptr().add(i + 12)), vld1q_f32(r_chunk.as_ptr().add(i + 12))));
                   i += 16;
               }
               while i + 4 <= l_chunk.len() {
                   vst1q_f32(l_chunk.as_mut_ptr().add(i), vdivq_f32(vld1q_f32(l_chunk.as_ptr().add(i)), vld1q_f32(r_chunk.as_ptr().add(i))));
                   i += 4;
               }
               for j in i..l_chunk.len() { l_chunk[j] /= r_chunk[j]; }
           });
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// All other targets — scalar (Rayon + LLVM auto-vectorisation)
// ═════════════════════════════════════════════════════════════════════════════
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use scalar_impl::*;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod scalar_impl {
    #[inline] pub fn simd_sigmoid_slice(data: &mut [f32]) {
        for x in data.iter_mut() { *x = 1.0 / (1.0 + (-*x).exp()); }
    }
    #[inline] pub fn simd_tanh_slice(data: &mut [f32]) {
        for x in data.iter_mut() { *x = x.tanh(); }
    }
    #[inline] pub fn simd_exp_slice(data: &mut [f32]) {
        for x in data.iter_mut() { *x = x.exp(); }
    }
    #[inline] pub fn simd_log_slice(data: &mut [f32]) {
        for x in data.iter_mut() { *x = x.ln(); }
    }
    #[inline] pub fn simd_sqrt_slice(data: &mut [f32]) {
        for x in data.iter_mut() { *x = x.sqrt(); }
    }
    #[inline] pub fn simd_silu_slice(data: &mut [f32]) {
        for x in data.iter_mut() { *x = *x / (1.0 + (-*x).exp()); }
    }
    #[inline] pub fn simd_gelu_slice(data: &mut [f32]) {
        for x in data.iter_mut() {
            let inner = 0.7978845608 * (*x + 0.044715 * *x * *x * *x);
            *x = 0.5 * *x * (1.0 + inner.tanh());
        }
    }
    #[inline] pub fn simd_relu_slice(data: &mut [f32]) {
        for x in data.iter_mut() { if *x < 0.0 { *x = 0.0; } }
    }
    #[inline]
    pub fn simd_add_slice(lhs: &mut [f32], rhs: &[f32]) {
        use rayon::prelude::*;
        lhs.par_iter_mut().zip(rhs.par_iter()).for_each(|(l, r)| {
            *l += r;
        });
    }
    #[inline]
    pub fn simd_mul_slice(lhs: &mut [f32], rhs: &[f32]) {
        use rayon::prelude::*;
        lhs.par_iter_mut().zip(rhs.par_iter()).for_each(|(l, r)| {
            *l *= r;
        });
    }
    #[inline]
    pub fn simd_sub_slice(lhs: &mut [f32], rhs: &[f32]) {
        use rayon::prelude::*;
        lhs.par_iter_mut().zip(rhs.par_iter()).for_each(|(l, r)| {
            *l -= r;
        });
    }
    #[inline]
    pub fn simd_div_slice(lhs: &mut [f32], rhs: &[f32]) {
        use rayon::prelude::*;
        lhs.par_iter_mut().zip(rhs.par_iter()).for_each(|(l, r)| {
            *l /= r;
        });
    }
}
