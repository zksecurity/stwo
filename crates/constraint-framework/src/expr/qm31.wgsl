// This shader contains implementations for QM31/CM31/M31 operations.
// It is stateless, i.e. it does not contain any storage variables, and also it does not include
// any entrypoint functions, which means that it can be used as a library in other shaders.
// Note that the variable names that are used in this shader cannot be used in other shaders.
const P: u32 = 0x7FFFFFFF;  // 2^31 - 1
const MODULUS_BITS: u32 = 31u;
const HALF_BITS: u32 = 16u;
const N_ROWS: u32 = ${N_ROWS};

const N_CONSTRAINTS: u32 = ${N_CONSTRAINTS};

const N_LANES: u32 = 16;
const N_EXTENDED_ROWS: u32 = N_ROWS * 4;
const N_ORIGINAL_ROWS: u32 = N_ROWS;
const N_COLUMNS: u32 = 2;

const R: CM31 = CM31(M31(2u), M31(1u));
const ONE = QM31(CM31(M31(1u), M31(0u)), CM31(M31(0u), M31(0u)));

alias M31  = u32;
alias CM31 = vec2<u32>;
alias QM31 = vec4<u32>;

fn m31_add(a: M31, b: M31) -> M31 {
    return M31(partial_reduce(a + b));
}

fn m31_sub(a: M31, b: M31) -> M31 {
    return m31_add(a, m31_neg(b));
}

fn m31_mul(a: M31, b: M31) -> M31 {
    // Split into 16-bit parts
    let a1 = a >> HALF_BITS;
    let a0 = a & 0xFFFFu;
    let b1 = b >> HALF_BITS;
    let b0 = b & 0xFFFFu;

    // Compute partial products
    let m0 = partial_reduce(a0 * b0);
    let m1 = partial_reduce(a0 * b1);
    let m2 = partial_reduce(a1 * b0);
    let m3 = partial_reduce(a1 * b1);

    // Combine middle terms with reduction
    let mid = partial_reduce(m1 + m2);

    // Combine parts with partial reduction
    let shifted_mid = partial_reduce(mid << HALF_BITS);
    let low = partial_reduce(m0 + shifted_mid);

    let high_part = partial_reduce(m3 + (mid >> HALF_BITS));

    // Final combination using Mersenne prime property
    let result = partial_reduce(
        partial_reduce((high_part << 1u)) + 
        partial_reduce((low >> MODULUS_BITS)) + 
        partial_reduce(low & P)
    );
    return M31(result);
}

fn m31_neg(a: M31) -> M31 {
    return M31(partial_reduce(P - a));
}

fn m31_square(x: M31) -> M31 {
    return m31_mul(x, x);
}

fn m31_pow3(x: M31) -> M31 {
    let x2 = m31_square(x);
    return m31_mul(x, x2);
}

fn m31_pow5(x: M31) -> M31 {
    let x2 = m31_square(x);
    let x4 = m31_square(x2);
    return m31_mul(x4, x);
}

fn m31_pow8(x: M31) -> M31  {
    let x2 = m31_square(x);
    let x4 = m31_square(x2);
    return m31_square(x4);
}

fn m31_pow128(x: M31) -> M31 {
    let x8   = m31_pow8(x);
    let x64  = m31_pow8(x8);
    return m31_square(x64); 
}

fn m31_pow256(x: M31) -> M31 {
    let x8   = m31_pow8(x);
    let x64  = m31_pow8(x8);
    let x128  = m31_square(x64);
    return m31_square(x128);
}

fn m31_inverse(x: M31) -> M31 {
    // Computes x^(2^31-2) using the same sequence as pow2147483645
    // This is equivalent to x^(P-2) where P = 2^31-1
    
    // t0 = x^5
    let t0 = m31_pow5(x);
    
    // t1 = x^15
    let t1 = m31_pow3(t0);
    
    // t2 = x^125
    let t2 = m31_mul(m31_pow8(t1), t0);
    
    // t3 = x^255
    let t3 = m31_mul(m31_square(t2), t0);
    
    // t4 = x^65535
    let t4 = m31_mul(m31_pow256(t3), t3);
    
    // t5 = x^16777215
    let t5 = m31_mul(m31_pow256(t4), t3);
    
    // result = x^2147483520
    var result = m31_pow128(t5);
    result = m31_mul(result, t2);
    
    return result;
}

fn cm31(a0: M31, b0: M31) -> CM31 {
    return vec2<u32>(a0, b0);
}

// Complex field operations for CM31
fn cm31_add(a: CM31, b: CM31) -> CM31 {
    return vec2<u32>(
        m31_add(a.x, b.x),
        m31_add(a.y, b.y)
    );
}

fn cm31_sub(a: CM31, b: CM31) -> CM31 {
    return vec2<u32>(
        m31_sub(a.x, b.x),
        m31_sub(a.y, b.y)
    );
}

fn cm31_mul(a: CM31, b: CM31) -> CM31 {
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    let ac = m31_mul(a.x, b.x);
    let bd = m31_mul(a.y, b.y);
    let ad = m31_mul(a.x, b.y);
    let bc = m31_mul(a.y, b.x);

    return vec2<u32>(
        m31_sub(ac, bd),
        m31_add(ad, bc)
    );
}

fn cm31_neg(a: CM31) -> CM31 {
    return vec2<u32>(m31_neg(a.x), m31_neg(a.y));
}

fn cm31_square(x: CM31) -> CM31 {
    return cm31_mul(x, x);
}

fn cm31_pow5(x: CM31) -> CM31 {
    return cm31_mul(cm31_square(x), x);
}

fn cm31_inverse(x: CM31) -> CM31 {
    // 1/(a + bi) = (a - bi)/(a² + b²)
    let a2       = m31_mul(x.x, x.x);
    let b2       = m31_mul(x.y, x.y);
    let denom    = m31_add(a2, b2);
    let denomInv = m31_inverse(denom);
    return vec2<u32>(
        m31_mul(x.x, denomInv),
        m31_neg(m31_mul(x.y, denomInv))
    );
}

fn qm31(a: CM31, b: CM31) -> QM31 {
    return vec4<u32>(a.x, a.y, b.x, b.y);
}

fn qm31_4(a: M31, b: M31, c: M31, d: M31) -> QM31 {
    return vec4<u32>(a, b, c, d);
}

// Quadratic extension field operations for QM31
fn qm31_add(u: QM31, v: QM31) -> QM31 {
    let a = cm31_add(vec2<u32>(u.x, u.y), vec2<u32>(v.x, v.y));
    let b = cm31_add(vec2<u32>(u.z, u.w), vec2<u32>(v.z, v.w));
    return qm31(a, b);
}

fn qm31_sub(u: QM31, v: QM31) -> QM31 {
    let a = cm31_sub(vec2<u32>(u.x, u.y), vec2<u32>(v.x, v.y));
    let b = cm31_sub(vec2<u32>(u.z, u.w), vec2<u32>(v.z, v.w));
    return qm31(a, b);
}

fn qm31_mul(u: QM31, v: QM31) -> QM31 {
    // (a + bu)(c + du) = (ac + rbd) + (ad + bc)u
    // where r = 2 + i is the irreducible polynomial coefficient
    let ua = vec2<u32>(u.x, u.y);
    let ub = vec2<u32>(u.z, u.w);
    let va = vec2<u32>(v.x, v.y);
    let vb = vec2<u32>(v.z, v.w);

    let ac  = cm31_mul(ua, va);
    let bd  = cm31_mul(ub, vb);
    let ad  = cm31_mul(ua, vb);
    let bc  = cm31_mul(ub, va);

    // r = 2 + i
    let r      = vec2<u32>(2u, 1u);
    let rbd    = cm31_mul(r, bd);

    let real   = cm31_add(ac, rbd);
    let imag   = cm31_add(ad, bc);
    return qm31(real, imag);
}

fn qm31_neg(q: QM31) -> QM31 {
    return qm31(
        cm31_neg(vec2<u32>(q.x, q.y)),
        cm31_neg(vec2<u32>(q.z, q.w))
    );
}

fn qm31_square(q: QM31) -> QM31 {
    return qm31_mul(q, q);
}

fn qm31_pow5(q: QM31) -> QM31 {
    let q2 = qm31_square(q);
    let q4 = qm31_square(q2);
    return qm31_mul(q4, q);
}

fn qm31_inverse(q: QM31) -> QM31 {
    let a   = vec2<u32>(q.x, q.y);
    let b   = vec2<u32>(q.z, q.w);
    let b2  = cm31_square(b);
    let r   = vec2<u32>(2u, 1u);            // 2 + i
    let rb2 = cm31_mul(r, b2);
    let a2  = cm31_square(a);
    let denomInv = cm31_inverse(cm31_sub(a2, rb2));

    let neg_b = cm31_neg(b);
    return qm31(
        cm31_mul(a, denomInv),
        cm31_mul(neg_b, denomInv)
    );
}

// Utility functions
fn partial_reduce(val: u32) -> u32 {
    let reduced = val - P;
    return select(val, reduced, reduced < val);
} 

const ZERO_FRACTION: Fraction =
    Fraction(vec4<u32>(0u), vec4<u32>(1u, 0u, 0u, 0u));

struct Fraction {
    numerator  : QM31,      // vec4<u32>
    denominator: QM31,
}

// Add two fractions: (a/b + c/d) = (ad + bc)/(bd)
fn fraction_add(x: Fraction, y: Fraction) -> Fraction {
    let num = qm31_add(
        qm31_mul(x.numerator,   y.denominator),
        qm31_mul(y.numerator,   x.denominator)
    );
    let den = qm31_mul(x.denominator, y.denominator);
    return Fraction(num, den);
}

fn fraction_eq(x: Fraction, y: Fraction) -> bool {
    return all(x.numerator   == y.numerator) &&
           all(x.denominator == y.denominator);
}

/// Returns the bit reversed index of `i` which is represented by `log_size` bits.
fn bit_reverse_index(i: u32, log_size: u32) -> u32 {
    if (log_size == 0u) {
        return i;
    }
    let bits = reverse_bits_u32(i);
    return bits >> (32u - log_size);
}

fn reverse_bits_u32(x: u32) -> u32 {
    var x_mut = x;
    var result = 0u;
    
    for (var i = 0u; i < 32u; i = i + 1u) {
        result = (result << 1u) | (x_mut & 1u);
        x_mut = x_mut >> 1u;
    }
    
    return result;
}

/// Returns the index of the offset element in a bit reversed circle evaluation
/// of log size `eval_log_size` relative to a smaller domain of size `domain_log_size`.
fn offset_bit_reversed_circle_domain_index(
    i: u32,
    domain_log_size: u32,
    eval_log_size: u32,
    offset: i32,
) -> u32 {
    var prev_index = bit_reverse_index(i, eval_log_size);
    let half_size = 1u << (eval_log_size - 1u);
    let step_size = i32(1u << (eval_log_size - domain_log_size - 1u)) * offset;
    
    if (prev_index < half_size) {
        let temp = i32(prev_index) + step_size;
        // Implement rem_euclid for positive modulo
        let m = i32(half_size);
        let rem = temp % m;
        prev_index = u32(select(rem + m, rem, rem >= 0));
    } else {
        let temp = i32(prev_index - half_size) - step_size;
        // Implement rem_euclid for positive modulo
        let m = i32(half_size);
        let rem = temp % m;
        prev_index = u32(select(rem + m, rem, rem >= 0)) + half_size;
    }
    
    return bit_reverse_index(prev_index, eval_log_size);
}

fn circle_domain_index_to_coset_index(i: u32, n: u32) -> u32 {
    if (i < (n / 2u)) {
        return 2u * i;
    } else {
        return 2u * (n - 1u - i) + 1u;
    }
}

fn coset_index_to_circle_domain_index(coset_index: u32, log_domain_size: u32) -> u32 {
    if (coset_index % 2u == 0u) {
        return coset_index / 2u;
    } else {
        return ((2u << log_domain_size) - coset_index) / 2u;
    }
}