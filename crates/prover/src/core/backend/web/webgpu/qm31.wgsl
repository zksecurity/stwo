// This shader contains implementations for QM31/CM31/M31 operations.
// It is stateless, i.e. it does not contain any storage variables, and also it does not include
// any entrypoint functions, which means that it can be used as a library in other shaders.
// Note that the variable names that are used in this shader cannot be used in other shaders.
const P: u32 = 0x7FFFFFFF;  // 2^31 - 1
const MODULUS_BITS: u32 = 31u;
const HALF_BITS: u32 = 16u;

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

fn m31_square(x: M31, n: u32) -> M31 {
    var result = x;
    for (var i = 0u; i < n; i += 1u) {
        result = m31_mul(result, result);
    }
    return result;
}

fn m31_pow5(x: M31) -> M31 {
    return m31_mul(m31_square(x, 2u), x);
}

fn m31_inverse(x: M31) -> M31 {
    // Computes x^(2^31-2) using the same sequence as pow2147483645
    // This is equivalent to x^(P-2) where P = 2^31-1
    
    // t0 = x^5
    let t0 = m31_mul(m31_square(x, 2u), x);
    
    // t1 = x^15
    let t1 = m31_mul(m31_square(t0, 1u), t0);
    
    // t2 = x^125
    let t2 = m31_mul(m31_square(t1, 3u), t0);
    
    // t3 = x^255
    let t3 = m31_mul(m31_square(t2, 1u), t0);
    
    // t4 = x^65535
    let t4 = m31_mul(m31_square(t3, 8u), t3);
    
    // t5 = x^16777215
    let t5 = m31_mul(m31_square(t4, 8u), t3);
    
    // result = x^2147483520
    var result = m31_square(t5, 7u);
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